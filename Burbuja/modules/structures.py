"""
structures.py

Data structures for Burbuja.
"""

import typing

from attrs import define, field
import numpy as np

from Burbuja.modules import base

@define
class Grid():
    """
    Represents a 3D grid for mass and density calculations in a water box.

    The grid is constructed to represent the mass and density of the system
    at various points in space (a finitized version of scalar fields).
    Densities can then be used to find bubbles in the system when the
    density is below a certain threshold.
    """
    approx_grid_space: float = field(default=0.1)
    boundaries: np.ndarray = field(factory=lambda: np.zeros(3))
    density_threshold: float = field(default=base.DEFAULT_DENSITY_THRESHOLD)
    neighbor_cells: int = field(default=base.DEFAULT_NEIGHBOR_CELLS)
    grid_space_x: float = field(default=0.1)
    grid_space_y: float = field(default=0.1)
    grid_space_z: float = field(default=0.1)
    xcells: int = field(default=0)
    ycells: int = field(default=0)
    zcells: int = field(default=0)
    mass_array: typing.Any = field(factory=lambda: np.zeros(0))
    densities: typing.Any = field(factory=lambda: np.zeros(0))
    total_system_volume: float = field(default=0.0)
    
    def initialize_cells(
            self,
            use_cupy: bool = False,
            use_float32: bool = False
            ) -> None:
        """
        Initialize the grid cell arrays for mass and density.

        Assigns the number of cells in each direction based on the box
        boundaries and the approximate grid spacing. Initializes the
        mass_array and densities arrays to zero.

        Args:
            use_cupy (bool, optional): Use CuPy arrays for GPU. Default is False.
            use_float32 (bool, optional): Use float32 precision. Default is False.

        Returns:
            None
        """
        L_x, L_y, L_z = self.boundaries[:]
        self.xcells = int((L_x + self.approx_grid_space) / self.approx_grid_space)
        self.ycells = int((L_y + self.approx_grid_space) / self.approx_grid_space)
        self.zcells = int((L_z + self.approx_grid_space) / self.approx_grid_space)
        # Now choose the actual grid space based on grid lengths and number of cells
        # in each direction
        self.grid_space_x = L_x / (self.xcells - 1)
        self.grid_space_y = L_y / (self.ycells - 1)
        self.grid_space_z = L_z / (self.zcells - 1)
        total_coordinates = self.xcells * self.ycells * self.zcells
        self.total_system_volume = L_x * L_y * L_z
        # Use float32 for CPU if requested (for precision comparison testing)
        dtype = np.float32 if use_float32 else np.float64
        self.mass_array = np.zeros(total_coordinates, dtype=dtype)
        self.densities = np.zeros(total_coordinates, dtype=dtype)
        return

    def calculate_cell_masses(
            self,
            coordinates: np.ndarray,
            mass_list: list,
            n_atoms: int,
            frame_id: int = 0,
            chunk_size: int = 1000,
            use_cupy: bool = False,
            use_float32: bool = False
            ) -> None:
        """
        Calculate the mass contained within each cell of the grid.

        Loops over all atoms and assigns their mass to the appropriate
        grid cell, using either CPU or GPU arrays.

        Args:
            coordinates (np.ndarray): Atomic coordinates, shape (n_frames, n_atoms, 3).
            mass_list (list): List of atomic masses.
            n_atoms (int): Number of atoms in the frame.
            frame_id (int, optional): Frame index. Default is 0.
            chunk_size (int, optional): Number of atoms per chunk. Default is 1000.
            use_cupy (bool, optional): Use CuPy arrays for GPU. Default is False.
            use_float32 (bool, optional): Use float32 precision. Default is False.

        Returns:
            None
        """
        if use_cupy:
            import cupy as cp

        xcells, ycells, zcells = self.xcells, self.ycells, self.zcells
        for start in range(0, n_atoms, chunk_size):
            end = min(start + chunk_size, n_atoms)
            coords_batch = coordinates[frame_id, start:end, :]
            mass_slice = mass_list[start:end]
                
            masses_batch = np.array(mass_slice, dtype=np.float32)
            # Use float32 for CPU if requested
            dtype = np.float32 if use_float32 else np.float64
            coords = coords_batch.astype(dtype)
            masses = masses_batch.astype(dtype)

            # Grid coordinates per atom
            grid_coords = np.zeros((end - start, 3), dtype=np.int32)
            grid_coords[:, 0] = np.floor(coords[:,0] / self.grid_space_x).astype(np.int32)
            grid_coords[:, 1] = np.floor(coords[:,1] / self.grid_space_y).astype(np.int32)
            grid_coords[:, 2] = np.floor(coords[:,2] / self.grid_space_z).astype(np.int32)

            xi, yi, zi = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]
            all_indices_cpu = np.ones(end - start, dtype=bool)  # Treat all atoms the same for now
            all_indices = all_indices_cpu
            if True:
                xi_w = xi[all_indices] #% xcells
                yi_w = yi[all_indices] #% ycells
                zi_w = zi[all_indices] #% zcells
                mw = masses[all_indices]
                # An assertion error here indicates a failure in box wrapping.
                assert (xi_w >= 0).all(), "xi_w contains negative indices"
                assert (yi_w >= 0).all(), "yi_w contains negative indices"
                assert (zi_w >= 0).all(), "zi_w contains negative indices"
                assert (xi_w < xcells).all(), "xi_w contains indices >= xcells"
                assert (yi_w < ycells).all(), "yi_w contains indices >= ycells"
                assert (zi_w < zcells).all(), "zi_w contains indices >= zcells"
                ids = xi_w * ycells * zcells + yi_w * zcells + zi_w
                np.add.at(self.mass_array, ids, mw)

        return

    def calculate_densities(
            self,
            unitcell_vectors,
            frame_id: int = 0,
            chunk_size: int = 1000,
            use_cupy: bool = False,
            use_float32: bool = False
            ) -> None:
        """
        Calculate the densities in each cell of the grid.

        Uses neighbor averaging to compute the density at each grid cell.
        Supports both CPU and GPU (CuPy) acceleration.

        Args:
            unitcell_vectors (np.ndarray): Unit cell vectors for the frame.
            frame_id (int, optional): Frame index. Default is 0.
            chunk_size (int, optional): Number of cells per chunk. Default is 1000.
            use_cupy (bool, optional): Use CuPy arrays for GPU. Default is False.
            use_float32 (bool, optional): Use float32 precision. Default is False.

        Returns:
            None
        """
        if use_cupy:
            import cupy as cp
            # Use cupy functions throughout
            array_lib = cp
            # Larger chunk sizes for GPU efficiency
            chunk_size = max(chunk_size, 10000)  # Minimum 10k for GPU
        else:
            array_lib = np
            
        grid_space_mean = np.mean([self.grid_space_x, self.grid_space_y, self.grid_space_z])    
        n_cells_to_spread = int(self.neighbor_cells * round(0.1 / grid_space_mean))
        
        xcells, ycells, zcells = self.xcells, self.ycells, self.zcells
        grid_shape = (xcells, ycells, zcells)
        N = xcells * ycells * zcells
        
        # Transfer mass array to GPU if using CuPy
        if use_cupy:
            grid_shape_array = cp.asarray(grid_shape, dtype=cp.int32)
        else:
            grid_shape_array = np.array(grid_shape)
        
        mass_array = self.mass_array
        # Use float32 for CPU if requested (for precision comparison testing)
        dtype = np.float32 if use_float32 else np.float64
        self.densities = np.zeros(N, dtype=dtype)
        mass_grid = mass_array.reshape(grid_shape)
        
        # Pre-compute neighbor offsets (once, on appropriate device)
        neighbor_range = array_lib.arange(-n_cells_to_spread, n_cells_to_spread + 1)
        dx, dy, dz = array_lib.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
        neighbor_offsets_box = array_lib.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
        neighbor_offsets_dist = array_lib.linalg.norm(neighbor_offsets_box, axis=1)
        
        neighbor_offsets_within_dist = neighbor_offsets_dist <= n_cells_to_spread
        neighbor_offsets = neighbor_offsets_box[neighbor_offsets_within_dist]
        M = neighbor_offsets.shape[0]
        
        # Get image offsets once
        if use_cupy:
            boundaries_gpu = cp.asarray(self.boundaries)
            image_offsets = base.get_periodic_image_offsets(unitcell_vectors, boundaries_gpu, grid_shape_array,
                                                            frame_id=frame_id, use_cupy=use_cupy)
        else:
            image_offsets = base.get_periodic_image_offsets(unitcell_vectors, self.boundaries, grid_shape_array,
                                                            frame_id=frame_id, use_cupy=use_cupy)
        # Calculate volume once
        volume = M * 1000.0 * self.grid_space_x * self.grid_space_y * self.grid_space_z
        
        # Estimate memory usage and adjust chunk size for GPU safety
        if use_cupy:
            estimated_memory_bytes = chunk_size * M * 4 * 3  # float32, 3 coords
            available_memory = cp.cuda.Device().mem_info[0]  # Available GPU memory
            
            if estimated_memory_bytes > available_memory * 0.6:  # Use 60% safety margin
                # Reduce chunk size if needed
                safe_chunk_size = int(available_memory * 0.6 / (M * 4 * 3))
                chunk_size = min(chunk_size, max(1000, safe_chunk_size))
        
        # Process in larger chunks for GPU efficiency
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            current_chunk_size = end - start
            
            # Generate all coordinates for this chunk at once
            global_indices = array_lib.arange(start, end)
            
            # Convert 1D indices to 3D coordinates efficiently (vectorized)
            coords = array_lib.empty((current_chunk_size, 3), dtype=array_lib.int32)
            coords[:, 0] = global_indices // (ycells * zcells)  # ix
            temp = global_indices % (ycells * zcells)
            coords[:, 1] = temp // zcells  # iy
            coords[:, 2] = temp % zcells   # iz
            
            # Expand coordinates with ALL neighbor offsets at once (key optimization)
            # Shape: (current_chunk_size, M, 3)
            coords_exp = coords[:, None, :] + neighbor_offsets[None, :, :]
            
            # Apply periodic boundary conditions (vectorized operations)
            # Handle z-direction
            out_of_bounds_z_lower = coords_exp[:, :, 2] < 0
            coords_exp[:, :, 0] += out_of_bounds_z_lower * image_offsets[0, 2]
            coords_exp[:, :, 1] += out_of_bounds_z_lower * image_offsets[1, 2]
            coords_exp[:, :, 2] += out_of_bounds_z_lower * image_offsets[2, 2]
            
            out_of_bounds_z_higher = coords_exp[:, :, 2] >= zcells
            coords_exp[:, :, 0] -= out_of_bounds_z_higher * image_offsets[0, 2]
            coords_exp[:, :, 1] -= out_of_bounds_z_higher * image_offsets[1, 2]
            coords_exp[:, :, 2] -= out_of_bounds_z_higher * image_offsets[2, 2]
            
            # Handle y-direction
            out_of_bounds_y_lower = coords_exp[:, :, 1] < 0
            coords_exp[:, :, 0] += out_of_bounds_y_lower * image_offsets[0, 1]
            coords_exp[:, :, 1] += out_of_bounds_y_lower * image_offsets[1, 1]
            
            out_of_bounds_y_higher = coords_exp[:, :, 1] >= ycells
            coords_exp[:, :, 0] -= out_of_bounds_y_higher * image_offsets[0, 1]
            coords_exp[:, :, 1] -= out_of_bounds_y_higher * image_offsets[1, 1]
            
            # Handle x-direction
            out_of_bounds_x_lower = coords_exp[:, :, 0] < 0
            coords_exp[:, :, 0] += out_of_bounds_x_lower * image_offsets[0, 0]
            
            out_of_bounds_x_higher = coords_exp[:, :, 0] >= xcells
            coords_exp[:, :, 0] -= out_of_bounds_x_higher * image_offsets[0, 0]
            
            # Validate coordinates (can be disabled in production for speed)
            if __debug__:
                assert (coords_exp[:, :, 0] >= 0).all()
                assert (coords_exp[:, :, 1] >= 0).all()
                assert (coords_exp[:, :, 2] >= 0).all()
                assert (coords_exp[:, :, 0] < xcells).all()
                assert (coords_exp[:, :, 1] < ycells).all()
                assert (coords_exp[:, :, 2] < zcells).all()
            
            # Extract neighbor masses and calculate densities (fully vectorized)
            if use_cupy:
                xi, yi, zi = cp.asnumpy(coords_exp[:, :, 0]), cp.asnumpy(coords_exp[:, :, 1]), cp.asnumpy(coords_exp[:, :, 2])
                neighbor_masses = cp.asarray(mass_grid[xi, yi, zi])
            else:
                xi, yi, zi = coords_exp[:, :, 0], coords_exp[:, :, 1], coords_exp[:, :, 2]
                neighbor_masses = mass_grid[xi, yi, zi]
            total_mass = array_lib.sum(neighbor_masses, axis=1)
            
            # Calculate densities for entire chunk
            # This converts the densities to units of g/L (1.66 factor)
            chunk_densities = total_mass / volume * 1.66
            
            # Store results
            if use_cupy:
                self.densities[start:end] = cp.asnumpy(chunk_densities)
            else:
                self.densities[start:end] = chunk_densities

            # Clean up large intermediate arrays to prevent memory buildup
            del coords_exp, xi, yi, zi, neighbor_masses, total_mass, chunk_densities, coords
            
            # Force garbage collection for GPU memory
            if use_cupy:
                cp.get_default_memory_pool().free_all_blocks()
        del self.mass_array

    def generate_bubble_object(
            self,
            use_cupy: bool = False,
            use_float32: bool = True
            ) -> "Bubble":
        """
        Generate a Bubble object from the grid densities data.

        Also prepares a DX file header for later output.

        Args:
            use_cupy (bool, optional): Use CuPy arrays for GPU. Default is False.
            use_float32 (bool, optional): Use float32 precision. Default is False.

        Returns:
            Bubble: The resulting Bubble object.
        """
        bubble_atoms = Bubble()
        bubble_atoms.density_threshold = self.density_threshold
        bubble_atoms.find(self.xcells, self.ycells, self.zcells, 
                          self.densities, grid_space_x=self.grid_space_x,
                          grid_space_y=self.grid_space_y,
                          grid_space_z=self.grid_space_z,
                          use_cupy=use_cupy, use_float32=use_float32)
        bubble_atoms.dx_header = self.make_dx_header()
        bubble_atoms.total_system_volume = self.total_system_volume
        return bubble_atoms
    
    def make_dx_header(self) -> dict:
        """
        Prepare the header information for a DX file.

        Returns:
            dict: DX file header information.
        """
        header = {}
        header["width"] = self.xcells
        header["height"] = self.ycells
        header["depth"] = self.zcells
        header["originx"] = 5.0 * self.grid_space_x
        header["originy"] = 5.0 * self.grid_space_y
        header["originz"] = 5.0 * self.grid_space_z
        header["resx"] = self.grid_space_x * 10.0
        header["resy"] = self.grid_space_y * 10.0
        header["resz"] = self.grid_space_z * 10.0
        return header
    
    def write_masses_dx(
            self,
            filename: str
            ) -> None:
        """
        Write the mass data to a DX file.

        Args:
            filename (str): Output filename.

        Returns:
            None
        """
        mass_grid = self.mass_array.reshape(self.xcells, self.ycells, self.zcells)
        base.write_data_array(self.make_dx_header(), mass_grid, filename)
        return

@define
class Bubble():
    """
    Represents a detected bubble or void region in a frame.

    Stores the coordinates and grid mask of bubble regions, and provides
    methods to write them to PDB or DX files for visualization.
    """
    atoms: dict = field(factory=dict)
    total_residues: int = field(default=1)
    total_atoms: int = field(default=0)
    total_bubble_volume: float = field(default=0.0)
    total_system_volume: float = field(default=0.0)
    densities: np.ndarray | None = None
    bubble_data: np.ndarray | None = None
    dx_header: str = field(default="")
    density_threshold: float = field(default=base.DEFAULT_DENSITY_THRESHOLD)

    def find(self,
         xcells: int,
         ycells: int,
         zcells: int,
         box_densities: np.ndarray,
         grid_space_x: float,
         grid_space_y: float,
         grid_space_z: float,
         use_cupy: bool = False,
         use_float32: bool = False
         ) -> None:
        """
        Identify bubble regions where density is below the threshold.

        Populates the bubble_data mask and atom coordinates for the bubble.
        Supports both CPU and GPU processing.

        Args:
            xcells (int): Number of grid cells in x direction.
            ycells (int): Number of grid cells in y direction.
            zcells (int): Number of grid cells in z direction.
            box_densities (np.ndarray): Flattened density array.
            grid_space_x (float): Grid spacing in x.
            grid_space_y (float): Grid spacing in y.
            grid_space_z (float): Grid spacing in z.
            use_cupy (bool, optional): Use CuPy arrays for GPU. Default is False.
            use_float32 (bool, optional): Use float32 precision. Default is False.

        Returns:
            None
        """
        # if use_cupy:
        #     import cupy as cp
        #     array_lib = cp
        #     # Ensure densities are on GPU
        #     if isinstance(box_densities, np.ndarray):
        #         box_densities = cp.asarray(box_densities)
        # else:
        array_lib = np
        # Ensure densities are on CPU
        if hasattr(box_densities, 'get'):  # Check if it's a CuPy array
            box_densities = box_densities.get()

        print(1)
        # Set precision for calculations
        float_dtype = np.float32 if use_float32 else np.float64

        print(2)
        # Reshape densities to 3D grid
        self.densities = box_densities.reshape((xcells, ycells, zcells))

        print(3)
        # Create bubble mask (vectorized operation)
        bubble_mask = box_densities < self.density_threshold

        print(4)
        self.total_atoms = int(array_lib.sum(bubble_mask))
        del box_densities  # Free memory

        if self.total_atoms == 0:
            # No bubbles found
            print(5)
            self.bubble_data = array_lib.zeros((xcells, ycells, zcells), dtype=bool)
            self.atoms = {}
            self.total_bubble_volume = 0.0
            return

        print(6)
        # Get indices of bubble cells (vectorized)
        bubble_indices = array_lib.where(bubble_mask)[0]
        print(6.5)

        del bubble_mask  # Free memory

        # Convert 1D indices to 3D coordinates (vectorized)
        iz = bubble_indices % zcells
        print(6.55)
        temp = bubble_indices // zcells
        print(6.56)

        del bubble_indices  # Free memory

        iy = temp % ycells
        print(6.57)
        ix = temp // ycells

        print(6.6)

        # Calculate physical coordinates (vectorized) with controlled precision
        # if use_cupy:
        #     # GPU calculations always use float32
        #     x_coords = ix * grid_space_x + grid_space_x/2
        #     y_coords = iy * grid_space_y + grid_space_y/2
        #     z_coords = iz * grid_space_z + grid_space_z/2
        # else:
        # CPU calculations use specified precision
        x_coords = (ix * float_dtype(grid_space_x) + float_dtype(grid_space_x) / 2).astype(float_dtype)
        print(6.7)
        y_coords = (iy * float_dtype(grid_space_y) + float_dtype(grid_space_y) / 2).astype(float_dtype)
        print(6.8)
        z_coords = (iz * float_dtype(grid_space_z) + float_dtype(grid_space_z) / 2).astype(float_dtype)

        # Create bubble_data array
        print(7)
        self.bubble_data = array_lib.zeros((xcells, ycells, zcells), dtype=bool)
        # if use_cupy:
        #     # Use advanced indexing for GPU
        #     self.bubble_data[ix, iy, iz] = True
        #     # Transfer coordinates to CPU for PDB writing
        #     x_coords_cpu = cp.asnumpy(x_coords)
        #     y_coords_cpu = cp.asnumpy(y_coords)
        #     z_coords_cpu = cp.asnumpy(z_coords)
        # else:
        print(8)
        self.bubble_data[ix, iy, iz] = True
        print(9)
        x_coords_cpu = x_coords
        y_coords_cpu = y_coords
        z_coords_cpu = z_coords
        print(10)

        # Generate PDB atom records (this part stays on CPU since it's string formatting)
        self.atoms = {}
        for i in range(self.total_atoms):
            atom_id = i + 1
            residue_id = self.total_residues
            x, y, z = x_coords_cpu[i], y_coords_cpu[i], z_coords_cpu[i]

            atom_pdb = "ATOM {:>6s}  BUB BUB  {:>4s}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00\n".format(
                str(atom_id), str(residue_id), x, y, z
            )
            self.atoms[atom_id] = atom_pdb
        print(11)

        # Calculate total bubble volume with controlled precision
        if use_float32:
            print(12)
            volume_per_cell = float_dtype(grid_space_x) * float_dtype(grid_space_y) * float_dtype(grid_space_z)
            print(13)
            self.total_bubble_volume = float(float_dtype(self.total_atoms) * volume_per_cell)
        else:
            print(14)
            self.total_bubble_volume = self.total_atoms * grid_space_x * grid_space_y * grid_space_z

        # Ensure final arrays are in expected format (CPU NumPy for compatibility)
        # if use_cupy:
        #     self.densities = cp.asnumpy(self.densities)
        #     self.bubble_data = cp.asnumpy(self.bubble_data)


    def write_pdb(
            self,
            filename: str
        ) -> None:
        """
        Write the bubble atom coordinates to a PDB file.

        Args:
            filename (str): Output filename.

        Returns:
            None
        """
        with open(filename, "w") as pdb:
            for key in self.atoms:
                pdb.write(self.atoms[key])
                pdb.write("TER\n")
            pdb.write("END\n")

    def write_densities_dx(
            self,
            filename: str
        ) -> None:
        """
        Write the density grid to a DX file.

        Args:
            filename (str): Output filename.

        Returns:
            None
        """
        base.write_data_array(self.dx_header, self.densities, filename)

    def write_bubble_dx(
            self,
            filename: str
        ) -> None:
        """
        Write the bubble mask to a DX file.

        Args:
            filename (str): Output filename.

        Returns:
            None
        """
        base.write_data_array(self.dx_header, self.bubble_data, filename)
