"""
structures.py

Data structure for Burbuja.
"""

import typing

from attrs import define, field
import numpy as np
import mdtraj

from Burbuja.modules import base

@define
class Grid():
    
    """
    A grid in in the shape of the wrapped water box (rectangular prism), that is
    constructed to represent the mass and density of the system at various
    points in space (a finitized version of scalar fields). The densities
    can then be used to find bubbles in the system when the density is below
    a certain threshold.
    """
    approx_coarse_grid_space: float = field(default=1.0)
    approx_fine_grid_space: float = field(default=0.1)
    boundaries: np.ndarray = field(factory=lambda: np.zeros(3))
    grid_coarse_space_x: float = field(default=1.0)
    grid_coarse_space_y: float = field(default=1.0)
    grid_coarse_space_z: float = field(default=1.0)
    grid_fine_space_x: float = field(default=0.1)
    grid_fine_space_y: float = field(default=0.1)
    grid_fine_space_z: float = field(default=0.1)
    coarse_xcells: int = field(default=0)
    coarse_ycells: int = field(default=0)
    coarse_zcells: int = field(default=0)
    fine_xcells: int = field(default=0)
    fine_ycells: int = field(default=0)
    fine_zcells: int = field(default=0)
    coarse_mass_array: typing.Any = field(factory=lambda: np.zeros(0))
    coarse_densities: typing.Any = field(factory=lambda: np.zeros(0))
    fine_mass_array: typing.Any = field(factory=lambda: np.zeros(0))
    fine_densities: typing.Any = field(factory=lambda: np.zeros(0))
    atom_coords: typing.Any = field(factory=lambda: np.zeros(0))
    atom_masses: typing.Any = field(factory=lambda: np.zeros(0))
    atoms_stored_per_cell: typing.Any = field(factory=lambda: np.zeros(0))
    coarse_cell_indexes: typing.List[int] = field(factory=list)
    
    def initialize_coarse_cells(
            self, 
            use_cupy=False,
            ) -> None:
        """
        Assign the number of cells in each direction based on the
        boundaries of the box and the approximate grid space.
        The grid space is then calculated based on the number of cells
        and the boundaries of the box.
        The mass_array and densities are initialized to zero - and are
        1D arrays (flattened 3D values).
        """
        L_x, L_y, L_z = self.boundaries[:]
        self.coarse_xcells = int(L_x / self.approx_coarse_grid_space)
        self.coarse_ycells = int(L_y / self.approx_coarse_grid_space)
        self.coarse_zcells = int(L_z / self.approx_coarse_grid_space)
        
        # Now choose the actual grid space based on grid lengths and number of cells
        # in each direction
        self.grid_coarse_space_x = L_x / self.coarse_xcells
        self.grid_coarse_space_y = L_y / self.coarse_ycells
        self.grid_coarse_space_z = L_z / self.coarse_zcells
        total_coordinates = self.coarse_xcells * self.coarse_ycells * self.coarse_zcells


        if use_cupy:
            import cupy as cp
            self.coarse_mass_array = cp.zeros(total_coordinates, dtype=cp.float32)
            self.coarse_densities = cp.zeros(total_coordinates, dtype=cp.float32)

        else:
            self.coarse_mass_array = np.zeros(total_coordinates)
            self.coarse_densities = np.zeros(total_coordinates)

        return
    

    def initialize_fine_cells_from_coarse_cells(
            self,
            use_cupy=False,
            ) -> None:
        """
        Assign the number of cells in each direction based on the
        boundaries of the coarse cells and the approximate grid space.
        The grid space is then calculated based on the number of cells
        and the boundaries of the box.
        The mass_array and densities are initialized to zero - and are
        1D arrays (flattened 3D values).
        """



        L_x, L_y, L_z = self.grid_coarse_space_x, self.grid_coarse_space_y, self.grid_coarse_space_z
        self.fine_xcells = int(L_x / self.approx_fine_grid_space)
        self.fine_ycells = int(L_y / self.approx_fine_grid_space)
        self.fine_zcells = int(L_z / self.approx_fine_grid_space)
        # Now choose the actual grid space based on grid lengths and number of cells
        # in each direction
        self.grid_fine_space_x = L_x / (self.fine_xcells)
        self.grid_fine_space_y = L_y / (self.fine_ycells)
        self.grid_fine_space_z = L_z / (self.fine_zcells)

        block_size = self.fine_xcells * self.fine_ycells * self.fine_zcells
        superblock_size = block_size * 27

        # If all coarse_cells have the same number of masses, allocate 2D arrays
        if not self.coarse_cell_indexes:
            raise ValueError("coarse_cells must not be empty")
        num_cells = len(self.coarse_cell_indexes)
        if use_cupy:
            import cupy as cp
            self.fine_mass_array = cp.zeros((num_cells, superblock_size), dtype=cp.float32)
            self.fine_densities = cp.zeros((num_cells, block_size), dtype=cp.float32)
        else:
            self.fine_mass_array = np.zeros((num_cells, superblock_size), dtype=np.float32)
            self.fine_densities = np.zeros((num_cells, block_size), dtype=np.float32)



    def apply_boundaries_to_protein(
            self, 
            structure: mdtraj.Trajectory,
            ) -> None:

        """
        Wrap all atoms within the boundaries of the box.
        """
        # TODO: don't use this! It's wrong - use instead a procedure like
        #  base.reshape_atoms_to_orthorombic() or see if this method can be
        #  left out entirely.
        L_x, L_y, L_z = self.boundaries[:]
        for i in range(structure.n_frames):
            for j in range(structure.n_atoms):
                while structure.xyz[i,j,0] > L_x:
                    structure.xyz[i,j,0] -= L_x
                while structure.xyz[i,j,0] < 0:
                    structure.xyz[i,j,0] += L_x

                while structure.xyz[i,j,1] > L_y:
                    structure.xyz[i,j,1] -= L_y
                while structure.xyz[i,j,1] < 0:
                    structure.xyz[i,j,1] += L_y

                while structure.xyz[i,j,2] > L_z:
                    structure.xyz[i,j,2] -= L_z
                while structure.xyz[i,j,2] < 0:
                    structure.xyz[i,j,2] += L_z
        return

    def calculate_coarse_cell_masses(
            self,
            coordinates: np.ndarray,
            mass_list: list,
            n_atoms: int,
            frame_id: int = 0,
            chunk_size: int = 5000,
            use_cupy: bool = False,
            store_atomic_information: bool = True
        ) -> None:
        """
        Calculate the mass contained within each cell of the grid, optionally storing atomic info.
        Uses CPU (NumPy) or GPU (CuPy + RawKernel) based on use_cupy.
        """
        # Coarse grid dimensions and total cells
        xcells, ycells, zcells = self.coarse_xcells, self.coarse_ycells, self.coarse_zcells
        n_cells = xcells * ycells * zcells

        # Estimate maximum atoms per coarse cell (upper bound)
        atoms_per_cell_allocated = int(
            200 * self.grid_coarse_space_x *
            self.grid_coarse_space_y * self.grid_coarse_space_z
        )

        if use_cupy:
            import cupy as cp
            # Allocate GPU arrays for coarse mass and, if needed, atomic details
            self.coarse_mass_array = cp.zeros(n_cells, dtype=cp.float32)
            if store_atomic_information:
                self.atom_coords = cp.zeros(
                    (n_cells, atoms_per_cell_allocated, 3), dtype=cp.float32
                )
                self.atom_masses = cp.zeros(
                    (n_cells, atoms_per_cell_allocated), dtype=cp.float32
                )
                self.atoms_stored_per_cell = cp.zeros(n_cells, dtype=cp.int32)

            # Define CUDA kernel for parallel atom insertion
            insert_atoms_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void insert_atoms(
                const float* coords, const float* masses,
                const int* ids, int* insert_ptr,
                float* atom_coords, float* atom_masses,
                const int max_atoms, const int n_atoms
            ){
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                if (i >= n_atoms) return;

                int cell_id = ids[i];
                int idx = atomicAdd(&insert_ptr[cell_id], 1);
                if (idx >= max_atoms) return;

                int base = (cell_id * max_atoms + idx) * 3;
                atom_coords[base + 0] = coords[3*i + 0];
                atom_coords[base + 1] = coords[3*i + 1];
                atom_coords[base + 2] = coords[3*i + 2];
                atom_masses[cell_id * max_atoms + idx] = masses[i];
            }
            ''', 'insert_atoms')
        else:
            # Allocate CPU arrays for coarse mass and, if needed, atomic details
            self.coarse_mass_array = np.zeros(n_cells, dtype=np.float32)
            if store_atomic_information:
                self.atom_coords = np.zeros(
                    (n_cells, atoms_per_cell_allocated, 3), dtype=np.float32
                )
                self.atom_masses = np.zeros(
                    (n_cells, atoms_per_cell_allocated), dtype=np.float32
                )
                self.atoms_stored_per_cell = np.zeros(n_cells, dtype=np.int32)

        # Process atoms in batches to limit memory usage
        for start in range(0, n_atoms, chunk_size):
            end = min(start + chunk_size, n_atoms)

            # Extract and cast coordinates and masses
            coords_batch = coordinates[frame_id, start:end, :].astype(np.float32)
            masses_batch = np.array(mass_list[start:end], dtype=np.float32)

            if use_cupy:
                #cp = __import__('cupy')
                coords = cp.asarray(coords_batch)
                masses = cp.asarray(masses_batch)
                mass_array = self.coarse_mass_array

                # build a 1×3 array of the cell‐spacings
                spacings = cp.array([
                    self.grid_coarse_space_x,
                    self.grid_coarse_space_y,
                    self.grid_coarse_space_z
                ], dtype=cp.float32)
                # coords is (n,3), spacings is (3,) → broadcast to (n,3)
                grid_coords = cp.floor(coords / spacings).astype(cp.int32)

            else:
                coords = coords_batch
                masses = masses_batch
                mass_array = self.coarse_mass_array

                # build a 1×3 array of the cell‐spacings
                spacings = np.array([
                    self.grid_coarse_space_x,
                    self.grid_coarse_space_y,
                    self.grid_coarse_space_z
                ], dtype=np.float32)
                # coords is (n,3), spacings is (3,) → broadcast to (n,3)
                grid_coords = np.floor(coords / spacings).astype(np.int32)

            xi, yi, zi = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]
            ids = xi * (ycells * zcells) + yi * zcells + zi

            if use_cupy:
                # Accumulate mass per cell
                cp.add.at(mass_array, ids, masses)
                # Store atomic details if requested
                if store_atomic_information:
                    # Launch CUDA kernel for parallel insertion on GPU
                    n_chunk = ids.size
                    threads_per_block = 256
                    blocks = (n_chunk + threads_per_block - 1) // threads_per_block

                    insert_atoms_kernel(
                        (blocks,), (threads_per_block,),
                        (
                            coords.ravel(), masses, ids,
                            self.atoms_stored_per_cell,
                            self.atom_coords.ravel(), self.atom_masses.ravel(),
                            atoms_per_cell_allocated, n_chunk
                        )
                    )
            else:
                # Accumulate mass per cell
                np.add.at(mass_array, ids, masses)
                # Store atomic details if requested
                if store_atomic_information:
                    # Fallback to Python loop on CPU
                    for i, cell_id in enumerate(ids):
                        idx = self.atoms_stored_per_cell[cell_id]
                        self.atom_coords[cell_id, idx, :] = coords[i]
                        self.atom_masses[cell_id, idx] = masses[i]
                        self.atoms_stored_per_cell[cell_id] += 1

    
        
    def calculate_fine_cell_masses_from_coarse_cells(
            self,
            box_lengths: np.ndarray,
            use_cupy: bool = False,
            ) -> None:
        """
        Calculate the mass contained within each cell of the grid.
        """
        if use_cupy:
            import cupy as cp

        

        #xcells, ycells, zcells = self.fine_xcells, self.fine_ycells, self.fine_zcells

        #n_cells = xcells * ycells * zcells

        # For each coarse cell index, store neighbor masses in a superblock of fine cells (33x33x33)
        grid_shape = (self.coarse_xcells, self.coarse_ycells, self.coarse_zcells)
        xcells, ycells, zcells = grid_shape
        fine_xcells, fine_ycells, fine_zcells = self.fine_xcells, self.fine_ycells, self.fine_zcells
        superblock_x, superblock_y, superblock_z = 3 * fine_xcells, 3 * fine_ycells, 3 * fine_zcells
        block_size = superblock_x * superblock_y * superblock_z
        neighbor_range = np.array([-1, 0, 1])
        dx, dy, dz = np.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
        neighbor_offsets = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
        # For each coarse cell index
        coarse_indices = np.array(self.coarse_cell_indexes)
        ix = coarse_indices // (ycells * zcells)
        iy = (coarse_indices % (ycells * zcells)) // zcells
        iz = coarse_indices % zcells
        for row, (iix, iiy, iiz) in enumerate(zip(ix, iy, iz)):
            # Find the min corner of the 3x3x3 coarse neighborhood
            min_cx = (iix - 1) % xcells
            min_cy = (iiy - 1) % ycells
            min_cz = (iiz - 1) % zcells
            # Get all 27 coarse cell indices in the neighborhood
            n_cx = (iix + neighbor_offsets[:, 0]) % xcells
            n_cy = (iiy + neighbor_offsets[:, 1]) % ycells
            n_cz = (iiz + neighbor_offsets[:, 2]) % zcells
            neighbor_idxs = n_cx * ycells * zcells + n_cy * zcells + n_cz
            all_idxs = np.unique(np.concatenate([[coarse_indices[row]], neighbor_idxs]))
            # Collect all atom coords and masses in the neighborhood
            coarse_cell_atom_coords = [self.atom_coords[i] for i in all_idxs]
            coarse_cell_atom_masses = [self.atom_masses[i] for i in all_idxs]
            coords = np.concatenate(coarse_cell_atom_coords, axis=0)
            masses = np.concatenate(coarse_cell_atom_masses, axis=0)
            # Wrap coordinates into the box
            coords = coords % box_lengths
            # Compute fine cell indices relative to superblock origin
            superblock_origin = np.array([min_cx * self.grid_coarse_space_x,
                                         min_cy * self.grid_coarse_space_y,
                                         min_cz * self.grid_coarse_space_z])
            rel_coords = coords - superblock_origin
            # Bin into fine cells
            fx = np.floor(rel_coords[:, 0] / self.grid_fine_space_x).astype(int)
            fy = np.floor(rel_coords[:, 1] / self.grid_fine_space_y).astype(int)
            fz = np.floor(rel_coords[:, 2] / self.grid_fine_space_z).astype(int)
            # Only keep atoms that fall within the superblock
            valid = (fx >= 0) & (fx < superblock_x) & (fy >= 0) & (fy < superblock_y) & (fz >= 0) & (fz < superblock_z)
            fx, fy, fz, masses = fx[valid], fy[valid], fz[valid], masses[valid]
            fine_ids = fx * superblock_y * superblock_z + fy * superblock_z + fz
            # Fill fine_mass_array
            if use_cupy:
                import cupy as cp
                self.fine_mass_array[row, :] = cp.zeros(block_size, dtype=cp.float32)
                cp.add.at(self.fine_mass_array[row, :], fine_ids, masses)
            else:
                self.fine_mass_array[row, :] = np.zeros(block_size, dtype=np.float32)
                np.add.at(self.fine_mass_array[row, :], fine_ids, masses)
        # self.fine_mass_array now has shape (num_coarse_indices, 35937) and is filled correctly
        return

    def calculate_fine_cell_densities_neighboring(
            self,
            use_cupy: bool = False
            ) -> None:
        """
        Calculate the densities in each cell of the grid, optionally using CuPy.
        Note that the densities are calculated by considering neighboring cells.
        """
        if use_cupy:
            import cupy as cp
        
        # Now choose the actual grid space based on grid lengths and number of cells in each direction
        grid_space_mean = np.mean([self.grid_fine_space_x, self.grid_fine_space_y, self.grid_fine_space_z])
        n_cells_to_spread = int(base.TOTAL_CELLS * round(0.1 / grid_space_mean))

        xcells, ycells, zcells = self.fine_xcells, self.fine_ycells, self.fine_zcells
        
        # Total fine grid size
        super_xcells = 3 * xcells
        super_ycells = 3 * ycells
        super_zcells = 3 * zcells

        # Start of central coarse cell (1,1,1) in fine grid
        fx_start = 1 * xcells
        fy_start = 1 * ycells
        fz_start = 1 * zcells

        

        if use_cupy:
            #self.fine_densities = cp.zeros(block_size, dtype=cp.float32).reshape(self.fine_mass_array)
            # Neighbors
            neighbor_range = cp.arange(-n_cells_to_spread, n_cells_to_spread + 1)
            dx, dy, dz = cp.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
            neighbor_offsets_box = cp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
            neighbor_offsets_dist = cp.linalg.norm(neighbor_offsets_box, axis=1)
            neighbor_offsets_within_dist = neighbor_offsets_dist <= n_cells_to_spread
            neighbor_offsets = neighbor_offsets_box[neighbor_offsets_within_dist]
            M = neighbor_offsets.shape[0]

             # Coordinates to integers
            x = cp.arange(super_xcells)
            y = cp.arange(super_ycells)
            z = cp.arange(super_zcells)
            ix, iy, iz = cp.meshgrid(x, y, z, indexing='ij')
            coords_all = cp.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)

            # Fine indices in each axis for the central coarse cell
            cix = cp.arange(fx_start, fx_start + xcells)
            ciy = cp.arange(fy_start, fy_start + ycells)
            ciz = cp.arange(fz_start, fz_start + zcells)

            # Create 3D grid and flatten to 1D indices
            I, J, K = cp.meshgrid(cix, ciy, ciz, indexing='ij')
            super_flat_indices = I + super_xcells * (J + super_ycells * K)
            super_flat_indices = super_flat_indices.ravel()
            
            I, J, K = np.meshgrid(np.arange(xcells), np.arange(ycells), np.arange(zcells), indexing='ij')
            flat_indices = I * (ycells * zcells) + J * zcells + K
            flat_indices = flat_indices.ravel()

        else:
            #self.fine_densities = np.zeros(block_size).reshape(self.fine_mass_array)
            # Neighbors
            neighbor_range = np.arange(-n_cells_to_spread, n_cells_to_spread + 1)
            dx, dy, dz = np.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
            neighbor_offsets_box = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
            neighbor_offsets_dist = np.linalg.norm(neighbor_offsets_box, axis=1)
            neighbor_offsets_within_dist = neighbor_offsets_dist <= n_cells_to_spread
            neighbor_offsets = neighbor_offsets_box[neighbor_offsets_within_dist]
            M = neighbor_offsets.shape[0]
            
            # Coordinates to integers
            x = np.arange(super_xcells)
            y = np.arange(super_ycells)
            z = np.arange(super_zcells)
            ix, iy, iz = np.meshgrid(x, y, z, indexing='ij')
            coords_all = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)

            # Fine indices in each axis for the central coarse cell
            cix = np.arange(fx_start, fx_start + xcells)
            ciy = np.arange(fy_start, fy_start + ycells)
            ciz = np.arange(fz_start, fz_start + zcells)

            # Create 3D grid and flatten to 1D indices
            I, J, K = np.meshgrid(cix, ciy, ciz, indexing='ij')
            super_flat_indices = I + super_xcells * (J + super_ycells * K)
            super_flat_indices = super_flat_indices.ravel()


            I, J, K = np.meshgrid(np.arange(xcells), np.arange(ycells), np.arange(zcells), indexing='ij')
            flat_indices = I * (ycells * zcells) + J * zcells + K
            flat_indices = flat_indices.ravel()


        #start = central_idx * block_size
        #end = start + block_size
        coords = coords_all[super_flat_indices]
        # Neighbor expanding masses
        coords_exp = coords[:, None, :] + neighbor_offsets[None, :, :]
        #image_offsets = base.get_periodic_image_offsets(unitcell_vectors, boundaries, np.array(grid_shape), 
        #                                        frame_id=frame_id, use_cupy=use_cupy)
        #out_of_bounds_z_lower = coords_exp[:, :, 2] < 0
        #coords_exp[:, :, 0] += out_of_bounds_z_lower * image_offsets[0, 2]
        #coords_exp[:, :, 1] += out_of_bounds_z_lower * image_offsets[1, 2]
        #coords_exp[:, :, 2] += out_of_bounds_z_lower * image_offsets[2, 2]
        #out_of_bounds_z_higher = coords_exp[:, :, 2] >= zcells
        #coords_exp[:, :, 0] -= out_of_bounds_z_higher * image_offsets[0, 2]
        #coords_exp[:, :, 1] -= out_of_bounds_z_higher * image_offsets[1, 2]
        #coords_exp[:, :, 2] -= out_of_bounds_z_higher * image_offsets[2, 2]
        #out_of_bounds_y_lower = coords_exp[:, :, 1] < 0
        #coords_exp[:, :, 0] += out_of_bounds_y_lower * image_offsets[0, 1]
        #coords_exp[:, :, 1] += out_of_bounds_y_lower * image_offsets[1, 1]
        #out_of_bounds_y_higher = coords_exp[:, :, 1] >= ycells
        #coords_exp[:, :, 0] -= out_of_bounds_y_higher * image_offsets[0, 1]
        #coords_exp[:, :, 1] -= out_of_bounds_y_higher * image_offsets[1, 1]
        #out_of_bounds_x_lower = coords_exp[:, :, 0] < 0
        #coords_exp[:, :, 0] += out_of_bounds_x_lower * image_offsets[0, 0]
        #out_of_bounds_x_higher = coords_exp[:, :, 0] >= xcells
        #coords_exp[:, :, 0] -= out_of_bounds_x_higher * image_offsets[0, 0]
        if use_cupy:
            assert cp.greater_equal(coords_exp, 0).all()
            assert cp.less(coords_exp[:, :, 0], super_xcells).all()
            assert cp.less(coords_exp[:, :, 1], super_ycells).all()
            assert cp.less(coords_exp[:, :, 2], super_zcells).all()
        else:
            assert (coords_exp[:, :, 0] >= 0).all(), "coords_exp[:, :, 0] contains negative indices"
            assert (coords_exp[:, :, 1] >= 0).all(), "coords_exp[:, :, 1] contains negative indices"
            assert (coords_exp[:, :, 2] >= 0).all(), "coords_exp[:, :, 2] contains negative indices"
            assert (coords_exp[:, :, 0] < super_xcells).all(), "coords_exp[:, :, 0] contains indices >= xcells"
            assert (coords_exp[:, :, 1] < super_ycells).all(), "coords_exp[:, :, 1] contains indices >= ycells"
            assert (coords_exp[:, :, 2] < super_zcells).all(), "coords_exp[:, :, 2] contains indices >= zcells"

        xi, yi, zi = coords_exp[:, :, 0], coords_exp[:, :, 1], coords_exp[:, :, 2]
        superblock_x, superblock_y, superblock_z = 3 * self.fine_xcells, 3 * self.fine_ycells, 3 * self.fine_zcells
        for coarse_block in range(self.fine_mass_array.shape[0]):
            # Access the fine cells stored in the coarse block (flattened 1D array)
            fine_cells = self.fine_mass_array[coarse_block, :]
            # Convert (xi, yi, zi) to 1D indices for the superblock
            flat_indices_neighbors = (xi * (superblock_y * superblock_z) + yi * superblock_z + zi).astype(int)
            # neighbor_masses shape: (N, M)
            neighbor_masses = fine_cells[flat_indices_neighbors]
            if use_cupy:
                total_mass = cp.sum(neighbor_masses, axis=1)
            else:
                total_mass = np.sum(neighbor_masses, axis=1)
            volume = M * 1000.0 * self.grid_fine_space_x * self.grid_fine_space_y * self.grid_fine_space_z
            densities = total_mass / volume * 1.66 # Convert amu/A^3 to g/ml using 1.66 factor
            self.fine_densities[coarse_block, flat_indices] = densities

        # Flatten fine_densities after the loop
        #self.fine_densities = self.fine_densities.flatten()
        

    def calculate_coarse_cell_densities_noneighboring(
            self,
            use_cupy: bool = False
            ) -> None:
        """
        Calculate the densities in each cell of the grid without considering neighboring cells.
        Each cell's density is simply its mass divided by its own volume.
        """
        N = self.coarse_mass_array.size
        cell_volume = self.grid_coarse_space_x * self.grid_coarse_space_y * self.grid_coarse_space_z * 1000.0  # Convert nm^3 to A^3
        densities = self.coarse_mass_array / cell_volume * 1.66 # Convert amu/A^3 to g/ml using 1.66 factor
        self.coarse_densities = densities
        # Find indices of cells with density < 0.7
        if use_cupy:
            import cupy as cp
            used_indices = cp.where(densities < 0.7)[0]
        else:
            used_indices = np.where(densities < 0.7)[0]
        num_cells_low_density = used_indices.shape[0]
        self.coarse_cell_indexes[:] = used_indices
        self.coarse_cell_indexes = self.coarse_cell_indexes[:num_cells_low_density]
        return

    def generate_bubble_object(self) -> "Bubble":
        """
        Generate a bubble object from the grid densities data.
        Also, prepare a DX file header in case it will be written later.
        """
        bubble_atoms = Bubble()
        bubble_atoms.find(self.fine_xcells, self.fine_ycells, self.fine_zcells,
                          self.coarse_xcells, self.coarse_ycells, self.coarse_zcells,
                          self.fine_densities, grid_space_x=self.grid_fine_space_x,
                          grid_space_y=self.grid_fine_space_y,
                          grid_space_z=self.grid_fine_space_z, cell_indexes=self.coarse_cell_indexes,
                          )
        bubble_atoms.dx_header = self.make_dx_header()
        return bubble_atoms
    
    def make_dx_header(self) -> dict:
        """
        Prepare the header information for a DX file.
        """
        header = {}
        header["width"] = self.fine_xcells
        header["height"] = self.fine_ycells
        header["depth"] = self.fine_zcells
        header["originx"] = 5.0 * self.grid_fine_space_x
        header["originy"] = 5.0 * self.grid_fine_space_y
        header["originz"] = 5.0 * self.grid_fine_space_z
        header["resx"] = self.grid_fine_space_x * 10.0
        header["resy"] = self.grid_fine_space_y * 10.0
        header["resz"] = self.grid_fine_space_z * 10.0
        return header
    
    def write_masses_dx(
            self, 
            filename: str
            ) -> None:
        """
        Write the mass data to a DX file.
        """
        mass_grid = self.fine_mass_array.reshape(self.fine_xcells, self.fine_ycells, self.fine_zcells)
        base.write_data_array(self.make_dx_header(), mass_grid, filename)
        return

# TODO: have a method in Grid to create a Bubble object
@define
class Bubble():
    """
    A Bubble object contains representations of the regions of the system
    where the density is below a certain threshold, indicating the presence
    of bubbles or vapor pockets. It stores the coordinates of these bubbles
    and can write them to a PDB file or DX file for visualization.
    """
    atoms: dict = field(factory=dict)
    total_residues: int = field(default=1)
    total_atoms: int = field(default=0)
    total_bubble_volume: float = field(default=0.0)
    densities: np.ndarray | None = None
    bubble_data: np.ndarray | None = None
    dx_header: str = field(default="")

    def find(self, fine_xcells, fine_ycells, fine_zcells,
         coarse_xcells, coarse_ycells, coarse_zcells,
         box_densities, grid_space_x, grid_space_y, grid_space_z, cell_indexes):

        total_fine_x = coarse_xcells * fine_xcells
        total_fine_y = coarse_ycells * fine_ycells
        total_fine_z = coarse_zcells * fine_zcells

        self.densities = np.zeros((total_fine_x, total_fine_y, total_fine_z))
        self.bubble_data = np.zeros((total_fine_x, total_fine_y, total_fine_z), dtype=bool)

        global_fine_indices = []

        # Create 3D grid of relative fine-cell indices inside a block
        dx = np.arange(fine_xcells)
        dy = np.arange(fine_ycells)
        dz = np.arange(fine_zcells)
        rel_grid = np.stack(np.meshgrid(dx, dy, dz, indexing="ij"), axis=-1).reshape(-1, 3)  # shape: (n_fine_per_block, 3)

        for i, coarse_index in enumerate(cell_indexes):
            # Get coarse cell 3D index
            cx, cy, cz = base.index_to_index3d(coarse_index, coarse_ycells, coarse_zcells)

            # Get fine densities for this coarse cell (flattened)
            fine_densities = box_densities[i]  # shape: (fine_xcells * fine_ycells * fine_zcells,)

            # Find which fine cells are below threshold
            low_mask = fine_densities < base.DENSITY_THRESHOLD
            if not np.any(low_mask):
                continue  # nothing to do
            
            print("found low density cells in coarse cell", i, "at index", coarse_index)
            # Relative fine-cell indices that are below threshold
            rel_indices = rel_grid[low_mask]  # shape: (num_low, 3)

            # Compute coarse cell's origin in fine grid
            fine_origin = np.array([cx * fine_xcells, cy * fine_ycells, cz * fine_zcells])

            # Global fine cell indices
            global_indices = rel_indices + fine_origin  # shape: (num_low, 3)
            global_fine_indices.append(global_indices)

            # Optional: mark in bubble_data
            for fx, fy, fz in global_indices:
                self.bubble_data[fx, fy, fz] = 1

        if global_fine_indices:
            global_fine_indices = np.vstack(global_fine_indices)  # shape: (total_low_fine_cells, 3)
        else:
            global_fine_indices = np.empty((0, 3), dtype=int)

        self.total_bubble_volume = np.sum(self.bubble_data) * grid_space_x * grid_space_y * grid_space_z

    def write_pdb(self, filename):
        with open(filename, "w") as pdb:
            for key in self.atoms:
                pdb.write(self.atoms[key])
                pdb.write("TER\n")
            pdb.write("END\n")

    def write_densities_dx(self, filename):
        base.write_data_array(self.dx_header, self.densities, filename)

    def write_bubble_dx(self, filename):
        base.write_data_array(self.dx_header, self.bubble_data, filename)