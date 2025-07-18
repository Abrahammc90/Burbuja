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
    approx_grid_space: float = field(default=0.1)
    boundaries: np.ndarray = field(factory=lambda: np.zeros(3))
    grid_space_x: float = field(default=0.1)
    grid_space_y: float = field(default=0.1)
    grid_space_z: float = field(default=0.1)
    xcells: int = field(default=0)
    ycells: int = field(default=0)
    zcells: int = field(default=0)
    mass_array: typing.Any = field(factory=lambda: np.zeros(0))
    densities: typing.Any = field(factory=lambda: np.zeros(0))
    
    def initialize_cells(
            self, 
            use_cupy=False
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
        self.xcells = int((L_x + self.approx_grid_space) / self.approx_grid_space)
        self.ycells = int((L_y + self.approx_grid_space) / self.approx_grid_space)
        self.zcells = int((L_z + self.approx_grid_space) / self.approx_grid_space)
        # Now choose the actual grid space based on grid lengths and number of cells
        # in each direction
        self.grid_space_x = L_x / (self.xcells - 1)
        self.grid_space_y = L_y / (self.ycells - 1)
        self.grid_space_z = L_z / (self.zcells - 1)
        total_coordinates = self.xcells * self.ycells * self.zcells
        if use_cupy:
            import cupy as cp
            self.mass_array = cp.zeros(total_coordinates, dtype=cp.float32)
            self.densities = cp.zeros(total_coordinates, dtype=cp.float32)
        else:
            self.mass_array = np.zeros(total_coordinates)
            self.densities = np.zeros(total_coordinates)
        return

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

    def calculate_cell_masses(
            self, 
            coordinates: np.ndarray,
            mass_list: list,
            n_atoms: int,
            frame_id: int = 0,
            chunk_size: int = 5000,
            use_cupy: bool = False
            ) -> None:
        """
        Calculate the mass contained within each cell of the grid.
        """
        if use_cupy:
            import cupy as cp

        xcells, ycells, zcells = self.xcells, self.ycells, self.zcells
        for start in range(0, n_atoms, chunk_size):
            end = min(start + chunk_size, n_atoms)
            coords_batch = coordinates[frame_id, start:end, :]
            mass_slice = mass_list[start:end]
                
            masses_batch = np.array(mass_slice, dtype=np.float32)
            if use_cupy:
                # Transfer to GPU
                coords = cp.asarray(coords_batch, dtype=cp.float32)
                masses = cp.asarray(masses_batch, dtype=cp.float32)
            else:
                coords = coords_batch.astype(np.float32)
                masses = masses_batch.astype(np.float32)

            # Grid coordinates per atom
            if use_cupy:
                grid_coords = cp.zeros((end - start, 3), dtype=cp.int32)
                grid_coords[:, 0] = cp.floor(coords[:,0] / self.grid_space_x).astype(cp.int32)
                grid_coords[:, 1] = cp.floor(coords[:,1] / self.grid_space_y).astype(cp.int32)
                grid_coords[:, 2] = cp.floor(coords[:,2] / self.grid_space_z).astype(cp.int32)

            else:
                grid_coords = np.zeros((end - start, 3), dtype=np.int32)
                grid_coords[:, 0] = np.floor(coords[:,0] / self.grid_space_x).astype(np.int32)
                grid_coords[:, 1] = np.floor(coords[:,1] / self.grid_space_y).astype(np.int32)
                grid_coords[:, 2] = np.floor(coords[:,2] / self.grid_space_z).astype(np.int32)
                
            xi, yi, zi = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]
            
            all_indices_cpu = np.ones(end - start, dtype=bool)  # Treat all atoms the same for now
            if use_cupy:
                all_indices = cp.asarray(all_indices_cpu, dtype=cp.bool_)
            else:
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
                if use_cupy:
                    cp.add.at(self.mass_array, ids, mw)
                else:
                    np.add.at(self.mass_array, ids, mw)

        return

    def calculate_densities(
            self, 
            unitcell_vectors,
            frame_id: int = 0,
            chunk_size: int = 1000, 
            use_cupy: bool = False
            ) -> None:
        """
        Calculate the densities in each cell of the grid, optionally using CuPy.
        Note that the densities
        """
        if use_cupy:
            import cupy as cp
        
        grid_space_mean = np.mean([self.grid_space_x, self.grid_space_y, self.grid_space_z])    
        n_cells_to_spread = int(base.TOTAL_CELLS * round(0.1 / grid_space_mean))
        
        xcells, ycells, zcells = self.xcells, self.ycells, self.zcells
        grid_shape = (xcells, ycells, zcells)
        N = xcells * ycells * zcells

        mass_grid = self.mass_array.reshape(grid_shape)

        if use_cupy:
            self.densities = cp.zeros(N, dtype=cp.float32)
            # Neighbors
            neighbor_range = cp.arange(-n_cells_to_spread, n_cells_to_spread + 1)
            dx, dy, dz = cp.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
            neighbor_offsets_box = cp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
            neighbor_offsets_dist = cp.linalg.norm(neighbor_offsets_box, axis=1)
            neighbor_offsets_within_dist = neighbor_offsets_dist <= n_cells_to_spread
            neighbor_offsets = neighbor_offsets_box[neighbor_offsets_within_dist]
            M = neighbor_offsets.shape[0]

            # Coordinates to integers
            x = cp.arange(xcells)
            y = cp.arange(ycells)
            z = cp.arange(zcells)
            ix, iy, iz = cp.meshgrid(x, y, z, indexing='ij')
            coords_all = cp.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)
        else:
            self.densities = np.zeros(N)

            # Neighbors
            neighbor_range = np.arange(-n_cells_to_spread, n_cells_to_spread + 1)
            dx, dy, dz = np.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
            neighbor_offsets_box = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
            neighbor_offsets_dist = np.linalg.norm(neighbor_offsets_box, axis=1)
            neighbor_offsets_within_dist = neighbor_offsets_dist <= n_cells_to_spread
            neighbor_offsets = neighbor_offsets_box[neighbor_offsets_within_dist]
            M = neighbor_offsets.shape[0]
            
            # Coordinates to integers
            x = np.arange(xcells)
            y = np.arange(ycells)
            z = np.arange(zcells)
            ix, iy, iz = np.meshgrid(x, y, z, indexing='ij')
            coords_all = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            coords = coords_all[start:end]

            # Neighbor expanding masses
            coords_exp = coords[:, None, :] + neighbor_offsets[None, :, :]
            image_offsets = base.get_periodic_image_offsets(unitcell_vectors, self.boundaries, np.array(grid_shape), 
                                                            frame_id=frame_id, use_cupy=use_cupy)
            out_of_bounds_z_lower = coords_exp[:, :, 2] < 0
            coords_exp[:, :, 0] += out_of_bounds_z_lower * image_offsets[0, 2]
            coords_exp[:, :, 1] += out_of_bounds_z_lower * image_offsets[1, 2]
            coords_exp[:, :, 2] += out_of_bounds_z_lower * image_offsets[2, 2]
            out_of_bounds_z_higher = coords_exp[:, :, 2] >= zcells
            coords_exp[:, :, 0] -= out_of_bounds_z_higher * image_offsets[0, 2]
            coords_exp[:, :, 1] -= out_of_bounds_z_higher * image_offsets[1, 2]
            coords_exp[:, :, 2] -= out_of_bounds_z_higher * image_offsets[2, 2]
            out_of_bounds_y_lower = coords_exp[:, :, 1] < 0
            coords_exp[:, :, 0] += out_of_bounds_y_lower * image_offsets[0, 1]
            coords_exp[:, :, 1] += out_of_bounds_y_lower * image_offsets[1, 1]
            out_of_bounds_y_higher = coords_exp[:, :, 1] >= ycells
            coords_exp[:, :, 0] -= out_of_bounds_y_higher * image_offsets[0, 1]
            coords_exp[:, :, 1] -= out_of_bounds_y_higher * image_offsets[1, 1]
            out_of_bounds_x_lower = coords_exp[:, :, 0] < 0
            coords_exp[:, :, 0] += out_of_bounds_x_lower * image_offsets[0, 0]
            out_of_bounds_x_higher = coords_exp[:, :, 0] >= xcells
            coords_exp[:, :, 0] -= out_of_bounds_x_higher * image_offsets[0, 0]
            if use_cupy:
                assert cp.greater_equal(coords_exp, 0).all()
                assert cp.less(coords_exp[:, :, 0], xcells).all()
                assert cp.less(coords_exp[:, :, 1], ycells).all()
                assert cp.less(coords_exp[:, :, 2], zcells).all()
            else:
                assert (coords_exp[:, :, 0] >= 0).all(), "coords_exp[:, :, 0] contains negative indices"
                assert (coords_exp[:, :, 1] >= 0).all(), "coords_exp[:, :, 1] contains negative indices"
                assert (coords_exp[:, :, 2] >= 0).all(), "coords_exp[:, :, 2] contains negative indices"
                assert (coords_exp[:, :, 0] < xcells).all(), "coords_exp[:, :, 0] contains indices >= xcells"
                assert (coords_exp[:, :, 1] < ycells).all(), "coords_exp[:, :, 1] contains indices >= ycells"
                assert (coords_exp[:, :, 2] < zcells).all(), "coords_exp[:, :, 2] contains indices >= zcells"

            xi, yi, zi = coords_exp[:, :, 0], coords_exp[:, :, 1], coords_exp[:, :, 2]
            neighbor_masses = mass_grid[xi, yi, zi]
            if use_cupy:
                total_mass = cp.sum(neighbor_masses, axis=1)
            else:
                total_mass = np.sum(neighbor_masses, axis=1)
            volume = M * 1000.0 * self.grid_space_x * self.grid_space_y * self.grid_space_z
            #densities = total_mass / volume * 1.66
            # TODO: what is 1.66? Ask Abraham and turn into a descriptive constant
            densities = total_mass / volume * 1.66
            self.densities[start:end] = densities

    def generate_bubble_object(self) -> "Bubble":
        """
        Generate a bubble object from the grid densities data.
        Also, prepare a DX file header in case it will be written later.
        """
        bubble_atoms = Bubble()
        bubble_atoms.find(self.xcells, self.ycells, self.zcells, 
                          self.densities, grid_space_x=self.grid_space_x,
                          grid_space_y=self.grid_space_y,
                          grid_space_z=self.grid_space_z)
        bubble_atoms.dx_header = self.make_dx_header()
        return bubble_atoms
    
    def make_dx_header(self) -> dict:
        """
        Prepare the header information for a DX file.
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
        """
        mass_grid = self.mass_array.reshape(self.xcells, self.ycells, self.zcells)
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

    def find(self, xcells, ycells, zcells, box_densities, grid_space_x,
             grid_space_y, grid_space_z):
        self.densities = np.resize(box_densities, (xcells, ycells, zcells))
        self.bubble_data = np.zeros((xcells, ycells, zcells), dtype=np.bool_)
        for i in range(len(box_densities)):
            ix, iy, iz = base.index_to_index3d(i, ycells, zcells)
            x = ix * grid_space_x
            y = iy * grid_space_y
            z = iz * grid_space_z

            #if box_densities[i] < 0.6:
            if box_densities[i] < base.DENSITY_THRESHOLD:
                self.total_atoms += 1
                #self.total_residues += 1
                x += grid_space_x/2
                y += grid_space_y/2
                z += grid_space_z/2
                #outfile.write("You got bubbles in {:.3f} {:.3f} {:.3f}\n".format(x, y, z))
                #print("You got bubbles in {:.3f} {:.3f} {:.3f}".format(x, y, z))
                atom_pdb = "ATOM {:>6s}  BUB BUB  {:>4s}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00\n".format(
                    str(self.total_atoms), str(self.total_residues), x, y, z
                )
                self.atoms[self.total_atoms] = atom_pdb
                self.bubble_data[ix, iy, iz] = 1

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