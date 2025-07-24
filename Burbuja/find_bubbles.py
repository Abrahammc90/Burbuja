# TODO: docstring

import sys
import numpy as np
import math
import cupy as cp
import time
from concurrent.futures import ProcessPoolExecutor

#import MDAnalysis as mda

# TODO: talk to Abraham about using attrs for these classes

class Atom:

    def __init__(self):
        self.crds = []
        self.mass = 0
        self.resname = ""
        self.resid = 0
        self.id = 0
        self.name = ""


    def get_attributes(self, atom):

        self.get_crds(atom)
        self.get_mass(atom)
        self.get_resname(atom)
        self.get_resid(atom)
        self.get_atomid(atom)
        self.get_atomname(atom)


    def get_resname(self, atom):
        resname = atom[17:20]
        self.resname = resname


    def get_resid(self, atom):
        resid = int(atom[23:26])
        self.resid = resid


    def get_atomid(self, atom):
        atomid = int(atom[6:11])
        self.id = atomid


    def get_atomname(self, atom):
        atomname = atom[12:16].strip(" ")
        self.name = atomname


    def get_crds(self, atom):
        x, y, z = float(atom[30:38]), float(atom[38:46]), float(atom[46:54])
        self.crds = [x, y, z]


    def set_crds(self, new_crds):
        self.crds = new_crds

    # TODO: find a better way to handle this. Use a common chem library? openmm.unit?
    def get_mass(self, atom):
        self.name = atom[12:16].strip(" ")

        if self.name[0] == "O":
            self.mass = 16
        elif self.name[0] == "N":
            self.mass = 14
        elif self.name[0] == "C":
            try:
                if self.name[1] == "L" or self.name[1] == "l":
                    self.mass = 35.5
            except IndexError:
                pass
            self.mass = 12
        elif self.name[0] == "F":
            self.mass = 19
        elif self.name[0] == "B":
            try:
                if self.name[1] == "R" or self.name[1] == "r":
                    self.mass = 79.9
            except IndexError:
                pass
            self.mass = 10.8
        elif self.name[0] == "I":
            self.mass = 126.9
        elif self.name[0] == "S":
            self.mass = 32
        elif self.name[0] == "P":
            self.mass = 31
        elif self.name[0:2] == "Na":
            self.mass = 23
        elif self.name[0] == "H" or isinstance(self.name[0], int) == True:
            self.mass = 1
        else:
            #print("WARNING:", self.name+": element not identified")
            #print("Setting mass to 0")
            self.mass = 0
        return

# TODO: replace with an object read by MDTraj (that would also
#  automatically handle masses)
class PDB():

    def __init__(self, filename):
        self.filename = filename
        self.box = None

    def parse_lines(self, lines):
        names, resnames, masses, coords = [], [], [], []

        atomic_masses = {
            'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0,
            'P': 31.0, 'S': 32.0, 'Cl': 35.5, 'F': 19.0,
            'Mg': 24.3, 'Zn': 65.4, 'Cu': 63.5, 'Na': 23.0,
            'Br': 79.9, 'I': 126.9,
        }

        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                name = line[12:16].strip()
                resname = line[17:20]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                # Heuristic: deduce element from atom name
                element = ''.join(filter(str.isalpha, name)).capitalize()
                element = element if element in atomic_masses else element[0]
                mass = atomic_masses[element]

                names.append(name)
                resnames.append(resname)
                masses.append(mass)
                coords.append((x, y, z))

        return names, resnames, masses, coords

    def read_pdb_parallel(self, lines_per_chunk=500_000):
        with open(self.filename, 'r') as f:
            all_lines = f.readlines()

        # Remove CRYST1 and non-atom lines if needed later
        chunks = [all_lines[i:i + lines_per_chunk] for i in range(0, len(all_lines), lines_per_chunk)]

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.parse_lines, chunks))

        # Merge lists
        self.names     = np.array(sum((r[0] for r in results), []), dtype='U4')
        self.resnames  = np.array(sum((r[1] for r in results), []), dtype='U4')
        self.masses    = np.array(sum((r[2] for r in results), []), dtype=np.float32)
        self.coords    = np.array(sum((r[3] for r in results), []), dtype=np.float32)

        return #self.names, self.names, self.masses, self.coords

    def read(self):
        #self.names = []
        #self.masses = []
        #self.coords = []

        atomic_masses = {
            'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0,
            'P': 31.0, 'S': 32.0, 'Cl': 35.5, 'F': 19.0,
            'Mg': 24.3, 'Zn': 65.4, 'Cu': 63.5, 'Na': 23.0,
            'Br': 79.9, 'I': 126.9,
        }

        with open(self.filename, 'r') as f:
            #masses = np.empty(total_atoms, dtype=np.float32)
            #coords = np.empty((total_atoms, 3), dtype=np.float32)
            #names = np.empty(total_atoms, dtype='U4')
#
            total_atoms = 0
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    total_atoms += 1

            print("Total atoms identified:", total_atoms)
            self.masses = np.empty(total_atoms, dtype=np.float32)
            self.coords = np.empty((total_atoms, 3), dtype=np.float32)
            #self.names = np.empty(total_atoms, dtype='U4')
            self.resnames = np.empty(total_atoms, dtype='U4')


        with open(self.filename, 'r') as f:
            idx = 0
            for line in f:
                if line.startswith("CRYST1"):
                    self.box = Box()
                    self.box.get_attributes(line)
                elif line.startswith("ATOM") or line.startswith("HETATM"):
                    name = line[12:16].strip()
                    resname = line[17:20]
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    element = ''.join(filter(str.isalpha, name)).capitalize()
                    element = element if element in atomic_masses else element[0]
                    if element not in atomic_masses:
                        #print("WARNING:", self.name+": element not identified")
                        #print("Setting mass to 0")
                        mass = 0
                    else:
                        mass = atomic_masses[element]

                    self.resnames[idx] = resname
                    self.masses[idx] = mass
                    self.coords[idx][0] = x
                    self.coords[idx][1] = y
                    self.coords[idx][2] = z
                    idx += 1

                #percentage = int((idx / total_atoms) * 100)
                #if idx == 1 or percentage > (idx - 1) / total_atoms * 100:
                #    print(f"Completed {percentage}%")

            #names=np.array(names)
            #masses=np.array(masses, dtype=np.float32)
            #coords=np.array(coords, dtype=np.float32)
    
    # replace with automated utilities in mdtraj
    def read_box(self):
        with open(self.filename, 'r') as f:
            for line in f:
                if line.startswith("CRYST1"):
                    self.box = Box()
                    self.box.get_attributes(line)


class Box():

    def __init__(self):
        self.length = [0, 0, 0]
        self.angles = [0, 0, 0]
        self.vectors = [[0, 0, 0], 
                        [0, 0, 0], 
                        [0, 0, 0]]
 
    
    def get_attributes(self, box_info):

        a = float(box_info[6:15])
        b = float(box_info[15:24])
        c = float(box_info[24:33])

        self.length = np.array([a, b, c])

        alpha = float(box_info[33:40])
        beta = float(box_info[40:47])
        gamma = float(box_info[47:54])

        alpha, beta, gamma = np.radians([alpha, beta, gamma])
        self.angles = np.array([alpha, beta, gamma])
        
        # Compute elements of the transformation matrix
        self.vectors = np.array([
        [a, b * np.cos(gamma), c * np.cos(beta)],
        [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],
        [0, 0, c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2)]
    ]).T
        #self.length = np.diag(self.vectors)
        #print(self.length)

    def reshape_atoms_to_orthorombic(self, pdb_coordinates):

        self.length = np.diag(self.vectors)
        for crds in pdb_coordinates:
            for j in range(2):
                scale3 = np.floor(crds[2]/self.length[2])
                crds[0] -= scale3*self.vectors[2][0]
                crds[1] -= scale3*self.vectors[2][1]
                crds[2] -= scale3*self.vectors[2][2]
                scale2 = np.floor(crds[1]/self.length[1])
                crds[0] -= scale2*self.vectors[1][0]
                crds[1] -= scale2*self.vectors[1][1]
                scale1 = np.floor(crds[0]/self.length[0])
                crds[0] -= scale1*self.vectors[0][0]

        alpha, beta, gamma = np.radians([90, 90, 90])
        self.angles = np.array([alpha, beta, gamma])

        return

class Grid():
    # TODO: docstrings
    def __init__(self, grid_space):
        self.mass_array = []
        self.coordinates = []
        self.grid_space = grid_space
        self.densities = []
        self.cells_i = []
        self.mass_array_dict = {}

    def get_boundaries(self, pdb):

        if not pdb.box:

            xmax, ymax, zmax = -np.inf, -np.inf, -np.inf
            xmin, ymin, zmin = np.inf, np.inf, np.inf
            for i in range(len(pdb.coordinates)):
            #for atom in atoms:
                resname = pdb.resname[i]
                coords = pdb.coords[i]
                if resname == "HOH" or resname == "WAT" or \
                    resname == "Cl-" or resname == "Na+":

                    x, y, z = coords[:]

                    if x > xmax:
                        xmax = x
                    if x < xmin:
                        xmin = x
                    if y > ymax:
                        ymax = y
                    if y < ymin:
                        ymin = y
                    if z > zmax:
                        zmax = z
                    if z < zmin:
                        zmin = z

            L_x = xmax-xmin
            L_y = ymax-ymin
            L_z = zmax-zmin
        
        else:
            L_x, L_y, L_z = pdb.box.length[:]
            #print(L_x, L_y, L_z)

        #L_x -= 1
        #L_y -= 1
        #L_z -= 1

        L_x = np.floor(L_x / self.grid_space) * self.grid_space
        L_y = np.floor(L_y / self.grid_space) * self.grid_space
        L_z = np.floor(L_z / self.grid_space) * self.grid_space
        self.boundaries = np.array([L_x, L_y, L_z])

    def initialize_cells(self):

        #print("Initializing grid")

        L_x, L_y, L_z = self.boundaries[:]

        x_range = range(0, int((L_x+self.grid_space)*100), int(self.grid_space*100))
        y_range = range(0, int((L_y+self.grid_space)*100), int(self.grid_space*100))
        z_range = range(0, int((L_z+self.grid_space)*100), int(self.grid_space*100))

        self.xcells = int((L_x + self.grid_space) / self.grid_space)
        self.ycells = int((L_y + self.grid_space) / self.grid_space)
        self.zcells = int((L_z + self.grid_space) / self.grid_space)

        #print(self.xcells, self.ycells, self.zcells)

        #print(total_xcoord * total_ycoord * total_zcoord)
        #exit()
        total_coordinates = self.xcells * self.ycells * self.zcells

        self.mass_array = np.array([0.0]*total_coordinates)
        self.densities = np.array([0.0]*total_coordinates)
        self.coordinates = np.array([[0.0, 0.0, 0.0]]*total_coordinates)
        #print(self.coordinates)
        #exit()

        i = 0
        for dx in x_range:
            dx /= 100
            for dy in y_range:
                dy /= 100
                for dz in z_range:
                    dz /= 100
                    self.coordinates[i][0] = dx
                    self.coordinates[i][1] = dy
                    self.coordinates[i][2] = dz
                    self.mass_array[i] = 0
                    self.mass_array_dict[(dx, dy, dz)] = 0
                    i += 1
    
    def initialize_cells_np(self):
        L_x, L_y, L_z = self.boundaries
        spacing = self.grid_space

        # Compute grid dimensions
        self.xcells = int((L_x + spacing) / spacing)
        self.ycells = int((L_y + spacing) / spacing)
        self.zcells = int((L_z + spacing) / spacing)
        total_coordinates = self.xcells * self.ycells * self.zcells

        # Only allocate what is strictly necessary
        self.mass_array = np.zeros(total_coordinates, dtype=np.float32)
        self.densities = np.zeros(total_coordinates, dtype=np.float32)

    def initialize_cells_cuda(self):
        L_x, L_y, L_z = self.boundaries
        spacing = self.grid_space

        # Compute grid dimensions
        self.xcells = int((L_x + spacing) / spacing)
        self.ycells = int((L_y + spacing) / spacing)
        self.zcells = int((L_z + spacing) / spacing)
        total_coordinates = self.xcells * self.ycells * self.zcells

        # Only allocate what is strictly necessary
        self.mass_array = cp.zeros(total_coordinates, dtype=cp.float32)
        self.densities = cp.zeros(total_coordinates, dtype=cp.float32)
            

    def apply_boundaries_to_protein(self, pdb):

        #Apply periodic boundary conditions to protein atoms
        for coords in pdb.coords:
            
            L_x, L_y, L_z = self.boundaries[:]
            
            while coords[0] > L_x:
                coords[0] -= L_x
            while coords[0] < 0:
                coords[0] += L_x

            while coords[1] > L_y:
                coords[1] -= L_y
            while coords[1] < 0:
                coords[1] += L_y

            while coords[2] > L_z:
                coords[2] -= L_z
            while coords[2] < 0:
                coords[2] += L_z


    def calculate_cell_masses_cuda(self, pdb, chunk_size=5000):
        spacing = self.grid_space
        xcells, ycells, zcells = self.xcells, self.ycells, self.zcells
        total_cells = xcells * ycells * zcells

        if not hasattr(self, 'mass_array') or self.mass_array.size != total_cells:
            self.mass_array = cp.zeros(total_cells, dtype=cp.float32)
        else:
            self.mass_array.fill(0)  # Reset if previously allocated

        num_atoms = len(pdb.coords)
        last_percent = -1

        for start in range(0, num_atoms, chunk_size):
            end = min(start + chunk_size, num_atoms)

            # Print progress every 5%
            percent = int(100 * end / num_atoms)
            if percent != last_percent and percent % 5 == 0:
                print(f"Calculating cell masses... {percent}% complete")
                last_percent = percent

            # Batch data
            coords_batch = pdb.coords[start:end]
            masses_batch = pdb.masses[start:end]
            resnames_batch = pdb.resnames[start:end]

            # Transfer to GPU
            coords = cp.asarray(coords_batch, dtype=cp.float32)
            masses = cp.asarray(masses_batch, dtype=cp.float32)

            # Grid coordinates per atom
            grid_coords = cp.floor(coords / spacing).astype(cp.int32)
            xi, yi, zi = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]

            # Create water/ion mask (still on CPU)
            resnames_cpu = resnames_batch
            is_water = cp.asarray([r in {"HOH", "WAT", "Cl-", "Na+"} for r in resnames_cpu], dtype=cp.bool_)

            # -------------------------------
            # Handle water/ion atoms
            # -------------------------------
            if cp.any(is_water):
                xi_w = xi[is_water] % xcells
                yi_w = yi[is_water] % ycells
                zi_w = zi[is_water] % zcells
                mw = masses[is_water]

                ids = xi_w * ycells * zcells + yi_w * zcells + zi_w
                cp.add.at(self.mass_array, ids, mw)

            # -------------------------------
            # Handle non-water atoms
            # -------------------------------
            is_nonwater = ~is_water
            if cp.any(is_nonwater):
                coords_nw = grid_coords[is_nonwater]
                mnw = masses[is_nonwater]

                neighbor_range = cp.arange(-1, 2, dtype=cp.int32)
                dx, dy, dz = cp.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
                offsets = cp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)  # (27, 3)

                # Neighbor cells per atom: shape (N_atoms, 27, 3)
                neighbors = coords_nw[:, None, :] + offsets[None, :, :]
                neighbors %= cp.array([xcells, ycells, zcells], dtype=cp.int32)

                # Compute flattened 1D indices for the grid
                xi_n = neighbors[:, :, 0]
                yi_n = neighbors[:, :, 1]
                zi_n = neighbors[:, :, 2]
                linear_idx = xi_n * ycells * zcells + yi_n * zcells + zi_n  # shape: (N_atoms, 27)

                # Expand each atom's mass to its 27 neighbors, multiplied by 2.0
                mnw_expanded = cp.tile(mnw[:, None] * 2.0, (1, 27))  # shape: (N_atoms, 27)

                # Accumulate masses in the global mass_array
                cp.add.at(self.mass_array, linear_idx.ravel(), mnw_expanded.ravel())

        #with open("mass_array_cuda", "w") as mass_file:
        #    for mass in self.mass_array:
        #        mass_file.write(str(mass)+"\n")

    def calculate_cell_masses(self, pdb, chunk_size=5000):
        spacing = self.grid_space
        xcells, ycells, zcells = self.xcells, self.ycells, self.zcells
        total_cells = xcells * ycells * zcells

        if not hasattr(self, 'mass_array') or self.mass_array.size != total_cells:
            self.mass_array = np.zeros(total_cells, dtype=np.float32)
        else:
            self.mass_array.fill(0)  # Reset if already allocated

        num_atoms = len(pdb.coords)
        last_percent = -1

        for start in range(0, num_atoms, chunk_size):
            end = min(start + chunk_size, num_atoms)

            # Progress update
            percent = int(100 * end / num_atoms)
            if percent != last_percent and percent % 5 == 0:
                print(f"Calculating cell masses... {percent}% complete")
                last_percent = percent

            # Load atom batch
            coords_batch = pdb.coords[start:end]
            masses_batch = pdb.masses[start:end]
            resnames_batch = pdb.resnames[start:end]

            coords = np.asarray(coords_batch, dtype=np.float32)
            masses = np.asarray(masses_batch, dtype=np.float32)

            # Convert to grid indices
            grid_coords = np.floor(coords / spacing).astype(np.int32)
            xi, yi, zi = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]

            # Identify water/ions (mask)
            is_water = np.array([r in {"HOH", "WAT", "Cl-", "Na+"} for r in resnames_batch], dtype=bool)

            # -------------------------------
            # Water and ion atoms
            # -------------------------------
            if np.any(is_water):
                xi_w = xi[is_water] % xcells
                yi_w = yi[is_water] % ycells
                zi_w = zi[is_water] % zcells
                mw = masses[is_water]

                ids = xi_w * ycells * zcells + yi_w * zcells + zi_w
                np.add.at(self.mass_array, ids, mw)

            # -------------------------------
            # Non-water atoms
            # -------------------------------
            is_nonwater = ~is_water
            if np.any(is_nonwater):
                coords_nw = grid_coords[is_nonwater]
                mnw = masses[is_nonwater]

                neighbor_range = np.arange(-1, 2, dtype=np.int32)
                dx, dy, dz = np.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
                offsets = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)  # (27, 3)

                # Compute neighbor coordinates for each atom
                neighbors = coords_nw[:, None, :] + offsets[None, :, :]
                neighbors %= np.array([xcells, ycells, zcells], dtype=np.int32)

                # Compute linear indices
                xi_n = neighbors[:, :, 0]
                yi_n = neighbors[:, :, 1]
                zi_n = neighbors[:, :, 2]
                linear_idx = xi_n * ycells * zcells + yi_n * zcells + zi_n  # shape: (N_atoms, 27)

                # Expand each atom’s mass to 27 neighbors
                mnw_expanded = np.tile(mnw[:, None] * 2.0, (1, 27))  # shape: (N_atoms, 27)

                # Accumulate to the mass array
                np.add.at(self.mass_array, linear_idx.ravel(), mnw_expanded.ravel())

        #with open("mass_array_cuda", "w") as mass_file:
        #    for mass in self.mass_array:
        #        mass_file.write(str(mass)+"\n") 

    def calculate_densities_cuda(self, chunk_size=1000, total_cells=6):
        xcells, ycells, zcells = self.xcells, self.ycells, self.zcells
        grid_shape = (xcells, ycells, zcells)
        N = xcells * ycells * zcells

        mass_grid = self.mass_array.reshape(grid_shape)
        self.densities = cp.zeros(N, dtype=cp.float32)

        # Neighbors
        neighbor_range = cp.arange(-total_cells, total_cells + 1)
        dx, dy, dz = cp.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
        neighbor_offsets = cp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
        M = neighbor_offsets.shape[0]  # 13^3 = 2197 if total_cells=6

        # Coordinates to integers
        x = cp.arange(xcells)
        y = cp.arange(ycells)
        z = cp.arange(zcells)
        ix, iy, iz = cp.meshgrid(x, y, z, indexing='ij')
        coords_all = cp.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            coords = coords_all[start:end]

            # Neighbor expanding masses
            coords_exp = coords[:, None, :] + neighbor_offsets[None, :, :]
            coords_exp %= cp.array([xcells, ycells, zcells])

            xi, yi, zi = coords_exp[:, :, 0], coords_exp[:, :, 1], coords_exp[:, :, 2]
            neighbor_masses = mass_grid[xi, yi, zi]

            total_mass = cp.sum(neighbor_masses, axis=1)
            volume = M * (self.grid_space ** 3)
            densities = total_mass / volume * 1.66

            self.densities[start:end] = densities

        #with open("densities_cuda", "w") as density_file:
        #    for idx in range(len(self.densities)):
        #        density = self.densities[idx]
        #        density_file.write(str(density)+"\n")


    def calculate_densities(self, chunk_size=1000, total_cells=6):
        xcells, ycells, zcells = self.xcells, self.ycells, self.zcells
        grid_shape = (xcells, ycells, zcells)
        N = xcells * ycells * zcells

        mass_grid = self.mass_array.reshape(grid_shape)
        self.densities = np.zeros(N, dtype=np.float32)

        # Vecinos
        neighbor_range = np.arange(-total_cells, total_cells + 1)
        dx, dy, dz = np.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
        neighbor_offsets = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
        M = neighbor_offsets.shape[0]  # 13^3 = 2197 if total_cells=6

        # Coordenadas de celdas en enteros
        x = np.arange(xcells)
        y = np.arange(ycells)
        z = np.arange(zcells)
        ix, iy, iz = np.meshgrid(x, y, z, indexing='ij')
        coords_all = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            coords = coords_all[start:end]

            # Expandir vecinos
            coords_exp = coords[:, None, :] + neighbor_offsets[None, :, :]
            coords_exp %= np.array([xcells, ycells, zcells])

            xi, yi, zi = coords_exp[:, :, 0], coords_exp[:, :, 1], coords_exp[:, :, 2]
            neighbor_masses = mass_grid[xi, yi, zi]

            total_mass = np.sum(neighbor_masses, axis=1)
            volume = M * (self.grid_space ** 3)
            densities = total_mass / volume * 1.66

            self.densities[start:end] = densities

        #with open("densities_cuda", "w") as density_file:
        #    for idx in range(len(self.densities)):
        #        density = self.densities[idx]
        #        density_file.write(str(density)+"\n")

class bubble():
    
    def __init__(self):

        self.atoms = {}
        self.total_residues = 1
        self.total_atoms = 0
        self.crds = []

    def find(self, xcells, ycells, zcells, box_densities, grid_space):

        for i in range(len(box_densities)):
            #x, y, z = grid_coordinates[i][:]
            x, y, z = index_to_coord(i, grid_space, ycells, zcells)
            
            if box_densities[i] < 0.6:
                self.total_atoms += 1
                #self.total_residues += 1
                x += grid_space/2
                y += grid_space/2
                z += grid_space/2
                #outfile.write("You got bubbles in {:.3f} {:.3f} {:.3f}\n".format(x, y, z))
                #print("You got bubbles in {:.3f} {:.3f} {:.3f}".format(x, y, z))
                atom_pdb = "ATOM {:>6s}  BUB BUB  {:>4s}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00\n".format(
                    str(self.total_atoms), str(self.total_residues), x, y, z
                )
                self.atoms[self.total_atoms] = atom_pdb

    def write_pdb(self):
        with open("bubbles.pdb", "w") as pdb:
            for key in self.atoms:
                pdb.write(self.atoms[key])
                pdb.write("TER\n")
            pdb.write("END\n")


def index_to_coord(index, grid_space, ycells, zcells):
    spacing = grid_space
    ix = index // (ycells * zcells)
    iy = (index % (ycells * zcells)) // zcells
    iz = index % zcells
    return (ix * spacing, iy * spacing, iz * spacing)


def load_npz_to_gpu(npz_file):
    data = np.load(npz_file)
    resnames = data['resnames']
    masses = cp.asarray(data['masses'])
    coords = cp.asarray(data['coords'])
    return resnames, coords, masses

def main():

    pdb_filename = sys.argv[1]
    grid_space = 1
    
    
    start_time = time.time()
    pdb = PDB(pdb_filename)
    pdb.read()
    #pdb.read_box()
    end_time = time.time()
    print("PDB reading time: " + str(end_time - start_time) + "\n")
    #pdb.read_pdb_parallel()
    

    if pdb.box:
        print("Box information found. Coordinates will be reshaped to orthorombic.")
        pdb.box.reshape_atoms_to_orthorombic(pdb.coords)
        #output_name = pdb_filename.split(".")[:-1]
        #output_name = "".join(output_name) + "_wrapped.pdb"
        #pdb.write(output_name)
        #print("PDB reshaped to orthorombic and saved as", output_name)
        print("PDB reshaped to orthorombic")
    
    box_grid = grid(grid_space)
    #np.savez_compressed("wrapped_data",
    #                    resnames=np.array(pdb.resnames),
    #                    masses=np.array(pdb.masses, dtype=np.float32),
    #                    coords=np.array(pdb.coords, dtype=np.float32))
    #print("Saved binary file: wrapped_data")
    
    start_time = time.time()
    box_grid.get_boundaries(pdb)
    end_time = time.time()
    print("Box wrapping time: " + str(end_time - start_time) + " seconds\n")

    #pdb.resnames, pdb.coords, pdb.masses = load_npz_to_gpu("wrapped_data.npz")
    start_time = time.time()
    box_grid.initialize_cells()
    end_time = time.time()
    print("Grid initialized time: " + str(end_time - start_time) + " seconds\n")
    start_time = time.time()
    box_grid.apply_boundaries_to_protein(pdb)
    end_time = time.time()
    print("Grid boundaries applied time: " + str(end_time - start_time) + " seconds\n")
    start_time = time.time()
    box_grid.calculate_cell_masses(pdb)
    end_time = time.time()
    print("Cell masses calculated time: " + str(end_time - start_time) + " seconds\n")
    
    start_time = time.time()
    box_grid.calculate_densities()
    #end_time = time.time()
    end_time = time.time()
    print("Density calculation time: " + str(end_time - start_time) + " seconds\n")
    #with open("init_time", "w") as f:
    #    print(end_time - start_time, file=f)
    start_time = time.time()
    bubble_atoms = bubble()
    bubble_atoms.find(box_grid.xcells, box_grid.ycells, box_grid.zcells, 
                      box_grid.densities, grid_space)
    bubble_atoms.write_pdb()
    end_time = time.time()
    print("PDB bubble writing time: " + str(end_time - start_time) + " seconds\n")

if __name__ == '__main__':
    main()
