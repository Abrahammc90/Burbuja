import sys
import numpy as np
import math
import cupy as cp
import time
from concurrent.futures import ProcessPoolExecutor

#import MDAnalysis as mda

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

                    #names.append(name)
                    #masses.append(mass)
                    #coords.append([x, y, z])

                    #self.names[idx] = name
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

        


class grid():

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

    def initialize(self):

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
        self.cells_i = np.array([[0, 0, 0]]*total_coordinates)
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
                    #self.mass_array[(dx, dy, dz)] = 0

        i = 0
        for cell_xi in range(self.xcells):
            for cell_yi in range(self.ycells):
                for cell_zi in range(self.zcells):
                    self.cells_i[i][0] = cell_xi
                    self.cells_i[i][1] = cell_yi
                    self.cells_i[i][2] = cell_zi
                    i += 1

        #print(self.coordinates[i-1])

        #for dx in range(-1, 2):
        #    while dx < 0:
        #        dx += self.xcells
        #    while dx >= self.xcells:
        #        dx -= self.xcells
        #    for dy in range(-1, 2):
        #       while dy < 0:
        #           dy += self.ycells
        #       while dy >= self.ycells:
        #           dy -= self.ycells
        #       for dz in range(-1, 2):
        #           while dz < 0:
        #               dz += self.zcells
        #           while dy >= self.ycells:
        #               dz -= self.zcells
        #           print(self.coordinates[dx*self.ycells*self.zcells + dy*self.zcells + dz])

        #print(self.coordinates[0])

        #exit()
        

    def apply_boundaries(self, pdb):

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

    def fill_gpu(self, pdb):
        spacing = self.grid_space
        xcells, ycells, zcells = self.xcells, self.ycells, self.zcells
        grid_shape = (xcells, ycells, zcells)

        # Initialize GPU mass array
        self.mass_array = cp.zeros(xcells * ycells * zcells, dtype=cp.float32)
        mass_grid = self.mass_array.reshape(grid_shape)

        # Transfer data to GPU
        coords = cp.asarray(pdb.coords)
        masses = cp.asarray(pdb.masses)
        resnames = np.array(pdb.resnames)

        # Convert physical coordinates to grid cell indices
        grid_coords = cp.floor(coords / spacing).astype(cp.int32)
        xi, yi, zi = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]

        # Identify water/ions and non-water atoms
        water_mask = cp.asarray([r in {"HOH", "WAT", "Cl-", "Na+"} for r in resnames])
        #water_mask = cp.isin(resnames, ["HOH", "WAT", "Cl-", "Na+"])
        # ---- Part 1: water/ions (no mass spreading) ----
        xi_w, yi_w, zi_w = xi[water_mask], yi[water_mask], zi[water_mask]
        mw = masses[water_mask]
        
        idx_w = (xi_w % xcells) * ycells * zcells + (yi_w % ycells) * zcells + (zi_w % zcells)
        cp.add.at(self.mass_array, idx_w, mw)

        # ---- Part 2: all other atoms (mass spreads to neighbors) ----
        if cp.any(~water_mask):
            coords_nw = grid_coords[~water_mask]  # (N, 3)
            mnw = masses[~water_mask]             # (N,)
            N = coords_nw.shape[0]

            neighbor_range = cp.arange(-1, 2)
            dx, dy, dz = cp.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
            neighbor_offsets = cp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)  # (M, 3)
            M = neighbor_offsets.shape[0]

            # Broadcast coordinates and neighbors to shape (N, M, 3)
            expanded_coords = coords_nw[:, None, :]  # (N, 1, 3)
            neighbors = expanded_coords + neighbor_offsets[None, :, :]  # (N, M, 3)
            neighbors %= cp.array([xcells, ycells, zcells])  # Apply periodic boundaries

            # Extract grid indices
            xi_n = neighbors[:, :, 0]
            yi_n = neighbors[:, :, 1]
            zi_n = neighbors[:, :, 2]

            # Compute flat indices (N, M)
            linear_idx = xi_n * ycells * zcells + yi_n * zcells + zi_n

            # Expand mass: each atom contributes mass to M neighbors
            #mnw_expanded = mnw[:, None] * 2.0  # (N, 1)
            #mnw_expanded = cp.broadcast_to(mnw_expanded, (N, M))
            mnw_expanded = cp.broadcast_to(mnw[:, None] * 2.0, linear_idx.shape).astype(cp.float32)

            # Flatten and scatter-add to the global mass array
            #flat_idx = linear_idx.ravel()
            #flat_mass = mnw_expanded.ravel()
            cp.add.at(self.mass_array, linear_idx.ravel(), mnw_expanded.ravel())

    def fill_gpu_chunked(self, pdb, chunk_size=5_000_000):

        spacing = self.grid_space
        xcells, ycells, zcells = self.xcells, self.ycells, self.zcells
        grid_shape = (xcells, ycells, zcells)
    
        # Crear mass_array si no existe
        self.mass_array = cp.zeros(xcells * ycells * zcells, dtype=cp.float32)
        mass_grid = self.mass_array.reshape(grid_shape)
    
        num_atoms = len(pdb.coords)
        last_percent = -1
        for start in range(0, num_atoms, chunk_size):
            end = min(start + chunk_size, num_atoms)

            # Progress tracking
            percent = int(100 * end / num_atoms)
            if percent != last_percent and percent % 1 == 0:
                print(f"[fill_gpu_chunked] {percent}% complete")
                last_percent = percent

            coords_batch = pdb.coords[start:end]
            masses_batch = pdb.masses[start:end]
            resnames_batch = pdb.resnames[start:end]
    
            # Transferir batch a GPU
            coords = cp.asarray(coords_batch)
            masses = cp.asarray(masses_batch)
            resnames = resnames_batch  # Lista o array CPU
    
            # Coordenadas en malla
            grid_coords = cp.floor(coords / spacing).astype(cp.int32)
            xi, yi, zi = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]
    
            # Máscara para agua/iones
            water_mask = cp.asarray([r in {"HOH", "WAT", "Cl-", "Na+"} for r in resnames])
    
            # Parte 1: agua/iones (sin extender masa)
            xi_w, yi_w, zi_w = xi[water_mask], yi[water_mask], zi[water_mask]
            mw = masses[water_mask]
            ids = (xi_w % xcells) * ycells * zcells + (yi_w % ycells) * zcells + (zi_w % zcells)
            cp.add.at(self.mass_array, ids, mw)
    
            # Parte 2: otros átomos (extender masa a vecinos)
            if cp.any(~water_mask):
                coords_nw = grid_coords[~water_mask]
                mnw = masses[~water_mask]
                N = coords_nw.shape[0]
    
                neighbor_range = cp.arange(-1, 2)
                dx, dy, dz = cp.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
                neighbor_offsets = cp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
    
                expanded_coords = coords_nw[:, None, :]
                neighbors = expanded_coords + neighbor_offsets[None, :, :]
                neighbors %= cp.array([xcells, ycells, zcells])
    
                xi_n = neighbors[:, :, 0]
                yi_n = neighbors[:, :, 1]
                zi_n = neighbors[:, :, 2]
    
                linear_idx = xi_n * ycells * zcells + yi_n * zcells + zi_n
    
                mnw_expanded = cp.broadcast_to(mnw[:, None] * 2.0, linear_idx.shape)
    
                flat_idx = linear_idx.ravel()
                flat_mass = mnw_expanded.ravel()
                cp.add.at(self.mass_array, flat_idx, flat_mass)

    def np_fill(self, pdb):
        spacing = self.grid_space
        xcells, ycells, zcells = self.xcells, self.ycells, self.zcells
        grid_shape = (xcells, ycells, zcells)

        # Initialize GPU mass array
        self.mass_array = np.zeros(xcells * ycells * zcells, dtype=np.float32)
        mass_grid = self.mass_array.reshape(grid_shape)

        # Transfer data to numpy
        coords = np.asarray(pdb.coords)
        masses = np.asarray(pdb.masses)
        resnames = np.array(pdb.resnames)

        # Convert physical coordinates to grid cell indices
        grid_coords = np.floor(coords / spacing).astype(np.int32)
        xi, yi, zi = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]

        # Identify water/ions and non-water atoms
        water_mask = np.isin(resnames, ["HOH", "WAT", "Cl-", "Na+"])

        # ---- Part 1: water/ions (no mass spreading) ----
        xi_w, yi_w, zi_w = xi[water_mask], yi[water_mask], zi[water_mask]
        mw = masses[water_mask]

        idx_w = (xi_w % xcells) * ycells * zcells + (yi_w % ycells) * zcells + (zi_w % zcells)
        np.add.at(self.mass_array, idx_w, mw)

        # ---- Part 2: all other atoms (mass spreads to neighbors) ----
        if np.any(~water_mask):
            coords_nw = grid_coords[~water_mask]
            mnw = masses[~water_mask]
            N = coords_nw.shape[0]

            neighbor_range = np.arange(-1, 2)
            dx, dy, dz = np.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
            neighbor_offsets = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)  # (27, 3)

            # Broadcast coordinates and neighbors to shape (N, M, 3)
            expanded_coords = coords_nw[:, None, :]  # (N, 1, 3)
            neighbors = expanded_coords + neighbor_offsets[None, :, :]  # (N, 27, 3)
            neighbors %= np.array([xcells, ycells, zcells])  # condiciones periódicas

            # Extract grid indices
            xi_n = neighbors[:, :, 0]
            yi_n = neighbors[:, :, 1]
            zi_n = neighbors[:, :, 2]

            # Compute flat indices (N, M)
            linear_idx = xi_n * ycells * zcells + yi_n * zcells + zi_n

            # Expand mass: each atom contributes mass to M neighbors
            mnw_expanded = np.broadcast_to(mnw[:, None] * 2.0, linear_idx.shape).astype(np.float32)

            #mnw_expanded = (mnw[:, None] * 2.0).astype(np.float32)

            # Flatten and scatter-add to the global mass array
            np.add.at(self.mass_array, linear_idx.ravel(), mnw_expanded.ravel())


    def fill(self, pdb):


        L_x, L_y, L_z = self.boundaries[:]

        for idx in range(len(pdb.coords)):

            coords = pdb.coords[idx]
            #print(coords)
            #exit()
            mass = pdb.masses[idx]
            resname = pdb.resnames[idx]

            x, y, z = coords[:]
            
            #print(x, y, z)

            x = np.floor(x / self.grid_space)# * self.grid_space
            y = np.floor(y / self.grid_space)# * self.grid_space
            z = np.floor(z / self.grid_space)# * self.grid_space
            
            #print(x, y, z)
            #print(int(x), int(y), int(z))
            cell_i = int(x)*self.ycells*self.zcells + int(y)*self.zcells + int(z)

            #print(self.cells_i[cell_i])
            #print()
            #
            #if idx > 10:
            #    exit()

            #print(x, y, z)
            #exit()

            if resname == "HOH" or resname == "WAT" or \
                    resname == "Cl-" or resname == "Na+":

                #if (x, y, z) not in self.mass_array:
                #    self.mass_array[(x, y, z)] = 0
                #try:
                #mass_i = x_i*self.ycells*self.zcells + y_i*self.zcells + z_i
                self.mass_array[cell_i] += float(mass)
                #except:
                #    print(idx)
                #    print(x, y, z)
                #    exit()

            else:

                x_i = self.cells_i[cell_i][0]
                y_i = self.cells_i[cell_i][1]
                z_i = self.cells_i[cell_i][2]
                
                x_range = range(int(x_i-1), int(x_i+2))
                y_range = range(int(y_i-1), int(y_i+2))
                z_range = range(int(z_i-1), int(z_i+2))

                for cell_xi in x_range:
                    #print(cell_xi)
                    while cell_xi >= self.xcells:
                        cell_xi -= self.xcells
                    while cell_xi < 0:
                        cell_xi += self.xcells
                    #print(cell_xi)
                    for cell_yi in y_range:
                        while cell_yi >= self.ycells:
                            cell_yi -= self.ycells
                        while cell_yi < 0:
                            cell_yi += self.ycells
                        for cell_zi in z_range:
                            while cell_zi >= self.zcells:
                                cell_zi -= self.zcells
                            while cell_zi < 0:
                                cell_zi += self.zcells
                        

                            cell_i = cell_xi*self.ycells*self.zcells + cell_yi*self.zcells + cell_zi
                            #print(self.coordinates[mass_i])
                            #print(self.mass_array[mass_i])
                            #exit()
                            #try:
                            #total_mass += self.mass_array[mass_i]
                            self.mass_array[cell_i] += float(mass*2)

        #self.mass_array = np.array(list(self.mass_array_dict.values()))
        
        #i = 0
        #for crds in self.mass_array_dict:
        #    self.mass_array[i] = self.mass_array_dict[crds]
        #    i += 1
        #    #print(i)

        with open("mass_array_cuda", "w") as mass_file:
            for mass in self.mass_array:
                mass_file.write(str(mass)+"\n")
                        

    def calculate_densities_gpu(self):
        total_cells = 3  # number of neighbors in each direction
        box_shape = (self.xcells, self.ycells, self.zcells)
        neighbor_range = cp.arange(-total_cells, total_cells + 1)

        # Full mass array, reshaped to 3D grid
        #mass_array_flat = cp.array(list(self.mass_array_dict.values()), dtype=cp.float32)
        mass_array_flat = cp.array(self.mass_array, dtype=cp.float32)
        mass_grid = mass_array_flat.reshape(box_shape)

        # Coordinates of each central grid cell (N, 3)
        coords = cp.array(self.coordinates, dtype=cp.int32)
        N = coords.shape[0]

        # Create neighbor deltas (flattened)
        dx, dy, dz = cp.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
        neighbor_offsets = cp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)  # (M, 3)
        M = neighbor_offsets.shape[0]

        # Expand coordinates to shape (N, M, 3)
        coords_expanded = coords[:, None, :]  # (N, 1, 3)
        neighbors = coords_expanded + neighbor_offsets[None, :, :]  # (N, M, 3)

        # Apply periodic boundary conditions
        neighbors %= cp.array(box_shape)

        # Unpack indices
        xi = neighbors[:, :, 0]
        yi = neighbors[:, :, 1]
        zi = neighbors[:, :, 2]

        # Gather mass values for all neighbor cells
        masses = mass_grid[xi, yi, zi]  # (N, M)

        # Sum mass and compute density
        total_mass = cp.sum(masses, axis=1)  # (N,)
        volume = M * (self.grid_space ** 3)
        densities = total_mass / volume * 1.66  # (N,)

        # Convert to CPU and store
        self.densities = {int(i): float(densities[i]) for i in range(N)}

        # Write to file
        #with open("densities_gpu", "w") as f:
        #    for i in range(N):
        #        f.write(f"{self.densities[i]}\n")

        print("Completed 100%")

    def calculate_densities_gpu_chunked(self, batch_size=1000):
        total_cells = 6
        box_shape = (self.xcells, self.ycells, self.zcells)
        neighbor_range = cp.arange(-total_cells, total_cells + 1)

        #mass_array_flat = cp.array(list(self.mass_array_dict.values()), dtype=cp.float32)
        mass_array_flat = cp.array(self.mass_array, dtype=cp.float32)
        mass_grid = mass_array_flat.reshape(box_shape)

        coords_all = cp.array(self.coordinates, dtype=cp.int32)
        N = coords_all.shape[0]

        dx, dy, dz = cp.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
        neighbor_offsets = cp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
        M = neighbor_offsets.shape[0]

        self.densities = {}

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            coords = coords_all[start:end]

            coords_expanded = coords[:, None, :]
            neighbors = coords_expanded + neighbor_offsets[None, :, :]
            neighbors %= cp.array(box_shape)

            xi, yi, zi = neighbors[:, :, 0], neighbors[:, :, 1], neighbors[:, :, 2]
            masses = mass_grid[xi, yi, zi]

            total_mass = cp.sum(masses, axis=1)
            volume = M * (self.grid_space ** 3)
            densities_batch = total_mass / volume * 1.66

            for local_i, global_i in enumerate(range(start, end)):
                self.densities[int(global_i)] = float(densities_batch[local_i])

            #print(f"Completed {int(100 * end / N)}%")

        with open("densities_cuda", "w") as density_file:
            for idx in self.densities:
                density = self.densities[idx]
                density_file.write(str(density)+"\n")


    def calculate_densities(self):

        #print(self.array)
        #print("")
        #self.densities = {}
        total = len(self.coordinates)
        total_cells = 6

        for i in range(len(self.cells_i)):

            volume = 0
            total_mass = 0

            #x_range = range(int((x-self.grid_space*total_cells)*100), int((x+self.grid_space*(total_cells+1))*100), int(self.grid_space*100))
            #y_range = range(int((y-self.grid_space*total_cells)*100), int((y+self.grid_space*(total_cells+1))*100), int(self.grid_space*100))
            #z_range = range(int((z-self.grid_space*total_cells)*100), int((z+self.grid_space*(total_cells+1))*100), int(self.grid_space*100))
            
            x_i = self.cells_i[i][0]
            y_i = self.cells_i[i][1]
            z_i = self.cells_i[i][2]
            
            #if x_i == self.xcells-1 or y_i == self.ycells-1 or z_i == self.zcells-1:
            #    continue
            #print(x_i-total_cells, y_i-total_cells, z_i-total_cells)

            x_range = range(int(x_i-total_cells), int(x_i+total_cells+1))
            y_range = range(int(y_i-total_cells), int(y_i+total_cells+1))
            z_range = range(int(z_i-total_cells), int(z_i+total_cells+1))

            for cell_xi in x_range:
                #print(cell_xi)
                while cell_xi >= self.xcells:
                    cell_xi -= self.xcells
                while cell_xi < 0:
                    cell_xi += self.xcells
                #print(cell_xi)
                for cell_yi in y_range:
                    while cell_yi >= self.ycells:
                        cell_yi -= self.ycells
                    while cell_yi < 0:
                        cell_yi += self.ycells
                    for cell_zi in z_range:
                        while cell_zi >= self.zcells:
                            cell_zi -= self.zcells
                        while cell_zi < 0:
                            cell_zi += self.zcells
                        

                        mass_i = cell_xi*self.ycells*self.zcells + cell_yi*self.zcells + cell_zi
                        #print(self.coordinates[mass_i])
                        #print(self.mass_array[mass_i])
                        #exit()
                        #try:
                        total_mass += self.mass_array[mass_i]
                        #except IndexError:
                        #    print(len(self.mass_array), mass_i)
                        #    print(cell_xi, cell_yi, cell_zi)
                        #    exit()
                        volume += self.grid_space**3
                        #ji += 1

            self.densities[i] = total_mass/volume * 1.66
            #print(total_mass/volume * 1.66)
            #print(self.densities[i])
            #exit()
            #print("")
            #print(self.densities[i])
            #exit()
            #percentage = int((i / total) * 100)
#
            #if i == 1 or percentage > (i - 1) / total * 100:
            #    print(f"Completed {percentage}%")

        #with open("densities_cuda_broken", "w") as density_file:
        #    for density in self.densities:
        #        density_file.write(str(density)+"\n")

    def np_calculate_densities(self):
        total_cells = 6  # number of neighbors in each direction
        box_shape = (self.xcells, self.ycells, self.zcells)
        neighbor_range = np.arange(-total_cells, total_cells + 1)

        # Full mass array, reshaped to 3D grid
        #mass_array_flat = np.array(list(self.mass_array_dict.values()), dtype=np.float32)
        mass_array_flat = np.array(self.mass_array, dtype=np.float32)
        mass_grid = mass_array_flat.reshape(box_shape)

        # Coordinates of each central grid cell (N, 3)
        coords = np.array(self.coordinates, dtype=np.int32)
        N = coords.shape[0]

        # Create neighbor deltas (flattened)
        dx, dy, dz = np.meshgrid(neighbor_range, neighbor_range, neighbor_range, indexing='ij')
        neighbor_offsets = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)  # (M, 3)
        M = neighbor_offsets.shape[0]

        # Expand coordinates to shape (N, M, 3)
        coords_expanded = coords[:, None, :]  # (N, 1, 3)
        neighbors = coords_expanded + neighbor_offsets[None, :, :]  # (N, M, 3)

        # Apply periodic boundary conditions
        neighbors %= np.array(box_shape)

        # Unpack indices
        xi = neighbors[:, :, 0]
        yi = neighbors[:, :, 1]
        zi = neighbors[:, :, 2]

        # Gather mass values for all neighbor cells
        masses = mass_grid[xi, yi, zi]  # (N, M)

        # Sum mass and compute density
        total_mass = np.sum(masses, axis=1)  # (N,)
        volume = M * (self.grid_space ** 3)
        densities = total_mass / volume * 1.66  # (N,)

        # Convert to CPU and store
        self.densities = {int(i): float(densities[i]) for i in range(N)}

        # Write to file
        #with open("densities_gpu", "w") as f:
        #    for i in range(N):
        #        f.write(f"{self.densities[i]}\n")

        print("Completed 100%")

class bubble():
    
    def __init__(self):

        self.atoms = {}
        self.total_residues = 1
        self.total_atoms = 0
        self.crds = []

    def find(self, grid_coordinates, box_densities, cell_identificators, 
             max_xcell, max_ycell, max_zcell, grid_space):

        for i in range(len(box_densities)):
            x, y, z = grid_coordinates[i][:]
            cell_i = cell_identificators[i]
            #if cell_i[0] == max_xcell-1 or cell_i[1] == max_ycell-1 or cell_i[2] == max_zcell-1:
            #    continue
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
        with open("bubbles_cuda.pdb", "w") as pdb:
            for key in self.atoms:
                pdb.write(self.atoms[key])
                pdb.write("TER\n")
            pdb.write("END\n")

def load_npz_to_gpu(npz_file):
    data = np.load(npz_file)
    resnames = data['resnames']
    masses = cp.asarray(data['masses'])
    coords = cp.asarray(data['coords'])
    return resnames, coords, masses

def main():

    pdb_filename = sys.argv[1]
    grid_space = 1
    
    start_time = time.perf_counter()
    
    pdb = PDB(pdb_filename)
    #pdb.read()
    pdb.read_box()
    end_time = time.perf_counter()
    print("PDB reading time: " + str(end_time - start_time) + "\n")
    #pdb.read_pdb_parallel()
    

    if pdb.box:
        print("Box information found. Coordinates will be reshaped to orthorombic.")
        #pdb.box.reshape_atoms_to_orthorombic(pdb.coords)
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
    print("Saved binary file: wrapped_data")
    

    box_grid.get_boundaries(pdb)
    end_time = time.perf_counter()
    print("Box wrapping time: " + str(end_time - start_time) + "\n")

    pdb.resnames, pdb.coords, pdb.masses = load_npz_to_gpu("wrapped_data.npz")
    box_grid.initialize()
    end_time = time.perf_counter()
    print("Grid initialized: " + str(end_time - start_time) + "\n")
    box_grid.apply_boundaries(pdb)
    end_time = time.perf_counter()
    print("Grid boundaries applied: " + str(end_time - start_time) + "\n")
    box_grid.fill_gpu_chunked(pdb)
    end_time = time.perf_counter()
    print("Cell masses calculated: " + str(end_time - start_time) + "\n")
    
    end_time = time.perf_counter()
    print("Grid creation time: " + str(end_time - start_time) + "\n")
    #box_grid.calculate_densities()
    #start_time = time.time()
    box_grid.calculate_densities_gpu_chunked()
    #end_time = time.time()
    end_time = time.perf_counter()
    print("Density calculation: " + str(end_time - start_time) + "\n")
    #with open("init_time", "w") as f:
    #    print(end_time - start_time, file=f)
    bubble_atoms = bubble()
    bubble_atoms.find(box_grid.coordinates, box_grid.densities, 
                      box_grid.cells_i, box_grid.xcells, box_grid.ycells,
                      box_grid.zcells, grid_space)
    bubble_atoms.write_pdb()
    end_time = time.perf_counter()
    print("PDB bubble writing: " + str(end_time - start_time) + "\n")

if __name__ == '__main__':
    main()
