import sys
import numpy as np
import math
import cupy as cp
import time

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

    def read(self):
        """Read a PDB file and extract atomic coordinates and CRYST1 line."""
        self.atoms = []
        self.cryst1 = None
        with open(self.filename, 'r') as f:
            self.lines = f.readlines()

        for line in self.lines:
            if line.startswith("CRYST1"):
                self.box = Box()
                self.box.get_attributes(line)
            elif line.startswith(("ATOM", "HETATM")):
                atom = Atom()
                atom.get_attributes(line)
                self.atoms.append(atom)

        return
    
    def write(self, output_filename):

        a, b, c = self.box.length[:]
        alpha, beta, gamma = 90, 90, 90
        CRYST1_line = "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} P 1\n".format(
            a, b, c, alpha, beta, gamma)
        with open(output_filename, "w") as pdb:
            pdb.write(CRYST1_line)
            atom_i = 0
            for line in self.lines:
                if line.startswith(("ATOM", "HETATM")):
                    crds = self.atoms[atom_i].crds
                    before_crds = line[:30]
                    after_crds = line[54:]

                    atom_str = before_crds + "{:8.3f}{:8.3f}{:8.3f}".format(
                        crds[0], crds[1], crds[2]) + after_crds

                    pdb.write(atom_str)
                    atom_i += 1

                elif not line.startswith("CRYST1"):
                    pdb.write(line)

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

    def reshape_atoms_to_orthorombic(self, atoms):

        self.length = np.diag(self.vectors)
        for atom in atoms:
            for j in range(2):
                scale3 = math.floor(atom.crds[2]/self.length[2])
                atom.crds[0] -= scale3*self.vectors[2][0]
                atom.crds[1] -= scale3*self.vectors[2][1]
                atom.crds[2] -= scale3*self.vectors[2][2]
                scale2 = math.floor(atom.crds[1]/self.length[1])
                atom.crds[0] -= scale2*self.vectors[1][0]
                atom.crds[1] -= scale2*self.vectors[1][1]
                scale1 = math.floor(atom.crds[0]/self.length[0])
                atom.crds[0] -= scale1*self.vectors[0][0]

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

    def get_boundaries(self, atoms, box_length):

        if np.all(box_length == 0):

            xmax, ymax, zmax = -np.inf, -np.inf, -np.inf
            xmin, ymin, zmin = np.inf, np.inf, np.inf
            for atom in atoms:
                if atom.resname == "HOH" or atom.resname == "WAT" or \
                    atom.resname == "Cl-" or atom.resname == "Na+":

                    x, y, z = atom.crds[:]

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

            L_x, L_y, L_z = box_length[:]
            #print(L_x, L_y, L_z)

        #L_x -= 1
        #L_y -= 1
        #L_z -= 1

        L_x = np.floor(L_x / self.grid_space) * self.grid_space
        L_y = np.floor(L_y / self.grid_space) * self.grid_space
        L_z = np.floor(L_z / self.grid_space) * self.grid_space
        self.boundaries = np.array([L_x, L_y, L_z])

    def initialize(self):

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
        

    def apply_boundaries(self, atoms):

        #Apply periodic boundary conditions to protein atoms
        for atom in atoms:
            
            L_x, L_y, L_z = self.boundaries[:]
            
            while atom.crds[0] > L_x:
                atom.crds[0] -= L_x
            while atom.crds[0] < 0:
                atom.crds[0] += L_x

            while atom.crds[1] > L_y:
                atom.crds[1] -= L_y
            while atom.crds[1] < 0:
                atom.crds[1] += L_y

            while atom.crds[2] > L_z:
                atom.crds[2] -= L_z
            while atom.crds[2] < 0:
                atom.crds[2] += L_z

    def fill_gpu(self, atoms):

        L_x, L_y, L_z = self.boundaries[:]

        # Transfer coordinate grid and mass array to GPU
        coords_gpu = cp.asarray(self.coordinates)
        mass_gpu = cp.asarray(self.mass_array)

        for atom in atoms:
            self.n_atoms = atom.id
            self.n_residues = atom.resid

            x, y, z = atom.crds
            x = np.floor(x / self.grid_space) * self.grid_space
            y = np.floor(y / self.grid_space) * self.grid_space
            z = np.floor(z / self.grid_space) * self.grid_space

            is_water_or_ion = atom.resname in {"HOH", "WAT", "Cl-", "Na+"}

            if is_water_or_ion:
                # Match exactly one coordinate on GPU
                match = cp.where(
                    (cp.abs(coords_gpu[:, 0] - x) < 1e-3) &
                    (cp.abs(coords_gpu[:, 1] - y) < 1e-3) &
                    (cp.abs(coords_gpu[:, 2] - z) < 1e-3)
                )[0]
                if match.size > 0:
                    mass_gpu[match] += atom.mass

            else:
                # Create range in 3D space
                dx_range = np.arange(x - self.grid_space, x + self.grid_space * 2, self.grid_space)
                dy_range = np.arange(y - self.grid_space, y + self.grid_space * 2, self.grid_space)
                dz_range = np.arange(z - self.grid_space, z + self.grid_space * 2, self.grid_space)

                for dx in dx_range:
                    dx = dx % L_x
                    for dy in dy_range:
                        dy = dy % L_y
                        for dz in dz_range:
                            dz = dz % L_z

                            match = cp.where(
                                (cp.abs(coords_gpu[:, 0] - dx) < 1e-3) &
                                (cp.abs(coords_gpu[:, 1] - dy) < 1e-3) &
                                (cp.abs(coords_gpu[:, 2] - dz) < 1e-3)
                            )[0]
                            if match.size > 0:
                                mass_gpu[match] += atom.mass * 2

        # Bring updated mass array back to CPU
        self.mass_array = cp.asnumpy(mass_gpu)

    def fill(self, atoms):


        L_x, L_y, L_z = self.boundaries[:]

        for atom in atoms:

            self.n_atoms = atom.id
            self.n_residues = atom.resid

            x, y, z = atom.crds[:]
            x = np.floor(x / self.grid_space) * self.grid_space
            y = np.floor(y / self.grid_space) * self.grid_space
            z = np.floor(z / self.grid_space) * self.grid_space
            
            if atom.resname == "HOH" or atom.resname == "WAT" or \
                    atom.resname == "Cl-" or atom.resname == "Na+":

                #if (x, y, z) not in self.mass_array:
                #    self.mass_array[(x, y, z)] = 0
                self.mass_array_dict[(x, y, z)] += float(atom.mass)

            else:
                
                x_range = range(int((x-self.grid_space)*100), int((x+self.grid_space*2)*100), int(self.grid_space*100))
                y_range = range(int((y-self.grid_space)*100), int((y+self.grid_space*2)*100), int(self.grid_space*100))
                z_range = range(int((z-self.grid_space)*100), int((z+self.grid_space*2)*100), int(self.grid_space*100))

                for dx in x_range:
                    dx /= 100
                    for dy in y_range:
                        dy /= 100
                        for dz in z_range:
                            dz /= 100

                            while dx > L_x:
                                dx -= L_x
                            while dx < 0:
                                dx += L_x
                            #
                            while dy > L_y:
                                dy -= L_y
                            while dy < 0:
                                dy += L_y
                            #
                            while dz > L_z:
                                dz -= L_z
                            while dz < 0:
                                dz += L_z

                            #if (dx, dy, dz) not in self.mass_array:
                            #    self.mass_array[(dx, dy, dz)] = 0
                            self.mass_array_dict[(dx, dy, dz)] += float(atom.mass*2)

        self.mass_array = np.array(list(self.mass_array_dict.values()))
        
        #i = 0
        #for crds in self.mass_array_dict:
        #    self.mass_array[i] = self.mass_array_dict[crds]
        #    i += 1
        #    #print(i)

        #with open("mass_array_cuda", "w") as mass_file:
        #    for mass in self.mass_array:
        #        mass_file.write(str(mass)+"\n")
                        

    def calculate_densities_gpu(self):
        total_cells = 3  # number of neighbors in each direction
        box_shape = (self.xcells, self.ycells, self.zcells)
        neighbor_range = cp.arange(-total_cells, total_cells + 1)

        # Full mass array, reshaped to 3D grid
        mass_array_flat = cp.array(list(self.mass_array_dict.values()), dtype=cp.float32)
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
        with open("densities_gpu", "w") as f:
            for i in range(N):
                f.write(f"{self.densities[i]}\n")

        print("Completed 100%")

    def calculate_densities_gpu_chunked(self, batch_size=1000):
        total_cells = 6
        box_shape = (self.xcells, self.ycells, self.zcells)
        neighbor_range = cp.arange(-total_cells, total_cells + 1)

        mass_array_flat = cp.array(list(self.mass_array_dict.values()), dtype=cp.float32)
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

            print(f"Completed {int(100 * end / N)}%")


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
            percentage = int((i / total) * 100)

            if i == 1 or percentage > (i - 1) / total * 100:
                print(f"Completed {percentage}%")

        #with open("densities_cuda", "w") as density_file:
        #    for density in self.densities:
        #        density_file.write(str(density)+"\n")

    def np_calculate_densities(self):
        total_cells = 6  # number of neighbors in each direction
        box_shape = (self.xcells, self.ycells, self.zcells)
        neighbor_range = np.arange(-total_cells, total_cells + 1)

        # Full mass array, reshaped to 3D grid
        mass_array_flat = np.array(list(self.mass_array_dict.values()), dtype=np.float32)
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
        with open("densities_gpu", "w") as f:
            for i in range(N):
                f.write(f"{self.densities[i]}\n")

        print("Completed 100%")

class bubble():
    
    def __init__(self, total_atoms, total_residues):

        self.atoms = {}
        self.total_residues = total_residues
        self.total_atoms = total_atoms
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
                x += grid_space/2
                y += grid_space/2
                z += grid_space/2

                print("You got bubbles in {:.3f} {:.3f} {:.3f}".format(x, y, z))

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

def main():

    pdb_filename = sys.argv[1]

    pdb = PDB(pdb_filename)
    pdb.read()
    #pdb_lines = open(pdb_filename).readlines()
    grid_space = 1

    #box = Box()
#
    #box.get_attributes(pdb.box)
    if pdb.box:
        print("Box information found. Coordinates will be reshaped to orthorombic.")
        pdb.box.reshape_atoms_to_orthorombic(pdb.atoms)
        output_name = pdb_filename.split(".")[:-1]
        output_name = "".join(output_name) + "_wrapped.pdb"
        pdb.write(output_name)
        print("PDB reshaped to orthorombic and saved as", output_name)

    #print(pdb.box.length)
    #exit()
    box_grid = grid(grid_space)
    #box_grid.get_boundaries(pdb.box.length)
    if pdb.box:
        box_grid.get_boundaries(pdb.atoms, pdb.box.length)
    else:
        box_grid.get_boundaries(pdb.atoms, np.array([0, 0, 0]))

    #print(box_grid.boundaries)
    #print(pdb.box.length)
    #print(xmin, xmax, ymin, ymax, zmin, zmax)
    #exit()

    box_grid.initialize()
    box_grid.apply_boundaries(pdb.atoms)
    box_grid.fill(pdb.atoms)
    
    #box_grid.calculate_densities()
    start_time = time.time()
    box_grid.calculate_densities_gpu_chunked()
    end_time = time.time()

    #with open("init_time", "w") as f:
    #    print(end_time - start_time, file=f)

    bubble_atoms = bubble(box_grid.n_atoms, box_grid.n_residues)
    bubble_atoms.find(box_grid.coordinates, box_grid.densities, 
                      box_grid.cells_i, box_grid.xcells, box_grid.ycells,
                      box_grid.zcells, grid_space)
    bubble_atoms.write_pdb()

if __name__ == '__main__':
    main()
