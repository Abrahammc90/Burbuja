import sys
import numpy as np
import math

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
        self.mass_array = {}
        self.grid_space = grid_space
        self.densities = {}

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

        for dx in x_range:
            dx /= 100
            for dy in y_range:
                dy /= 100
                for dz in z_range:
                    dz /= 100
                    self.mass_array[(dx, dy, dz)] = 0.0

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
                self.mass_array[(x, y, z)] += float(atom.mass)

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
                            self.mass_array[(dx, dy, dz)] += float(atom.mass*2)

        #with open("mass_array", "w") as mass_file:
        #    for crds in self.mass_array:
        #        mass_file.write(str(self.mass_array[crds])+"\n")
                        

    def calculate_densities(self):

        #print(self.array)
        #print("")
        self.densities = {}
        total = len(self.mass_array)
        total_cells = 6
        for i, crds in enumerate(self.mass_array):

            x, y, z = crds[:]
            L_x, L_y, L_z = self.boundaries[:]

            #print((x-self.grid_space*total_cells), (y-self.grid_space*total_cells), (z-self.grid_space*total_cells))

            volume = 0
            total_mass = 0

            x_range = range(int((x-self.grid_space*total_cells)*100), int((x+self.grid_space*(total_cells+1))*100), int(self.grid_space*100))
            y_range = range(int((y-self.grid_space*total_cells)*100), int((y+self.grid_space*(total_cells+1))*100), int(self.grid_space*100))
            z_range = range(int((z-self.grid_space*total_cells)*100), int((z+self.grid_space*(total_cells+1))*100), int(self.grid_space*100))

            for dx in x_range:
                dx /= 100
                for dy in y_range:
                    dy /= 100
                    ji = 0
                    for dz in z_range:
                        dz /= 100

                        #Apply periodic boundary conditions to the cell crds
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

                        #print([dx, dy, dz])
                        #print(self.mass_array[(dx, dy, dz)])
                        #exit()
                        
                        #if (dx, dy, dz) in self.mass_array:
                        total_mass += self.mass_array[(dx, dy, dz)]
                        volume += self.grid_space**3
                        #ji += 1

            self.densities[crds] = total_mass/volume * 1.66
            #print("")
            #print(self.densities[crds])
            #exit()
            
            percentage = int((i / total) * 100)

            if i == 1 or percentage > (i - 1) / total * 100:
                print(f"Completed {percentage}%")

        with open("densities", "w") as density_file:
            for crds in self.densities:
                density_file.write(str(self.densities[crds])+"\n")


class bubble():

    def __init__(self, total_atoms, total_residues):

        self.atoms = {}
        self.total_residues = total_residues
        self.total_atoms = total_atoms
        self.crds = []

    def find(self, box_densities, grid_space):

        for crds in box_densities:
            x, y, z = crds[:]

            if box_densities[crds] < 0.6:
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
        with open("bubbles.pdb", "w") as pdb:
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
    
    box_grid.calculate_densities()

    bubble_atoms = bubble(box_grid.n_atoms, box_grid.n_residues)
    bubble_atoms.find(box_grid.densities, grid_space)
    bubble_atoms.write_pdb()

if __name__ == '__main__':
    main()
