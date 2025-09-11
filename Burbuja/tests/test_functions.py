import numpy as np
from modules.structures import Grid, Bubble
from Burbuja.modules import base


# Test for calculate_fine_cell_densities_neighboring
def test_find():
    
    # Grid setup: 2 coarse cells per axis, each 2 Å, fine grid 1 Å
    grid = Grid(approx_coarse_grid_space=0.2, approx_fine_grid_space=0.1)
    grid.boundaries = np.array([0.8, 0.8, 0.8])  # Enough to hold 3 coarse cells (3x2 Å = 6 Å)
    grid.initialize_coarse_cells()
    xcells, ycells, zcells = grid.coarse_xcells, grid.coarse_ycells, grid.coarse_zcells

    coarse_boundaries = np.array([grid.grid_coarse_space_x * 3, grid.grid_coarse_space_y * 3, grid.grid_coarse_space_z * 3])
    #grid.coarse_cell_indexes = list(range((xcells*ycells*zcells)))
    grid.coarse_cell_indexes = [28]
    grid.initialize_fine_cells_from_coarse_cells()
    fx, fy, fz = grid.fine_xcells, grid.fine_ycells, grid.fine_zcells
    

    # Place one atom at the center of each fine cell in all coarse cells
    atoms = []
    masses = []
    mass = 0.0
    for cx in range(xcells):
        for cy in range(ycells):
            for cz in range(zcells):
                for ix in range(fx):
                    for iy in range(fy):
                        for iz in range(fz):
                            x = cx * grid.grid_coarse_space_x + ix * grid.grid_fine_space_x + grid.grid_fine_space_x / 2
                            y = cy * grid.grid_coarse_space_y + iy * grid.grid_fine_space_y + grid.grid_fine_space_y / 2
                            z = cz * grid.grid_coarse_space_z + iz * grid.grid_fine_space_z + grid.grid_fine_space_z / 2
                            atoms.append([x, y, z])
                            #mass += 1.0
                            masses.append(mass)
    n_atoms = len(atoms)
    coords = np.array(atoms).reshape(1, n_atoms, 3)
    grid.calculate_coarse_cell_masses(coords, masses, n_atoms)
    grid.calculate_fine_cell_masses_from_coarse_cells(coarse_boundaries)
    base.TOTAL_CELLS = 2
    grid.calculate_fine_cell_densities_neighboring()

    bubbles = Bubble()
    bubbles.find(fx, fy, fz,
                 xcells, ycells, zcells,
                 grid.fine_densities, grid.grid_fine_space_x, grid.grid_fine_space_y, grid.grid_fine_space_z,
                 grid.coarse_cell_indexes)

    print(bubbles.bubble_data)
    
    

# Test for calculate_fine_cell_densities_neighboring
def test_calculate_fine_cell_densities_neighboring():
    
    # Grid setup: 2 coarse cells per axis, each 2 Å, fine grid 1 Å
    grid = Grid(approx_coarse_grid_space=0.2, approx_fine_grid_space=0.1)
    grid.boundaries = np.array([0.8, 0.8, 0.8])  # Enough to hold 3 coarse cells (3x2 Å = 6 Å)
    grid.initialize_coarse_cells()
    xcells, ycells, zcells = grid.coarse_xcells, grid.coarse_ycells, grid.coarse_zcells

    coarse_boundaries = np.array([grid.grid_coarse_space_x * 3, grid.grid_coarse_space_y * 3, grid.grid_coarse_space_z * 3])
    grid.coarse_cell_indexes = list(range((xcells*ycells*zcells)))
    #grid.coarse_cell_indexes = [28]
    grid.initialize_fine_cells_from_coarse_cells()
    fx, fy, fz = grid.fine_xcells, grid.fine_ycells, grid.fine_zcells
    

    # Place one atom at the center of each fine cell in all coarse cells
    atoms = []
    masses = []
    mass = 0.0
    for cx in range(xcells):
        for cy in range(ycells):
            for cz in range(zcells):
                for ix in range(fx):
                    for iy in range(fy):
                        for iz in range(fz):
                            x = cx * grid.grid_coarse_space_x + ix * grid.grid_fine_space_x + grid.grid_fine_space_x / 2
                            y = cy * grid.grid_coarse_space_y + iy * grid.grid_fine_space_y + grid.grid_fine_space_y / 2
                            z = cz * grid.grid_coarse_space_z + iz * grid.grid_fine_space_z + grid.grid_fine_space_z / 2
                            atoms.append([x, y, z])
                            mass += 1.0
                            masses.append(mass)
    n_atoms = len(atoms)
    coords = np.array(atoms).reshape(1, n_atoms, 3)
    grid.calculate_coarse_cell_masses(coords, masses, n_atoms)
    grid.calculate_fine_cell_masses_from_coarse_cells(coarse_boundaries)
    base.TOTAL_CELLS = 2
    grid.calculate_fine_cell_densities_neighboring()

    
    #for cx in range(xcells):
    #    for cy in range(ycells):
    #        for cz in range(zcells):
    total_mass = np.zeros(fx*fy*fz, dtype=np.float32)
    for coarse_idx in range(len(grid.coarse_cell_indexes)):
        i = 0
        mass = 0
        volume = 0
        for ix in range(fx*3):
            for iy in range(fy*3):
                for iz in range(fz*3):
                    x = ix * grid.grid_fine_space_x + grid.grid_fine_space_x / 2
                    y = iy * grid.grid_fine_space_y + grid.grid_fine_space_y / 2
                    z = iz * grid.grid_fine_space_z + grid.grid_fine_space_z / 2

                    x -= grid.grid_coarse_space_x
                    y -= grid.grid_coarse_space_y
                    z -= grid.grid_coarse_space_z

                    x = round(x, 2)
                    y = round(y, 2)
                    z = round(z, 2)
                    
                    x_thres = ix * grid.grid_fine_space_x
                    y_thres = iy * grid.grid_fine_space_y
                    z_thres = iz * grid.grid_fine_space_z
                    mass_i = (ix % 2) * fy * fz + (iy % 2) * fz + (iz % 2)

                    #x, y, z = grid.fine_coordinates_array[0][i][:]
                    #if (x >= x_thres-0.25 and y >= y_thres-0.25 and z >= z_thres-0.25) and (x <= x_thres+0.25 and y <= y_thres+0.25 and z <= z_thres+0.25):
                    if (x >= -0.25 and y >= -0.25 and z >= -0.25) and (x <= 0.25 and y <= 0.25 and z <= 0.25):
                        mass += grid.fine_mass_array[coarse_idx][i]
                        volume += grid.grid_fine_space_x * grid.grid_fine_space_y * grid.grid_fine_space_z * 1000
                    #total_mass[mass_i] = mass
                    i += 1
                    #print(f"Fine cell {i} at ({x}-{x+grid.grid_fine_space_x}, {y}-{y+grid.grid_fine_space_y}, {z}-{z+grid.grid_fine_space_z})")
                    #print(f"Fine cell {i} atom coordinates from fine_coordinates_array:\n{grid.fine_coordinates_array[0][i]}")
                    #print(f"Fine cell {i} mass: {grid.fine_mass_array[0][i]}")
                    #print()
        
        #print(neighbor_masses)
        density = mass / volume * 1.66  # Convert to g/L
        print(density)
        #for coarse_idx in range(len(grid.coarse_cell_indexes)):
        densities = grid.fine_densities[coarse_idx, :]
        print(densities)
    #x = grid.flat_indices // (fy * fz)
    #y = (grid.flat_indices % (fy * fz)) // fz
    #z = grid.flat_indices % fz
    #print(x)
    #print(y)
    #print(z)


# Test for fine cell masses with superblock implementation
def test_fine_cell_masses_superblock():
    # Grid setup: 2 coarse cells per axis, each 2 Å, fine grid 1 Å
    grid = Grid(approx_coarse_grid_space=0.2, approx_fine_grid_space=0.1)
    grid.boundaries = np.array([0.6, 0.6, 0.6])  # Enough to hold 3 coarse cells (3x2 Å = 6 Å)
    grid.coarse_xcells = 3
    grid.coarse_ycells = 3
    grid.coarse_zcells = 3

    grid.fine_xcells = 2
    grid.fine_ycells = 2
    grid.fine_zcells = 2

    grid.grid_coarse_space_x = 0.2
    grid.grid_coarse_space_y = 0.2
    grid.grid_coarse_space_z = 0.2
    grid.grid_fine_space_x = 0.1
    grid.grid_fine_space_y = 0.1
    grid.grid_fine_space_z = 0.1

    coarse_boundaries = np.array([grid.grid_coarse_space_x * 3, grid.grid_coarse_space_y * 3, grid.grid_coarse_space_z * 3])
    total_coarse_cells = grid.coarse_xcells * grid.coarse_ycells * grid.coarse_zcells

    grid.coarse_mass_array = np.zeros(total_coarse_cells, dtype=np.float32)
    
    block_size = grid.fine_xcells * grid.fine_ycells * grid.fine_zcells
    superblock_size = block_size * total_coarse_cells

    #grid.coarse_cell_indexes = list(range(grid.coarse_xcells * grid.coarse_ycells * grid.coarse_zcells))
    grid.coarse_cell_indexes = [13]
    grid.fine_mass_array = np.zeros((total_coarse_cells, superblock_size), dtype=np.float32)
    grid.fine_coordinates_array = np.zeros((total_coarse_cells, superblock_size, 3), dtype=np.float32)

    print(f"Coarse cells: {grid.coarse_xcells}x{grid.coarse_ycells}x{grid.coarse_zcells}, "
          f"Fine cells: {grid.fine_xcells}x{grid.fine_ycells}x{grid.fine_zcells}")

    fx, fy, fz = grid.fine_xcells, grid.fine_ycells, grid.fine_zcells
    xcells, ycells, zcells = grid.coarse_xcells, grid.coarse_ycells, grid.coarse_zcells

    # Place one atom at the center of each fine cell in all coarse cells
    atoms = []
    mass = 0.0
    grid.atom_coords = np.zeros((total_coarse_cells, block_size, 3), dtype=np.float32)
    grid.atom_masses = np.zeros((total_coarse_cells, block_size), dtype=np.float32)
    coarse_idx = 0
    for cx in range(xcells):
        for cy in range(ycells):
            for cz in range(zcells):
                fine_idx = 0
                for ix in range(fx):
                    for iy in range(fy):
                        for iz in range(fz):
                            x = cx * grid.grid_coarse_space_x + ix * grid.grid_fine_space_x + grid.grid_fine_space_x / 2
                            y = cy * grid.grid_coarse_space_y + iy * grid.grid_fine_space_y + grid.grid_fine_space_y / 2
                            z = cz * grid.grid_coarse_space_z + iz * grid.grid_fine_space_z + grid.grid_fine_space_z / 2
                            
                            atoms.append([x, y, z])
                            mass += 1.0

                            grid.atom_coords[coarse_idx, fine_idx, :] = (x, y, z)  # Store atom coordinates
                            grid.atom_masses[coarse_idx, fine_idx] = mass  # Store atom mass
                            grid.coarse_mass_array[coarse_idx] += mass  # Increment coarse cell mass
                            fine_idx += 1
                coarse_idx += 1
    n_atoms = len(atoms)
    coords = np.array(atoms).reshape(1, n_atoms, 3)

    coords = grid.atom_coords
    coords_flat = coords.reshape(-1, 3)

    # Calculate fine cell masses with superblock logic
    grid.calculate_fine_cell_masses_from_coarse_cells(coarse_boundaries)
    #exit()
    # Visualize fine mass distribution for first coarse cell’s superblock
    i = 0
    for cx in range(xcells):
        for cy in range(ycells):
            for cz in range(zcells):
                for ix in range(fx):
                    for iy in range(fy):
                        for iz in range(fz):
                            x = cx * grid.grid_coarse_space_x + ix * grid.grid_fine_space_x
                            y = cy * grid.grid_coarse_space_y + iy * grid.grid_fine_space_y
                            z = cz * grid.grid_coarse_space_z + iz * grid.grid_fine_space_z
                        
                            #assert grid.atom_coords[0, i, 0].any() > x or grid.atom_coords[0, i, 0].any() < x + grid.grid_fine_space_x, \
                            #    f"Atom coordinates x {grid.atom_coords[0, i, 0]} not within fine cell {i} bounds ({x}-{x+grid.grid_fine_space_x})"
                            #assert grid.atom_coords[0, i, 1].any() > y or grid.atom_coords[0, i, 1].any() < y + grid.grid_fine_space_y, \
                            #    f"Atom coordinates y {grid.atom_coords[0, i, 1]} not within fine cell {i} bounds ({y}-{y+grid.grid_fine_space_y})"
                            #assert grid.atom_coords[0, i, 2].any() > z or grid.atom_coords[0, i, 2].any() < z + grid.grid_fine_space_z, \
                            #    f"Atom coordinates z {grid.atom_coords[0, i, 2]} not within fine cell {i} bounds ({z}-{z+grid.grid_fine_space_z})"

                            print(f"Fine cell {i} at ({x}-{x+grid.grid_fine_space_x}, {y}-{y+grid.grid_fine_space_y}, {z}-{z+grid.grid_fine_space_z})")
                            print(f"Fine cell {i} atom coordinates from atom_coords:\n{coords_flat[i]}")
                            print(f"Fine cell {i} atom coordinates from fine_coordinates_array:\n{grid.fine_coordinates_array[0][i]}")
                            print(f"Fine cell {i} mass: {grid.fine_mass_array[0][i]}")
                            print()
                            i += 1
    #total_mass = float(grid.fine_xcells * grid.fine_ycells * grid.fine_zcells * 27)
    #print(f"Total mass in fine cells should be {total_mass} and you got", np.sum(grid.fine_mass_array[0]))
    print("Test finished!")

# Test for coarse cell density calculation
def test_calculate_coarse_cell_density():
    # Create grid with 1 coarse cell of 10A per side
    grid = Grid(approx_coarse_grid_space=10.0)
    grid.boundaries = np.array([10.0, 10.0, 10.0])
    grid.initialize_coarse_cells()

    # Generate 10 atoms in [0, 10) in each axis
    n_atoms = 10
    coords = np.random.uniform(0, 10, size=(1, n_atoms, 3))
    masses = [1.0] * n_atoms

    # Calculate masses
    grid.calculate_coarse_cell_masses(coords, masses, n_atoms)

    # Calculate densities
    grid.calculate_coarse_cell_densities_noneighboring()

    cell_volume = grid.grid_coarse_space_x * grid.grid_coarse_space_y * grid.grid_coarse_space_z * 1000.0
    expected_density = 10.0 / cell_volume * 1.66
    actual_density = grid.coarse_densities[0]
    print(f"Coarse cell density: {actual_density} (expected {expected_density})")
    assert np.isclose(actual_density, expected_density), f"Expected {expected_density}, got {actual_density}"
    print("Density test passed!")

# Simple test for calculate_coarse_cell_masses
def test_calculate_coarse_cell_masses():
    # Create grid with 1 coarse cell of 10A per side
    grid = Grid(approx_coarse_grid_space=5.0)
    grid.boundaries = np.array([10.2, 10.2, 10.2])
    grid.initialize_coarse_cells()

    print(f"Coarse cells: {grid.coarse_xcells}x{grid.coarse_ycells}x{grid.coarse_zcells}")
    print(f"Coarse grid space: {grid.grid_coarse_space_x}, "
          f"{grid.grid_coarse_space_y}, {grid.grid_coarse_space_z}")
    #exit()
    # Generate 10 atoms in [0, 10) in each axis
    n_atoms = 10
    coords = np.random.uniform(0, 10, size=(1, n_atoms, 3))
    masses = [1.0] * n_atoms

    # Calculate masses
    grid.calculate_coarse_cell_masses(coords, masses, n_atoms, use_cupy=True)

    total_mass = np.sum(grid.coarse_mass_array)
    print(f"Total mass in coarse cells: {total_mass} (should be 10.0)")
    assert np.isclose(total_mass, 10.0), f"Expected 10.0, got {total_mass}"
    print("Total atoms stored should be 10:", np.sum(grid.atoms_stored_per_cell))
    assert np.sum(grid.atoms_stored_per_cell) == 10, "Expected 10 atoms stored in coarse cells"
    assert np.sum(grid.atom_masses) == 10.0, "Expected 10.0 masses stored in coarse cells, you got " + str(np.sum(grid.atom_masses))
    
    # Print per coarse cell data
    for i in range(grid.coarse_xcells * grid.coarse_ycells * grid.coarse_zcells):
        x = i // (grid.coarse_ycells * grid.coarse_zcells) * grid.grid_coarse_space_x
        y = (i % (grid.coarse_ycells * grid.coarse_zcells)) // grid.coarse_zcells * grid.grid_coarse_space_y
        z = i % grid.coarse_zcells * grid.grid_coarse_space_z

        assert grid.atom_coords[i, :, 0].any() > x or grid.atom_coords[i, :, 0].any() < x + grid.grid_coarse_space_x, \
            f"Atom coordinates x {grid.atom_coords[i, :, 0]} not within coarse cell {i} bounds ({x}-{x+grid.grid_coarse_space_x})"
        assert grid.atom_coords[i, :, 1].any() > y or grid.atom_coords[i, :, 1].any() < y + grid.grid_coarse_space_y, \
            f"Atom coordinates y {grid.atom_coords[i, :, 1]} not within coarse cell {i} bounds ({y}-{y+grid.grid_coarse_space_y})"
        assert grid.atom_coords[i, :, 2].any() > z or grid.atom_coords[i, :, 2].any() < z + grid.grid_coarse_space_z, \
            f"Atom coordinates z {grid.atom_coords[i, :, 2]} not within coarse cell {i} bounds ({z}-{z+grid.grid_coarse_space_z})"

        #print(f"Coarse cell {i} at ({x}-{x+grid.grid_coarse_space_x}, {y}-{y+grid.grid_coarse_space_y}, {z}-{z+grid.grid_coarse_space_z})")
        #print(f"Coarse cell {i} mass: {grid.coarse_mass_array[i]}")
        #print(f"Coarse cell {i} atom coordinates:\n{grid.atom_coords[i]}")
        #print(f"Coarse cell {i} atom masses: {grid.atom_masses[i]}")
        #print(f"Coarse cell {i} atoms stored: {grid.atoms_stored_per_cell[i]}")
        #print()

    print("Test passed!")
test_find()