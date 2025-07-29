import numpy as np
from modules.structures import Grid

# Test for fine cell masses with superblock implementation
def test_fine_cell_masses_superblock():
    # Grid setup: 2 coarse cells per axis, each 2 Å, fine grid 1 Å
    grid = Grid(approx_coarse_grid_space=2.0, approx_fine_grid_space=1.0)
    grid.boundaries = np.array([6.0, 6.0, 6.0])  # Enough to hold 3 coarse cells (3x2 Å = 6 Å)
    grid.initialize_coarse_cells()
    grid.coarse_cell_indexes = list(range(grid.coarse_xcells * grid.coarse_ycells * grid.coarse_zcells))
    grid.initialize_fine_cells_from_coarse_cells()

    print(f"Coarse cells: {grid.coarse_xcells}x{grid.coarse_ycells}x{grid.coarse_zcells}, "
          f"Fine cells: {grid.fine_xcells}x{grid.fine_ycells}x{grid.fine_zcells}")

    # Determine number of fine cells per coarse cell
    fx, fy, fz = grid.fine_xcells, grid.fine_ycells, grid.fine_zcells

    # 3x3x3 superblock coarse cells
    cx, cy, cz = 3, 3, 3
    total_coarse_cells = cx * cy * cz
    atoms_per_coarse = fx * fy * fz
    total_atoms = total_coarse_cells * atoms_per_coarse

    coords = np.zeros((1, total_atoms, 3), dtype=np.float32)
    masses = [1.0] * total_atoms

    coarse_spacing = np.array([
        grid.grid_coarse_space_x,
        grid.grid_coarse_space_y,
        grid.grid_coarse_space_z,
    ])
    fine_spacing = np.array([
        grid.grid_fine_space_x,
        grid.grid_fine_space_y,
        grid.grid_fine_space_z,
    ])

    atom_idx = 0
    for cx_i in range(cx):
        for cy_i in range(cy):
            for cz_i in range(cz):
                # Origin of the current coarse cell
                coarse_origin = np.array([cx_i, cy_i, cz_i]) * coarse_spacing
                for fx_i in range(fx):
                    for fy_i in range(fy):
                        for fz_i in range(fz):
                            fine_offset = (np.array([fx_i, fy_i, fz_i]) + 0.5) * fine_spacing
                            coords[0, atom_idx, :] = coarse_origin + fine_offset
                            atom_idx += 1

    # Feed data to the grid
    grid.calculate_coarse_cell_masses(coords, masses, total_atoms)

    # Print per coarse cell data
    for i in range(grid.coarse_xcells * grid.coarse_ycells * grid.coarse_zcells):
        print(f"Coarse cell {i} mass: {grid.coarse_mass_array[i]}")
        print(f"Coarse cell {i} atom coordinates:\n{grid.atom_coords[i]}")
        print(f"Coarse cell {i} atom masses: {grid.atom_masses[i]}")
        print(f"Coarse cell {i} atoms stored: {grid.atoms_stored_per_cell[i]}")
        print()

    # Calculate fine cell masses with superblock logic
    grid.calculate_fine_cell_masses_from_coarse_cells(grid.boundaries)

    # Visualize fine mass distribution for first coarse cell’s superblock
    coarse_idx = 0
    block_size = grid.fine_xcells * grid.fine_ycells * grid.fine_zcells
    fine_masses = grid.fine_mass_array[coarse_idx, :block_size]
    block = fine_masses.reshape(grid.fine_xcells, grid.fine_ycells, grid.fine_zcells)
    
    print(f"Fine mass array for coarse cell {coarse_idx} (first block in superblock):")
    for ix in range(grid.fine_xcells):
        for iy in range(grid.fine_ycells):
            row = [f"{block[ix, iy, iz]:.1f}" for iz in range(grid.fine_zcells)]
            print(f"x={ix}, y={iy}: " + " ".join(row))
        print()

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
    grid.calculate_coarse_cell_masses(coords, masses, n_atoms)

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
test_calculate_coarse_cell_masses()