"""
test_compare.py

Compare a CPU calculation with a GPU calculation step by step
to identify any differences in the intermediate or final results.
"""

import os

import pytest
import numpy as np
import mdtraj

import Burbuja.burbuja as burbuja

TEST_DIRECTORY = os.path.dirname(__file__)
DATA_DIRECTORY = os.path.join(TEST_DIRECTORY, "data")

import Burbuja.modules.base as base
import Burbuja.modules.parse as parse
import Burbuja.modules.structures as structures

@pytest.mark.needs_cupy
def test_tb_traj_cupy():
    """
    This system is a trajectory with multiple frames, and has bubbles
    in some of the earlier frames. Test both CPU and GPU implementations.
    """
    grid_resolution: float = 0.1
    dcd_filename = os.path.join(DATA_DIRECTORY, "tb_traj.dcd")
    prmtop_filename = os.path.join(DATA_DIRECTORY, "tryp_ben.prmtop")
    mdtraj_structure = mdtraj.load(dcd_filename, top=prmtop_filename)
    mdtraj_struct_frame0 = mdtraj_structure[0]

    n_atoms = mdtraj_struct_frame0.n_atoms
    coordinates = mdtraj_struct_frame0.xyz
    unitcell_vectors = mdtraj_struct_frame0.unitcell_vectors
    masses = []
    for atom in mdtraj_struct_frame0.topology.atoms:
        mass = atom.element.mass if atom.element else 0.0
        masses.append(mass)
    
    lengths = base.reshape_atoms_to_orthorombic(coordinates, unitcell_vectors, 0)
    box_grid_cpu = structures.Grid(
        approx_grid_space=grid_resolution,
        boundaries=lengths)
    box_grid_gpu = structures.Grid(
        approx_grid_space=grid_resolution,
        boundaries=lengths)
    # Make sure all results from initialize are the same
    box_grid_cpu.initialize_cells(use_cupy=False)
    box_grid_gpu.initialize_cells(use_cupy=True)
    assert np.isclose(box_grid_cpu.xcells, box_grid_gpu.xcells)
    assert np.isclose(box_grid_cpu.ycells, box_grid_gpu.ycells)
    assert np.isclose(box_grid_cpu.zcells, box_grid_gpu.zcells)
    assert np.isclose(box_grid_cpu.grid_space_x, box_grid_gpu.grid_space_x)
    assert np.isclose(box_grid_cpu.grid_space_y, box_grid_gpu.grid_space_y)
    assert np.isclose(box_grid_cpu.grid_space_z, box_grid_gpu.grid_space_z)
    assert np.isclose(box_grid_cpu.total_system_volume, box_grid_gpu.total_system_volume)
    assert box_grid_cpu.mass_array.shape == box_grid_gpu.mass_array.shape
    assert box_grid_cpu.densities.shape == box_grid_gpu.densities.shape

    # Make sure results from calculate_cell_masses are the same
    box_grid_cpu.calculate_cell_masses(
        coordinates, masses, n_atoms, frame_id=0, use_cupy=False)
    box_grid_gpu.calculate_cell_masses(
        coordinates, masses, n_atoms, frame_id=0, use_cupy=True)
    assert np.isclose(box_grid_cpu.mass_array, box_grid_gpu.mass_array).all()

    # Make sure results from calculate_densities are the same
    box_grid_cpu.calculate_densities(
        unitcell_vectors, frame_id=0, use_cupy=False)
    box_grid_gpu.calculate_densities(
        unitcell_vectors, frame_id=0, use_cupy=True)
    assert np.isclose(box_grid_cpu.densities, box_grid_gpu.densities).all()

    bubbles_cpu = box_grid_cpu.generate_bubble_object(use_cupy=False)
    bubbles_gpu = box_grid_gpu.generate_bubble_object(use_cupy=True)

    assert bubbles_cpu.total_bubble_volume == bubbles_gpu.total_bubble_volume
    assert bubbles_cpu.total_atoms == bubbles_gpu.total_atoms
    