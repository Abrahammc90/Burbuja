"""
burbuja.py

Manage all stages of Burbuja, including use as a command-line tool or
as the API.
"""

import os
import time
import typing
import pathlib
import argparse

import numpy as np
import mdtraj

import Burbuja.modules.base as base
import Burbuja.modules.parse as parse
import Burbuja.modules.structures as structures

BIG_FILE_CHUNK_SIZE = 100000  # Number of atoms to process in each chunk for big files

def burbuja(
        structure: str | mdtraj.Trajectory,
        grid_resolution: float = 0.1,
        use_cupy: bool = False,
        use_float32: bool = False,
        density_threshold: float = base.DEFAULT_DENSITY_THRESHOLD,
        neighbor_cells: int = base.DEFAULT_NEIGHBOR_CELLS
        ) -> typing.List[structures.Bubble]:
    """
    Detect bubbles in a structure or trajectory and return a list of
    Bubble objects (one per frame).

    This is the main API function for bubble detection. It supports both
    PDB files and MDTraj trajectory objects. For each frame, a grid is
    constructed, densities are calculated, and a Bubble object is returned.

    Args:
        structure (str or mdtraj.Trajectory):
            Path to a structure file (e.g., PDB, DCD) or an MDTraj
            trajectory object.
        grid_resolution (float, optional):
            Grid spacing in nanometers. Default is 0.1.
        use_cupy (bool, optional):
            Use CuPy for GPU acceleration. Default is False.
        use_float32 (bool, optional):
            Use float32 precision for calculations. Default is False.
        density_threshold (float, optional):
            Density threshold for void detection (g/L). Default is 0.25.
        neighbor_cells (int, optional):
            Number of cells from the central cell to include in the density
            average. Default is 4.

    Returns:
        list[Bubble]: List of Bubble objects, one per frame.

    Example:
        >>> import mdtraj
        >>> from Burbuja import burbuja
        >>> traj = mdtraj.load('traj.dcd', top='top.prmtop')
        >>> bubbles = burbuja(traj, grid_resolution=0.1, use_cupy=True)
        >>> for i, bubble in enumerate(bubbles):
        ...     print(f"Frame {i}: Bubble volume = {bubble.total_bubble_volume:.3f} nm^3")
    """

    bubbles = []
    if use_float32:
        mydtype = np.float32
    else:
        mydtype = np.float64
    if isinstance(structure, str):
        a, b, c, alpha, beta, gamma = parse.get_box_information_from_pdb_file(structure)
        n_frames, n_atoms = parse.get_num_frames_and_atoms_from_pdb_file(structure)
        coordinates = np.zeros((n_frames, n_atoms, 3), dtype=mydtype)
        masses = np.zeros(n_atoms, dtype=mydtype)
        unitcell_vectors0 = np.array([
            [a, b * np.cos(gamma), c * np.cos(beta)],
            [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],
            [0, 0, c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2)]],
            dtype=mydtype)
        unitcell_vectors0 = np.transpose(unitcell_vectors0, axes=(1, 0))
        unitcell_vectors = np.repeat(unitcell_vectors0[np.newaxis, :, :], n_frames, axis=0)
        if n_frames > 1:
            print("Warning: The PDB file contains multiple frames, and unit cell vectors are "
                  "assumed to be constant across frames. If the unit cell vectors changed during "
                  "the generation of this trajectory, you must load a different trajectory file "
                  "format, such as a DCD file, and provide the topology file to Burbuja in order "
                  "for the correct unit cell vectors to be used for each frame.")
        print("Filling out coordinates and masses from PDB structure...")
        start_time = time.time()
        parse.fill_out_coordinates_and_masses(structure, coordinates, masses, n_frames, n_atoms)
        print("Coordinates and masses filled in {:.2f} seconds.".format(time.time() - start_time))
        
    else:
        n_frames = structure.n_frames
        n_atoms = structure.n_atoms
        coordinates = structure.xyz
        unitcell_vectors = structure.unitcell_vectors
        masses = []
        for atom in structure.topology.atoms:
            mass = atom.element.mass if atom.element else 0.0
            masses.append(mass)

    for frame_id in range(n_frames):
        lengths = base.reshape_atoms_to_orthorombic(coordinates, unitcell_vectors, frame_id)
        box_grid = structures.Grid(
            approx_grid_space=grid_resolution,
            boundaries=lengths,
            density_threshold=density_threshold,
            neighbor_cells=neighbor_cells
        )
        box_grid.initialize_cells(use_cupy=use_cupy, use_float32=use_float32)
        box_grid.calculate_cell_masses(coordinates, masses, n_atoms, frame_id, use_cupy=use_cupy)
        print("Cell masses calculated in {:.2f} seconds.".format(time.time() - start_time))
        box_grid.calculate_densities(unitcell_vectors, frame_id=frame_id, use_cupy=use_cupy)
        bubble = box_grid.generate_bubble_object(use_cupy=use_cupy)
        bubbles.append(bubble)
    return bubbles

def has_bubble(
        structure: mdtraj.Trajectory,
        grid_resolution: float = 0.1,
        use_cupy: bool = False,
        dx_filename_base: str | None = None,
        density_threshold: float = base.DEFAULT_DENSITY_THRESHOLD,
        minimum_bubble_fraction: float = base.DEFAULT_MINIMUM_BUBBLE_FRACTION,
        neighbor_cells: int = base.DEFAULT_NEIGHBOR_CELLS
    ) -> bool:
    """
    Quickly check if a structure or trajectory contains a significant
    bubble.

    This function runs bubble detection and returns True if any frame
    contains a bubble whose volume exceeds the minimum_bubble_fraction
    of the system volume. Optionally, writes DX files for visualization
    if dx_filename_base is provided.

    Args:
        structure (str or mdtraj.Trajectory):
            Path to a structure file or MDTraj trajectory object.
        grid_resolution (float, optional):
            Grid spacing in nanometers. Default is 0.1.
        use_cupy (bool, optional):
            Use CuPy for GPU acceleration. Default is False.
        dx_filename_base (str, optional):
            If provided, write DX files for each frame with a bubble.
            Default is None.
        density_threshold (float, optional):
            Density threshold for void detection (g/L). Default is 0.25.
        minimum_bubble_fraction (float, optional):
            Minimum fraction of system volume for a bubble to be
            considered significant. Default is 0.005.
        neighbor_cells (int, optional):
            Number of cells from the central cell to include in the
            density average. Default is 4.

    Returns:
        bool: True if a significant bubble is found, False otherwise.

    Example:
        >>> from Burbuja import has_bubble
        >>> import mdtraj
        >>> traj = mdtraj.load('traj.dcd', top='top.prmtop')
        >>> contains_bubble = has_bubble(traj, use_cupy=True, dx_filename_base='bubble_output')
        >>> print("Contains bubble?", contains_bubble)
    """
    bubbles = burbuja(structure, grid_resolution, use_cupy=use_cupy,
                      density_threshold=density_threshold,
                      neighbor_cells=neighbor_cells)
    found_bubble = False
    
    for i, bubble in enumerate(bubbles):
        if bubble.total_bubble_volume > minimum_bubble_fraction * bubble.total_system_volume:
            found_bubble = True
            if dx_filename_base is not None:
                dx_filename = f"{dx_filename_base}_frame_{i}.dx"
                bubble.write_bubble_dx(dx_filename)
                print(f"Bubble detected with volume: {bubble.total_bubble_volume:.3f} nm^3. Frame: {i}. "
                    f"Bubble volume map file: {dx_filename}")
            else:
                break
    return found_bubble

def main():
    def main():
        """
        Command-line interface for Burbuja bubble detection.

        Parses command-line arguments, loads the structure and topology as
        needed, and runs bubble detection. Prints results and optionally
        writes DX files for visualization.

        Usage:
            python -m Burbuja.burbuja STRUCTURE_FILE [options]

        For a full list of options, see the user guide or run with
        -h/--help.
        """
        argparser = argparse.ArgumentParser(
            description="Automatically detect bubbles and vapor pockets and local "
                "voids within molecular dynamics simulation structures making use "
                "of explicit solvent.")
        # ...existing code...
        args = argparser.parse_args()
        args = vars(args)
        structure_file = pathlib.Path(args["structure_file"])
        topology_file = pathlib.Path(args["topology"]) if args["topology"] else None
        grid_resolution = args["grid_resolution"]
        use_cupy = args["use_cupy"]
        detailed_output = args["detailed_output"]
        density_threshold = args["density_threshold"]
        minimum_bubble_fraction = args["minimum_bubble_fraction"]
        neighbor_cells = args["neighbor_cells"]

        if topology_file is None:
            structure = str(structure_file)
        else:
            structure = mdtraj.load(structure_file, top=topology_file)
        if detailed_output:
            structure_file_base = os.path.splitext(structure_file.name)[0]
            dx_filename_base = f"{structure_file_base}_bubble"
        else:
            dx_filename_base = None
    
        time_start = time.time()
        has_bubble_result = has_bubble(structure, grid_resolution, use_cupy=use_cupy,
                                       dx_filename_base=dx_filename_base, 
                                       density_threshold=density_threshold,
                                       minimum_bubble_fraction=minimum_bubble_fraction,
                                       neighbor_cells=neighbor_cells)
        time_end = time.time()
        elapsed_time = time_end - time_start
        print(f"Bubble detection completed in {elapsed_time:.2f} seconds.")
    
        if has_bubble_result:
            print("The structure has a bubble.")
        else:
            print("No bubble detected in structure.")

    if __name__ == "__main__":
        main()
