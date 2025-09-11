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
        coarse_grid_resolution: float = 1.0,
        fine_grid_resolution: float = 0.1,
        use_cupy: bool = False
        ) -> typing.List[structures.Bubble]:
    """
    Perform bubble detection and analysis on the structure.
    """

    # TODO: make structure able to take a list or a structure object from mdtraj
    #  the list being for either multiple frames or pieces of a very large structure.
    start_time = time.time()
    bubbles = []
    if isinstance(structure, str):
        a, b, c, alpha, beta, gamma = parse.get_box_information_from_pdb_file(structure)
        n_frames, n_atoms = parse.get_num_frames_and_atoms_from_pdb_file(structure)
        coordinates = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        masses = np.zeros(n_atoms, dtype=np.float32)
        unitcell_vectors0 = np.array([
            [a, b * np.cos(gamma), c * np.cos(beta)],
            [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],
            [0, 0, c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2)]],
            dtype=np.float32)
        unitcell_vectors0 = np.transpose(unitcell_vectors0, axes=(1, 0))
        unitcell_vectors = np.repeat(unitcell_vectors0[np.newaxis, :, :], n_frames, axis=0)
        if n_frames > 1:
            print("Warning: The PDB file contains multiple frames, and unit cell vectors are "
                  "assumed to be constant across frames. If the unit cell vectors changed during "
                  "the generation of this trajectory, you must load a different trajectory file "
                  "format, such as a DCD file, and provide the topology file to Burbuja in order "
                  "for the correct unit cell vectors to be used for each frame.")
        parse.fill_out_coordinates_and_masses(structure, coordinates, masses, n_frames, n_atoms)
        
    else:
        n_frames = structure.n_frames
        n_atoms = structure.n_atoms
        coordinates = structure.xyz
        #unitcell_vectors = np.transpose(structure.unitcell_vectors, axes=(0, 2, 1))
        unitcell_vectors = structure.unitcell_vectors
        masses = []
        for atom in structure.topology.atoms:
            mass = atom.element.mass if atom.element else 0.0
            masses.append(mass)

    for frame_id in range(n_frames):
        lengths = base.reshape_atoms_to_orthorombic(coordinates, unitcell_vectors, n_atoms, frame_id)
        box_grid = structures.Grid(
            approx_coarse_grid_space=coarse_grid_resolution,
            approx_fine_grid_space=fine_grid_resolution,
            boundaries=lengths)
        box_grid.initialize_coarse_cells(use_cupy=use_cupy)
        box_grid.calculate_coarse_cell_masses(coordinates, masses, n_atoms, frame_id, use_cupy=use_cupy, store_atomic_information=True)
        box_grid.calculate_coarse_cell_densities_noneighboring(use_cupy=use_cupy)
        box_grid.initialize_fine_cells_from_coarse_cells(use_cupy=use_cupy)
        box_grid.calculate_fine_cell_masses_from_coarse_cells(
            lengths, use_cupy=use_cupy)
        box_grid.calculate_fine_cell_densities_neighboring(use_cupy=use_cupy)
        bubble = box_grid.generate_bubble_object()
        bubbles.append(bubble)
    return bubbles

def has_bubble(
        structure: mdtraj.Trajectory,
        coarse_grid_resolution: float = 1.0,
        fine_grid_resolution: float = 0.1,
        use_cupy: bool = False,
        dx_filename_base: str | None = None
    ) -> bool:
    """
    Check if the structure has a bubble based on density threshold.
    """
    bubbles = burbuja(structure, coarse_grid_resolution, fine_grid_resolution, use_cupy=use_cupy)
    found_bubble = False
    
    for i, bubble in enumerate(bubbles):
        if bubble.total_bubble_volume > base.MINIMUM_BUBBLE_VOLUME:
            found_bubble = True
            if dx_filename_base is not None:
                dx_filename = f"{dx_filename_base}_frame_{i}.dx"
                bubble.write_bubble_dx(dx_filename)
                #dens_filename = f"{dx_filename_base}_density_frame_{i}.dx"
                #bubble.write_densities_dx(dens_filename)
                print(f"Bubble detected with volume: {bubble.total_bubble_volume} nm^3. Frame: {i}. "
                    f"Bubble volume map file: {dx_filename}")
            else:
                break
    return found_bubble

def main():
    argparser = argparse.ArgumentParser(
        description="Automatically detect bubbles and vapor pockets and local "
            "voids within molecular dynamics simulation structures making use "
            "of explicit solvent.")
    argparser.add_argument(
        "structure_file", metavar="STRUCTURE_FILE", type=str, 
        help="Path to the input structure or trajectory file - most likely a "
        "PDB file. If a PDB file is provided, a special procedure is used to "
        "process the atom information that saves time and memory. However, if "
        "a different file format is provided, then one must also provide the "
        "topology file (--topology argument), and MDTraj will be used to access "
        "molecular structure information. For large atom counts, it is recommended "
        "for one to use a PDB format, as MDTraj can have trouble reading files "
        "with large numbers of atoms.")
    argparser.add_argument(
        "-t", "--topology", dest="topology",
        metavar="TOPOLOGY", type=str, default=None,
        help="Path to the topology file used by mdtraj to characterize atoms. "
        "If provided, MDTraj will be used to read the structure and "
        "topology files. If this argument is not provided, the structure file "
        "is assumed to be in PDB format, and a special procedure will be used "
        "to efficiently read the atom information needed for bubble detection. "
        "Default: None.")
    argparser.add_argument(
        "-cr", "--coarse_grid_resolution", dest="coarse_grid_resolution",
        metavar="COARSE_GRID_RESOLUTION", type=float, default=1.0,
        help="Coarse grid resolution in nanometers for coarse cell generation. "
        "Default: 1.0 nm.")
    argparser.add_argument(
        "-fr", "--fine_grid_resolution", dest="fine_grid_resolution",
        metavar="FINE_GRID_RESOLUTION", type=float, default=0.1,
        help="Coarse grid resolution in nanometers for fine cell generation. "
        "Default: 0.1 nm.")
    argparser.add_argument(
        "-c", "--use_cupy", dest="use_cupy", default=False,
        help="Enable CuPy for GPU acceleration. Default: False.",
        action="store_true")
    argparser.add_argument(
        "-d", "--detailed_output", dest="detailed_output", default=False,
        help="Enable detailed output, which includes bubble volumes per frame, "
        "and also DX files for bubble visualization. Default: False.",
        action="store_true")
    args = argparser.parse_args()
    args = vars(args)
    structure_file = pathlib.Path(args["structure_file"])
    topology_file = pathlib.Path(args["topology"]) if args["topology"] else None
    coarse_grid_resolution = args["coarse_grid_resolution"]
    fine_grid_resolution = args["fine_grid_resolution"]
    use_cupy = args["use_cupy"]
    detailed_output = args["detailed_output"]

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
    has_bubble_result = has_bubble(structure, coarse_grid_resolution, fine_grid_resolution, 
                                   use_cupy=use_cupy, dx_filename_base=dx_filename_base)
    time_end = time.time()
    elapsed_time = time_end - time_start
    print(f"Bubble detection completed in {elapsed_time:.2f} seconds.")
    
    if has_bubble_result:
        print("The structure has a bubble.")
    else:
        print("No bubble detected in structure.")

if __name__ == "__main__":
    main()