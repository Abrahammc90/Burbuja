"""
parse.py

This module contains functions for parsing and processing molecular structures
efficiently for large files.
"""

import re

import numpy as np
import mdtraj

unusual_element_names = {
    "POT": "K",  # Potassium
    "SOD": "Na",  # Sodium
    "CLA": "Cl",  # Chlorine
}


def get_box_information_from_pdb_file(pdb_filename):
    """
    Extract box information from a PDB file.
    
    Parameters:
    pdb_filename (str): Path to the PDB file.
    
    Returns:
    tuple: A tuple containing the box dimensions (x, y, z).
    """
    with open(pdb_filename, 'r') as file:
        for line in file:
            if line.startswith("CRYST1"):
                box_info_search = re.search(
                    r"CRYST1\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+"  # a, b, c
                    r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)",  # alpha, beta, gamma
                    line)
                if box_info_search:
                    a = 0.1 * float(box_info_search.group(1))
                    b = 0.1 * float(box_info_search.group(2))
                    c = 0.1 * float(box_info_search.group(3))
                    # Angles are not used in this context, but can be extracted if needed
                    alpha = float(box_info_search.group(4)) * (np.pi / 180.0)
                    beta = float(box_info_search.group(5)) * (np.pi / 180.0)
                    gamma = float(box_info_search.group(6)) * (np.pi / 180.0)
                    return a, b, c, alpha, beta, gamma
                
                else:
                    # If no CRYST1 line is found, raise an error
                    raise ValueError("No CRYST1 line found in the PDB file - "
                                     "box information cannot be extracted.")
                
def get_num_frames_and_atoms_from_pdb_file(pdb_filename):
    """
    Count the number of frames in a PDB file.
    
    Parameters:
    pdb_filename (str): Path to the PDB file.
    
    Returns:
    int: The number of frames in the PDB file.
    """
    with open(pdb_filename, 'r') as file:
        frame_count = 0
        atom_count = 0
        for line in file:
            if line.startswith("MODEL"):
                frame_count += 1
            if frame_count <= 1:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_count += 1
        if frame_count == 0:
            frame_count = 1  # If no ENDMDL lines, assume single frame
        return frame_count, atom_count

def get_mass_from_element_symbol(element_symbol, name_with_spaces):
    try:
        # First try to find a sensible element symbol from columns 76-77
        element = mdtraj.core.element.get_by_symbol(element_symbol)
    except KeyError:
        # otherwise, deduce element from first two characters of atom name
        # remove digits found in some hydrogen atom names
        symbol = name_with_spaces[0:2].strip().lstrip("0123456789")
        if symbol in unusual_element_names:
            symbol = unusual_element_names[symbol]

        try:
            # Some molecular dynamics PDB files, such as gromacs with ffamber force
            # field, include 4-character hydrogen atom names beginning with "H".
            # Hopefully elements like holmium (Ho) and mercury (Hg) will have fewer than four
            # characters in the atom name.  This problem is the fault of molecular
            # dynamics code authors who feel the need to make up their own atom
            # nomenclature because it is too tedious to read that provided by the PDB.
            # These are the same folks who invent their own meanings for biochemical terms
            # like "dipeptide".  Clowntards.
            if len(name_with_spaces) == 4 and name_with_spaces[0:1] == "H":
                element = mdtraj.core.element.hydrogen
            else:
                element = mdtraj.core.element.get_by_symbol(symbol)
        
        except KeyError:
            # OK, I give up
            #element = None
            # Don't give up!!
            symbol = name_with_spaces[0:1].strip().lstrip("0123456789")
            try:
                if len(name_with_spaces) == 4 and name_with_spaces[0:1] == "H":
                    element = mdtraj.core.element.hydrogen
                else:
                    element = mdtraj.core.element.get_by_symbol(symbol)
            except KeyError:
                # If we still can't find the element, return None
                element = None

    if element is None:
        mass = 0.0
    else:
        mass = element.mass
    return mass

def fill_out_coordinates_and_masses(pdb_filename, coordinates, mass_list, n_frames, n_atoms):
    """
    Fill out the coordinates array with data from a PDB file.
    
    Parameters:
    pdb_filename (str): Path to the PDB file.
    coordinates (np.ndarray): Array to fill with coordinates.
    n_frames (int): Number of frames in the PDB file.
    n_atoms (int): Number of atoms in the PDB file.
    
    Returns:
    np.ndarray: Filled coordinates array.
    """
    with open(pdb_filename, 'r') as file:

        frame_id = 0
        atom_id = 0
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                if atom_id < n_atoms:
                    coords = [0.1 * float(line[30:38]), 0.1 * float(line[38:46]), 0.1 * float(line[46:54])]
                    coordinates[frame_id, atom_id, :] = coords
                    
                    name_with_spaces = line[12:16]
                    element_symbol = line[76:78].strip()
                    
                    mass = get_mass_from_element_symbol(element_symbol, name_with_spaces)
                    if mass == 0.0:
                        print(f"Warning: No mass found for atom {name_with_spaces} in frame {frame_id}. "
                              "Assuming mass of 0.0.")
                        print("element_symbol:", element_symbol, "name_with_spaces:", name_with_spaces)
                        #exit()
                    mass_list[atom_id] = mass
                    atom_id += 1
                if atom_id == n_atoms:
                    atom_id = 0
                    frame_id += 1
                    if frame_id == n_frames:
                        break
    return