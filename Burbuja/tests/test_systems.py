"""
test_systems.py

Test the systems within the Burbuja test data set.
"""

import os

import pytest
import mdtraj

import Burbuja.burbuja as burbuja

TEST_DIRECTORY = os.path.dirname(__file__)
DATA_DIRECTORY = os.path.join(TEST_DIRECTORY, "data")

def test_bad_box():
    """
    This system has a badly-wrapped triclinic box, so there are some
    planar bubbles in the structure.
    """
    pdb_filename = os.path.join(DATA_DIRECTORY, "bad_box.pdb")
    mdtraj_structure = mdtraj.load(pdb_filename)
    result_str = burbuja.has_bubble(pdb_filename)
    results_mdtraj = burbuja.has_bubble(mdtraj_structure)
    assert result_str == results_mdtraj, \
        "Results should be the same for string and mdtraj input."
    assert result_str == True, \
        "There should be a bubble in the bad_box.pdb structure."
    return
    
def test_bound_meta():
    """
    This system does have a spherical bubble.
    """
    pdb_filename = os.path.join(DATA_DIRECTORY, "bound_meta.pdb")
    mdtraj_structure = mdtraj.load(pdb_filename)
    result_str = burbuja.has_bubble(pdb_filename)
    results_mdtraj = burbuja.has_bubble(mdtraj_structure)
    assert result_str == results_mdtraj, \
        "Results should be the same for string and mdtraj input."
    assert result_str == True, \
        "There should be a bubble in the bound_meta.pdb structure."
    return

def test_bubble_unknown():
    """
    This system has a box that is too large, although the structure
    appears to have normal density, so there are planar bubbles -
    a subtle problem we want to be able to detect.
    """
    pdb_filename = os.path.join(DATA_DIRECTORY, "bubble_unknown.pdb")
    mdtraj_structure = mdtraj.load(pdb_filename)
    result_str = burbuja.has_bubble(pdb_filename)
    results_mdtraj = burbuja.has_bubble(mdtraj_structure)
    assert result_str == results_mdtraj, \
        "Results should be the same for string and mdtraj input."
    assert result_str == True, \
        "There should be a bubble in the bubble_unknown.pdb structure."
    return

def test_hsp90():
    """
    This system has no bubbles, although the protein does wrap around
    between the periodic boundaries, so there's a bulge into the other
    side of the box, which at first glance might look like a bubble,
    but Burbuja should detect that it is not a bubble.
    """
    pdb_filename = os.path.join(DATA_DIRECTORY, "hsp90.pdb")
    mdtraj_structure = mdtraj.load(pdb_filename)
    result_str = burbuja.has_bubble(pdb_filename)
    results_mdtraj = burbuja.has_bubble(mdtraj_structure)
    assert result_str == results_mdtraj, \
        "Results should be the same for string and mdtraj input."
    assert result_str == False, \
        "There should not be a bubble in the hsp90.pdb structure."
    return

def test_tb_traj():
    """
    This system is a trajectory with multiple frames, and has bubbles
    in some of the earlier frames.
    """
    # The trajectory has a changing volume, so the PDB version will not work.
    #pdb_filename = os.path.join(DATA_DIRECTORY, "tb_traj.pdb")
    dcd_filename = os.path.join(DATA_DIRECTORY, "tb_traj.dcd")
    prmtop_filename = os.path.join(DATA_DIRECTORY, "tryp_ben.prmtop")
    mdtraj_structure = mdtraj.load(dcd_filename, top=prmtop_filename)
    #result_str = burbuja.has_bubble(pdb_filename)
    results_mdtraj = burbuja.has_bubble(mdtraj_structure)
    #assert result_str == results_mdtraj, \
    #    "Results should be the same for string and mdtraj input."
    assert results_mdtraj == True, \
        "There should not be a bubble in the tb_traj.pdb structure."
    return

def test_tb_wrapped_bubble():
    """
    This system has a large bubble that should be detected.
    """
    pdb_filename = os.path.join(DATA_DIRECTORY, "tb_wrapped_bubble.pdb")
    mdtraj_structure = mdtraj.load(pdb_filename)
    result_str = burbuja.has_bubble(pdb_filename)
    results_mdtraj = burbuja.has_bubble(mdtraj_structure)
    assert result_str == results_mdtraj, \
        "Results should be the same for string and mdtraj input."
    assert result_str == True, \
        "There should be a bubble in the tb_wrapped_bubble.pdb structure."
    return

def test_triclinic_box_trypsin():
    """
    This system is a properly wrapped triclinic box with no bubbles.
    """
    pdb_filename = os.path.join(DATA_DIRECTORY, "triclinic_box_trypsin.pdb")
    mdtraj_structure = mdtraj.load(pdb_filename)
    result_str = burbuja.has_bubble(pdb_filename)
    results_mdtraj = burbuja.has_bubble(mdtraj_structure)
    assert result_str == results_mdtraj, \
        "Results should be the same for string and mdtraj input."
    assert result_str == False, \
        "There should not be a bubble in the triclinic_box_trypsin.pdb structure."
    return

@pytest.mark.needs_cupy
def test_triclinic_box_trypsin_cupy():
    """
    This system is a properly wrapped triclinic box with no bubbles. Test
    both CPU and GPU implementations.
    """
    pdb_filename = os.path.join(DATA_DIRECTORY, "triclinic_box_trypsin.pdb")
    result_numpy= burbuja.burbuja(pdb_filename)
    result_cupy = burbuja.burbuja(pdb_filename, use_cupy=True)
    for bubble_numpy, bubble_cupy in zip(result_numpy, result_cupy):
        assert bubble_numpy.total_bubble_volume == bubble_cupy.total_bubble_volume, \
            "Bubble volumes should match between numpy and cupy implementations."
        assert bubble_numpy.densities.shape == bubble_cupy.densities.shape, \
            "Bubble densities should have the same shape between numpy and cupy implementations."
    return

@pytest.mark.needs_cupy
def test_tb_traj_cupy():
    """
    This system is a trajectory with multiple frames, and has bubbles
    in some of the earlier frames. Test both CPU and GPU implementations.
    """
    dcd_filename = os.path.join(DATA_DIRECTORY, "tb_traj.dcd")
    prmtop_filename = os.path.join(DATA_DIRECTORY, "tryp_ben.prmtop")
    mdtraj_structure = mdtraj.load(dcd_filename, top=prmtop_filename)
    result_numpy= burbuja.burbuja(mdtraj_structure)
    result_cupy = burbuja.burbuja(mdtraj_structure, use_cupy=True)
    for bubble_numpy, bubble_cupy in zip(result_numpy, result_cupy):
        assert bubble_numpy.total_bubble_volume == bubble_cupy.total_bubble_volume, \
            "Bubble volumes should match between numpy and cupy implementations."
        assert bubble_numpy.densities.shape == bubble_cupy.densities.shape, \
            "Bubble densities should have the same shape between numpy and cupy implementations."
    return