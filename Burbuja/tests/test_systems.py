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

def test_membrane():
    """
    This system is a membrane with no bubbles.
    """
    pdb_filename = os.path.join(DATA_DIRECTORY, "membrane_system.pdb")
    result_str = burbuja.has_bubble(pdb_filename)
    assert result_str == False, \
        "There should not be a bubble in the membrane_system.pdb structure."
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
    for i, (bubble_numpy, bubble_cupy) in enumerate(zip(result_numpy, result_cupy)):
        # Use relative tolerance for volume comparison due to float32 vs float64 precision
        vol_numpy = bubble_numpy.total_bubble_volume
        vol_cupy = bubble_cupy.total_bubble_volume
        
        if vol_numpy == 0 and vol_cupy == 0:
            # Both are zero - perfect match
            continue
        elif vol_numpy == 0 or vol_cupy == 0:
            # One is zero, other is not - check if the non-zero is very small
            max_vol = max(vol_numpy, vol_cupy)
            assert max_vol < 1e-6, f"Frame {i}: One implementation found bubble, other didn't. Volumes: {vol_numpy:.6f} vs {vol_cupy:.6f}"
        else:
            # Both non-zero - use relative tolerance
            rel_diff = abs(vol_numpy - vol_cupy) / max(vol_numpy, vol_cupy)
            assert rel_diff < 1e-4, f"Frame {i}: Bubble volumes differ too much. NumPy: {vol_numpy:.6f}, CuPy: {vol_cupy:.6f}, rel_diff: {rel_diff:.6f}"
        
        assert bubble_numpy.densities.shape == bubble_cupy.densities.shape, \
            f"Frame {i}: Bubble densities should have the same shape between numpy and cupy implementations."
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
    
    # Track frames with bubbles for each implementation
    numpy_bubble_frames = []
    cupy_bubble_frames = []
    
    for i, (bubble_numpy, bubble_cupy) in enumerate(zip(result_numpy, result_cupy)):
        # Use relative tolerance for volume comparison due to float32 vs float64 precision
        vol_numpy = bubble_numpy.total_bubble_volume
        vol_cupy = bubble_cupy.total_bubble_volume
        
        # Track which frames have bubbles
        if vol_numpy > 1e-6:
            numpy_bubble_frames.append(i)
        if vol_cupy > 1e-6:
            cupy_bubble_frames.append(i)
        
        if vol_numpy == 0 and vol_cupy == 0:
            # Both are zero - perfect match
            continue
        elif vol_numpy == 0 or vol_cupy == 0:
            # One is zero, other is not - check if the non-zero is very small
            max_vol = max(vol_numpy, vol_cupy)
            assert max_vol < 1e-4, f"Frame {i}: One implementation found bubble, other didn't. Volumes: {vol_numpy:.6f} vs {vol_cupy:.6f}"
        else:
            # Both non-zero - use generous tolerance due to cumulative float32 vs float64 differences
            rel_diff = abs(vol_numpy - vol_cupy) / max(vol_numpy, vol_cupy)
            assert rel_diff < 0.15, f"Frame {i}: Bubble volumes differ too much. NumPy: {vol_numpy:.6f}, CuPy: {vol_cupy:.6f}, rel_diff: {rel_diff:.6f}"
        
        assert bubble_numpy.densities.shape == bubble_cupy.densities.shape, \
            f"Frame {i}: Bubble densities should have the same shape between numpy and cupy implementations."
    
    # Ensure both implementations detect bubbles in the same general frames
    bubble_frame_overlap = set(numpy_bubble_frames) & set(cupy_bubble_frames)
    min_required_overlap = max(1, min(len(numpy_bubble_frames), len(cupy_bubble_frames)) * 0.8)
    
    assert len(bubble_frame_overlap) >= min_required_overlap, \
        f"Implementations should detect bubbles in similar frames. NumPy: {numpy_bubble_frames}, CuPy: {cupy_bubble_frames}, Overlap: {list(bubble_frame_overlap)}"
    
    return