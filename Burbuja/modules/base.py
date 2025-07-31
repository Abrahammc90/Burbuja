"""
base.py

Base functions and constants for Burbuja.
"""

import numpy as np
import mdtraj

DENSITY_THRESHOLD = 0.25 #0.01  # Density threshold for bubble detection
MINIMUM_BUBBLE_VOLUME = 1.5  # Minimum volume for a bubble to be considered significant
TOTAL_CELLS = 4

def reshape_atoms_to_orthorombic(
        coordinates: np.ndarray,
        unitcell_vectors: np.ndarray,
        n_atoms: int,
        frame_id: int = 0,
        use_cupy: bool = False
        ) -> np.ndarray:
    """
    Wrap the system waterbox based on the orthorhombic unit cell vectors.
    This will always end up being a rectangular box with 90 degree angles,
    although, in general, the side lengths will not be equal.
    """
    assert unitcell_vectors is not None, \
        "Unit cell vectors are required within the mdtraj structure."
    
    vectors = unitcell_vectors[frame_id,:,:]
    lengths = np.diag(vectors)
    coords = coordinates[frame_id, :, :]
    for _ in range(2):
        scale3 = np.floor(coords[:, 2] / lengths[2])
        coords[:, 0] -= scale3 * vectors[2, 0]
        coords[:, 1] -= scale3 * vectors[2, 1]
        coords[:, 2] -= scale3 * vectors[2, 2]
        scale2 = np.floor(coords[:, 1] / lengths[1])
        coords[:, 0] -= scale2 * vectors[1, 0]
        coords[:, 1] -= scale2 * vectors[1, 1]
        scale1 = np.floor(coords[:, 0] / lengths[0])
        coords[:, 0] -= scale1 * vectors[0, 0]

    return lengths

def index_to_index3d(index, ycells, zcells):
    """
    Convert a 1D index to a 3D index (ix, iy, iz) for a grid with ycells and zcells.
    """
    ix = index // (ycells * zcells)
    iy = (index % (ycells * zcells)) // zcells
    iz = index % zcells
    return (ix, iy, iz)

def write_data_array(header, data, filename):
    """
    Write a data array to a file in the DX format.
    """
    ourfile = open(filename, 'w')
    width = header['width']
    height = header['height']
    depth = header['depth']
    originx = header['originx']
    originy = header['originy']
    originz = header['originz']
    resx = header['resx']
    resy = header['resy']
    resz = header['resz']
    total_points = width*height*depth
    header_text = """# Data from metaD_to_dx.py
#
# ENERGY (kcal/mol)
#
object 1 class gridpositions counts %d %d %d
origin  %8.6e  %8.6e  %8.6e
delta %8.6e 0.000000e+00 0.000000e+00
delta 0.000000e+00 %8.6e 0.000000e+00
delta 0.000000e+00 0.000000e+00 %8.6e
object 2 class gridconnections counts %d %d %d
object 3 class array type double rank 0 items %d data follows
""" % (width, height, depth, originx, originy, originz, resx, resy, resz, width, height, depth, total_points)

    tailer = """
attribute "dep" string "positions"
object "regular positions regular connections" class field
component "positions" value 1
component "connections" value 2
component "data" value 3"""
    ourfile.write(header_text)
    data_list = []
    counter = 0
    for i in range(width):
        for j in range(height):
            for k in range(depth):
                data_list.append("%8.6e" % data[i,j,k])
                if (counter % 3 == 2) or (counter == total_points-1):
                    ourfile.write(" ".join(data_list))
                    ourfile.write("\n")
                    data_list = []
                counter += 1

    ourfile.write(tailer)

def get_periodic_image_offsets(
        unitcell_vectors: np.ndarray, 
        lengths: np.ndarray, 
        grid_shape: np.ndarray,
        frame_id: int = 0,
        use_cupy: bool = False
        ) -> np.ndarray:
    """
    When a neighbor of a grid cell is outside the grid, this function
    indicates the index offsets to apply to the coordinates.
    """
    if use_cupy:
        import cupy as cp
        resolution = cp.asarray(np.divide(lengths, grid_shape))
        image_offsets = cp.zeros((3, 3), dtype=cp.int32)
        unitcell_vectors_frame = cp.asarray(unitcell_vectors[frame_id, :, :])
    else:
        resolution = np.divide(lengths, grid_shape)
        image_offsets = np.zeros((3, 3), dtype=np.int32)
        unitcell_vectors_frame = unitcell_vectors[frame_id, :, :]
    for i in range(3):
        image_offsets[i, 0] = unitcell_vectors_frame[i, 0] // resolution[i]
        image_offsets[i, 1] = unitcell_vectors_frame[i, 1] // resolution[i]
        image_offsets[i, 2] = unitcell_vectors_frame[i, 2] // resolution[i]
    del resolution
    del unitcell_vectors_frame
    return image_offsets
