"""
base.py

Base functions and constants for Burbuja.
"""

import numpy as np

# Density threshold for bubble detection
DEFAULT_DENSITY_THRESHOLD = 0.25 
# Minimum fraction of the total system volume for a bubble to be considered significant
# Radial number of neighbor cells to include in density averaging
DEFAULT_NEIGHBOR_CELLS = 4
DEFAULT_MINIMUM_BUBBLE_VOLUME = 0.1

def reshape_atoms_to_orthorombic(
        coordinates: np.ndarray,
        unitcell_vectors: np.ndarray,
        frame_id: int = 0,
        ) -> np.ndarray:
    """
    Wrap the system coordinates based on orthorhombic unit cell vectors.

    This function wraps all atoms in the given frame into a rectangular
    box with 90 degree angles, using the provided unit cell vectors.
    The side lengths may differ, but the box will always be orthorhombic.

    Args:
        coordinates (np.ndarray): Atomic coordinates, shape (n_frames, n_atoms, 3).
        unitcell_vectors (np.ndarray): Unit cell vectors, shape (n_frames, 3, 3).
        frame_id (int, optional): Frame index to use. Default is 0.

    Returns:
        np.ndarray: Side lengths of the orthorhombic box for the frame.
    """
    MAX_ITER = 10000
    assert unitcell_vectors is not None, \
        "Unit cell vectors are required within the mdtraj structure."
    
    vectors = unitcell_vectors[frame_id,:,:]
    lengths = np.diag(vectors)
    coords = coordinates[frame_id, :, :]
    
    for iteration in range(MAX_ITER):
        scale3 = np.floor(coords[:, 2] / lengths[2])
        coords[:, 0] -= scale3 * vectors[2, 0]
        coords[:, 1] -= scale3 * vectors[2, 1]
        coords[:, 2] -= scale3 * vectors[2, 2]
        scale2 = np.floor(coords[:, 1] / lengths[1])
        coords[:, 0] -= scale2 * vectors[1, 0]
        coords[:, 1] -= scale2 * vectors[1, 1]
        scale1 = np.floor(coords[:, 0] / lengths[0])
        coords[:, 0] -= scale1 * vectors[0, 0]
        
        if not (np.any(scale1) or np.any(scale2) or np.any(scale3)):
            break
    else:
        raise RuntimeError(f"Coordinate wrapping did not converge after {MAX_ITER} iterations")

    return lengths

def index_to_index3d(
        index: int, 
        ycells: int, 
        zcells: int
        ) -> tuple:
    """
    Convert a 1D index to a 3D index (ix, iy, iz) for a grid.

    Args:
        index (int): 1D index.
        ycells (int): Number of grid cells in y direction.
        zcells (int): Number of grid cells in z direction.

    Returns:
        tuple: (ix, iy, iz) 3D indices for the grid.
    """
    ix = index // (ycells * zcells)
    iy = (index % (ycells * zcells)) // zcells
    iz = index % zcells
    return (ix, iy, iz)

def write_data_array(
        header: dict, 
        data: np.ndarray, 
        filename: str
        ) -> None:
    """
    Write a 3D data array to a file in the OpenDX format.

    Args:
        header (dict): DX file header information (width, height, depth, etc).
        data (np.ndarray): 3D data array to write.
        filename (str): Output filename.

    Returns:
        None
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
""" % (width, height, depth, originx, originy, originz, resx, resy, resz, width, 
       height, depth, total_points)

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
    Compute periodic image offsets for grid boundary crossing.

    When a neighbor of a grid cell is outside the grid, this function
    returns the index offsets to apply to the coordinates to wrap them
    back into the periodic box. To ensure consistency between CPU and
    GPU, the calculation is always performed on the CPU and transferred
    to the GPU if needed.

    Args:
        unitcell_vectors (np.ndarray): Unit cell vectors, shape (n_frames, 3, 3).
        lengths (np.ndarray): Box side lengths for the frame.
        grid_shape (np.ndarray): Number of grid cells in each direction (x, y, z).
        frame_id (int, optional): Frame index to use. Default is 0.
        use_cupy (bool, optional): If True, return a CuPy array. Default is False.

    Returns:
        np.ndarray or cupy.ndarray: Image offsets for periodic wrapping, shape (3, 3).
    """
    # Always compute on CPU first for consistency
    resolution = np.divide(lengths, grid_shape)
    image_offsets_cpu = np.zeros((3, 3), dtype=np.int32)
    unitcell_vectors_frame = unitcell_vectors[frame_id, :, :]

    # Do these computations on CPU to avoid precision issues
    if use_cupy:
        import cupy as cp
        unitcell_vectors_frame_cpu = cp.asnumpy(unitcell_vectors_frame)
        resolution_cpu = cp.asnumpy(resolution)
    else:
        unitcell_vectors_frame_cpu = unitcell_vectors_frame
        resolution_cpu = resolution
        
    for i in range(3):
        image_offsets_cpu[i, 0] = unitcell_vectors_frame_cpu[i, 0] // resolution_cpu[i]
        image_offsets_cpu[i, 1] = unitcell_vectors_frame_cpu[i, 1] // resolution_cpu[i]
        image_offsets_cpu[i, 2] = unitcell_vectors_frame_cpu[i, 2] // resolution_cpu[i]

    # Transfer to GPU if needed
    if use_cupy:
        image_offsets = cp.asarray(image_offsets_cpu)
    else:
        image_offsets = image_offsets_cpu
    
    del resolution
    del unitcell_vectors_frame
    return image_offsets
