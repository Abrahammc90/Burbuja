API Reference
=============

This section provides detailed documentation for the most common Python API functions and objects in Burbuja.

.. contents::
	:local:
	:depth: 2

burbuja() Function
------------------

.. py:function:: burbuja(structure, grid_resolution=0.1, use_cupy=False, use_float32=False, density_threshold=0.25, neighbor_cells=4)

	Detect bubbles in a structure or trajectory and return a list of Bubble objects (one per frame).

	:param structure: Path to a structure file (e.g., PDB, DCD) or an mdtraj.Trajectory object.
	:type structure: str or mdtraj.Trajectory
	:param grid_resolution: Grid spacing in nanometers. Default: 0.1
	:type grid_resolution: float
	:param use_cupy: Use CuPy for GPU acceleration. Default: False
	:type use_cupy: bool
	:param use_float32: Use float32 precision for calculations. Default: True
	:type use_float32: bool
	:param density_threshold: Density threshold for void detection (g/L). Default: 0.25
	:type density_threshold: float
	:param neighbor_cells: Number of cells from the central cell to include in the density average. Default: 4
	:type neighbor_cells: int
	:return: List of Bubble objects, one per frame.
	:rtype: list[Bubble]

	**Example:**

	.. code-block:: python

		import mdtraj
		from Burbuja import burbuja
		traj = mdtraj.load('traj.dcd', top='top.prmtop')
		bubbles = burbuja(traj, grid_resolution=0.1, use_cupy=True)
		for i, bubble in enumerate(bubbles):
			 print(f"Frame {i}: Bubble volume = {bubble.total_bubble_volume:.3f} nm^3")

has_bubble() Function
---------------------

.. py:function:: has_bubble(structure, grid_resolution=0.1, use_cupy=False, dx_filename_base=None, density_threshold=0.25, minimum_bubble_fraction=0.005, neighbor_cells=4)

	Quickly check if a structure or trajectory contains a significant bubble.

	:param structure: Path to a structure file or an mdtraj.Trajectory object.
	:type structure: str or mdtraj.Trajectory
	:param grid_resolution: Grid spacing in nanometers. Default: 0.1
	:type grid_resolution: float
	:param use_cupy: Use CuPy for GPU acceleration. Default: True
	:type use_cupy: bool
	:param dx_filename_base: If provided, write DX files for each frame with a bubble. Default: None
	:type dx_filename_base: str or None
	:param density_threshold: Density threshold for void detection (g/L). Default: 0.25
	:type density_threshold: float
	:param minimum_bubble_fraction: Minimum fraction of system volume for a bubble to be considered significant. Default: 0.005
	:type minimum_bubble_fraction: float
	:param neighbor_cells: Number of cells from the central cell to include in the density average. Default: 4
	:type neighbor_cells: int
	:return: True if a significant bubble is found, False otherwise.
	:rtype: bool

	**Example:**

	.. code-block:: python

		from Burbuja import has_bubble
		import mdtraj
		traj = mdtraj.load('traj.dcd', top='top.prmtop')
		contains_bubble = has_bubble(traj, use_cupy=True, dx_filename_base='bubble_output')
		print("Contains bubble?", contains_bubble)

Bubble Object
-------------

The Bubble object represents a detected bubble or void region in a frame. It is returned by the `burbuja()` function.

**Attributes:**

- ``densities``: 3D NumPy array of grid cell densities.
- ``bubble_data``: 3D NumPy array (bool) indicating which cells are part of a bubble.
- ``atoms``: Dictionary of atom records for bubble visualization (PDB format), as an alternative to DX file.
- ``total_atoms``: Number of grid cells identified as part of a bubble.
- ``total_bubble_volume``: Total volume of the bubble (nm³).
- ``total_system_volume``: Total volume of the system (nm³).
- ``dx_header``: Header information for DX file output.
- ``density_threshold``: Density threshold used for bubble detection.

**Methods:**

- ``find(...)``: Identify bubble regions in the grid (used internally).
- ``write_pdb(filename)``: Write bubble atoms to a PDB file for visualization.
- ``write_densities_dx(filename)``: Write the density grid to a DX file.
- ``write_bubble_dx(filename)``: Write the bubble mask to a DX file.

**Example:**

.. code-block:: python

	from Burbuja import burbuja
	import mdtraj
	traj = mdtraj.load('traj.dcd', top='top.prmtop')
	bubbles = burbuja(traj)
	for i, bubble in enumerate(bubbles):
		 print(f"Frame {i}: Bubble volume = {bubble.total_bubble_volume:.3f} nm^3")
		 print(f"System volume: {bubble.total_system_volume:.3f} nm^3")
		 bubble.write_pdb(f"bubble_frame_{i}.pdb")
		 bubble.write_bubble_dx(f"bubble_frame_{i}.dx")

See the tutorials for more advanced usage and visualization examples.