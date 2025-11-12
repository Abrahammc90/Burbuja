Burbuja User Guide
==================

Command-Line Usage
------------------
Burbuja can be used as a command-line tool for detecting bubbles, 
vapor pockets, and voids in explicit solvent molecular dynamics (MD) 
simulation structures and trajectories.

.. note::
    There is also a :doc:`Python API<api>` for more advanced usage; see the API 
    documentation for details.

To run Burbuja from the command line:

.. code-block:: bash

	burbuja STRUCTURE_FILE [options]

Required Argument
~~~~~~~~~~~~~~~~~

**STRUCTURE_FILE**
	Path to the input structure or trajectory file (typically a PDB file). If a PDB file is provided, Burbuja uses a fast, memory-efficient procedure. For other formats, you must also provide a topology file with ``--topology``.

Options
~~~~~~~

- ``-t``, ``--topology TOPOLOGY``
	Path to a topology file (e.g., PRMTOP, PSF) for use with MDTraj. Required if STRUCTURE_FILE is not a PDB. If omitted, STRUCTURE_FILE is assumed to be a PDB.

- ``-r``, ``--grid_resolution GRID_RESOLUTION``
	Grid resolution in nanometers for bubble detection. Default: ``0.1`` nm.

- ``-c``, ``--use_cupy``
	Enable GPU acceleration using CuPy. Default: ``False`` (CPU only).

- ``-d``, ``--detailed_output``
	Enable detailed output, including per-frame bubble volumes and DX files for visualization. Default: ``False``.

- ``-D``, ``--density_threshold DENSITY_THRESHOLD``
	Density threshold (g/L) for void detection. Cells with neighbor-averaged density below this are considered voids. Default: ``0.25`` g/L.

- ``-m``, ``--minimum_bubble_volume MINIMUM_BUBBLE_VOLUME``
	Minimum volume of any contiguous bubble to be considered significant. Default: ``0.005``.

- ``-n``, ``--neighbor_cells NEIGHBOR_CELLS``
	Number of cells from the central cell to include in the density average (neighbor search radius). Default: ``4``.

Examples
~~~~~~~~

Detect bubbles in a PDB file (CPU):

.. code-block:: bash

	burbuja mysystem.pdb

Detect bubbles in a trajectory with a topology file:

.. code-block:: bash

	burbuja traj.dcd -t top.prmtop

Use a custom grid resolution and enable GPU acceleration:

.. code-block:: bash

	burbuja mysystem.pdb -r 0.05 -c

Set a custom density threshold and minimum bubble volume:

.. code-block:: bash

	burbuja mysystem.pdb -D 0.2 -m 0.01

Enable detailed output (per-frame volumes and DX files):

.. code-block:: bash

	burbuja mysystem.pdb -d

Notes
~~~~~

- For large systems, PDB input is recommended for speed and memory efficiency.
- If using a non-PDB format, you must provide a compatible topology file.
- GPU acceleration requires CuPy to be installed and a compatible GPU.
- DX files (if detailed output is enabled) can be visualized with molecular graphics 
  programs such as VMD or PyMOL.

For more details, see the full documentation and API guide.


