Getting Started
===============

Welcome to Burbuja! This guide will walk you through installing and setting up Burbuja for automated bubble detection.

What is Burbuja?
------------------

Burbuja is an automated bubble-detection tool for finding vapor pockets and local voids within molecular dynamics 
simulation structures making use of explicit solvent.

It provides:

- Automated bubble presence detection with PDB file or any MDTraj-readable format
- Optionally determine more detailed bubble properties such as volume, location, and shape
- Support for bubble analysis of trajectories
- GPU acceleration with CuPy for large systems

Installation
------------

The easiest, quickest way to install Burbuja is to use Mamba. If you don't already have 
Mamba installed, Download the Miniforge install script and run.

.. code-block:: bash

    curl -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
    bash Miniforge3-$(uname)-$(uname -m).sh


Fill out the prompts as they appear.

Once this has been done, set up a new environment:

.. code-block:: bash

    mamba create -n BURBUJA python=3.11 --yes
    mamba activate BURBUJA

If you wish to benefit from GPU acceleration, install CuPy. See the CuPy documentation (https://docs.cupy.dev/en/stable/install.html) for detailed installation instruction for your own system using PyPI, Conda, or from Source, but if you're in a hurry, the following command should work:

.. code-block:: bash

    mamba install cupy

Next, simply install Burbuja. All remaining dependencies should be handled automatically:

.. code-block:: bash

    git clone https://github.com/Abrahammc90/Burbuja.git
    cd Burbuja
    python -m pip install .

One may then optionally run unit tests:

.. code-block:: bash

    pytest

Important Options and Hints
---------------------------

* In general, Burbuja programs can be run with the '-h' argument to see all available options. Please see https://burbuja.readthedocs.io/en/latest for a detailed description of programs and options.

For a complete tutorial, see the :doc:`tutorials` section.

Troubleshooting
---------------

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the :doc:`user_guide` for detailed usage instructions
2. Review the :doc:`api` for complete API documentation
3. See https://burbuja.readthedocs.io/en/latest for Burbuja-specific help
4. Submit issues to the project repository

Next Steps
----------

Now that you have Burbuja installed, you can:

- Follow the :doc:`tutorials` for step-by-step examples
- Read the :doc:`user_guide` for detailed usage information
- Explore the :doc:`api` reference for complete documentation
- Check out the :doc:`developer_guide` if you want to contribute