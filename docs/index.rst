.. Burbuja documentation master file, created by
   Lane Votapka on Thu Mar 15 13:55:56 2018.

Welcome to Burbuja's documentation!
=========================================================

Burbuja is an automated bubble-detection tool for finding vapor pockets and local voids within 
molecular dynamics simulation structures making use of explicit solvent.

**Key Features:**

- Automated bubble presence detection with PDB file or any MDTraj-readable format
- Optionally determine more detailed bubble properties such as volume, location, and shape
- Support for bubble analysis of trajectories
- GPU acceleration with CuPy for large systems

.. grid:: 1 1 2 2

    .. grid-item-card:: Getting Started
      :margin: 0 3 0 0
      
      Install Burbuja and get up and running quickly with a basic example.

      .. button-link:: ./getting_started.html
         :color: primary
         :outline:
         :expand:

         To the Getting Started Guide

      

    .. grid-item-card:: Tutorials
      :margin: 0 3 0 0
      
      Step-by-step tutorials covering common usage and advanced features.

      .. button-link:: ./tutorials.html
         :color: primary
         :outline:
         :expand:

         To the Tutorials

      

    .. grid-item-card::  User Guide
      :margin: 0 3 0 0
      
      Comprehensive guide covering all aspects of using Burbuja.

      .. button-link:: ./user_guide.html
         :color: primary
         :outline:
         :expand:

         To the User Guide
      
      

    .. grid-item-card:: API Reference
      :margin: 0 3 0 0
      
      Complete technical reference for all modules, classes, and functions.

      .. button-link:: ./api.html
         :color: primary
         :outline:
         :expand:

         To the API Reference

      

    .. grid-item-card::  Developer Guide
      :margin: 0 3 0 0
      
      Learn how to contribute to Burbuja and extend its functionality.

      .. button-link:: ./developer_guide.html
         :color: primary
         :outline:
         :expand:

         To the Developer Guide

Quick Start Example
===================

Once Burbuja is installed, a full example workflow requires only a PDB file. The example can be run with the following commands:

.. code-block:: bash

   cd tests/
   python ~/Burbuja/Burbuja/burbuja.py data/tb_wrapped_bubble.pdb

For detailed installation instructions, see the :doc:`getting_started` guide.

.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   getting_started
   tutorials
   user_guide
   api
   developer_guide