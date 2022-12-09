.. Test_Documentation documentation master file, created by
   sphinx-quickstart on Wed Sep  9 14:56:29 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GenSol: Generic linear Solver for photonic integrated circuits
==================================================================

GenSol is a open source package for linear simulation of photonic circuit, based on the well known scattering matrix method.
It features include:
    - A collection of pre-defined building blocks for easy definition of circuits;
    - Easy calculation of S-parameters of a photonic circuit;
    - Parametric building blocks;
    - Hierarchical circuit definition;
    - Monitors inside the circuit for calculation of power flow.
    - Loading and exporting scattering matrices in `InPulse <https://cordis.europa.eu/project/id/824980>`_ data format.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :numbered:

   installation
   how_it_works

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :numbered:

   api_summary
   jupyter/examples
   reference-manual

   





Indices and tables
========================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
