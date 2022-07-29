.. currentmodule:: solver

*************
Basic Syntax
*************

This page provides an overview of the most used Methods and Classes in the GenSol package. This is by no meand a full API guide (which can be found :doc:`here <reference-manual>` ), but just shows the most common Methods, grouped by functionality.


Circuit Definition
===================

Classes and Methods for building a circuit:

.. rubric:: Definition of Solver
.. autosummary::

    Solver
    Solver.__enter__

.. rubric:: Adding and connecting componets
.. autosummary::

    Model
    Model.put
    Solver.put
    connect

.. rubric:: Naming external pins
.. autosummary::

    Pin
    Pin.put

.. rubric:: Adding monitors
.. autosummary::

    add_structure_to_monitors
    Solver.monitor_structure

.. rubric:: Available Models
.. autosummary::

   Waveguide
   UserWaveguide
   BeamSplitter
   Splitter1x2
   Splitter1x2Gen
   PhaseShifter
   PushPullPhaseShifter
   PolRot
   Attenuator
   LinearAttenuator
   Mirror
   PerfectMirror
   FPR
   FPR_NxM
   FPRGaussian
   TH_PhaseShifter

Circuit Simulation
===================

Methods for running the simulation:

.. autosummary::

    Solver.solve
    solve


Data Extraction
==================

Calsses and Methods for extraction of the data after the simulation is run:

.. rubric:: General Methods
.. autosummary::

    SolvedModel
    SolvedModel.get_T
    SolvedModel.get_A
    SolvedModel.get_PH
    SolvedModel.get_output

.. rubric:: Methos for sweeps
.. autosummary::
    
    SolvedModel.get_data
    SolvedModel.get_full_output
    
.. rubric:: Data from monitors
.. autosummary::
    
    SolvedModel.get_monitor


