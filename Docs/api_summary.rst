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

    Attenuator
    BeamSplitter
    FPR
    FPRGaussian
    FPR_NxM
    LinearAttenuator
    LinearNDInterpolator
    Mirror
    ..Model_from_NazcaCM
    PerfectMirror
    PhaseShifter
    PolRot
    ProtectedPartial
    PushPullPhaseShifter
    Splitter1x2
    Splitter1x2Gen
    TH_PhaseShifter
    UserWaveguide
    Waveguide

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
    SolvedModel.get_full_data

.. rubric:: Data from monitors
.. autosummary::
    
    SolvedModel.get_monitor

Export and import of models
==============================
.. autosummary::
    
    Model_from_InPulse
    SolvedModel.export_InPulse
