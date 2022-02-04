.. currentmodule:: solver

*************
API Reference
*************

Direct methods
===============

.. autosummary::
    :toctree: generated/

    add_param
    add_structure_to_monitors
    connect
    connect_all
    debugfile
    diag_blocks
    logfile
    putpin
    raise_pins
    set_default_params
    solve
    update_default_params

Model
=============

Base Models
---------------------

.. autosummary::
   :toctree: generated/

   Model
   SolvedModel

Methods
---------------------

.. autosummary::
    :toctree: generated/

    Model.expand_mode
    Model.get_A
    Model.get_PH
    Model.get_T
    Model.get_output
    Model.get_pin_modes
    Model.inspect
    Model.pin_mapping
    Model.print_S
    Model.prune
    Model.put
    Model.show_free_pins
    Model.solve
    Model.update_params


    SolvedModel.get_data
    SolvedModel.get_full_output
    SolvedModel.get_monitor
    SolvedModel.set_intermediate



Functional Models
---------------------

.. autosummary::
   :toctree: generated/

   Waveguide
   UserWaveguide
   GeneralWaveguide
   BeamSplitter
   GeneralBeamSplitter
   Splitter1x2
   Splitter1x2Gen
   PhaseShifter
   PushPullPhaseShifter
   PolRot
   Attenuator
   LinearAttenuator
   Mirror
   PerfectMirror
   FPR_NxM
   TH_PhaseShifter
   Model_from_NazcaCM

Solver
=============

.. autosummary::
   :toctree: generated/

    Solver

Methods
---------------------



.. autosummary::
    :toctree: generated/

    Solver.__enter__
    Solver.add_param
    Solver.add_structure
    Solver.connect
    Solver.connect_all
    Solver.cut_structure
    Solver.flatten
    Solver.flatten_top_level
    Solver.inspect
    Solver.map_pins
    Solver.maps_all_pins
    Solver.monitor_structure
    Solver.prune
    Solver.put
    Solver.remove_structure
    Solver.set_param
    Solver.show_connections
    Solver.show_default_params
    Solver.show_free_pins
    Solver.show_pin_mapping
    Solver.show_structures
    Solver.solve
    Solver.update_params


    



Structure
=============

.. autosummary::
   :toctree: generated/

   Structure

Methods
---------------------


.. autosummary::
    :toctree: generated/


    Structure.add_conn
    Structure.add_pin
    Structure.createS
    Structure.get_S_back
    Structure.cut_connections
    Structure.get_in_from
    Structure.get_model
    Structure.get_out_to
    Structure.intermediate
    Structure.join
    Structure.pin
    Structure.print_conn
    Structure.print_pindic
    Structure.print_pins
    Structure.remove_connections
    Structure.remove_pin
    Structure.reset
    Structure.sel_input
    Structure.sel_output
    Structure.split_in_out
    Structure.update_params

S_matrix
=============

.. autosummary::
   :toctree: generated/

   S_matrix

Methods
---------------------


.. autosummary::
    :toctree: generated/

    S_matrix.S_print
    S_matrix.add
    S_matrix.det
    S_matrix.int_complete
    S_matrix.matrix


Pin
=============

.. autosummary::
   :toctree: generated/

   Pin

Methods
---------------------


.. autosummary::
    :toctree: generated/

    Pin.put

