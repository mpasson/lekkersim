# -------------------------------------------
#
# GenSol Package
#
# Python tool for simulation of abstract photonic circuits
#
# @author: Marco Passoni
#
# ------------------------------------------
from typing import Any, List, Dict, Tuple, Callable

import numpy as np
from copy import copy
from copy import deepcopy
from lekkersim.pin import Pin
from lekkersim.structure import Structure, StructurePin
from lekkersim import sol_list
from lekkersim import logger
from lekkersim.utils import map_args
from lekkersim.model import SolvedModel


class Solver:
    """Class Solver
    This class defines the simulations. It contains all the structures of the optical componets, and has the methods for running the simulation and accessing the results.

    Args:
        structures (list) : list of structures in the solver. Default is None (empty list)
        connections (dict) : dictionary of tuples (structure (Structure), pin (str)) containing connections {(structure1,pin1):(structure2,pin2)}. Default is None (empty dictionary)
        param_dic (dict) :  dictionary of parameters {param_name (str) : param_value (usually float)}. Default is None (empty dictionary). The provided values are assume as the default parameters
        param_mapping (dict): dict for re-definition of the parameters. Useful to include some kind of physic level knowledge in the model. The dictionary is build as:
            >>> {'old_name' : (func, {'new_name' : new_default_vaule})}.
            where func is a function whose parameter are provided by the second dictionary. When simulating, the values of the parameters in the 'new_name' dictionary are passed to func and the retunrded values assigned to 'old_name'.

    """

    structures: List[Structure]

    space = ""
    depth = 0

    def __init__(
        self,
        name: str = None,
        structures: List[Structure] = None,
        connections: Dict[StructurePin, StructurePin] = None,
        pin_mapping: Dict[Pin, StructurePin] = None,
        param_dic: Dict[str, Any] = None,
        param_mapping: Dict[str, str] = None,
    ) -> None:
        """Creator"""
        self.structures = structures if structures is not None else []
        self.connections = connections if connections is not None else {}
        self.connections_list = []
        self.param_dic = param_dic if param_dic is not None else {}
        self.pin_mapping = pin_mapping if pin_mapping is not None else {}
        self.default_params = {"wl": None}
        self.default_params.update(self.param_dic)
        self.param_mapping = {} if param_mapping is None else param_mapping
        self.monitor_st = {}
        for pin1, pin2 in self.connections.items():
            self.connections_list.append(pin1)
            self.connections_list.append(pin2)
            pin1[0].add_conn(pin1[1], *pin2)
            pin2[0].add_conn(pin2[1], *pin1)
        if len(set(self.connections_list)) != len(self.connections_list):
            raise ValueError("Same pin connected multiple time")
        self.free_pins = []
        for st in self.structures:
            for pin in st.pin_list:
                self.free_pins.append(pin)
        for pin in self.connections_list:
            self.free_pins.remove(pin)
        self.name = name

    def __enter__(self):
        """Make the Solver the active solver

        Usage:
            >>> with Solver() as MySol:
            >>>     stuff


        Until the with statement is closed, every change (for example, from put methods) will be applied to MySol
        """
        sol_list.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Delete the solver from the list of the active ones."""
        sol_list.pop()

    def __str__(self):
        """Formatter for printing"""
        if self.name is None:
            return f"Solver object (id={id(self)})"
        else:
            return f"Solver of {self.name} (id={id(self)})"

    def is_empty(self) -> bool:
        """Checks if solver is empy

        Returns:
            Bool: False is model has no pins, True otherwise

        """
        if len(self.structures) > 0:
            return False
        else:
            return True

    def add_structure(self, structure: Structure) -> None:
        """Add a structure to the solver

        Args:
            structure (Structure) : structure to be added

        Returns:
            None
        """
        if structure not in self.structures:
            self.structures.append(structure)
            for pin in structure.pin_list:
                self.free_pins.append(pin)
        else:
            raise ValueError("Structure already present")

        inv_mapping = {
            old_name: new_name for new_name, old_name in structure.param_mapping.items()
        }
        default_dic = {}
        if structure.model is not None:
            default_params = structure.model.default_params
        elif structure.solver is not None:
            default_params = structure.solver.default_params
        else:
            default_params = {}

        for key, value in default_params.items():
            if key in ["R", "w", "pol"]:
                continue
            if key in inv_mapping:
                default_dic[inv_mapping[key]] = value
            else:
                default_dic[key] = value
        self.default_params.update(default_dic)

    def cut_structure(self, structure: Structure) -> None:
        """Remove structure from solver, cutting all the connections to other structures (pins in connected structure are not removed, but freed again)

        Args:
            structure (Structure) : structure to be removed

        Returns:
            None
        """
        if structure not in self.structures:
            raise Exception("Structure {structure} is not in solver {self}")
        self.structures.remove(structure)
        for st in structure.connected_to:
            st.cut_connections(structure)
        copy_dic = copy(self.connections)
        for (st1, pin1), (st2, pin2) in copy_dic.items():
            if st1 is structure or st2 is structure:
                self.connections.pop((st1, pin1))
                self.free_pins.append((st2, pin2))
                self.free_pins.append((st1, pin1))
                self.connections_list.remove((st2, pin2))
                self.connections_list.remove((st1, pin1))
        copy_dic = copy(self.free_pins)
        for st, pin in copy_dic:
            if st is structure:
                self.free_pins.remove((st, pin))
        copy_dic = copy(self.pin_mapping)
        for pinname, (st, pin) in copy_dic.items():
            if st is structure:
                self.pin_mapping.pop(pinname)

    def remove_structure(self, structure: Structure) -> None:
        """Remove structure from solver, also removing all the connections to other structures

        Args:
            structure (Structure) : structure to be removed

        Returns:
            None
        """
        if structure not in self.structures:
            raise Exception("Structure {structure} is not in solver {self}")
        self.structures.remove(structure)
        for st in structure.connected_to:
            st.remove_connections(structure)
        copy_dic = copy(self.connections)
        for (st1, pin1), (st2, pin2) in copy_dic.items():
            if st1 is structure:
                self.connections.pop((st1, pin1))
            if st2 is structure:
                self.connections.pop((st1, pin1))
        copy_dic = copy(self.free_pins)
        for st, pin in copy_dic:
            if st is structure:
                self.free_pins.remove((st, pin))
        copy_dic = copy(self.pin_mapping)
        for pinname, (st, pin) in copy_dic.items():
            if st is structure:
                self.pin_mapping.pop(pinname)

    def monitor_structure(
        self, structure: Structure = None, name: str = "Monitor"
    ) -> None:
        """Add structure to the ones to be monitored for internal modes

        Args:
            structure (Structure): structure to be added
            name (string): name to the associated with this monitor. Default is 'Monitor'
        """
        if structure not in self.structures:
            raise ("Error: structure {structure} not in {self}")
        self.monitor_st[structure] = name

    def connect(
        self,
        structure1: Structure,
        pin1: Pin | str,
        structure2: Structure,
        pin2: Pin | str,
    ) -> None:
        """Connect two different structures in the solver by the specified pins

        Args:
            structure1 (Structure) : first structure
            pin1 (Pin | str) : pin or pin name of first structure
            structure2 (Structure) : second structure
            pin2 (Pin | str) : pin or pin name of second structure

        Returns:
            None
        """
        if isinstance(pin1, str):
            pin1 = structure1.pin[pin1][1]
        if isinstance(pin2, str):
            pin2 = structure2.pin[pin2][1]
        if (structure1, pin1) in self.connections_list:
            if (structure1, pin1) in self.connections and self.connections[
                (structure1, pin1)
            ] == (structure2, pin2):
                return
            if (structure2, pin2) in self.connections and self.connections[
                (structure2, pin2)
            ] == (structure1, pin1):
                return
            raise ValueError("Pin already connected")
        if (structure2, pin2) in self.connections_list:
            raise ValueError("Pin already connected")
        self.connections_list.append((structure1, pin1))
        self.connections_list.append((structure2, pin2))
        self.connections[(structure1, pin1)] = (structure2, pin2)
        self.free_pins.remove((structure1, pin1))
        self.free_pins.remove((structure2, pin2))
        structure1.add_conn(pin1, structure2, pin2)
        structure2.add_conn(pin2, structure1, pin1)

    def connect_all(
        self,
        structure1: Structure,
        basename1: str,
        structure2: Structure,
        basename2: str,
    ) -> None:
        """Connect the two structures using all the pins with the matching basename

        Args:
            structure1 (Structure): first structure
            basename1 (str): basename of pin in structure1
            structure2 (Structure): second structure
            basename2 (str): basename of pin in structure2

        Returns:
            None
        """
        modes1 = set(structure1.get_pin_modenames(pin1))
        modes2 = set(structure2.get_pin_modenames(pin2))
        if modes1 != modes2:
            logger.error(
                f"{pin1} in {structure1} and {pin2} in {structure2} and have differents modes: only matching modes connected"
            )
        modes = modes1.intersection(modes2)
        for m in modes:
            p1 = Pin(basename1, m)
            p2 = Pin(basename2, m)
            self.connect(structure1, p1, structure2, p2)

    def show_free_pins(self) -> None:
        """Print all pins of the structure in the solver whcih are not connected. If a pin mapping exists, it is also reported"""
        print("Free pins of solver: %50s)" % (self))
        for st, pin in self.free_pins:
            try:
                pinname = list(self.pin_mapping.keys())[
                    list(self.pin_mapping.values()).index((st, pin))
                ]
                print("  (%50s, %5s) --> %5s" % (st, pin, pinname))
            except ValueError:
                print("  (%50s, %5s)" % (st, pin))

    def show_structures(self) -> None:
        """Print all structures in the solver"""
        print("Structures and pins of solver: %50s)" % (self))
        for st in self.structures:
            print("  %50s" % (st))

    def show_connections(self) -> None:
        """Print all connected pins in the solver"""
        print("Connection of solver: %50s)" % (self))
        for c1, c2 in self.connections.items():
            print("  (%50s, %5s) <--> (%50s, %5s)" % (c1 + c2))

    def show_pin_mapping(self) -> None:
        """If a pin mapping is defined, print only mapped pins"""
        try:
            for pinname, (st, pin) in self.pin_mapping.items():
                print("  %5s --> (%50s, %5s)" % (pinname, st, pin))
        except AttributeError:
            print("  No mapping defined")

    def map_pins(self, pin_mapping: Dict[Pin | str, StructurePin]):
        """Add mapping of pins

        it the pin mapping is provided with a string, only the basename can be set. For setting the mode name too, please use the Pin object

        Args:
            pin_mapping (dict): dictionary of pin mapping in the form {pin object or pin name : (structure (Structure), pin (str) )}

        Returns:
            None
        """
        converted_pin_mapping = {}
        for pin, (structure, target) in pin_mapping.items():
            _to = Pin(pin) if isinstance(pin, str) else pin
            _from = (
                structure.pin[target]
                if isinstance(target, str)
                else (structure, target)
            )
            converted_pin_mapping[_to] = _from

        self.pin_mapping.update(converted_pin_mapping)

    def solve(self, **kwargs) -> SolvedModel:
        """Calculates the scattering matrix of the solver

        Args:
            kwargs (dict) : paramters in the form param_name=param_value

        Returs:
            SolvedModel : model containing the scattering matrix
        """
        logger.debug(f"Solving {self}")
        if len(self.free_pins) > len(self.pin_mapping):
            logger.warning(
                f"{self}:Running solve without complete mapping: some pins will not be not accessible"
            )
        func = None
        monitor_mapping = None
        self.update_params(kwargs)
        ns = 1
        for par, value in self.param_dic.items():
            self.param_dic[par] = np.reshape(value, -1)
            if len(self.param_dic[par]) == 1:
                continue
            if ns == 1:
                ns = len(self.param_dic[par])
            elif ns != len(self.param_dic[par]):
                raise ValueError("Calling solve with parameters of different shapes")

        for par, value in self.param_dic.items():
            self.param_dic[par] = (
                np.array([value[0] for i in range(ns)]) if len(value) == 1 else value
            )
        for st in self.structures:
            st.update_params(self.param_dic)
        st_list = [st for st in self.structures if st not in self.monitor_st]
        if len(st_list) == 1:
            st_list[0].createS()
        while len(st_list) != 1:
            st_list_pins = [st.pin_count for st in st_list]
            st_list = [st_list[i] for i in np.argsort(st_list_pins)]
            source_st = st_list[0].gone_to
            connected_to = [
                st_list[0].connected_to[i]
                for i in np.argsort(
                    [_.gone_to.pin_count for _ in st_list[0].connected_to]
                )
            ]
            for st in connected_to + st_list[1:]:
                if st.gone_to in st_list:
                    tar_st = st.gone_to
                    break
            new_st = source_st.join(tar_st)
            st_list.remove(source_st)
            st_list.remove(tar_st)
            st_list.append(new_st)
        self.main = st_list[0]

        if self.monitor_st != {}:
            st_list = list(self.monitor_st.keys())
            if len(st_list) == 1:
                st_list[0].createS()
            while len(st_list) != 1:
                source_st = st_list[0].gone_to
                for st in st_list[0].connected_to + st_list[1:]:
                    if st.gone_to in st_list:
                        tar_st = st.gone_to
                        break
                new_st = source_st.join(tar_st)
                st_list.remove(source_st)
                st_list.remove(tar_st)
                st_list.append(new_st)
            self.monitor = st_list[0]
            self.total = self.main.join(self.monitor)
            func, pins = self.main.intermediate(self.monitor, self.pin_mapping)
            monitor_mapping = {
                f"{self.monitor_st[st]}_{pin}": i for (st, pin), i in pins.items()
            }

        else:
            self.total = self.main

        for st in self.structures:
            st.reset()
        mod = self.total.get_model(self.pin_mapping, name=self.name)
        mod.solved_params = deepcopy(self.param_dic)
        if func is not None:
            mod.set_intermediate(func, monitor_mapping)
        self.param_dic = {}
        return mod

    def set_param(self, name: str, value: Any = None) -> None:
        """Set a value for one parameter. This is assued as the new default

        Args:
            name (str) : name of the parameter
            value (usually float) : value of the parameter

        Returns:
            None
        """
        self.default_params.update({name: value})

    def put(
        self,
        source_pin: str | Pin = None,
        target_pin: StructurePin = None,
        param_mapping: Dict[str, str] = None,
    ) -> Structure:
        """Function for putting a Solver in another Solver object, and eventually specify connections

        This function creates a Structure object for the Solver and place it in the current active Solver
        If both pins and pint are provided, the connection also is made.

        Args:
            source_pin (str): pin of model to be connected
            target_pin (tuple): tuple (structure (Structure) , pin (str)) existing structure and pin to which to connect pins of model
            param_mapping (dict): dictionary of {oldname (str) : newname (str)} containing the mapping of the names of the parameters

        Returns:
            Structure: the Structure instance created from the Solver
        """
        if isinstance(source_pin, str):
            for pin in self.pin_mapping:
                if pin.name == source_pin:
                    source_pin = pin
                    break
            else:
                raise ValueError(f"Pin {source_pin} not found in {self}")

        if param_mapping is None:
            param_mapping = {}
        # ST=Structure(solver=deepcopy(self),param_mapping=param_mapping)
        ST = Structure(solver=self, param_mapping=param_mapping)
        sol_list[-1].add_structure(ST)
        if (source_pin is not None) and (target_pin is not None):
            sol_list[-1].connect(ST, source_pin, target_pin[0], target_pin[1])

        #        default_dic={}
        #        for key, value in self.default_params.items():
        #            if key in ['R','w','wl']: continue
        #            if key in param_mapping:
        #                default_dic[param_mapping[key]] = value
        #            else:
        #                default_dic[key] = value
        #        sol_list[-1].default_params.update(default_dic)

        return ST

    def shallow_copy(self):
        """Build a shallow copy of the solver

        This function build a copy of the original solver. Structure objject of the new solver are copies of the old ones. If a structure reference to a model or a solver, the copied structure will still refer to the old model or solver.

        Returs:
            Solver: Shallow copy of the solver
        """
        copy_dic = {}
        new_structures = []
        for st in self.structures:
            if st.model is not None:
                new_st = Structure(
                    model=st.model,
                    param_mapping={val: key for key, val in st.param_mapping.items()},
                )
            elif st.solver is not None:
                new_st = Structure(
                    solver=st.solver,
                    param_mapping={val: key for key, val in st.param_mapping.items()},
                )
            else:
                new_st = deepcopy(st)
            new_structures.append(new_st)
            copy_dic[st] = new_st
        new_connections = {}
        for (st1, pin1), (st2, pin2) in self.connections.items():
            new_connections[(copy_dic[st1], pin1)] = (copy_dic[st2], pin2)
        sol = Solver(
            structures=new_structures,
            connections=new_connections,
            param_dic=self.default_params.copy(),
        )
        pin_mapping = {
            name: (copy_dic[st], pin) for name, (st, pin) in self.pin_mapping.items()
        }
        sol.map_pins(pin_mapping)
        sol.param_mapping = self.param_mapping.copy()
        return sol

    def flatten_top_level(self) -> bool:
        """Flatten top level of a solver

        Reduce the depth of the solver of one level, starting from the top.
        If depth is already 1, nothing is done and the function returns False

        Returns:
            bool: True if reduction is done, False if depth is already 1
        """

        solvers = [
            structure for structure in self.structures if structure.solver is not None
        ]
        for st in solvers:
            st.solver = st.solver.shallow_copy()
        old_conn = copy(self.connections)
        old_mapping = copy(self.pin_mapping)
        if solvers == []:
            return False

        for st in solvers:
            self.cut_structure(st)
            if st.solver.param_mapping != {}:
                logger.error(
                    "During flattening: solver {st.solver} has custom parametes. Simulation results will not be correct."
                )
            for lower_st in st.solver.structures:
                self.add_structure(lower_st)
            for pin1, pin2 in st.solver.connections.items():
                self.connect(*pin1, *pin2)
        for tup1, tup2 in old_conn.items():
            if tup1[0] in solvers:
                if tup2[0] in solvers:
                    self.connect(
                        *tup1[0].solver.pin_mapping[tup1[1]],
                        *tup2[0].solver.pin_mapping[tup2[1]],
                    )
                else:
                    self.connect(*tup1[0].solver.pin_mapping[tup1[1]], *tup2)
            elif tup2[0] in solvers:
                self.connect(*tup1[0], *tup2[0].solver.pin_mapping[tup2[1]])
        new_mapping = {}
        for pin, tup in old_mapping.items():
            if tup[0] in solvers:
                new_mapping[pin] = tup[0].solver.pin_mapping[tup[1]]
        self.map_pins(new_mapping)

        for st in solvers:
            up_dic = {}
            for lower_st in st.solver.structures:
                up_dic[lower_st] = {}
                for top, middle in st.param_mapping.items():
                    if middle in lower_st.param_mapping:
                        bottom = lower_st.param_mapping.pop(middle)
                        up_dic[lower_st][top] = bottom
                    elif top not in lower_st.param_mapping:
                        up_dic[lower_st][top] = middle
            for lower_st in st.solver.structures:
                lower_st.param_mapping.update(up_dic[lower_st])

        return True

    def flatten(self) -> None:
        """Collapse the hyerarchycal structure of the solver in only one level.

        Returns:
            None
        """
        dec = True
        while dec:
            dec = self.flatten_top_level()
        return None

    def _inspect(self, max_depth: int = None) -> None:
        """Recursive function for printing one step of the solver hierarchy

        Args:
            max_depth (int, optional): Maximum depth for printing the circuit. Default is None (full circuit is printed)

        Returns:
            None
        """
        for s in self.structures:
            if max_depth is None or self.__class__.depth < max_depth:
                print(f"{self.space}  {s}")
            if s.solver is not None:
                self.__class__.space = self.__class__.space + "  "
                self.__class__.depth += 1
                s.solver._inspect(max_depth=max_depth)
                self.__class__.space = self.__class__.space[:-2]
                self.__class__.depth -= 1

    def inspect(self, max_depth: int = None) -> None:
        """Print the full hierarchy of the solver

        Args:
            max_depth (int, optional): Maximum depth for printing the circuit. Default is None (full circuit is printed)

        Returns:
            None
        """
        print(f"{self.space}{self}")
        self._inspect(max_depth=max_depth)

    def show_default_params(self) -> None:
        """Print the names of all the top-level parameters and corresponding default value"""
        print(f"Default params of {self}:")
        for name, par in self.default_params.items():
            print(f"  {name:10}: {par}")

    def maps_all_pins(self) -> None:
        """Function for automatically map all pins.

        It scans the unmapped pins and raise them at top level wiht the same name. If one or more pins have the same name, it fails.
        It ignores any pin already mapped by the user.
        """
        for st, pin in self.free_pins:
            if (st, pin) in self.pin_mapping.values():
                continue
            if pin in self.pin_mapping:
                raise Exception("Pins double naming present, cannot map authomatically")
            self.pin_mapping[pin] = (st, pin)

    def update_params(self, update_dic: Dict[str, Any]) -> None:
        """Update the parameters of solver, setting defaults when value is not provides. It takes care of any parameter added with add_param.

        Args:
            update_dic (dict) : dictionary of parameters in the from {param_name (str) : param_value (usually float)}

        Returns:
            None
        """
        # self.param_dic.update(self.default_params)
        # self.param_dic.update(update_dic)
        start_dic = {}
        for name, (func, args) in self.param_mapping.items():
            new_args = {key: value for key, value in args.items()}
            for key, value in new_args.items():
                if key in update_dic:
                    new_args[key] = update_dic[key]
                new_value = func(**new_args)
            start_dic.update({name: new_value})
        self.param_dic.update(self.default_params)
        self.param_dic.update(update_dic)
        self.param_dic.update(start_dic)

    def add_param(
        self, old_name: str, func: Callable, default: Dict[str, Any] = None
    ) -> None:
        """Define a paramter of the solver in term of new paramter(s)

        Args:
            old_name (str) : name of the old parameter to set
            func (function): function linking old parameter to new parameter(s)
            default (dic)  : default value(s) of the new parameter(s).
                Can be None. In this case, introspection is used to try fo find the new parameter(s) and default from func. An error is raised if not possible.
        """
        if default is None:
            var_names = func.__code__.co_varnames
            var_def = func.__defaults__
            if len(var_names) != len(var_def):
                raise ValueError("Not all default provided")
            default = {key: value for key, value in zip(var_names, var_def)}
        up = {old_name: (func, default)}
        self.param_mapping.update(up)
        self.default_params.pop(old_name)
        self.default_params.update(default)

    def prune(self) -> bool:
        """Remove dead branch in the solver hierarchy (the ones ending with an empy solver)

        Returns:
            bool: True if Solver is empty
        """
        # print(f'Entered in {solver}')
        not_empty = []
        copy_list = copy(self.structures)
        for st in copy_list:
            if st.model is not None:
                if st.model.is_empty():
                    self.remove_structure(st)
                else:
                    not_empty.append(st)
                continue
            if st.solver is not None:
                if st.solver.prune():
                    self.remove_structure(st)
                else:
                    not_empty.append(st)
        return len(not_empty) == 0

    def split(self) -> List:
        """Identify sub-circuits inside the solver and splits them into multiple solvers

        Returns:
            list: list of the solvers containing the sub_circuits
        """
        sets = []
        for st in self.structures:
            connected_sets = []
            for _set in sets:
                for target in st.connected_to:
                    if target in _set:
                        _set.add(st)
                        connected_sets.append(_set)
            for _set in connected_sets:
                sets.remove(_set)
            if len(connected_sets) == 0:
                sets.append(set([st] + st.connected_to))
            else:
                sets.append(set().union(*connected_sets))
        solvers = []
        for i, _set in enumerate(sets):
            connections = {
                t1: t2 for t1, t2 in self.connections.items() if t1[0] in _set
            }
            mapping = {
                name: pin for name, pin in self.pin_mapping.items() if pin[0] in _set
            }
            solvers.append(
                Solver(
                    name=f"{self.name}_{i}",
                    structures=list(_set),
                    connections=connections,
                    pin_mapping=mapping,
                    param_mapping=self.param_mapping,
                    param_dic=self.param_dic,
                )
            )
        return solvers


sol_list.append(Solver())


def putpin(name: str, tup: StructurePin) -> None:
    """Maps a pin of the current active solver

    Args:
        name (str) : name of the new pin
        tup (tuple) : tuple of (structure (Structure), pin (str)) containing the data to the pin to be mapped
    """
    sol_list[-1].map_pins({name: tup})


def connect(tup1: StructurePin, tup2: StructurePin) -> None:
    """Connect two structures in the active Solver

    Args:
        tup1 (tuple) : tuple of (structure (Structure), pin (str)) containing the data of the first pin
        tup1 (tuple) : tuple of (structure (Structure), pin (str)) containing the data of the second pin
    """
    sol_list[-1].connect(tup1[0], tup1[1], tup2[0], tup2[1])


def connect_all(
    structure1: Structure, pin1: str, structure2: Structure, pin2: str
) -> None:
    """Connect in the active solver the two structures using all the pins with the matching basename

    Args:
        structure1 (Structure): first structure
        pin1 (str): basename of pin in structure1
        structure2 (Structure): second structure
        pin2 (str): basename of pin in structure2

    Returns:
        None
    """
    sol_list[-1].connect_all(structure1, pin1, structure2, pin2)


def add_param(old_name: str, func: Callable, default: Dict[str, Any] = None) -> None:
    """Define a paramter of the active solver in term of new paramter(s)

    Args:
        old_name (str) : name of the old parameter to set
        func (function): function linking old parameter to new parameter(s)
        default (dic)  : default value(s) of the new parameter(s).
            Can be None. In this case, introspection is used to try fo find the new parameter(s) and default from func. An error is raised if not possible.
    """
    sol_list[-1].add_param(old_name, func, default=default)


def set_default_params(dic: Dict[str, Any]) -> None:
    """Set default parameters for the solver

    The provided dict will oervwrite the default parameters. All pre-existing parameters will be deleted

     Args:
         dic (dict): dictionary of the default parameters {param_name (str) : default_value (usually float)}
    """
    sol_list[-1].default_params = dic


def update_default_params(dic: Dict[str, Any]) -> None:
    """Update default parameters for the solver

    The provided dict will upadte the default parametes. Not included pre-existing parmeters will be kept.

    Args:
        dic (dict): dictionary of the default parameters {param_name (str) : default_value (usually float)}
    """
    sol_list[-1].default_params.update(dic)


def raise_pins() -> None:
    """Raise all pins in the solver. It reuiqres unique pin naming, otherwaise an error is raised

    It scans the unmapped pins and raise them at top level wiht the same name. If one or more pins have the same name, it fails.
    It ignores any pin already mapped by the user.
    """
    sol_list[-1].maps_all_pins()


def solve(**kwargs) -> SolvedModel:
    """Solve active solver and returns the model

    Args:
        **kwargs : parameters for the simulation

    Returns:
        Model: Model of the active solver.
    """
    return sol_list[-1].solve(**kwargs)


def add_structure_to_monitors(structure: Structure, name: str = "Monitor") -> None:
    """Add structure to the ones to be monitored for internal modes. It effects the active solver

    Args:
        structure (Structure): structure to be added
        name (string): name to the associated with this monitor. Default is 'Monitor'
    """
    sol_list[-1].monitor_structure(structure=structure, name=name)
