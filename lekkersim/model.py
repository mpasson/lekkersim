# -------------------------------------------
#
# GenSol Package
#
# Python tool for simulation of abstract photonic circuits
#
# @author: Marco Passoni
#
# -------------------------------------------


"""File containing the model calls and related methods"""

from __future__ import annotations
from typing import Any, List, Dict, Tuple, Callable, TYPE_CHECKING, Union, Optional, Set

from collections import defaultdict
import functools
from copy import deepcopy
from copy import copy
import yaml
import io
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import pandas as pd
from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.integrate import quad_vec

import lekkersim.structure
from lekkersim.pin import Pin
from lekkersim import sol_list
from lekkersim import logger
import lekkersim
import lekkersim.log
from lekkersim.utils import line, GaussianBeam, ProtectedPartial


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def diag_blocks(array_list: List[np.ndarray]) -> np.ndarray:
    """Function building a block diagonal array for list of array

    Args:
        array_list (list): list of ndarrays. Each array is a 2D square array.

    Returns:
         ndarray: 2D array with the arrays in array_list as blocks in the diagonal
    """
    for A in array_list:
        if np.shape(A)[0] != np.shape(A)[1]:
            raise ValueError("Matrix is not square")
    Nl = [np.shape(A)[0] for A in array_list]
    N = sum(Nl)
    M = np.zeros((N, N), dtype=complex)
    m = 0
    for n, A in zip(Nl, array_list):
        M[m : m + n, m : m + n] = A
        m += n
    return M


class Model:
    """Class Model

    It contains the model (definition of the scattering matrix) for each base photonic building block
    This is the general template for the class, no model is defined here.
    Each model is a separate class based on the main one.
    The only use case for using this class directly is to call it without arguments to create an empy
    model that will be eliminated if prune is called

    """

    def __init__(
        self,
        pin_dic: Dict[Pin, int] = None,
        param_dic: Dict[str, Any] = None,
        Smatrix: np.ndarray = None,
    ) -> None:
        """Initialize the model

        Args:
            pin_dic (dictionary): Dictionary of pins and relative position in the scattering matrix
            param_dic (dictionary): dictionary {'param_name':param_value} containing the definition of
                the model's parameters.
            Smatrix (ndarray) : Fixed S_matrix of the model

        """
        self.pin_dic = {} if pin_dic is None else pin_dic
        self.N = len(self.pin_dic)
        if Smatrix is not None:
            if self.N != np.shape(Smatrix)[-1]:
                self.N = np.shape(Smatrix)[-1]
        self.S = np.identity(self.N, complex) if Smatrix is None else Smatrix
        self.param_dic = {} if param_dic is None else param_dic
        self.default_params = deepcopy(self.param_dic)
        self.update_pins()

    def update_pins(self) -> dict[str, Pin]:
        pins = {}
        for pin in self.pin_dic:
            if pin.name not in pins:
                pins[pin.name] = pin
            else:
                raise ValueError(
                    f"In Model {self}: Pin {repr(pin)} and Pin {repr(pins[pin.name])} maps to same pin name."
                )
        self.pin = pins
        return self.pin

    def is_empty(self) -> bool:
        """Checks if model is empy

        Returns:
            Bool: False is model has no pins, True otherwise

        """
        if len(self.pin_dic) > 0:
            return False
        else:
            return True

    def _expand_S(self) -> np.ndarray:
        self.N = self.N // self.np
        S = self._create_S()
        self.N = self.N * self.np
        # self.S= diag_blocks(self.np*[S])
        return diag_blocks(self.np * [S])

    def inspect(self) -> None:
        """Function that print self. It mocks the same function of Solver"""
        print(f"{self}")

    def create_S(self) -> np.ndarray:
        """Function for returning the scattering matrix of the model

        Returns:
            ndarray: Scattering matrix of the model
        """
        return self.S

    def print_S(self, func: Callable = np.abs) -> None:
        """Function for nice printing of scattering matrix in agreement with pins

        Args:
            func (callable): function to be applied at the scattering matrix before returning.
                if None (default), the raw complex coefficients are printed
        """
        func = (lambda x: x) if func is None else func
        a = list(self.pin_dic.keys())
        ind = list(self.pin_dic.values())
        indsort = np.argsort(a)
        a = [a[i] for i in indsort]
        indsort = np.array(ind)[indsort]
        for pin, i in self.pin_dic.items():
            print(pin, i)
        S = self.create_S()
        I, J = np.meshgrid(indsort, indsort, indexing="ij")
        S = S[0, I, J] if len(np.shape(S)) == 3 else S[I, J]
        S = func(S) if func is not None else S
        st = "            "
        for p in a:
            st += f" {p:8} "
        st += "\n"
        for i, pi in enumerate(a):
            st += f" {pi:8} "
            for j, pj in enumerate(a):
                pr = S[i, j]
                st += f" {pr:8.4f} "
            st += "\n"
        print(st)

    def S2PD(self, func: Callable = None) -> pd.DataFrame:
        """Function for returning the Scattering Matrix as a PD Dataframe

        Args:
            func (callable): function to be applied at the scattering matrix before returning.
                if None (default), the raw complex coefficients are returned

        Returns:
            Pandas DataFrame: Scattering Matrix with name of pins
        """
        a = [pin.name for pin in self.pin_dic]
        ind = list(self.pin_dic.values())
        indsort = np.argsort(a)
        a = [a[i] for i in indsort]
        indsort = np.array(ind)[indsort]
        S = self.create_S()
        I, J = np.meshgrid(indsort, indsort, indexing="ij")
        S = S[0, I, J] if len(np.shape(S)) == 3 else S[I, J]
        S = func(S) if func is not None else S
        data = pd.DataFrame(data=S, index=a, columns=a)
        return data

    def get_T(self, pin1: str, pin2: str) -> float:
        """Function for returning the energy transmission between two ports

        If the two ports are the same the returned value has the meaning of reflection

        Args:
            pin1 (str): Name of input pin
            pin2 (str): Name of output pin

        Returns:
            float: Energy transmission between the ports
        """
        if np.shape(self.S)[0] > 1:
            logger.warning(
                f"{self}:Using get_T on a sweep solve. Consider using get_data"
            )
        return (
            np.abs(
                self.S[
                    0,
                    self.pin_dic[self.pin[pin1]],
                    self.pin_dic[self.pin[pin2]],
                ]
            )
            ** 2.0
        )

    def get_PH(self, pin1: str, pin2: str) -> float:
        """Function for returning the phase of the transmission between two ports

        If the two ports are the same the returned value has the meaning of reflection

        Args:
            pin1 (str): Name of input pin
            pin2 (str): Name of output pin

        Returns:
            float: Phase of the transmission between the ports
        """
        if np.shape(self.S)[0] > 1:
            logger.warning(
                f"{self}:Using get_PH on a sweep solve. Consider using get_data"
            )
        return np.angle(
            self.S[0, self.pin_dic[self.pin[pin1]], self.pin_dic[self.pin[pin2]]]
        )

    def get_A(self, pin1: str, pin2: str) -> complex:
        """Function for returning complex amplitude of the transmission between two ports

        Args:
            pin1 (str): Name of input pin
            pin2 (str): Name of output pin

        Returns:
            float: Complex amplitude of the transmission between the ports
        """
        if np.shape(self.S)[0] > 1:
            logger.warning(
                f"{self}:Using get_A on a sweep solve. Consider using get_data"
            )
        return self.S[0, self.pin_dic[self.pin[pin1]], self.pin_dic[self.pin[pin2]]]

    def expand_mode(self, mode_list: List[str]):
        """This function expands the model by adding additional modes.

        For each pin a number of pins equal the length of mode_list will be created. The pin names will be "{pinname}_{modename}}".
        Each mode will have the same behavior.

        Args:
            mode_list (list) : list of strings containing the modenames.

        Returns:
            Model : new model with expanded modes
        """
        for pin in self.pin_dic:
            if pin.mode_name is not None:
                raise Exception("Model already has modes")
        self.np = len(mode_list)
        self.mode_list = mode_list
        new_pin_dic = {}
        for pin, n in self.pin_dic.items():
            for i, mode in enumerate(self.mode_list):
                new_pin_dic[Pin(pin.name, mode)] = i * self.N + n
        self.pin_dic = new_pin_dic
        self.pin: dict[str, Pin] = {pin.name: pin for pin in self.pin_dic}
        self._create_S = self.create_S
        self.create_S = self._expand_S
        self.N = self.N * self.np
        return self

    def get_output(
        self, input_dic: Dict[str, float | complex], power: bool = True
    ) -> Dict[str, float]:
        """Returns the outputs from all ports of the model given the inputs amplitudes

        Args:
            input_dic (dict): dictionary {pin_name (str) : input_amplitude (complex)}.
                Dictionary containing the complex amplitudes at each input port. Missing port are assumed
                with amplitude 0.0.
            power (bool): If True, returned values are power transmissions. If False, complex amplitudes
                are instead returned. Default is True.

        Returns:
            dict: Dictionary containing the outputs in the form {pin_name (str) : output (float or complex)}
        """
        if np.shape(self.S)[0] > 1:
            logger.warning(
                f"{self}:Using get_output on a sweep solve. Consider using get_full_output"
            )
        input_pin_dic: dict[Pin, float | complex] = {
            self.pin[name]: value for name, value in input_dic.items()
        }

        l1 = list(self.pin_dic.keys())
        l2 = list(input_pin_dic.keys())
        for pin in l2:
            l1.remove(pin)
        if l1 != []:
            for pin in l1:
                input_pin_dic[pin] = 0.0 + 0.0j
        u = np.zeros(self.N, complex)
        for pin, i in self.pin_dic.items():
            u[i] = input_pin_dic[pin]
        d = np.dot(self.S[0, :, :], u)
        out_dic = {}
        for pin, i in self.pin_dic.items():
            out_dic[pin.name] = np.abs(d[i]) ** 2.0 if power else d[i]
        return out_dic

    def put(
        self,
        source_pin: Optional[str | Pin] = None,
        target_pin: Optional[lekkersim.StructurePin] = None,
        param_mapping: Optional[dict] = None,
    ) -> lekkersim.Structure:
        """Function for putting a model in a Solver object, and eventually specify connections

         This function creates a Structure object for the model and place it in the current active Solver
         If both pins and pint are provided, the connection also is made.

        Args:
             source_pin (str): pin of model to be connected
             target_pin (tuple): tuple (structure (Structure) , pin (str)) existing structure and pin to
                which to connect pins of model
             param_mapping (dict): dictionary of {oldname (str) : newname (str)} containing the mapping
                of the names of the parameters

         Returns:
             Structure: the Structure instance created from the model
        """
        if isinstance(source_pin, str):
            source_pin = self.pin[source_pin]

        param_mapping = param_mapping or {}
        ST = lekkersim.structure.Structure(model=self, param_mapping=param_mapping)
        sol_list[-1].add_structure(ST)
        if source_pin is not None and target_pin is not None:
            sol_list[-1].connect(ST, source_pin, target_pin[0], target_pin[1])
        return ST

    def solve(self, **kargs) -> lekkersim.SolvedModel:
        """Function for returning the solved model

        This function is to align the behavior of the Model and Solver class.

        Args:
            kwargs: dictionary of the parameters {param_name (str) : param_value (usually float)}

        Returns:
            model: solved model of self
        """
        logger.debug(f"Solving Model {self}")
        self.param_dic.update(self.default_params)
        ns = 1
        for name in kargs:
            kargs[name] = np.reshape(kargs[name], -1)
            if len(kargs[name]) == 1:
                continue
            if ns == 1:
                ns = len(kargs[name])
            else:
                if ns != len(kargs[name]):
                    raise Exception("Different lengths between parameter arrays")
        up_dic = {}
        S_list = []
        for i in range(ns):
            for name, values in kargs.items():
                up_dic[name] = values[0] if len(values) == 1 else values[i]
            self.param_dic.update(up_dic)
            S_list.append(self.create_S())
        return SolvedModel(
            pin_dic=self.pin_dic,
            param_dic=kargs,
            Smatrix=np.array(S_list),
            name=type(self).__name__,
        )

    def show_free_pins(self) -> None:
        """Function for printing pins of model"""
        print(f"Pins of model {self} (id={id(self)})")
        for pin, n in self.pin_dic.items():
            print(f"{pin:5s}:{n:5}")
        print("")

    def pin_mapping(self, pin_mapping: Dict[Pin, Pin]):
        """Function for changing the names of the pins of a model

        Args:
            pin_mapping (dict): Dictionary containing the mapping of the pin names.
                Format is {'oldname' : 'newname'}

        Returns:
            model: model of self with updated pin names
        """
        for pin in copy(self.pin_dic):
            if pin in pin_mapping:
                n = self.pin_dic.pop(pin)
                self.pin_dic[pin_mapping[pin]] = n
        return self

    def update_params(self, update_dic: Dict[str, Any]) -> None:
        """Update the parameters of model, setting defaults when value is not provides

        Args:
            update_dic (dict) : dictionary of parameters in the from
                {param_name (str) : param_value (usually float)}
        """
        self.param_dic.update(self.default_params)
        self.param_dic.update(update_dic)

    def prune(self) -> None:
        """Check if the model is empty

        Returns:
            bool: True if the model is empty, False otherwise
        """
        return self.pin_dic == {}

    def get_pin_modes(self, basename: str) -> List[str]:
        """Parse the pins for locating the pins with the same base name

        Assumes for the pins a name in the form pinname_modename.

        Args:
            basename (str): base name of the pin

        Returns:
            list: list of modenames for which pinname==pin
        """
        return [pin.mode_name for pin in self.pin_dic if pin.basename == basename]

    def get_pin_basenames(self) -> List[str]:
        """Returns a list of the basenames of the pins"""
        basenames = [pin.basename for pin in self.pin_dic]
        return list(set(basenames))

    def __str__(self):
        """Formatter function for printing"""
        return f"Model object (id={id(self)}) with pins: {list(self.pin_dic)}"


class SolvedModel(Model):
    """Class for storing data of a solver mode.

    Do not use this class directly. It is returned from all solve methods. It is convenient for extracting data
    """

    def __init__(
        self,
        pin_dic: Dict[Pin, int],
        param_dic: Dict[str, Any],
        Smatrix: np.ndarray,
        int_func: Optional[Callable] = None,
        monitor_mapping: Optional[Dict[str, Tuple[lekkersim.Structure, str]]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the model

        Args:
            pin_dic (dictionary): list of strings containing the model's pin names
            param_dic (dictionary): dictionary {'param_name':param_value} containing the definition of
                the model's parameters.
            Smatrix (ndarray) : Fixed S_matrix of the model
            int_func (Callable): function for returning the modal coefficient in between two part of the scattering matrix
            monitor_mapping (Dict): Dictionry mapping the name of the monitor to the connected pin
            name (str): name of the solved model

        """
        super().__init__(pin_dic=pin_dic, param_dic=param_dic, Smatrix=Smatrix)
        self.solved_params = deepcopy(param_dic)
        self.ns = np.shape(Smatrix)[0]
        self.int_func = int_func
        self.monitor_mapping = {} if monitor_mapping is None else monitor_mapping
        self.name = name

    def set_intermediate(
        self,
        int_func: Callable,
        monitor_mapping: Dict[str, Tuple[lekkersim.Structure, str]],
    ):
        """Methods for setting the function and mapping for monitors

        Args:
            int_func (callable): Function linking the modal amplitudes at the monitor port to the
                inputs amplitudes
            monitor_mapping (dict): Dictionary connecting the name of the monitor ports with the
                index in the amplitude arrays
        """
        self.int_func = int_func
        self.monitor_mapping = monitor_mapping

    def get_data(self, pin1: str, pin2: str) -> pd.DataFrame:
        """Function for returning transmission data between two ports

        Args:
            pin1 (str): name of the input pin
            pin2 (str): name of the output pin

        Returns:
            pandas DataFrame: Dataframe containing the data. It contains one column per parameter
                given to solve, plus the following:
                    'T'         : Transmission in absolute units
                    'dB'        : Transmission in dB units
                    'Phase'     : Phase of the transmission
                    'Amplitude' : Complex amplitude of the transmission

        """
        params = {}
        if self.ns == 1:
            params = deepcopy(self.solved_params)
        else:
            for name, values in self.solved_params.items():
                if len(values) == 1:
                    params[name] = np.array([values[0] for i in range(self.ns)])
                elif len(values) == self.ns:
                    params[name] = values
                else:
                    raise Exception("Not able to convert to pandas")

        i1, i2 = self.pin_dic[self.pin[pin1]], self.pin_dic[self.pin[pin2]]
        params["T"] = np.abs(self.S[:, i1, i2]) ** 2.0
        params["dB"] = 20.0 * np.log10(np.abs(self.S[:, i1, i2]))
        params["Phase"] = np.angle(self.S[:, i1, i2])
        params["Amplitude"] = self.S[:, i1, i2]
        pan = pd.DataFrame.from_dict(params)
        return pan

    def get_full_output(
        self, input_dic: Dict[str, float], power: bool = True
    ) -> pd.DataFrame:
        """Function for getting the output do the system given the inputs

        Args:
            input_dic (dict): Dictionary of the input amplitudes. Format is
                {'pinname' : amplitude (float or complex)}. Missing pins are  assumed to have 0 amplitude.
            power (bool): if True, power (in absolute units) between the ports is returned, otherwise the
                complex amplitude is returned. Default is True

        Returns:
            pandas DataFrame: DataFrame with the outputs. It has one column for each parameter
                given to solve plus one column for each pin.
        """

        input_pin_dic: dict[Pin, float | complex] = {
            self.pin[name]: value for name, value in input_dic.items()
        }

        params = {}
        if self.ns == 1:
            params = deepcopy(self.solved_params)
        else:
            for name, values in self.solved_params.items():
                if len(values) == 1:
                    params[name] = np.array([values[0] for i in range(self.ns)])
                elif len(values) == self.ns:
                    params[name] = values
                else:
                    raise Exception("Not able to convert to pandas")

        l1 = list(self.pin_dic.keys())
        l2 = list(input_pin_dic.keys())
        for pin in l2:
            l1.remove(pin)
        if l1 != []:
            for pin in l1:
                input_pin_dic[pin] = 0.0 + 0.0j
        u = np.zeros(self.N, complex)
        for pin, i in self.pin_dic.items():
            u[i] = input_pin_dic[pin]

        output = np.matmul(self.S, u)

        for pin, i in self.pin_dic.items():
            params[pin.name] = np.abs(output[:, i]) ** 2.0 if power else output[:, i]
        pan = pd.DataFrame.from_dict(params)
        return pan

    def get_full_data(self) -> pd.DataFrame:
        """Returns the scattering matrix for all the solved parametes in form of padas DataFrame"""
        params: dict[Union[str, tuple], Any] = {}
        if self.ns == 1:
            _copy = deepcopy(self.solved_params)
            params.update(_copy)
        else:
            for name, values in self.solved_params.items():
                if len(values) == 1:
                    params[name] = np.array([values[0] for i in range(self.ns)])
                elif len(values) == self.ns:
                    params[name] = values
                else:
                    raise Exception("Not able to convert to pandas")
        for p1, i1 in self.pin_dic.items():
            for p2, i2 in self.pin_dic.items():
                params[(p1, p2)] = self.S[:, i1, i2]
        return pd.DataFrame(params)

    def _build_metadata(
        self,
        parameter_name_mapping: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """Builds the metadata dict for the InPulse scattering matrix export

        Args:
            parameter_name_mapping (dict): mapping between parameter name in the model and exported file.
                The for of the dictionary is {<model-parameter-name>:<exported-parameter-name>}.
                Not mapped parameter are exported with their origina name.
            units (dict): Units of the exported parameters, in the form: {<exported-parameter-name>:<unit>}
                For parameters with no units explicit map to None is advised.

        Returns:
            dict: nested dictionary of the metadata
        """
        metadata = {
            "_schema": "InPulse S-Matrix RAW DATA",
            "_schema_version": "1.0",
            "_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "_description": "Scattering matrix model exported from gensol",
            "units": {
                "abs2": None,
                "phase": "rad",
            },
        }
        parameter_name_mapping = (
            {} if parameter_name_mapping is None else parameter_name_mapping
        )
        units = {} if units is None else units
        solved_parameters_units = {
            parameter_name_mapping.get(par, par): None for par in self.solved_params
        }
        for par in solved_parameters_units:
            if par not in units:
                logger.warning(
                    f"Export to InPulse SM: parameter '{par}' has no unit. If None unit is correct, provide it explicitly."
                )
            else:
                solved_parameters_units[par] = units[par]
        metadata["units"].update(solved_parameters_units)
        metadata["port_modes"] = {
            name: [_ for _ in self.get_pin_modes(name)]
            for name in self.get_pin_basenames()
        }

        metadata["smatrix_map"] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

        for pinin in self.pin_dic:
            for pinout in self.pin_dic:
                metadata["smatrix_map"][pinin.basename][pinout.basename][
                    pinin.mode_name
                ][pinout.mode_name] = f"{pinin}//{pinout}"

        metadata["smatrix_map"] = default_to_regular(metadata["smatrix_map"])

        return metadata

    def _build_data(self) -> pd.DataFrame:
        """Builds the dataframe used to export data to InPulse scattering matrix"""
        full_data = self.get_full_data()
        for column in full_data.columns:
            if column in self.solved_params:
                continue
            col = full_data.pop(column)
            fullin, fullout = column
            name = f"{fullin}//{fullout}"
            full_data[f"{name}:abs2"] = np.abs(col) ** 2.0
            full_data[f"{name}:phase"] = np.angle(col)
        return full_data

    def export_InPulse(
        self,
        filename: str = "exported_model.csvy",
        parameter_name_mapping: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, str]] = None,
    ) -> str:
        """Export scattering matrix in InPulse format

        Args:
            filename (str): name of the file to export. Default to exported_model.csvy
            parameter_name_mapping (dict): mapping between parameter name in the model and exported file.
                The for of the dictionary is {<model-parameter-name>:<exported-parameter-name>}.
                Not mapped parameter are exported with their origina name.
            units (dict): Units of the exported parameters, in the form: {<exported-parameter-name>:<unit>}
                For parameters with no units explicit map to None is advised.

        Returs:
            str: string representing the scattering matrix in InPulse
        """
        metadata = self._build_metadata(
            units=units, parameter_name_mapping=parameter_name_mapping
        )
        metadata = yaml.dump(metadata)

        data = self._build_data()
        if parameter_name_mapping is not None:
            data.rename(columns=parameter_name_mapping, inplace=True)
        data = data.to_csv(index=False)

        to_export = f"{metadata}---\n{data}"
        with open(filename, "w") as f:
            f.write(to_export)
        return to_export

    def get_monitor(
        self, input_dic: Dict[str, float], power: bool = True
    ) -> pd.DataFrame:
        """Function for returning data from monitors

        This function returns the mode coefficients if inputs and outputs of every structure selected
            as monitors

        Args:
            input_dic (dict): Dictionary of the input amplitudes. Format is
                {'pinname' : amplitude (float or complex)}. Missing pins are assumed to have 0 amplitude.

        Returns:
            pandas DataFrame: DataFrame with the amplitude at the ports. It has one column for each
                parameter given to solve plus two columns for monitor port.
        """
        input_pin_dic: dict[Pin, float | complex] = {
            self.pin[name]: value for name, value in input_dic.items()
        }

        params = {}
        if self.ns == 1:
            params = deepcopy(self.solved_params)
        else:
            for name, values in self.solved_params.items():
                if len(values) == 1:
                    params[name] = np.array([values[0] for i in range(self.ns)])
                elif len(values) == self.ns:
                    params[name] = values
                else:
                    raise Exception("Not able to convert to pandas")

        u, d = self.int_func(input_pin_dic)
        for pin, i in self.monitor_mapping.items():
            params[f"{pin}_i"] = np.abs(u[:, i]) ** 2.0 if power else u[:, i]
            params[f"{pin}_o"] = np.abs(d[:, i]) ** 2.0 if power else d[:, i]

        pan = pd.DataFrame.from_dict(params)
        return pan

        return u, d


class Waveguide(Model):
    """Model of a simple waveguide"""

    def __init__(self, L: float, n: float = 1.0, wl: float = 1.0) -> None:
        """Initialize the model

        Args:
            L (float) : length of the waveguide
            n (float or complex): effective index of the waveguide
            wl (float) : default wavelength of the waveguide
        """
        self.pin_dic = {
            Pin("a0"): 0,
            Pin("b0"): 1,
        }
        self.N = 2
        self.S = np.identity(self.N, complex)
        self.L = L
        self.param_dic = {"wl": wl}
        self.n = n
        self.default_params = deepcopy(self.param_dic)
        self.update_pins()

    def create_S(self) -> np.ndarray:
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        wl = self.param_dic["wl"]
        n = self.n
        self.S = np.zeros((self.N, self.N), complex)
        self.S[0, 1] = np.exp(2.0j * np.pi * n / wl * self.L)
        self.S[1, 0] = np.exp(2.0j * np.pi * n / wl * self.L)
        return self.S

    def __str__(self):
        """Formatter function for printing"""
        return f"Model of waveguide of lenght {self.L:.3f} and index {self.n:.3f} (id={id(self)})"


class UserWaveguide(Model):
    """Template for a user defined waveguide"""

    def __init__(
        self,
        L: float,
        func: Callable,
        param_dic: Dict[str, Any] = None,
        allowedmodes: Dict[str, Dict] = None,
    ) -> None:
        """Initialize the model

        Args:
            L (float): length of the waveguide
            func (function): index function of the waveguide
            param_dic (dict): dictionary of the default parameters to be used (common to all modes)
            allowedmodes (dict): Dict of allowed modes and settings. Form is name:extra.
                extra is a dictionary containing the extra parameters to be passed to func
                Default is for 1 mode, with no name and no parameters
        """

        self.allowed = {None: {}} if allowedmodes is None else allowedmodes
        self.pin_dic = {}

        for i, mode in enumerate(self.allowed):
            self.pin_dic[Pin("a0", mode)] = 2 * i
            self.pin_dic[Pin("b0", mode)] = 2 * i + 1

        self.N = len(self.pin_dic)
        self.S = np.identity(self.N, complex)
        self.param_dic = deepcopy(param_dic) if param_dic else {}
        self.default_params = deepcopy(self.param_dic)

        self.index_func = func
        self.L = L
        self.update_pins()

    def create_S(self) -> np.ndarray:
        """Created the scattering Matrix"""
        wl = self.param_dic["wl"]
        S_list = []
        for mode, extra in self.allowed.items():
            self.param_dic.update(extra)
            n = self.index_func(**self.param_dic)
            S = np.zeros((2, 2), complex)
            S[0, 1] = np.exp(2.0j * np.pi * n / wl * self.L)
            S[1, 0] = np.exp(2.0j * np.pi * n / wl * self.L)
            S_list.append(S)
        self.S = diag_blocks(S_list)
        return self.S


class BeamSplitter(Model):
    """Model of variable ration beam splitter"""

    def __init__(self, ratio: float = 0.5, t: float = None, phase: float = 0.0) -> None:
        """Initialize the model

        Args:
            ratio (float) : Power coupling coefficient. It is also the splitting ratio if t is not provided.
            t (float): Power transmission coefficient. If None (default) it is calculated from the ratio assuming
                no loss in the component.
            phase (float) : phase shift of the transmitted ray (in unit of pi). Default to 0.0
        """
        self.pin_dic = {
            Pin("a0"): 0,
            Pin("a1"): 1,
            Pin("b0"): 2,
            Pin("b1"): 3,
        }
        self.N = 4
        self.ratio = ratio
        self.phase = phase
        p1 = np.pi * self.phase
        c = 1.0j * np.sqrt(self.ratio)
        t = np.sqrt(1.0 - self.ratio) if t is None else np.sqrt(t)
        self.S = np.zeros((self.N, self.N), complex)
        self.param_dic = {}
        self.default_params = deepcopy(self.param_dic)
        # self.S[:2,2:]=np.array([[t,c],[c,-t]])
        # self.S[2:,:2]=np.array([[t,c],[c,-t]])
        self.S[:2, 2:] = np.exp(2.0j * np.pi * phase) * np.array([[t, c], [c, t]])
        self.S[2:, :2] = np.exp(2.0j * np.pi * phase) * np.array([[t, c], [c, t]])
        self.update_pins()

    def __str__(self):
        """Formatter function for printing"""
        return f"Model of beam-splitter with ratio {self.ratio:.3} (id={id(self)})"


class Splitter1x2(Model):
    """Model of 1x2 Splitter"""

    def __init__(self) -> None:
        """Initialize the model"""
        self.pin_dic = {
            Pin("a0"): 0,
            Pin("b0"): 1,
            Pin("b1"): 2,
        }
        self.N = 3
        self.S = (
            1.0
            / np.sqrt(2.0)
            * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], complex)
        )
        self.param_dic = {}
        self.default_params = deepcopy(self.param_dic)
        self.update_pins()

    def __str__(self):
        return f"Model of 1x2 splitter (id={id(self)})"


class Splitter1x2Gen(Model):
    """Model of 1x2 Splitter with possible reflection between the 2 port side.
    TODO: verify this model makes sense
    """

    def __init__(self, cross: float = 0.0, phase: float = 0.0) -> None:
        """Initialize the model

        Args:
            cross (float) : ratio of reflection (power ratio)
            phase (float) : phase shift of the reflected ray (in unit of pi)
        """
        self.pin_dic = {
            Pin("a0"): 0,
            Pin("b0"): 1,
            Pin("b1"): 2,
        }
        self.N = 3
        self.param_dic = {}
        self.default_params = deepcopy(self.param_dic)
        c = np.sqrt(cross)
        t = np.sqrt(0.5 - cross)
        p1 = np.pi * phase
        self.S = np.array(
            [
                [0.0, t, t],
                [t, 0.0, c * np.exp(1.0j * p1)],
                [t, c * np.exp(-1.0j * p1), 0.0],
            ],
            complex,
        )
        self.update_pins()


class PhaseShifter(Model):
    """Model of multimode variable phase shifter"""

    def __init__(self, param_name: str = "PS", param_default: float = 0.0) -> None:
        """Initialize the model

        Args:
            param_name (str): name of the parameter of the Phase Shifter
            param_default (float): default value of the Phase Shift in pi units
        """
        self.param_dic = {}
        self.pin_dic = {
            Pin("a0"): 0,
            Pin("b0"): 1,
        }
        self.N = 2
        self.pn = param_name
        self.param_dic = {param_name: param_default}
        self.default_params = deepcopy(self.param_dic)
        self.update_pins()

    def create_S(self) -> np.ndarray:
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        S = np.zeros((self.N, self.N), complex)
        S[0, 1] = np.exp(1.0j * np.pi * self.param_dic[self.pn])
        S[1, 0] = np.exp(1.0j * np.pi * self.param_dic[self.pn])
        self.S = S
        return self.S


class PushPullPhaseShifter(Model):
    """Model of multimode variable phase shifter"""

    def __init__(self, param_name: str = "PS") -> None:
        """Initialize the model

        Args:
            param_name (str) : name of the parameter of the Phase Shifter
        """
        self.param_dic = {}
        self.pin_dic = {
            Pin("a0"): 0,
            Pin("b0"): 1,
            Pin("a1"): 2,
            Pin("b1"): 3,
        }
        self.N = 4
        self.pn = param_name
        self.param_dic = {param_name: 0.0}
        self.default_params = deepcopy(self.param_dic)
        self.update_pins

    def create_S(self) -> np.ndarray:
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        S1 = np.zeros((2, 2), complex)
        S1[0, 1] = np.exp(0.5j * np.pi * self.param_dic[self.pn])
        S1[1, 0] = np.exp(0.5j * np.pi * self.param_dic[self.pn])
        S2 = np.zeros((2, 2), complex)
        S2[0, 1] = np.exp(-0.5j * np.pi * self.param_dic[self.pn])
        S2[1, 0] = np.exp(-0.5j * np.pi * self.param_dic[self.pn])
        self.S = diag_blocks([S1, S2])
        return self.S

    def __str__(self):
        """Formatter function for printing"""
        return f"Model of variable Push-Pull phase shifter (id={id(self)})"


class PolRot(Model):
    """Model of a 2 modes polarization rotator"""

    def __init__(self, angle: float = None, angle_name: str = "angle") -> None:
        """Initialize the model

        If angle is provided the rotation is fixed to that value. If not, the rotation is assumed
            variable and the angle will be fetched form the parameter dictionary.

        Args:
            angle (float) : fixed value of the rotation angle (in pi units). Default is None
            angle_name (str) : name of the angle parameter
        """
        self.pin_dic = {
            Pin("a0", mode_name="pol0"): 0,
            Pin("a0", mode_name="pol1"): 1,
            Pin("b0", mode_name="pol0"): 2,
            Pin("b0", mode_name="pol1"): 3,
        }
        self.N = 4
        self.param_dic = {}
        if angle is None:
            self.fixed = False
            self.angle_name = angle_name
            self.param_dic = {angle_name: 0.0}
        else:
            self.fixed = True
            c = np.cos(np.pi * angle)
            s = np.sin(np.pi * angle)
            self.S = np.zeros((self.N, self.N), complex)
            self.S[:2, 2:] = np.array([[c, s], [-s, c]])
            self.S[2:, :2] = np.array([[c, -s], [s, c]])
        self.default_params = deepcopy(self.param_dic)
        self.update_pins()

    def create_S(self) -> np.ndarray:
        """Function for returning the scattering matrix of the model

        Returns:
            ndarray: Scattering matrix of the model
        """
        if self.fixed:
            return self.S
        else:
            angle = self.param_dic[self.angle_name]
            c = np.cos(np.pi * angle)
            s = np.sin(np.pi * angle)
            S = np.zeros((self.N, self.N), complex)
            S[:2, 2:] = np.array([[c, s], [-s, c]])
            S[2:, :2] = np.array([[c, -s], [s, c]])
            return S


class Attenuator(Model):
    """Model of attenuator in dB"""

    def __init__(self, loss: float = 0.0) -> None:
        """Initialize the model

        Args:
            loss: value of the loss (in dB)
        """
        self.param_dic = {}
        self.pin_dic = {
            Pin("a0"): 0,
            Pin("b0"): 1,
        }
        self.N = 2
        self.loss = loss
        self.S = np.zeros((self.N, self.N), complex)
        self.S[0, 1] = 10.0 ** (-0.05 * loss)
        self.S[1, 0] = 10.0 ** (-0.05 * loss)
        self.default_params = deepcopy(self.param_dic)
        self.update_pins()


class LinearAttenuator(Model):
    """Model of attenuator in absolute unit"""

    def __init__(self, c: float = 1.0) -> None:
        """Initialize the model

        Args:
            c (float): fraction of power transmitted:
                1.0 -> no loss
                0.3 -> 30% of the power is transmitted
                0.0 -> no light transmitted
        """
        self.param_dic = {}
        self.pin_dic = {
            Pin("a0"): 0,
            Pin("b0"): 1,
        }
        self.N = 2
        self.S = np.zeros((self.N, self.N), complex)
        self.S[0, 1] = np.sqrt(c)
        self.S[1, 0] = np.sqrt(c)
        self.default_params = deepcopy(self.param_dic)
        self.update_pins()


class Mirror(Model):
    """Model of partially reflected Mirror"""

    def __init__(self, ref: float = 0.5, phase: float = 0.0) -> None:
        """Initialize the model

        Args:
            ref (float) : ratio of reflected power
            phase (float): phase shift of the reflected ray (in pi units)
        """
        self.pin_dic = {
            Pin("a0"): 0,
            Pin("b0"): 1,
        }
        self.param_dic = {}
        self.default_params = deepcopy(self.param_dic)
        self.N = 2
        self.ref = ref
        self.phase = phase
        t = np.sqrt(self.ref)
        c = np.sqrt(1.0 - self.ref)
        p1 = np.pi * self.phase
        self.S = np.array(
            [[t * np.exp(1.0j * p1), c], [-c, t * np.exp(-1.0j * p1)]], complex
        )
        self.update_pins()


class PerfectMirror(Model):
    """Model of perfect mirror (only one port), 100% reflection"""

    def __init__(self, phase: float = 0.0) -> None:
        """Initialize the model

        Args:
            phase (float): phase of the reflected ray (in pi unit)
        """
        self.pin_dic = {Pin("a0"): 0}
        self.param_dic = {}
        self.default_params = deepcopy(self.param_dic)
        self.N = 1
        self.phase = phase
        p1 = np.pi * self.phase
        self.S = np.array([[np.exp(1.0j * p1)]], complex)
        self.update_pins()


class FPR_NxM(Model):
    """Model of Free Propagation Region. TODO: check this model makes sense"""

    def __init__(self, N: int, M: int, phi: float = 0.1) -> None:
        """Initialize the model

        Args:
            N (int) : number of input ports
            M (int) : number of output ports
            phi (float) : phase difference between adjacent ports
        """
        self.param_dic = {}
        self.default_params = deepcopy(self.param_dic)
        self.pin_dic = {Pin(f"a{i}"): i for i in range(N)}
        self.pin_dic.update({Pin(f"b{i}"): N + i for i in range(M)})
        Sint = np.zeros((N, M), complex)
        for i in range(N):
            for j in range(M):
                Sint[i, j] = np.exp(
                    -1.0j * np.pi * phi * (i - 0.5 * N + 0.5) * (j - 0.5 * M + 0.5)
                )
        Sint2 = np.conj(np.transpose(Sint))
        self.S = np.concatenate(
            [
                np.concatenate([np.zeros((N, N), complex), Sint / np.sqrt(M)], axis=1),
                np.concatenate([Sint2 / np.sqrt(N), np.zeros((M, M), complex)], axis=1),
            ],
            axis=0,
        )
        self.update_pins()


class Ring(Model):
    """Model of ring resonator filter"""

    def __init__(self, R: float, n: float, alpha: float, t: float) -> None:
        """Initialize the model

        Args:
            R (float) : radius of the ring
            n (float) : effective index of the waveguide in the ring
            alpha (float) : one trip loss coefficient (remaining complex amplitude)
            t (float) : transmission of the beam splitter (complex amplitude)
        """
        self.pin_dic = {
            Pin("a0"): 0,
            Pin("b0"): 1,
        }
        self.N = 2
        self.S = np.identity(self.N, complex)
        self.R = R
        self.n = n
        self.alpha = alpha
        self.t = t
        self.param_dic = {"wl": None}
        self.default_params = deepcopy(self.param_dic)
        self.update_pins()

    def create_S(self) -> np.ndarray:
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        wl = self.param_dic["wl"]
        n = self.n
        t = self.t
        ex = np.exp(-4.0j * np.pi**2.0 / wl * n * self.R)
        b = (-self.alpha + t * ex) / (-self.alpha * t + ex)
        self.S = np.zeros((self.N, self.N), complex)
        self.S[0, 1] = b
        self.S[1, 0] = b
        return self.S


class TH_PhaseShifter(Model):
    """Model of thermal phase shifter (dispersive waveguide + phase shifter)"""

    def __init__(
        self,
        L: float,
        Neff: Callable,
        R: float = None,
        w: float = None,
        wl: float = None,
        pol: int = None,
        param_name: str = "PS",
    ) -> None:
        """Initialize the model

        Args:
            L (float) : length of the waveguide
            Neff (function): function returning the effective index of the wavegude. It must be a function of
                wl,R,w, and pol
            wl (float) : default wavelength of the waveguide
            w  (float) : default width of the waveguide
            R  (float) : default bending radius of the waveguide
            pol (int)  : default mode of the waveguide
            param_name (str) : name of the parameter of the Phase Shifter
        """
        self.pin_dic = {
            Pin("a0"): 0,
            Pin("b0"): 1,
        }
        self.N = 2
        self.Neff = Neff
        self.L = L
        self.pn = param_name
        self.param_dic = {"R": R, "w": w, "wl": wl, "pol": pol, param_name: 0.0}
        self.default_params = deepcopy(self.param_dic)
        self.update_pins()

    def create_S(self) -> np.ndarray:
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        wl = self.param_dic["wl"]
        n = self.Neff(**self.param_dic)
        self.S = np.zeros((self.N, self.N), complex)
        self.S[0, 1] = np.exp(
            1.0j * np.pi * (2.0 * n / wl * self.L + self.param_dic[self.pn])
        )
        self.S[1, 0] = np.exp(
            1.0j * np.pi * (2.0 * n / wl * self.L + self.param_dic[self.pn])
        )
        return self.S


class AWGfromVPI(Model):
    """Define custom model from a VPI scattering matrix format"""

    def __init__(self, filename: str, pol: str = "TE", fsr: float = 1.0) -> None:
        """Instantiate an AWG from a VPI S-matrix.

        Args:
            filename (str): Name of the file where the S-matrix is stored.
            pol (str): Polarization that is considered, while the other value is discarded. Default is TE.
            fsr (float): ratio between the provided portion spectrum and the fpr of the AWG.
                if wavelength outside the spectrum are provided they will be reconstructed assuming periodicity.
        """
        with open(filename) as f:
            data = f.read()

            data = data.split("\n\n")[2:]

        coeff = {}
        pins = []
        for t in data:
            if "NullTransferFunction" in t:
                continue
            p = t.split("\n")
            if len(p) == 1:
                continue
            pin_in = p[0]
            pin_out = p[1]
            if pol not in pin_in:
                continue
            if pol not in pin_out:
                continue
            pin_in = (
                pin_in.split()[2].split("#")[0][0] + pin_in.split()[2].split("#")[1]
            )
            pin_out = (
                pin_out.split()[2].split("#")[0][0] + pin_out.split()[2].split("#")[-1]
            )
            if pin_in not in pins:
                pins.append(pin_in)
            if pin_out not in pins:
                pins.append(pin_out)
            dd = io.StringIO(t)
            ar = np.loadtxt(dd)
            LAM = ar[:, 0]
            coeff[(pin_in, pin_out)] = ar[:, 1] * np.exp(
                1.0j * np.pi / 180 * 0 * ar[:, 2]
            )

        pins.sort()
        S = np.zeros((len(LAM), len(pins), len(pins)), dtype=complex)
        for i, ipin in enumerate(pins):
            for j, jpin in enumerate(pins):
                if (ipin, jpin) in coeff:
                    S[:, i, j] = coeff[(ipin, jpin)]

        self.pin_dic = {pin: i for i, pin in enumerate(pins)}
        self.N = len(pins)
        self.param_dic = {}
        self.default_params = {}

        self.S_func = interp1d(LAM, S, axis=0)

        self.fsr = (LAM[-1] - LAM[0]) / fsr
        self.lamc = (LAM[-1] + LAM[0]) / 2.0

    def create_S(self) -> np.ndarray:
        lam = self.param_dic["wl"]
        self.S = self.S_func(
            self.lamc
            - 0.5 * self.fsr
            + np.mod(lam - self.lamc + 0.5 * self.fsr, self.fsr)
        )
        return self.S


class FPR(Model):
    """Simplified model of FPR circle mount"""

    def __init__(
        self, n: int, m: int, R: float, d1: float, d2: float, Ri: float = None
    ) -> None:
        """Initialize the model

        Args:
            n (int): Number of inputs arms.
            m (int): Number of output arms.
            R (float): Radius the output part of FPR (input and output are on the same radius)
            d1 (float): Distance of inputs on the input plane.
            d2 (float): Distance of output on the output plane.

        Returns:
            Model: Model of the FPR region

        """
        Ri = 0.5 * R if Ri is None else Ri
        self.n, self.m = n, m
        self.d1, self.d2 = d1, d2
        self.R = R
        _d1 = {Pin(f"a{i:03}"): i for i in range(n)}
        _d2 = {Pin(f"b{i:03}"): n + i for i in range(m)}
        dic = {**_d1, **_d2}
        super().__init__(pin_dic=dic)
        t1 = self.d1 / Ri * np.array(line(n))
        t2 = self.d2 / R * np.array(line(m))
        T1, T2 = np.meshgrid(t1, t2, indexing="ij")
        DY = R * np.sin(T2) - Ri * np.sin(T1)
        DX = R * np.cos(T2) - Ri * (1.0 - np.cos(T1))
        self.DR = np.sqrt(DY**2.0 + DX**2.0)
        self.S = np.zeros((n + m, n + m), dtype=complex)

    def create_S(self) -> np.ndarray:
        """Creates the scattering matrix

        Returns:
            2darray: Scattering matrix of the FPR

        """
        lam = self.param_dic["wl"]
        n = self.n
        Mat = np.exp(2.0j * np.pi / lam * self.DR) / np.sqrt(max(self.m, self.n))
        self.S[:n, n:] = Mat
        self.S[n:, :n] = np.transpose(Mat)
        return self.S


class FPRGaussian(Model):
    """Simplified model of FPR circle mount based on Gaussian beams."""

    def __init__(
        self,
        n: int,
        m: int,
        R: float,
        d1: float,
        d2: float,
        w1: float,
        w2: float,
        n_slab: float,
        Ri: float = None,
    ) -> None:
        """Initialize the model

        Args:
            n (int): Number of inputs arms.
            m (int): Number of output arms.
            R (float): Radius the output part of FPR (input and output are on the same radius).
            d1 (float): Center-to-center distance of inputs on the input plane.
            d2 (float): Center-to-center distance of output on the output plane in.
            n_slab (float | callable): Effective index of the slab mode.
            R_i (float): Input radius. If it is not specified it is set equal to the output one. Default is None.

        Returns:
            Model: Model of the FPR region

        """
        Ri = R if Ri is None else Ri
        self.n, self.m = n, m
        self.d1, self.d2 = d1, d2
        self.Ri = Ri
        self.R = R
        self.n_slab = n_slab
        self.w1 = w1
        self.w2 = w2
        _d1 = {Pin(f"a{i:03}"): i for i in range(n)}
        _d2 = {Pin(f"b{i:03}"): n + i for i in range(m)}
        dic = {**_d1, **_d2}
        super().__init__(pin_dic=dic)

        self.t1 = self.d1 / Ri * line(n)
        self.t2 = self.d2 / R * line(m)
        self.pos1 = [
            (Ri * (1 - np.cos(self.t1[i])), Ri * np.sin(self.t1[i])) for i in range(n)
        ]
        self.pos2 = [(R * np.cos(self.t2[i]), R * np.sin(self.t2[i])) for i in range(m)]

        self.S = np.zeros((n + m, n + m), dtype=complex)

    def create_S(self):
        """Creates the scattering matrix

        Returns:
            2darray: Scattering matrix of the FPR

        """
        lam = self.param_dic["wl"]
        if type(self.n_slab) == float or type(self.n_slab) == int:
            n_slab = self.n_slab
        elif callable(self.n_slab):
            n_slab = self.n_slab(lam)

        n = self.n
        m = self.m
        mat = np.zeros((n, m), dtype=complex)
        z = 0.5 * self.R
        for i in range(n):
            beam1 = GaussianBeam(
                w0=self.w1,
                n=n_slab,
                wl=lam,
                z0=self.pos1[i][0],
                x0=self.pos1[i][1],
                theta=-np.rad2deg(self.t1[i]),
            )
            for j in range(m):
                beam2 = GaussianBeam(
                    w0=self.w2,
                    n=n_slab,
                    wl=lam,
                    z0=self.pos2[j][0],
                    x0=self.pos2[j][1],
                    theta=np.rad2deg(self.t2[j]),
                )

                def to_integrate(x):
                    integrand = beam1.field(z=z, x=x) * np.conjugate(
                        beam2.field(z=z, x=x)
                    )
                    return np.array([integrand.real, integrand.imag])

                res = quad_vec(to_integrate, -np.inf, np.inf)
                mat[i, j] = res[0][0] + 1j * res[0][1]

        self.S[:n, n:] = mat
        self.S[n:, :n] = np.transpose(mat)
        return self.S

    def show(self, ax: matplotlib.axes.Axes = None) -> matplotlib.axes.Axes:
        """Shows the input and output ports of the FPR.

        Args:
            ax (matplotlib.axes.Axes): Axis onto which the plot is mapped.

        Returns:
            matplotlib.axes.Axes: The axis.

        """
        x_in = [el[0] for el in self.pos1]
        x_out = [el[0] for el in self.pos2]
        y_in = [el[1] for el in self.pos1]
        y_out = [el[1] for el in self.pos2]
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(x_in, y_in, label="input")
        ax.scatter(x_out, y_out, label="output")
        ax.legend(loc="lower center")
        ax.set_xlabel("x [$\mu$m]")
        ax.set_ylabel("y [$\mu$m]")
        ax.set_title(f"FPR: R$_i$={self.Ri} $\mu$m, R$_o$={self.R} $\mu$m")

        return ax


class CWA(Model):
    """Simple model fo a Coupled Waveguide Array"""

    def __init__(self, N: int, L: float, n: float = 1.0, k: float = 0.1):
        """Initialize the model

        Args:
            N (int): Number of waveguides in the array.
            L (float): Length of the CWA section
            n (float): Index of the single waveguide in the array.
            k (float): Coupling coefficient between waveguides.

        Returns:
            Model: Model of a CWA

        """
        self.NW = N
        self.n = n
        self.k = k
        self.L = L
        d1 = {Pin(f"a{i}"): i for i in range(N)}
        d2 = {Pin(f"b{i}"): N + i for i in range(N)}
        dic = {**d1, **d2}
        super().__init__(pin_dic=dic)
        i, j = list(range(N)), list(range(N))
        I, J = np.meshgrid(i, j, indexing="ij")
        self.RM = np.exp(2.0j * np.pi * I * J / self.NW) / np.sqrt(self.NW)
        self.S = np.zeros((2 * N, 2 * N), dtype=complex)

    def create_S(self) -> np.ndarray:
        B = [
            2.0 * np.pi * self.n / self.param_dic.get("wl", 1.0)
            + 2.0 * self.k * np.cos(2.0 * np.pi * n / self.NW)
            for n in range(self.NW)
        ]
        P = np.diag(np.exp(1.0j * np.array(B) * self.L))
        S = np.dot(np.dot(self.RM, P), self.RM)
        self.S[: self.NW, self.NW :] = S
        self.S[self.NW :, : self.NW] = S
        return self.S


class Model_from_InPulse(Model):
    """Class for model from InPulse S-Matrix RAW DATA file"""

    def __init__(
        self,
        file: str,
        parameter_name_mapping: Optional[Dict[str, str]] = None,
        mode_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize the model

        Args:
            file (str): Path of the file to be loaded.
            parameter_name_mapping (dict[str,str]): mapping of the parameter name between the file and the model.
                The format is {'<file-parameter-name>':'<model-parameter-name>'}
                Parameters not provided retain theur original name.
                Example:
                Wavelength is usually called 'wl' in gensol, but it may be 'wavelength' in the file.
                To uniform the model to the other provide the mapping {"wavelength":"wl"}
            mode_mapping (dict[str,str]): mapping between the mode names in the file and the model.
                The format is {'<file-mode-name>':'<model-mode-name>'}
                If nothing is provided, all the modes are loaded with their original name.
                This will create multiple pins in the model with name '<pin-name>_<mode-name>'
                If a dictionary is provided, only the mapped modes are loaded and the names are changed accordingly.
                If a mode is mapped to the empty string ("" or ''), the result port name is just <pin-name>

        Returns:
            None
        """
        self._file = file
        parameter_name_mapping = (
            {} if parameter_name_mapping is None else parameter_name_mapping
        )
        pin_dic, param_dic = self._process_file()
        self.parameters_names = [
            parameter_name_mapping.get(par, par) for par in self.parameters_names
        ]
        if mode_mapping is not None:
            pin_mapping = {}
            for pin in pin_dic:
                try:
                    name, mode = pin.split("_")
                except ValueError:
                    name, mode = pin, ""
                mode = mode_mapping.get(mode)
                if mode is not None:
                    pin_mapping[pin] = name if mode == "" else f"{name}_{mode}"

            pin_dic = {target: i for i, (pin, target) in enumerate(pin_mapping.items())}

            new_map_columns = {}
            for (pin_in, pin_out), column in self.map_columns.items():
                new_in = pin_mapping.get(pin_in)
                new_out = pin_mapping.get(pin_out)
                if new_in is not None and new_out is not None:
                    new_map_columns[(new_in, new_out)] = column
            self.map_columns = new_map_columns

        if not pin_dic:
            raise ValueError(
                "Loaded model from InPulse has no pins. Hint: check mode mapping."
            )

        super().__init__(pin_dic=pin_dic, param_dic=param_dic)

    def __get_pin_dic(self, port_modes: Dict) -> Dict[str, int]:
        """
        Creates the pin dictionary of the model from the file

        Args:
            port_modes (dict): The nested dictionary containing the "port_modes" part of the metadata

        Returns:
            dict: pin dictionary in the form {<pin-name> : <pin-number>}
        """
        pin_dic = {}
        _i = 0
        for pin, modes in port_modes.items():
            for mode in modes:
                pin_dic[Pin(pin, mode)] = _i
                _i += 1
        return pin_dic

    def __get_pin_map(
        self, smatrix_map: Dict
    ) -> Tuple[Set[str], Dict[Tuple[str, str], str]]:
        """
        Retrieve the mapping between the ports and the columns in the data section.

        Args:
            smatrix_map (dict): The nested dictionary containing the "smatrix_map" part of the metadata

        Returns:
            set: set of the columns names
            map_columns: map between pairs of pins and the related column
        """
        map_columns = {}
        columns = set()
        for pin, target in smatrix_map.items():
            for pin_target, modes in target.items():
                for modein, modes_out in modes.items():
                    for modeout, column in modes_out.items():
                        if column:
                            pin1 = Pin(pin, modein)
                            pin2 = Pin(pin_target, modeout)
                            map_columns[(pin1, pin2)] = column
                            columns.add(column)
        return columns, map_columns

    def _process_file(self) -> Tuple[Dict, Dict]:
        """
        Load the data from the file and proccess the content

        Args;
            None

        Returns:
            dict: pin dictionry of the model
            dict: parameters dictionary of the models
        """
        with open(self._file, "r") as file:
            text = file.read()
        metadata, data = text.split("---")
        metadata = metadata.replace("\t", "    ")
        metadata = yaml.safe_load(metadata)

        pin_dic = self.__get_pin_dic(metadata["port_modes"])
        columns, self.map_columns = self.__get_pin_map(metadata["smatrix_map"])

        param_dic = {
            name: parameter_info["mean"]
            for name, parameter_info in metadata.get("csm_parameters", {}).items()
        }

        data = io.StringIO(data)
        data = pd.read_csv(data)
        self.parameters_names = [col for col in data.columns if ":" not in col]
        parameters_values = data[self.parameters_names].values
        self.functions_columns = {}
        for column in columns:
            ampl = np.asarray(data[f"{column}:abs2"].values)
            phase = np.asarray(data[f"{column}:phase"].values)
            amplitude = np.sqrt(ampl) * np.exp(1.0j * phase)
            if len(self.parameters_names) >= 2:
                self.functions_columns[column] = LinearNDInterpolator(
                    parameters_values, amplitude
                )
            else:
                self.functions_columns[column] = interp1d(
                    np.squeeze(parameters_values), amplitude
                )

        return pin_dic, param_dic

    def create_S(self) -> np.ndarray:
        """Returns the scattering matrix of the model"""
        S = np.zeros((self.N, self.N), dtype=complex)
        parameters_values = tuple(
            self.param_dic[parameter] for parameter in self.parameters_names
        )
        amplitudes = {
            col: func(*parameters_values)
            for col, func in self.functions_columns.items()
        }

        for (pin_in, pin_out), column in self.map_columns.items():
            S[self.pin_dic[pin_in], self.pin_dic[pin_out]] = amplitudes[column]
        self.S = S
        return S


if __name__ == "__main__":
    wg = Waveguide(n=1.0, L=100.0)
    print(wg.solve(wl=1.55).S2PD())
