# -------------------------------------------
#
# GenSol Package
#
# Python tool for simulation of abstract photonic circuits
#
# @author: Marco Passoni
#
# -------------------------------------------


"""File containing the model calls and related methods
"""
from __future__ import annotations
import functools
from typing import Any, List, Dict, Tuple, Callable, TYPE_CHECKING, Union

import matplotlib.axes
import numpy as np
import solver.structure
from solver import sol_list
from solver import logger
import solver
import solver.log
from solver.utils import line, GaussianBeam
from copy import deepcopy
from copy import copy
import pandas as pd
from scipy.interpolate import interp1d
import io
from scipy.integrate import quad_vec
import matplotlib.pyplot as plt


if TYPE_CHECKING:
    import nazca as nd


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


class ProtectedPartial(functools.partial):
    """Like partial, but keywords provided at creation cannot be overwritten al call time"""

    def __call__(self, /, *args, **keywords):
        keywords = {**keywords, **self.keywords}
        return self.func(*self.args, *args, **keywords)


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
        pin_dic: Dict[str, int] = None,
        param_dic: Dict[str, Any] = None,
        Smatrix: np.ndarray = None,
    ) -> None:
        """Initialize the model

        Args:
            pin_dic (dictionary): list of strings containing the model's pin names
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
        a = list(self.pin_dic.keys())
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
        return np.abs(self.S[0, self.pin_dic[pin1], self.pin_dic[pin2]]) ** 2.0

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
        return np.angle(self.S[0, self.pin_dic[pin1], self.pin_dic[pin2]])

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
        return self.S[0, self.pin_dic[pin1], self.pin_dic[pin2]]

    def expand_mode(self, mode_list: List[str]):
        """This function expands the model by adding additional modes.

        For each pin a number of pins equal the length of mode_list will be created. The pin names will be
            "{pinname}_{modename}}".
        Each mode will have the same behavior.

        Args:
            mode_list (list) : list of strings containing the modenames.

        Returns:
            Model : new model with expanded modes
        """
        self.np = len(mode_list)
        self.mode_list = mode_list
        new_pin_dic = {}
        for name, n in self.pin_dic.items():
            for i, mode in enumerate(self.mode_list):
                new_pin_dic[f"{name}_{mode}"] = i * self.N + n
        self.pin_dic = new_pin_dic
        self._create_S = self.create_S
        self.create_S = self._expand_S
        self.N = self.N * self.np
        return self

    def get_output(
        self, input_dic: Dict[str, float], power: bool = True
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
        l1 = list(self.pin_dic.keys())
        l2 = list(input_dic.keys())
        for pin in l2:
            l1.remove(pin)
        if l1 != []:
            for pin in l1:
                input_dic[pin] = 0.0 + 0.0j
        u = np.zeros(self.N, complex)
        for pin, i in self.pin_dic.items():
            u[i] = input_dic[pin]
        d = np.dot(self.S[0, :, :], u)
        out_dic = {}
        for pin, i in self.pin_dic.items():
            if power:
                out_dic[pin] = np.abs(d[i]) ** 2.0
            else:
                out_dic[pin] = d[i]
        return out_dic

    def put(
        self, source_pin: str = None, target_pin=None, param_mapping={}
    ) -> solver.Structure:
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
        # ST=solver.structure.Structure(model=deepcopy(self),param_mapping=param_mapping)
        ST = solver.structure.Structure(model=self, param_mapping=param_mapping)
        sol_list[-1].add_structure(ST)
        if (source_pin is not None) and (target_pin is not None):
            sol_list[-1].connect(ST, source_pin, target_pin[0], target_pin[1])
        return ST

    def solve(self, **kargs) -> solver.SolvedModel:
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
            pin_dic=self.pin_dic, param_dic=kargs, Smatrix=np.array(S_list)
        )

    def show_free_pins(self) -> None:
        """Function for printing pins of model"""
        print(f"Pins of model {self} (id={id(self)})")
        for pin, n in self.pin_dic.items():
            print(f"{pin:5s}:{n:5}")
        print("")

    def pin_mapping(self, pin_mapping: Dict[str, str]):
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

    def get_pin_modes(self, pin: str) -> List[Tuple[str, str]]:
        """Parse the pins for locating the pins with the same base name

        Assumes for the pins a name in the form pinname_modename.

        Args:
            pin (str): base name of the pin

        Returns:
            list: list of modenames for which pinname==pin
        """
        li = []
        for _pin in self.pin_dic:
            try:
                pinname, modename = _pin.split("_")
            except ValueError:
                pinname, modename = pin, ""
            if pinname == pin:
                li.append((pinname, modename))
        return li

    def __str__(self):
        """Formatter function for printing"""
        return f"Model object (id={id(self)}) with pins: {list(self.pin_dic)}"


class SolvedModel(Model):
    """Class for storing data of a solver mode.

    Do not use this class directly. It is returned from all solve methods. It is convenient for extracting data
    """

    def __init__(
        self,
        pin_dic: Dict[str, int] = None,
        param_dic: Dict[str, Any] = None,
        Smatrix: np.ndarray = None,
        int_func: Callable = None,
        monitor_mapping: Dict[str, Tuple[solver.Structure, str]] = None,
    ) -> None:
        """Initialize the model

        Args:
            pin_dic (dictionary): list of strings containing the model's pin names
            param_dic (dictionary): dictionary {'param_name':param_value} containing the definition of
                the model's parameters.
            Smatrix (ndarray) : Fixed S_matrix of the model
            int_func (Callable): function for returning the modal coefficient in between two part of the scattering matrix
            monitor_mapping (Dict): Dictionry mapping the name of the monitor to the connected pin

        """
        super().__init__(pin_dic=pin_dic, param_dic=param_dic, Smatrix=Smatrix)
        self.solved_params = deepcopy(param_dic)
        self.ns = np.shape(Smatrix)[0]
        self.int_func = int_func
        self.monitor_mapping = {} if monitor_mapping is None else monitor_mapping

    def set_intermediate(
        self,
        int_func: Callable,
        monitor_mapping: Dict[str, Tuple[solver.Structure, str]],
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
        params["T"] = np.abs(self.S[:, self.pin_dic[pin1], self.pin_dic[pin2]]) ** 2.0
        params["dB"] = 20.0 * np.log10(
            np.abs(self.S[:, self.pin_dic[pin1], self.pin_dic[pin2]])
        )
        params["Phase"] = np.angle(self.S[:, self.pin_dic[pin1], self.pin_dic[pin2]])
        params["Amplitude"] = self.S[:, self.pin_dic[pin1], self.pin_dic[pin2]]
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
        l2 = list(input_dic.keys())
        for pin in l2:
            l1.remove(pin)
        if l1 != []:
            for pin in l1:
                input_dic[pin] = 0.0 + 0.0j
        u = np.zeros(self.N, complex)
        for pin, i in self.pin_dic.items():
            u[i] = input_dic[pin]

        output = np.matmul(self.S, u)

        for pin, i in self.pin_dic.items():
            params[pin] = np.abs(output[:, i]) ** 2.0 if power else output[:, i]
        pan = pd.DataFrame.from_dict(params)
        return pan

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

        u, d = self.int_func(input_dic)
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
        self.pin_dic = {"a0": 0, "b0": 1}
        self.N = 2
        self.S = np.identity(self.N, complex)
        self.L = L
        self.param_dic = {"wl": wl}
        self.n = n
        self.default_params = deepcopy(self.param_dic)

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

        self.allowed = {"": {}} if allowedmodes is None else allowedmodes
        self.pin_dic = {}

        for i, mode in enumerate(self.allowed):
            if mode == "":
                self.pin_dic["a0"] = 2 * i
                self.pin_dic["b0"] = 2 * i + 1
            else:
                self.pin_dic[f"a0_{mode}"] = 2 * i
                self.pin_dic[f"b0_{mode}"] = 2 * i + 1

        self.N = len(self.pin_dic)
        self.S = np.identity(self.N, complex)
        self.param_dic = deepcopy(param_dic) if param_dic else {}
        self.default_params = deepcopy(self.param_dic)

        self.index_func = func
        self.L = L

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
        self.pin_dic = {"a0": 0, "a1": 1, "b0": 2, "b1": 3}
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

    def __str__(self):
        """Formatter function for printing"""
        return f"Model of beam-splitter with ratio {self.ratio:.3} (id={id(self)})"


class Splitter1x2(Model):
    """Model of 1x2 Splitter"""

    def __init__(self) -> None:
        """Initialize the model"""
        self.pin_dic = {"a0": 0, "b0": 1, "b1": 2}
        self.N = 3
        self.S = (
            1.0
            / np.sqrt(2.0)
            * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], complex)
        )
        self.param_dic = {}
        self.default_params = deepcopy(self.param_dic)

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
        self.pin_dic = {"a0": 0, "b0": 1, "b1": 2}
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


class PhaseShifter(Model):
    """Model of multimode variable phase shifter"""

    def __init__(self, param_name: str = "PS", param_default: float = 0.0) -> None:
        """Initialize the model

        Args:
            param_name (str): name of the parameter of the Phase Shifter
            param_default (float): default value of the Phase Shift in pi units
        """
        self.param_dic = {}
        self.pin_dic = {"a0": 0, "b0": 1}
        self.N = 2
        self.pn = param_name
        self.param_dic = {param_name: param_default}
        self.default_params = deepcopy(self.param_dic)

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
        self.pin_dic = {"a0": 0, "b0": 1, "a1": 2, "b1": 3}
        self.N = 4
        self.pn = param_name
        self.param_dic = {param_name: 0.0}
        self.default_params = deepcopy(self.param_dic)

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
        self.pin_dic = {"a0_pol0": 0, "a0_pol1": 1, "b0_pol0": 2, "b0_pol1": 3}
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
        self.pin_dic = {"a0": 0, "b0": 1}
        self.N = 2
        self.loss = loss
        self.S = np.zeros((self.N, self.N), complex)
        self.S[0, 1] = 10.0 ** (-0.05 * loss)
        self.S[1, 0] = 10.0 ** (-0.05 * loss)
        self.default_params = deepcopy(self.param_dic)


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
        self.pin_dic = {"a0": 0, "b0": 1}
        self.N = 2
        self.S = np.zeros((self.N, self.N), complex)
        self.S[0, 1] = np.sqrt(c)
        self.S[1, 0] = np.sqrt(c)
        self.default_params = deepcopy(self.param_dic)


class Mirror(Model):
    """Model of partially reflected Mirror"""

    def __init__(self, ref: float = 0.5, phase: float = 0.0) -> None:
        """Initialize the model

        Args:
            ref (float) : ratio of reflected power
            phase (float): phase shift of the reflected ray (in pi units)
        """
        self.pin_dic = {"a0": 0, "b0": 1}
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


class PerfectMirror(Model):
    """Model of perfect mirror (only one port), 100% reflection"""

    def __init__(self, phase: float = 0.0) -> None:
        """Initialize the model

        Args:
            phase (float): phase of the reflected ray (in pi unit)
        """
        self.pin_dic = {"a0": 0}
        self.param_dic = {}
        self.default_params = deepcopy(self.param_dic)
        self.N = 1
        self.phase = phase
        p1 = np.pi * self.phase
        self.S = np.array([[np.exp(1.0j * p1)]], complex)


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
        self.pin_dic = {f"a{i}": i for i in range(N)}
        self.pin_dic.update({f"b{i}": N + i for i in range(M)})
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
        self.pin_dic = {"a0": 0, "b0": 1}
        self.N = 2
        self.S = np.identity(self.N, complex)
        self.R = R
        self.n = n
        self.alpha = alpha
        self.t = t
        self.param_dic = {"wl": None}
        self.default_params = deepcopy(self.param_dic)

    def create_S(self) -> np.ndarray:
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        wl = self.param_dic["wl"]
        n = self.n
        t = self.t
        ex = np.exp(-4.0j * np.pi ** 2.0 / wl * n * self.R)
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
        self.pin_dic = {"a0": 0, "b0": 1}
        self.N = 2
        self.Neff = Neff
        self.L = L
        self.pn = param_name
        self.param_dic = {"R": R, "w": w, "wl": wl, "pol": pol, param_name: 0.0}
        self.default_params = deepcopy(self.param_dic)

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
        d1 = {f"a{i:03}": i for i in range(n)}
        d2 = {f"b{i:03}": n + i for i in range(m)}
        dic = {**d1, **d2}
        super().__init__(pin_dic=dic)
        t1 = self.d1 / Ri * np.array(line(n))
        t2 = self.d2 / R * np.array(line(m))
        T1, T2 = np.meshgrid(t1, t2, indexing="ij")
        DY = R * np.sin(T2) - Ri * np.sin(T1)
        DX = R * np.cos(T2) - Ri * (1.0 - np.cos(T1))
        self.DR = np.sqrt(DY ** 2.0 + DX ** 2.0)
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
        d1 = {f"a{i:03}": i for i in range(n)}
        d2 = {f"b{i:03}": n + i for i in range(m)}
        dic = {**d1, **d2}
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
        if ax in None:
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
        d1 = {f"a{i}": i for i in range(N)}
        d2 = {f"b{i}": N + i for i in range(N)}
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


class Model_from_NazcaCM(Model):
    """Class for model from a nazca cell with compact models"""

    def __init__(
        self,
        cell: nd.Cell,
        ampl_model: str = None,
        loss_model: str = None,
        optlength_model: str = None,
        allowed: Dict[str, Dict[str, Any]] = None,
    ) -> None:
        """Initialize the model

        This class is for building scattering matrices from the connectivity infomation built in nazca.

        In the definition of the scattering matrices, tree models can be provided.
            - ampl_model: This model returns directly the amplitude complex coefficient A.
            - loss_model: This model returns the loss used for the calculation of the modulus of the amplitude (in dB):
                abs(A)^2 = 10^loss/10.0
            - optlength_model: This model returns the optical length used for the calculation of the phase of the
                amplitude: angle(A) = 2.0*pi/wl*optlenght. It assumes the same unit as the wavelegnth.

        The priority of usage withing the models is the following.
            1. ampl: is the model for amplitude is found, the others are ignored. Amplitude between the same pin
                (reflection) is assumed 0 if not otherwaise specified.
            2. loss and optlength: if ampl  is not found, a model will be build using the available information
                between loss and optlength. If either one is missing, it will be assumed as 0
                (i.d. no loss means abs(A)=1, no phase means A purely real), but a warning will be raised.

        Args:
            cell (Nazca Cell): it expects a Nazca cell with some compact model defined.
            ampl_model (str): model to the used to diectly get the scattering matrix amplitudes
            loss_model (str): model to be used to estimate the loss
            optlength_model (str): model to be used for the optical length (phase)
            allowed (dict): mapping {Mode:extra}. The allowed mode in the cell and the extra information to pass to
                the compact model to build the optical length.
        """
        self.name = getattr(cell, "cell_name", cell.name)
        self.pin_dic = {}
        self.param_dic = {}
        self.default_params = {}
        opt_conn = {}
        n = 0

        # Checking for ampl model
        for name, pin in cell.pin.items():
            for mode, extra_in in allowed.items():
                opt = list(pin.path_nb_iter(ampl_model, extra=extra_in))
                if len(opt) != 0:
                    opt_conn[(pin, mode)] = opt
                    n += 1
        if n != 0:
            n = 0
            self.CM = {}
            pins = set([pin for pin, mode in opt_conn])
            for pi in pins:
                for mi in allowed:
                    pin_in = pi.basename if mi == "" else "_".join([pi.basename, mi])
                    self.pin_dic[pin_in] = n
                    n += 1
                    for po in pins:
                        for mo in allowed:
                            pin_out = (
                                po.basename if mo == "" else "_".join([po.basename, mo])
                            )
                            tup = tup = (pin_in, pin_out)
                            self.CM[tup] = 0.0

            logger.debug(f"Model for {self.name}: using amplitude model {ampl_model}")
            self.N = len(self.pin_dic)
            for (pin, mode), conn in opt_conn.items():
                for stuff in conn:
                    target = stuff[0]
                    CM = stuff[1]
                    extra_out = stuff[5]
                    pin_in = (
                        pin.basename if mode == "" else "_".join([pin.basename, mode])
                    )
                    try:
                        modeo = (
                            mode
                            if extra_out is None
                            else list(allowed.keys())[
                                list(allowed.values()).index(extra_out)
                            ]
                        )
                    except ValueError:
                        logger.error(
                            f"Model for {self.name}: mode {extra_out} is not in allowed {allowed}: ignored"
                        )
                        continue
                    pin_out = (
                        target.basename
                        if modeo == ""
                        else "_".join([target.basename, modeo])
                    )
                    tup = (pin_in, pin_out)
                    self.CM[tup] = (
                        self.wraps(ProtectedPartial(CM, **allowed[mode]))
                        if callable(CM)
                        else CM
                    )
        else:
            opt_conn = {}
            for name, pin in cell.pin.items():
                for mode, extra_in in allowed.items():
                    opt = {
                        (x[0], str(x[-1])): x[1:]
                        for x in pin.path_nb_iter(optlength_model, extra=extra_in)
                    }
                    lss = {
                        (x[0], str(x[-1])): x[1:]
                        for x in pin.path_nb_iter(loss_model, extra=extra_in)
                    }
                    for target in set(opt.keys()).union(set(lss.keys())):
                        if (pin, mode) not in opt_conn:
                            opt_conn[(pin, mode)] = {}
                        tup1 = (
                            opt[target]
                            if target in opt
                            else (0.0, None, None, None, allowed[mode])
                        )
                        tup2 = (
                            lss[target]
                            if target in lss
                            else (0.0, None, None, None, allowed[mode])
                        )
                        opt_conn[(pin, mode)][target] = tup1 + tup2
            logger.debug(
                f"Model for {self.name}: using optical length model {optlength_model} and loss model {loss_model}"
            )
            pins = set([pin for pin, mode in opt_conn])
            self.CM = {}
            for pi in pins:
                for mi in allowed:
                    pin_in = pi.basename if mi == "" else "_".join([pi.basename, mi])
                    self.pin_dic[pin_in] = n
                    n += 1
                    for po in pins:
                        for mo in allowed:
                            pin_out = (
                                po.basename if mo == "" else "_".join([po.basename, mo])
                            )
                            tup = tup = (pin_in, pin_out)
                            self.CM[tup] = 0.0

            self.N = len(self.pin_dic)

            for (pin, mode), conn in opt_conn.items():
                for (target, extra_target), stuff in conn.items():
                    OM = stuff[0]
                    extra_om = stuff[4]
                    AM = stuff[5]
                    extra_am = stuff[9]
                    if extra_om != extra_am:
                        if extra_om == allowed[mode]:
                            extra_om = extra_am

                    pin_in = (
                        pin.basename if mode == "" else "_".join([pin.basename, mode])
                    )
                    try:
                        modeo = (
                            mode
                            if extra_om is None
                            else list(allowed.keys())[
                                list(allowed.values()).index(extra_om)
                            ]
                        )
                    except ValueError:
                        logger.error(
                            f"Model for {self.name}: mode {extra_om} is not in allowed {allowed}: ignored"
                        )
                        continue
                    pin_out = (
                        target.basename
                        if modeo == ""
                        else "_".join([target.basename, modeo])
                    )
                    tup = (pin_in, pin_out)
                    OMt = ProtectedPartial(OM, **allowed[mode]) if callable(OM) else OM
                    AMt = ProtectedPartial(AM, **allowed[mode]) if callable(AM) else AM
                    self.CM[tup] = self.__class__.generator(OMt, AMt)

    @staticmethod
    def call_partial(func: functools.partial, *args, **kwargs) -> Any:
        """Evaluate a partial function with args ad kwargs provided and the args and kwargs saved at partial creation

        If any of the keywork arguments is not supported by the original function (the one used to create the partial),
        the keyword is ignored and a warning is logged.

        Args:
            func (functools.partial): partial function to be used
            args: positional arguments
            kwargs: keyword arguments

        Returns:
            Any: whatever the function returns after the unaccepted keyworkd arguments are removed.
        """
        while True:
            to_remove = getattr(func, "to_remove", [])
            for key in to_remove:
                func.keywords.pop(key, None)
                kwargs.pop(key, None)
            try:
                return func(*args, **kwargs)
            except TypeError as e:
                if "got an unexpected keyword argument" not in e.args[0]:
                    raise e
                else:
                    var = e.args[0].split("'")[-2]
                try:
                    func.to_remove.append(var)
                except AttributeError:
                    func.to_remove = [var]
                obj = func
                while True:
                    try:
                        name = obj.__name__
                        break
                    except AttributeError:
                        obj = obj.func
                code = obj.__code__
                logger.warning(
                    f'Function "{name}" in  file "{code.co_filename.split("/")[-1]}", line {code.co_firstlineno} does not support argument "{var}", so it is ignored. To remove this warning, add **kwargs to the function definition.'
                )

    @staticmethod
    def generator(
        OM: functools.partial, AM: functools.partial
    ) -> Callable[..., complex]:
        """Static method for generating the function creating the scattering matrix element from the compact models

        Args:
            OM (functools.partial): Compact model for Optical Length
            AM (functools.partial): Compact model for Loss

        Returns:
            function: function to crete the element of the scattering matrix.

        """

        def TOT(**kwargs) -> complex:
            OML = (
                Model_from_NazcaCM.call_partial(OM, **kwargs)
                if callable(OM)
                else copy(OM)
            )
            AML = (
                Model_from_NazcaCM.call_partial(AM, **kwargs)
                if callable(AM)
                else copy(AM)
            )
            return 10.0 ** (0.05 * AML) * np.exp(2.0j * np.pi / kwargs.get("wl") * OML)

        return TOT

    @staticmethod
    def wraps(func: functools.partial) -> Callable[..., complex]:
        """Static method for generating the function creating the scattering matrix element from the amplitude
        compact model.

        Args:
            func (functools.partial): Compact model for Amplitude

        Returns:
            function: function to create the element of the scattering matrix.
        """

        def wrapper(**kwargs):
            Inner = (
                Model_from_NazcaCM.call_partial(func, **kwargs)
                if callable(func)
                else copy(func)
            )
            return Inner

        return wrapper

    @classmethod
    def check_init(
        cls,
        cell: nd.Cell,
        ampl_model: str = None,
        loss_model: str = None,
        optlength_model: str = None,
        allowed: Dict[str, Dict[str, Any]] = None,
    ) -> Model_from_NazcaCM:
        """Alternative Creator. Mainly used for debugging

        This creator will directly try to solve the obtained model before returning.

        Args:
            cls (TYPE): DESCRIPTION.
            cell (TYPE): DESCRIPTION.
            ampl_model (TYPE, optional): DESCRIPTION. Defaults to None.
            loss_model (TYPE, optional): DESCRIPTION. Defaults to None.
            optlength_model (TYPE, optional): DESCRIPTION. Defaults to None.
            allowed (TYPE, optional): DESCRIPTION. Defaults to None.

        Returns:
            Model: Model of the cell.

        Raises:
            RuntimeError: if calling solve on the Model fails.
        """
        try:
            obj = cls(
                cell=cell,
                ampl_model=ampl_model,
                loss_model=loss_model,
                optlength_model=optlength_model,
                allowed=allowed,
            )
            obj.solve(wl=1.55)
            return obj
        except AttributeError:
            raise RuntimeError(f"Model for cell {obj.name} is not solved")

    @classmethod
    def nazca_init(
        cls,
        cell: nd.Cell,
        ampl_model: str = None,
        loss_model: str = None,
        optlength_model: str = None,
        allowed: Dict[str, Dict[str, Any]] = None,
    ) -> Union[Model_from_NazcaCM, solver.Solver]:
        """Alternative Creator to be used inside Nazca Integration

        This alternative creator will check if a model can be obtained from the Nazca cell. If not, an empty Solver will
        be returned instead. For a more detailed description on how to use this class see the __init__ method.

        Args:
            cell (Nazca Cell): it expects a Nazca cell with some compact model defined.
            ampl_model (str): model to the used to directly get the scattering matrix amplitudes
            loss_model (str): model to be used to estimate the loss
            optlength_model (str): model to be used for the optical length (phase)
            allowed (dict): mapping {Mode:extra}. The allowed mode in the cell and the extra information to
                pass to the compact model to build the optical length.

        Returns:
            Model or Solver: Model of the cell or empy Solver.

        """
        obj = cls(
            cell=cell,
            ampl_model=ampl_model,
            loss_model=loss_model,
            optlength_model=optlength_model,
            allowed=allowed,
        )
        if obj.is_empty():
            logger.debug(f"Model of cell {obj.name} is empy")
            return solver.Solver(name=obj.name)
        else:
            return obj

    def create_S(self) -> np.ndarray:
        """Creates the scattering matrix

        Returns:
            ndarray: Scattering Matrix
        """
        self.S = np.zeros((self.N, self.N), dtype="complex")
        for (pin1, pin2), CM in self.CM.items():
            CM = self.CM[(pin1, pin2)]
            self.S[self.pin_dic[pin1], self.pin_dic[pin2]] = (
                CM(**self.param_dic) if callable(CM) else CM
            )
        return self.S

    def __str__(self):
        """Formatter function for printing"""
        return f"Model (id={id(self)}) from Nazca {self.name:20}"
