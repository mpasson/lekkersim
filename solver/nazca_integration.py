from typing import Any, Callable, Self, Union

import numpy as np
from copy import copy
from copy import deepcopy
from itertools import tee
import functools

import nazca as nd

from .model import Model
from .structure import Structure
from .log import logger
from .utils import ProtectedPartial
from .sol import Solver


class Model_from_NazcaCM(Model):
    """Class for model from a nazca cell with compact models"""

    def __init__(
        self,
        cell: nd.Cell,
        ampl_model: str = None,
        loss_model: str = None,
        optlength_model: str = None,
        allowed: dict[str, dict[str, Any]] = None,
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
        allowed: dict[str, dict[str, Any]] = None,
    ) -> Self:
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
        allowed: dict[str, dict[str, Any]] = None,
    ) -> Union[Self, Solver]:
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
            return Solver(name=obj.name)
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


def get_solver_from_nazca(
    Cell,
    fullreturn=False,
    drc=True,
    prune=True,
    infolevel=0,
    atol=1e-4,
    ampl_model="ampl",
    optlength_model="optlen",
    loss_model="optloss",
    allowed=None,
):
    """Build a Solver Object from a Nazca Cell

    Args:
        Cell (Cell): Nazca cell
        fullreturn (bool): If True, the returned object is a dictionary containing all the solver of all sub-cells in the hierarchy. If False (default) only the top solver is returned.
        drc (bool): if True (default) pin connections violating drc will not be connected
        prune (bool): Remove empty branches from the solver structure. Default is True
        infolevel (int): Regulates the amount of info written to stdout. Default is 0 (no info added). The higher the number the more the information
        atol (flost): absolute tolerance of pin connection
        ampl_model (string): Name of the tracker to get the scattering amplidtude commpact model. It takes precedence over the loss and optical length model.
        optlength_model (string): Name of the tracker to get the optical length compact model. It is only used if ampl_model is not found in the cell.
        loss_model (string): Name of the tracker to get the loss compact model. It is only used if ampl_model is not found in the cell.

    Returns:
        Solver or dict: see fullreturn
    """

    allowed = {"": dict(pol=0, mode=0)} if allowed is None else allowed

    models = {}
    structures = {}
    cells_with_model = []

    it1, it2 = tee(nd.cell_iter(Cell, hierarchy="full", topdown=False))

    for params in it1:
        if params.cell_create:
            if params.cell.auxiliary:
                continue
            if "bbox" in params.cell.cell_name:
                continue
            if "icon" in params.cell.cell_name:
                continue
            if "stub" in params.cell.cell_name:
                continue
            # if set([t.cell.cnode for t in params.branch]).intersection(cells_with_model)!=set(): continue

            if infolevel > 0:
                logger.debug(f"{params.cell.cnode}")
            # logger.debug(params.cell.model_info)
            if params.cell.model_info["model"] is None:
                models[params.cell.cnode] = Model_from_NazcaCM.nazca_init(
                    params.cell,
                    ampl_model=ampl_model,
                    loss_model=loss_model,
                    optlength_model=optlength_model,
                    allowed=allowed,
                )
            else:
                models[params.cell.cnode] = params.cell.model_info["model"]
            if not models[params.cell.cnode].is_empty():
                cells_with_model.append(params.cell.cnode)
                if infolevel > 1:
                    logger.debug(f"  {models[params.cell.cnode]}")
                if infolevel > 0:
                    logger.debug("")
                continue

            params.iters["instance"], itcopy = tee(params.iters["instance"])
            for inode, xya, flip in itcopy:
                if inode.instance.cell.auxiliary:
                    continue
                if "bbox" in inode.instance.cell.cell_name:
                    continue
                if "stub" in inode.instance.cnode.up.cell.cell_name:
                    continue
                if "icon" in inode.instance.cnode.up.cell.cell_name:
                    continue
                param_mapping = (
                    inode.instance.model_info["param_mapping"]
                    if "param_mapping" in inode.instance.model_info
                    else {}
                )
                if infolevel > 1:
                    logger.debug(f"  {inode}")
                if isinstance(models[inode.instance.cell.cnode], Model):
                    ST = Structure(
                        model=models[inode.instance.cell.cnode],
                        param_mapping=param_mapping,
                    )
                if isinstance(models[inode.instance.cell.cnode], Solver):
                    ST = Structure(
                        solver=models[inode.instance.cell.cnode],
                        param_mapping=param_mapping,
                    )
                structures[inode] = ST
                models[params.cell.cnode].add_structure(ST)

            for inode, xya, flip in params.iters["instance"]:
                if inode.instance.cell.auxiliary:
                    continue
                if "bbox" in inode.instance.cell.cell_name:
                    continue
                if "stub" in inode.instance.cnode.up.cell.cell_name:
                    continue
                if "icon" in inode.instance.cnode.up.cell.cell_name:
                    continue
                if infolevel > 0:
                    logger.debug(f"  {inode}")
                for name, pin in inode.instance.pin.items():
                    # if infolevel>1: logger.debug(f'    {name} : {pin}')
                    if name not in structures[inode].get_pin_basenames():
                        continue
                    if name in ["org"]:
                        continue
                    if pin.type == "bbox":
                        continue
                    if "stub" in pin.cnode.up.cell.cell_name:
                        continue
                    if "icon" in pin.cnode.up.cell.cell_name:
                        continue
                    if infolevel > 0:
                        logger.debug(f"    {name} : {pin}")
                    new_nb_geo = []
                    for (tnode, pointer) in pin.nb_geo:
                        if infolevel > 1:
                            logger.debug(f"      {tnode}")
                        if tnode.type == "bbox":
                            continue
                        if tnode.name.endswith("org"):
                            continue
                        if "bbox" in tnode.name:
                            continue
                        if tnode.io is None:
                            continue
                        new_nb_geo.append((tnode, pointer))
                    connected = False
                    for (tnode, pointer) in new_nb_geo:
                        if tnode.name.isdigit() and len(new_nb_geo) > 1:
                            continue
                        if drc == True and pin.xs != tnode.xs:
                            continue
                        xya = list(nd.diff(pin, tnode))
                        xya[2] = xya[2] % 360.0
                        if infolevel > 0:
                            logger.debug(f"      {tnode} : {xya}")
                        mode_out = structures[inode].get_pin_modenames(name)
                        if infolevel > 2:
                            logger.debug(f"      {mode_out}, {xya}")
                        if (
                            np.all(np.isclose(xya, [0.0, 0.0, 180.0], atol=atol))
                            and tnode.up.name
                            in structures[tnode.cnode].get_pin_basenames()
                        ):
                            models[params.cell.cnode].connect_all(
                                structures[inode],
                                name,
                                structures[tnode.cnode],
                                tnode.up.name,
                            )
                            connected = True
                    if connected:
                        continue
                    for (tnode, pointer) in new_nb_geo:
                        if tnode.name.isdigit() and len(new_nb_geo) > 1:
                            continue
                        if drc == True and pin.xs != tnode.xs:
                            continue
                        xya = list(nd.diff(pin, tnode))
                        xya[2] = xya[2] % 360.0
                        if infolevel > 0:
                            logger.debug(f"      {tnode} : {xya}")
                        mode_out = structures[inode].get_pin_modenames(name)
                        if infolevel > 2:
                            logger.debug(f"      {mode_out}, {xya}")
                        if np.all(
                            np.isclose(xya, [0.0, 0.0, 0.0], atol=atol)
                        ) or np.all(np.isclose(xya, [0.0, 0.0, 360.0], atol=atol)):
                            for mi in mode_out:
                                name_out = name if mi == "" else "_".join([name, mi])
                                name_in = (
                                    tnode.name
                                    if mi == ""
                                    else "_".join([tnode.name, mi])
                                )
                                models[params.cell.cnode].map_pins(
                                    {name_in: structures[inode].pin[name_out]}
                                )
                                if infolevel > 1:
                                    logger.debug(f"        {name_in} : {name_out}")
            if infolevel > 0:
                logger.debug("\n")

    if prune:
        for cnode, solver in copy(models).items():
            if solver.prune():
                models.pop(cnode)

    if len(models) == 0:
        nd.main_logger(f"Impossible to buld solver for cell {Cell.name}", "error")
        return None

    if fullreturn:
        return models
    else:
        return models[Cell.cnode]
