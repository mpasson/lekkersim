# -------------------------------------------
#
# GenSol Package
#
# Python tool for simulation of abstract photonic circuits
#
# @author: Marco Passoni
#
# -------------------------------------------


""" Initialization file
"""
import os
import setuptools_git_versioning


def _get_version():
    try:
        import poetry_dynamic_versioning as pdv

        config = pdv._get_config_from_path(os.path.split(os.path.dirname(__file__))[0])
        return pdv._get_version(config=config)
    except:
        from ._version import __version__

        return __version__


__version__ = _get_version()

import logging

logger = logging.getLogger(__name__)

sol_list = []

from .log import logfile, debugfile
from .scattering import S_matrix
from .structure import Structure
from .sol import *
from .model import *
