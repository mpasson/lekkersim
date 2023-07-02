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

try:
    from ._version import __version__
except ModuleNotFoundError:
    from hatch_vcs.version_source import VCSVersionSource
    import toml

    basedir = os.path.split(os.path.dirname(__file__))[0]
    config = toml.load(os.path.join(basedir, "pyproject.toml"))
    vcs_version = VCSVersionSource(basedir, config["tool"]["hatch"]["version"])
    __version__ = vcs_version.get_version_data()["version"]

import logging

logger = logging.getLogger(__name__)

sol_list = []

from .log import logfile, debugfile
from .scattering import S_matrix
from .structure import Structure
from .sol import *
from .model import *
