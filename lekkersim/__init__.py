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

__version__ = setuptools_git_versioning.get_version(
    root=os.path.join(*os.path.split(os.path.dirname(__file__))[:-1])
)

import logging

logger = logging.getLogger(__name__)

sol_list = []

from .log import logfile, debugfile
from .scattering import S_matrix
from .structure import Structure
from .sol import *
from .model import *
