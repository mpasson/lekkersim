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
from ._version import __version__

import logging

logger = logging.getLogger(__name__)

sol_list = []

from .log import logfile, debugfile
from .pin import Pin
from .scattering import S_matrix
from .structure import Structure
from .sol import *
from .model import *
