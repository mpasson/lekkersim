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

import logging

logger = logging.getLogger(__name__)

sol_list = []

from .version import __version__, git_clean
from solver.log import logfile, debugfile, logger
from solver.scattering import S_matrix
from solver.structure import Structure
from solver.sol import *
from solver.model import *


