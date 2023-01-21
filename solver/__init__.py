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
from .log import logfile, debugfile, logger
from .scattering import S_matrix
from .structure import Structure
from .sol import *
from .model import *
from .nazca_integration import get_solver_from_nazca, Model_from_NazcaCM
