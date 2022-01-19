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
from solver.sol import *
from solver.structure import *
from solver.model import *
