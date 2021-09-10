#-------------------------------------------
#
# GenSol Package
#
# Python tool for simulation of abstract photonic circuits
#
# @author: Marco Passoni
#
#-------------------------------------------


""" Initialization file
"""

import logging
logger = logging.getLogger(__name__)

sol_list=[]

from version import __version__
from  solver.log import logfile, debugfile
from  solver.sol import *
from  solver.structure import *
from  solver.model import *

