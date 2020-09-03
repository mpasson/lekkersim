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


sol_list=[]

from  solver.sol import *
from  solver.structure import *
from  solver.model import *

sol_list=[Solver()]
