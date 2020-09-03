#-------------------------------------------
#
# GenSol Package
#
# Python tool for simulation of abstract photonic circuits
#
# @author: Marco Passoni
#
#------------------------------------------

import numpy as np
from copy import copy
from copy import deepcopy
from solver.structure import Structure
from solver import sol_list


        
class Solver:
    """Class Solver
    This class defines the simulations. It contains all the structures of the optical componets, and has the methods for running the simulation and accessing the results.
    """
    def __init__(self,structures=None, connections=None, param_dic=None, default_params=None, name=None):
        """Creator
        Args:
            structures (list) : list of structures in the solver. Default is None (empty list)
            connections (dict) : dictionary of tuples (structure (Structure), pin (str)) containing connections {(structure1,pin1):(structure2,pin2)}. Default is None (empty dictionary)
            param_dic (dict) :  dictionary of parameters {param_name (str) : param_value (usually float)}. Default is None (empty dictionary)
            default_params (dict) : dictionary of default parameters {param_name (str) : param_value (usually float)}. Default is None (empty dictionary)
        """
        self.structures=structures if structures is not None else []
        self.connections=connections if connections is not None else {}
        self.connections_list=[]
        self.param_dic=param_dic if param_dic is not None else {}
        self.pin_mapping={}
        self.default_params=default_params if default_params is not None else {}
        for pin1,pin2 in self.connections.items():
            self.connections_list.append(pin1)
            self.connections_list.append(pin2)
        if len(set(self.connections_list))!=len(self.connections_list):
            raise ValueError('Same pin connected multiple time')
        self.free_pins=[]
        for st in self.structures:
            for pin in st.pin_list:
                self.free_pins.append(pin)
        for pin in self.connections_list:
            self.free_pins.remove(pin)
        self.name=name

    def __enter__(self):
        """__enter__ function for the with statement. Add the newly created solver to sol_list.
        """
        sol_list.append(self)
        return self

    def __exit__(self,*args):
        """__enter__ function for the with statement. Remove last element of sol_list
        """
        sol_list.pop()
        
    def __str__(self):
        """Formatter for printing
        """
        if self.name is None:
            return f'Solver object (id={id(self)})'
        else:
            return f'Solver of {self.name} (id={id(self)})'

    def add_structure(self,structure):
        """Add a structure to the solver
        Args:
            structure (Structure) : structure to be added
        """
        if structure not in self.structures: 
            self.structures.append(structure)
            for pin in structure.pin_list:
                self.free_pins.append(pin)
        else:
            raise ValueError('Structure already present')
        
    def remove_structure(self,structure):
        """Remove structure from solver, also removing all the connections to other structures
        Args:
            structure (Structure) : structure to be removed          
        """
        if structure not in self.structures:
            raise Exception('Structure {structure} is not in solver {self}')
        self.structures.remove(structure)
        for st in structure.connected_to:
            st.remove_connections(structure)
        copy_dic=copy(self.connections)
        for (st1,pin1),(st2,pin2) in copy_dic.items():
            if st1 is structure: self.connections.pop((st1,pin1))
            if st2 is structure: self.connections.pop((st1,pin1))
        copy_dic=copy(self.free_pins)
        for st,pin in copy_dic:
            if st is structure: self.free_pins.remove((st,pin))
        copy_dic=copy(self.pin_mapping)
        for pinname,(st,pin) in copy_dic.items():
            if st is structure: self.pin_mapping.pop(pinname)
                
        
            


    def connect(self,structure1,pin1,structure2,pin2):
        """Connect two different structures in the solver by the specified pins
        Args:
            structure1 (Structure) : first structure
            pin1 (str) : pin of first structure
            structure2 (Structure) : second structure
            pin2 (str) : pin of second structure
        """
        if (structure1,pin1) in self.connections_list:
            if  (structure1,pin1) in self.connections and self.connections[(structure1,pin1)]==(structure2,pin2) : return
            if  (structure2,pin2) in self.connections and self.connections[(structure2,pin2)]==(structure1,pin1) : return
            raise ValueError('Pin already connected')
        if (structure2,pin2) in self.connections_list: 
            raise ValueError('Pin already connected')
        self.connections_list.append((structure1,pin1))
        self.connections_list.append((structure2,pin2))
        self.connections[(structure1,pin1)]=(structure2,pin2)
        self.free_pins.remove((structure1,pin1))
        self.free_pins.remove((structure2,pin2))
        structure1.add_conn(pin1,structure2,pin2)    
        structure2.add_conn(pin2,structure1,pin1)    


    def show_free_pins(self):
        """Print all pins of the structure in the solver whcih are not connected. If a pin mapping exists, it is also reported
        """
        print('Free pins of solver: %50s)' % (self))
        for st,pin in self.free_pins:
            try:
                pinname=list(self.pin_mapping.keys())[list(self.pin_mapping.values()).index((st,pin))]
                print ('(%50s, %5s) --> %5s' % (st,pin,pinname))
            except ValueError:
                print ('(%50s, %5s)' % (st,pin))
        print('')

    def show_structures(self):
        """Print all structures in the solver
        """
        print('Structures and pins of solver: %50s)' % (self))
        for st in self.structures:
            print ('%50s' % (st))
        print('')
    


    def show_connections(self):
        """Print all connected pins in the solver
        """
        print('Connection of solver: %50s)' % (self))
        for c1,c2 in self.connections.items():
                print ('(%50s, %5s) <--> (%50s, %5s)' % (c1+c2))
        print('')

               

    def show_pin_mapping(self):
        """If a pin mapping is defined, print only mapped pins
        """
        try:
            for pinname,(st,pin) in self.pin_mapping.items():
                print ('%5s --> (%50s, %5s)' % (pinname,st,pin))
        except AttributeError:
            print('No mapping defined')
            

    def map_pins(self,pin_mapping):
        """Add mapping of pins
        Args:
            pin_mapping (dict): dictionary of pin mapping in the form {pin_name (str) : (structure (Structure), pin (str) )}
        """
        self.pin_mapping.update(pin_mapping)


    def solve(self,**kwargs):
        """Calculates the scattering matrix of the solver
        Args:
            kwargs (dict) : paramters in the form param_name=param_value
        Returs:
            Model : model containing the scattering matrix
        """
        self.param_dic.update(self.default_params)
        self.param_dic.update(kwargs)
        for st in self.structures:
            st.update_params(self.param_dic)
        st_list=copy(self.structures)
        for st in st_list:
            if len(st.connected_to)==0:
                st.createS()
        #if len(st_list)==1:
        #    st_list[0].createS()
        while len(st_list)!=1:
            source_st=st_list[0].gone_to
            if len(st_list[0].connected_to)==0:
                tar_st=st_list[1]
            else:
                tar_st=st_list[0].connected_to[0].gone_to
            #print('Started join step')
            #print('Source structure: %50s inside %50s' % (source_st,source_st.gone_to))
            #print('Target structure: %50s inside %50s' % (tar_st,tar_st.gone_to))
            #print('\nSource connections')
            #source_st.print_conn()
            #print('\nTarget connections')
            #tar_st.print_conn()
            new_st=source_st.join(tar_st)
            st_list.remove(source_st)
            st_list.remove(tar_st)
            st_list.append(new_st)
            #print('\nGener. structure: %50s' % (new_st))
            #print('Generated connections')
            #new_st.print_conn()
            #print('\n')
        self.Final=st_list[0]
        for st in self.structures:
            st.reset()
        mod=st_list[0].get_model(self.pin_mapping)
        return mod


    def set_param(self,name,value=None):
        """Set a value for one parameter
        Args:
            name (str) : name of the parameter
            value (usually float) : value of the parameter
        """
        self.param_dic[name]=value

    def put(self,pins=None,pint=None,param_mapping={}):
        """Function for putting a Solver in another Solver object, and eventually specify connections
        This function creates a Structure object for the Solver and place it in the current active Solver
        If both pins and pint are provided, the connection also is made. 
        Args:
            pins (str): pin of model to be connected
            pint (tuple): tuple (structure (Structure) , pin (str)) existing structure and pin to which to connect pins of model
            param_mapping (dict): dictionary of {oldname (str) : newname (str)} containning the mapping of the names of the parameters
        Returns:
            Structure: the Structure instance created from the Solver
        """
        ST=Structure(solver=deepcopy(self),param_mapping=param_mapping)
        sol_list[-1].add_structure(ST)
        if (pins is not None) and (pint is not None):
            sol_list[-1].connect(ST,pins,pint[0],pint[1])
        return ST

    def inspect(self):
        """Print the full hierarchy of the solver. Just a wrapper for help.print()
        """
        help.print(self)


class Pin():
    """Helper class for more user friendly pin mapping (same as Nazca sintax
    """
    def __init__(self,name):
        """Creator
        Args:
            name (str) : name of the pin
        """
        self.name=name

    def put(self,tup):
        """Maps the pins in the tuple to self.name
        Args:
            tup (tuple) : tuple of (structure (Structure), pin (str)) containing the data to the pin to be mapped
        """
        sol_list[-1].map_pins({self.name:tup})

def putpin(name,tup):
    """Maps a pin of the current active solver
    Args:
        name (str) : name of the new pin
        tup (tuple) : tuple of (structure (Structure), pin (str)) containing the data to the pin to be mapped
    """ 
    sol_list[-1].map_pins({name:tup})

def connect(tup1,tup2):
    """Connect two structures in the active Solver
    Args:
        tup1 (tuple) : tuple of (structure (Structure), pin (str)) containing the data of the first pin
        tup1 (tuple) : tuple of (structure (Structure), pin (str)) containing the data of the second pin
    """
    sol_list[-1].connect(tup1[0],tup1[1],tup2[0],tup2[1])

def set_default_params(dic):
    """Set default parameters for the solver
    Args:
        dic (dict): dictionary of the default parameters {param_name (str) : default_value (usually float)}
    """
    sol_list[-1].default_params=dic


class helper():
    """Helper class for some operation on solvers
    """
    def __init__(self):
        """Creator
        """
        self.space=''

    def print(self,solver):
        """Print the full hierarchy of the solver
        Args:
            solver (Solver) : solver to be printed
        """
        print(f'{self.space}{solver}')
        for s in solver.structures:
            if s.solver is not None:
                self.space=self.space+'  '
                self.print(s.solver)
                self.space=self.space[:-2]    
            elif s.model is not None:
                print(f'{self.space}  {s.model}')
            else:
                print(f'{self.space}  {s}')

    def prune(self,solver):
        """Remove dead branch in the solver hierarchy (the ones ending with an empy solver)
        Args:
            solver (Solver) : solver to be pruned
        Returns:
            bool: True if Solver is empty
        """
        #print(f'Entered in {solver}')
        if not isinstance(solver,Solver):
            return False
        not_empty=[]
        copy_list=copy(solver.structures)
        for st in copy_list:
            if st.model is not None:
                not_empty.append(st)
                continue
            if st.solver is not None:
                if self.prune(st.solver):
                    solver.remove_structure(st)
                else:
                    not_empty.append(st)
        return len(not_empty)==0

"""Renaming some of the inner function to make them available at top level
"""
help=helper()
solver_print=help.print
prune=help.prune

 

