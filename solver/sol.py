import numpy as np
from copy import copy
from copy import deepcopy
from solver.structure import Structure
from solver import sol_list


        
class Solver:
    def __init__(self,structures=[],connections={},param_dic={},default_params={},name=None):
        self.structures=[]
        self.connections={}
        self.connections_list=[]
        self.param_dic={}
        self.pin_mapping={}
        self.default_params=default_params
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
        self.structures=[]
        self.connections={}
        self.connections_list=[]
        self.param_dic={}
        self.pin_mapping={}
        sol_list.append(self)
        return self

    def __exit__(self,*args):
        sol_list.pop()
        
    def __str__(self):
        if self.name is None:
            return f'Solver object (id={id(self)})'
        else:
            return f'Solver of {self.name} (id={id(self)})'

    def add_structure(self,structure):
        if structure not in self.structures: 
            self.structures.append(structure)
            for pin in structure.pin_list:
                self.free_pins.append(pin)
        else:
            raise ValueError('Structure already present')
        
    def remove_structure(self,structure):
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
        print('Free pins of solver: %50s)' % (self))
        for st,pin in self.free_pins:
            try:
                pinname=list(self.pin_mapping.keys())[list(self.pin_mapping.values()).index((st,pin))]
                print ('(%50s, %5s) --> %5s' % (st,pin,pinname))
            except ValueError:
                print ('(%50s, %5s)' % (st,pin))
        print('')

    def show_structures(self):
        print('Structures and pins of solver: %50s)' % (self))
        for st in self.structures:
            print ('%50s' % (st))
        print('')
    


    def show_connections(self):
        print('Connection of solver: %50s)' % (self))
        for c1,c2 in self.connections.items():
                print ('(%50s, %5s) <--> (%50s, %5s)' % (c1+c2))
        print('')

               

    def show_pin_mapping(self):
        try:
            for pinname,(st,pin) in self.pin_mapping.items():
                print ('%5s --> (%50s, %5s)' % (pinname,st,pin))
        except AttributeError:
            print('No mapping defined')
            

    def map_pins(self,pin_mapping):
        self.pin_mapping.update(pin_mapping)


    def solve(self,**kwargs):
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
        self.param_dic[name]=value

    def put(self,pins=None,pint=None,param_mapping={}):
        ST=Structure(solver=deepcopy(self),param_mapping=param_mapping)
        sol_list[-1].add_structure(ST)
        if (pins is not None) and (pint is not None):
            sol_list[-1].connect(ST,pins,pint[0],pint[1])
        return ST

    def inspect(self):
        help.print(self)


class Pin():
    def __init__(self,name):
        self.name=name

    def put(self,tup):
        sol_list[-1].map_pins({self.name:tup})

def putpin(name,tup):
    sol_list[-1].map_pins({name:tup})

def connect(tup1,tup2):
    sol_list[-1].connect(tup1[0],tup1[1],tup2[0],tup2[1])

def set_default_params(dic):
    sol_list[-1].default_params=dic


class helper():
    def __init__(self):
        self.space=''

    def print(self,solver):
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

help=helper()
solver_print=help.print
prune=help.prune

 

