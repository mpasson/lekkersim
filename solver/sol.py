import numpy as np
from copy import copy
from solver.structure import Structure
from solver import sol_list


class Solver:
    def __init__(self,structures=[],connections={},param_dic={}):
        self.structures=structures
        self.connections=connections
        self.connections_list=[]
        self.param_dic=param_dic
        self.pin_mapping={}
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
        

    def add_structure(self,structure):
        if structure not in self.structures: 
            self.structures.append(structure)
            for pin in structure.pin_list:
                self.free_pins.append(pin)
        else:
            raise ValueError('Structure already present')
        
    def connect(self,structure1,pin1,structure2,pin2):
        if (structure1,pin1) in self.connections_list: 
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


    def solve(self):
        for st in self.structures:
            try:
                st.model.param_dic.update(self.param_dic)
            except AttributeError:
                pass
            try:
                st.solver.param_dic.update(self.param_dic)
            except AttributeError:
                pass
        st_list=copy(self.structures)
        while len(st_list)!=1:
            source_st=st_list[0].gone_to
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

    def put(self,pins=None,pint=None):
        ST=Structure(solver=self)
        sol_list[-1].add_structure(ST)
        if (pins is not None) and (pint is not None):
            sol_list[-1].connect(ST,pins,pint[0],pint[1])
        return ST

def putpin(name,tup):
    sol_list[-1].map_pins({name:tup})

def connect(tup1,tup2):
    sol_list[-1].connect(tup1[0],tup1[1],tup2[0],tup2[1])



