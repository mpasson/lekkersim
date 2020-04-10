import numpy as np
from copy import copy

class Solver:
    def __init__(self,structures=[],connections={},param_dic={}):
        self.structures=structures
        self.connections=connections
        self.connections_list=[]
        self.param_dic=param_dic
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

    def show_free_pins(self):
        for st,pin in self.free_pins:
            print ('(%50s, %5s)' % (st,pin))

