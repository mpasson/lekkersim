import numpy as np
from solver.scattering import S_matrix
from copy import deepcopy
from copy import copy
import solver.model as mod
from solver.log import logger


class Structure:
    """Class defining a single element of the photonic circuit

    Args:
        pin_list (list) : list of str containing the names of the pins. Default is empty list
        model (Model) : model object from which the structue may be derived. Defauls is none
        solver (Solver) : solver object from which the structue may be derived. Defauls is none
        param_mapping (dict): dictionary of {oldname (str) : newname (str)} containning the mapping of the names of the parameters. Default is empty dict
    """

    def __init__(self, pin_list=[], model=None, solver=None, param_mapping=None):
        """Creator"""
        self.pin_list = []
        self.pin_dic = {}
        for i, pin in enumerate(pin_list):
            self.pin_list.append((self, pin))
            self.pin_dic[(self, pin)] = i
        self.in_pins = {}
        self.out_pins = {}
        self.structures = []
        self.conn_dict = {}
        self.connected_to = []
        self.gone_to = self
        self.N = len(pin_list)
        self.model = model
        self.sn = 1
        self.param_dic = {}
        if model is not None:
            for pin, i in model.pin_dic.items():
                self.pin_list.append((self, pin))
                self.pin_dic[(self, pin)] = i
        if solver is not None:
            for pin in solver.pin_mapping:
                self.pin_list.append((self, pin))
                self.pin_dic[(self, pin)] = None
        self.solver = solver
        self.param_mapping = {}
        param_mapping = {} if param_mapping is None else param_mapping
        for oldname, newname in param_mapping.items():
            self.param_mapping[newname] = oldname

    def __str__(self):
        """Formatter for printing"""
        if self.model is not None:
            return f"Structure (id={id(self)}) containing {str(self.model)}"
        elif self.solver is not None:
            return f"Structure (id={id(self)}) containing {str(self.solver)}"
        else:
            return f"{self.__class__.__name__} instance at {id(self)}>"

    @property
    def pin(self):
        """Alllows the feeding of the sintax structure.pin['pin_name'] where the tuple of structure and pin is required

        Returns:
            dict: Dictionary of the pins, in the format {'pinname' : (self, pinname)}
        """
        dic = {pin: (self, pin) for sl, pin in self.pin_list}
        return dic

    def get_pin_basenames(self):
        """Return the set of the basename of the pins"""
        return set([pin[1].split("_")[0] for pin in self.pin_list])

    def get_pin_modenames(self, target):
        """Return list of mode names given a pin basename"""
        li = []
        for pin in self.pin_list:
            try:
                pinname, modename = pin[1].split("_")
            except ValueError:
                pinname, modename = pin[1], ""
            if pinname == target:
                li.append(modename)
        return li

    def get_pins(self, basename=None):
        """Return list of the pins tuple

        Args:
            basename (str): if pins are in the form 'basename_modename', only the ones with the corresponding basename are returned

        Returs:
            list: list of pin tuples
        """
        if basename is None:
            return self.pin_list
        else:
            return [pin for pin in self.pin_list if pin.split("_")[0] == basename]

    def update_params(self, param_dic):
        """Updated the parametes dictionary of the represented optic componet

        Args:
            param_dic (dictionary): dcitionary {'param_name':param_value} containing value of the parameters to be updated

        Returns:
            None
        """
        update_dic = deepcopy(param_dic)
        for newname, oldname in self.param_mapping.items():
            if oldname in update_dic:
                update_dic.pop(oldname)
            if newname in param_dic:
                update_dic[oldname] = update_dic.pop(newname)
        # print(self,param_dic,self.param_mapping,update_dic)
        self.param_dic = update_dic
        if self.model is not None:
            self.model.update_params(update_dic)
        if self.solver is not None:
            self.solver.update_params(update_dic)

    def createS(self):
        """Creates the scattering matrix of the components

        Returns:
            ndarray: Scattering Matrix
        """
        if self.model is not None:
            self.Smatrix = self.model.solve(**self.param_dic).create_S()
        if self.solver is not None:
            model = self.solver.solve(**self.param_dic)
            for pin, i in model.pin_dic.items():
                self.pin_dic[(self, pin)] = i
            self.Smatrix = model.create_S()
        self.N = np.shape(self.Smatrix)[-1]
        self.ns = np.shape(self.Smatrix)[0]
        return self.Smatrix

    def print_pindic(self):
        """Print the mappping between the pins and the entries of the scatterng matrix"""
        for (st, pin), i in self.pin_dic.items():
            print("(%50s, %5s) : %3i" % (st, pin, i))

    def print_conn(self):
        """Print the connection of this structure to other ones"""
        for (st1, pin1), (st2, pin2) in self.conn_dict.items():
            print("(%50s, %5s) --> (%50s, %5s)" % (st1, pin1, st2, pin2))

    def reset(self):
        """Reset the mapping that keeps track of the solving"""
        self.gone_to = self
        self.param_dic = {}

    def add_pin(self, pin):
        """Add pin to structure

        Args:
            pin (str) : name of the pin

        Returns:
            None
        """
        if pin in self.pin_list:
            raise Exception("Pin already present, nothing is done")
        else:
            self.pin_list.append(pin)
            self.pin_dic[pin] = self.N
            self.N += 1

    def sel_input(self, pin_list):
        """Divide pins to be connected providing inputs pins

        Args:
            pin_list (list) : list of tuples (structure (Structure), pin_name (str)) providing input pins

        Returns:
            None
        """
        self.in_list = []
        self.out_list = copy(self.pin_list)
        for pin in pin_list:
            self.in_list.append(pin)
            self.out_list.remove(pin)

    def sel_output(self, pin_list):
        """Divide pins to be connected providing output pins

        Args:
            pin_list (list) : list of tuples (structure (Structure), pin_name (str)) providing output pins

        Returns:
            None
        """
        self.out_list = []
        self.in_list = copy(self.pin_list)
        # print(self.in_list)
        for pin in pin_list:
            self.out_list.append(pin)
            self.in_list.remove(pin)

    def split_in_out(self, in_pins=None, out_pins=None):
        """Created the scattering matrix object with left pins separated from right pins

        Args:
            in_pins (list) : list of tuples (structure (Structure), pin_name (str)) providing "left" pins
            out_pins (list) : list of tuples (structure (Structure), pin_name (str)) providing "right" pins

        Returns:
            None
        """
        # print('Splitting structure: ',self)
        in_pins = self.in_list if in_pins is None else in_pins
        out_pins = self.out_lists if out_pins is None else out_pins

        N = len(in_pins)
        M = len(out_pins)
        self.Sproc = S_matrix(N, M, ns=self.ns)
        for i, p in enumerate(in_pins):
            self.in_pins[p] = i
            for j, q in enumerate(in_pins):
                self.Sproc.S21[:, i, j] = self.Smatrix[
                    :, self.pin_dic[p], self.pin_dic[q]
                ]
        for i, p in enumerate(in_pins):
            for j, q in enumerate(out_pins):
                self.Sproc.S22[:, i, j] = self.Smatrix[
                    :, self.pin_dic[p], self.pin_dic[q]
                ]
        for i, p in enumerate(out_pins):
            self.out_pins[p] = i
            for j, q in enumerate(in_pins):
                self.Sproc.S11[:, i, j] = self.Smatrix[
                    :, self.pin_dic[p], self.pin_dic[q]
                ]
        for i, p in enumerate(out_pins):
            for j, q in enumerate(out_pins):
                self.Sproc.S12[:, i, j] = self.Smatrix[
                    :, self.pin_dic[p], self.pin_dic[q]
                ]

    def get_S_back(self):
        """Recreates the scattering matrix as ndarray"""
        self.Smatrix = np.concatenate(
            [
                np.concatenate([self.Sproc.S21, self.Sproc.S22], axis=-1),
                np.concatenate([self.Sproc.S11, self.Sproc.S12], axis=-1),
            ],
            axis=-2,
        )
        N = self.Sproc.N
        for pin, i in self.in_pins.items():
            self.pin_dic[pin] = i
        for pin, i in self.out_pins.items():
            self.pin_dic[pin] = i + N
        del self.Sproc
        self.in_pins = {}
        self.out_pins = {}

    def print_pins(self):
        """Print all pins of the structures, divided in self pins and contained pins"""
        print("Self pins:")
        for c, pinname in self.pin_list:
            if c is self:
                print(pinname)
        if len(self.structures) == 0:
            print("No additional Pins")
        else:
            print("Additional Pins:")
            for cc in self.structures:
                for c, pinname in self.pin_list:
                    if c is cc:
                        print(c, pinname)

    def add_conn(self, pin, target, target_pin):
        """Add connection between a self pin and a pin in another structure

        Args:
            pin (str) : name of self pin
            target (Structure) : structure to which to connect
            target_pin (str) : name of the pin in the target structure to whcih to connect

        Returns:
            None
        """
        tup = self.conn_dict.get((self, pin))
        if tup is not None:
            if tup != (target, target_pin):
                raise Exception("Pin already connected")
        else:
            self.conn_dict[(self, pin)] = (target, target_pin)
        if target not in self.connected_to:
            self.connected_to.append(target)

    def remove_pin(self, pin):
        """Remove pin from structure

        Args:
            pin (str) : name of pin to remove

        Returns:
            None
        """
        if (self, pin) in self.conn_dict:
            self.conn_dict.pop((self, pin))
        else:
            raise Exception(f"Pin {pin} not in conn_dict")
        if (self, pin) in self.pin_list:
            self.pin_list.remove((self, pin))
            self.pin_dic.pop((self, pin))
        else:
            raise Exception(f"Pin {pin} not in conn_dict")

    def cut_connections(self, target):
        """Remove all connection to target structure. Pins in self are kept

        Args:
            target (Structure) : structore to which to remove connnetions

        Returns:
            None
        """
        if target not in self.connected_to:
            raise Exception(
                f"Structure {target} is not connected to {self}. Impossible to remove."
            )
        self.connected_to.remove(target)
        copy_dic = copy(self.conn_dict)
        for (s, pin), (t, tpin) in copy_dic.items():
            if t is target:
                if (s, pin) in self.conn_dict:
                    self.conn_dict.pop((s, pin))
                else:
                    raise Exception(f"Pin {pin} not in conn_dict")

    def remove_connections(self, target):
        """Remove all connection to target structure. Remove also from self all pins involved in the connections

        Args:
            target (Structure) : structore to which to remove connnetions

        Returns:
            None
        """
        if target not in self.connected_to:
            raise Exception(
                f"Structure {target} is not connected to {self}. Impossible to remove."
            )
        self.connected_to.remove(target)
        copy_dic = copy(self.conn_dict)
        for (s, pin), (t, tpin) in copy_dic.items():
            if t is target:
                self.remove_pin(pin)

    def get_out_to(self, st):
        """Find pins of self with are connected to a target structure

        Args:
            st (Structure) : target structure

        Returns:
            pin_list : list of tuple (structure (Structure), pin_name (str)) contaning the list of pins which connects to the target structure
        """
        pin_list = []
        target_list = [st] + st.structures
        for (loc_c, loc_name), (tar_c, tar_name) in self.conn_dict.items():
            if tar_c in target_list:
                pin_list.append((loc_c, loc_name))
        return pin_list

    def get_in_from(self, st):
        """Find pins of self with are connected from a target structure

        Args:
            st (Structure) : target structure

        Returns:
            pin_list : list of tuple (structure (Structure), pin_name (str)) contaning the list of pins which connects from the target structure
        """
        pin_list = []
        loc_list = [self] + self.structures
        for (source_c, source_name), (loc_c, loc_name) in st.conn_dict.items():
            if loc_c in loc_list:
                pin_list.append((loc_c, loc_name))
        return pin_list

    def join(self, st):
        """Join two structures to create the one cotaining the merged structure

        Args:
            st (Structure) : target structure to join to self

        Returns:
            Structure : new structure combining the two starting one
        """
        self.createS()
        st.createS()
        # safety checks
        if st in self.structures:
            raise Exception("Cannot add structure: already containded")
        for st1 in st.structures:
            if st1 in self.structures:
                raise Exception("Cannot add structure: already containded")
        # retriving list of pins
        new_st = Structure()
        loc_out = self.get_out_to(st)
        tar_in = st.get_in_from(self)
        # print(loc_out)
        # print(tar_in)
        # Checking connectivity
        if len(loc_out) != len(tar_in):
            raise Exception("Connectivity problem: Different number of pins")
        for pin1 in loc_out:
            if pin1 != st.conn_dict[self.conn_dict[pin1]]:
                raise Exception("Connectivity problem: Not Symmetric")
        # Correct ordering of connection pins
        tar_in = []
        for pin in loc_out:
            tar_in.append(self.conn_dict[pin])

        # getting list of pins for local and target
        # print(loc_out)
        self.sel_output(loc_out)
        st.sel_input(tar_in)
        # print('Structure:',self)
        # print('In pins:',self.in_list)
        # print('Out pins:',self.out_list)

        # print('Structure:',st)
        # print('In pins:',st.in_list)
        # print('Out pins:',st.out_list)

        # Splittig scattering matrces of local and target
        self.split_in_out(self.in_list, self.out_list)
        st.split_in_out(st.in_list, st.out_list)

        # print(self.Sproc)
        # self.Sproc.S_print()

        # Joining S matrices for new one
        new_st.Sproc = self.Sproc.add(st.Sproc)
        new_st.in_pins = self.in_pins
        new_st.out_pins = st.out_pins
        new_st.get_S_back()

        # Generating new structures list
        if len(self.structures) == 0:
            new_st.structures.append(self)
            self.gone_to = new_st
        else:
            for st1 in self.structures:
                new_st.structures.append(st1)
                st1.gone_to = new_st
        if len(st.structures) == 0:
            new_st.structures.append(st)
            st.gone_to = new_st
        else:
            for st1 in st.structures:
                new_st.structures.append(st1)
                st1.gone_to = new_st
        # generating new pin lists
        add_pins = self.pin_list + st.pin_list
        for pin in loc_out + tar_in:
            add_pins.remove(pin)
        for pin in add_pins:
            new_st.add_pin(pin)

        # generating new connected to
        for st1 in self.connected_to + st.connected_to:
            if (st1 not in new_st.connected_to) and (st1 not in new_st.structures):
                new_st.connected_to.append(st1)

        # upgrading connection_dic
        for (st_source, pin_source), (st_target, pin_target) in {
            **self.conn_dict,
            **st.conn_dict,
        }.items():
            if not (
                (st_source in new_st.structures) and (st_target in new_st.structures)
            ):
                new_st.conn_dict[(st_source, pin_source)] = (st_target, pin_target)

        if self.ns == st.ns:
            new_st.ns = self.ns

        return new_st

    def intermediate(self, st, pin_mapping):
        """Used to generate the function for monitoring the modes between two structures

        This function is use to generate tge function to coumpe the modes amplitude at selected internal ports between 2 structures.

        Args:
            st (Structure) : structure towards which to monitor the ports. Connectivity between the structures has to be already specified.
            pin_mapping (dict): dictionary of pin mapping in the form {pin_name (str) : (structure (Structure), pin (str) )}.
                used when building the function to keep track of the correct pin naming

        Returns:
            Function: function for returnin modal coefficient given a dictionary with the inputs
            dict: dictionary containing the mapping between ports of st and scattering matrix entry
        """

        loc_out = self.get_out_to(st)
        tar_in = st.get_in_from(self)
        if len(loc_out) != len(tar_in):
            raise Exception("Connectivity problem: Different number of pins")
        for pin1 in loc_out:
            if pin1 != st.conn_dict[self.conn_dict[pin1]]:
                raise Exception("Connectivity problem: Not Symmetric")
        # Correct ordering of connection pins
        tar_in = []
        for pin in loc_out:
            tar_in.append(self.conn_dict[pin])

        self.sel_output(loc_out)
        st.sel_input(tar_in)

        # Splittig scattering matrces of local and target
        self.split_in_out(self.in_list, self.out_list)
        st.split_in_out(st.in_list, st.out_list)

        def solve_inter(dic):
            """This function calculates the couefficient of the internal modes

            Args:
                dic (dict): dictionary {pin_name (str) : input_amplitude (complex)}. Dictionary containing the complex amplitudes at each input port. Missing port are assumed wiht amplitude 0.0

            Returs:
                numpy array: array of the coefficient of the input modes
                numpy array: array of the coefficient of the output modes
            """
            u = np.zeros(len(self.in_list), dtype=complex)
            d = np.zeros(len(st.out_list), dtype=complex)
            for name, value in dic.items():
                mapped = pin_mapping[name]
                if mapped in self.in_list:
                    u[self.in_pins[mapped]] = value
                if mapped in st.out_list:
                    d[st.out_pins[mapped]] = value
            uo, do = self.Sproc.int_complete(st.Sproc, u, d)
            return uo, do

        return solve_inter, st.in_pins

    def get_model(self, pin_mapping=None):
        """Retunrn model corresponding to structure

        Args:
            pin_mapping (dict) : dictionary contain the information of pin mapping {pin_name (str) : (structure (Structure), pin (str))}

        Returns:
            SolvedModel : model object containing the scattering matrix od the structure
        """
        Smod = np.zeros((self.ns, self.N, self.N), complex)
        if pin_mapping is None:
            pin_mapping = self.solver.pin_mapping
        # if len(pin_mapping)!=len(self.pin_dic):
        # raise Exception('Not all pins mapped correctly')
        #    logger.warning(f'{self}:Running solve without complete mapping: some pins will not be not accessible')
        # for i,pin_name in enumerate(pin_mapping):
        #    pin_dic[pin_name]=i
        #   #print(i,pin_name,pin_mapping[pin_name])
        #    for j,pin_namej in enumerate(pin_mapping):
        #        Smod[:,i,j]=self.Smatrix[:,self.pin_dic[pin_mapping[pin_name]],self.pin_dic[pin_mapping[pin_namej]]]
        pin_dic = {name: self.pin_dic[pin] for name, pin in pin_mapping.items()}
        MOD = mod.SolvedModel(
            pin_dic=pin_dic, param_dic=self.param_dic, Smatrix=self.Smatrix
        )
        return MOD

    # def return_model(self):
    #    return self.model

    # def get_T(self,pin1,pin2):
    #    self.createS()
    #    return np.abs(self.Smatrix[self.pin_dic[(self,pin1)],self.pin_dic[(self,pin2)]])**2.0

    # def get_output(self,input_dic,power=True):
    #    try:
    #        ret=self.model.get_output(input_dic,power=power)
    #        return ret
    #    except AttributeError:
    #        raise ValueError('This structure does not have a model')
