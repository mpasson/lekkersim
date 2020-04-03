import numpy as np
from scattering import S_matrix
from copy import deepcopy
from copy import copy

class Structure():
    def __init__(self,pin_list=[],model=None):
        self.pin_list=[]
        self.pin_dic={}
        for i,pin in enumerate(pin_list):
            self.pin_list.append((self,pin))
            self.pin_dic[(self,pin)]=i
        self.in_pins={}
        self.out_pins={}
        self.structures=[]
        self.conn_dict={}       
        self.connected_to=[]
        self.gone_to=self
        self.N=len(pin_list) 
        self.model=model   
        if model is not None:
            for pin,i in model.pin_dic.items():
                self.pin_list.append((self,pin))
                self.pin_dic[(self,pin)]=i
                self.Smatrix=model.create_S()


    def add_pin(self,pin):
        if pin in self.pin_list:
            raise Exception('Pin already present, nothing is done')
        else:
            self.pin_list.append(pin)
            self.pin_dic[pin]=self.N
            self.N+=1
        

    def sel_input(self,pin_list):
        self.in_list=[]
        self.out_list=copy(self.pin_list)
        for pin in pin_list:
            self.in_list.append(pin)
            self.out_list.remove(pin)

    def sel_output(self,pin_list):
        self.out_list=[]
        self.in_list=copy(self.pin_list)
        print(self.in_list)
        for pin in pin_list:
            self.out_list.append(pin)
            self.in_list.remove(pin)



    def split_in_out(self,in_pins,out_pins):
        N=len(in_pins)
        M=len(out_pins)
        self.Sproc=S_matrix(N,M)
        for i,p in enumerate(in_pins): 
            self.in_pins[p]=i
            for j,q in enumerate(in_pins): 
                self.Sproc.S12[i,j]=self.Smatrix[self.pin_dic[p],self.pin_dic[q]]
        for i,p in enumerate(in_pins): 
            for j,q in enumerate(out_pins): 
                self.Sproc.S22[i,j]=self.Smatrix[self.pin_dic[p],self.pin_dic[q]]
        for i,p in enumerate(out_pins): 
            self.out_pins[p]=i
            for j,q in enumerate(in_pins): 
                self.Sproc.S11[i,j]=self.Smatrix[self.pin_dic[p],self.pin_dic[q]]
        for i,p in enumerate(out_pins): 
            for j,q in enumerate(out_pins): 
                self.Sproc.S21[i,j]=self.Smatrix[self.pin_dic[p],self.pin_dic[q]]


    def get_S_back(self):
        self.Smatrix=np.vstack([np.hstack([self.Sproc.S21,self.Sproc.S22]),np.hstack([self.Sproc.S11,self.Sproc.S12])])
        N=self.Sproc.N
        for pin,i in self.in_pins.items():
            self.pin_dic[pin]=i
        for pin,i in self.out_pins.items():
            self.pin_dic[pin]=i+N
        del self.Sproc
        self.in_pins={}
        self.out_pins={}
        



    def print_pins(self):
        print('Self pins:')
        for c,pinname in self.pin_list:
            if c is self:
                print(pinname)    
        if len(self.structures)==0:
            print('No additional Pins')    
        else:
            print('Additional Pins:')
            for cc in self.structures:
                for c,pinname in self.pin_list:
                    if c is cc:
                        print(c,pinname)    

    def add_conn(self,pin,target,target_pin):
        if (self,pin) in self.conn_dict:
            raise Exception('Pin already connected')
        else:
            self.conn_dict[(self,pin)]=(target,target_pin)
        if target not in self.connected_to:
           self.connected_to.append(target)

    def get_out_to(self,st):
        pin_list=[]
        target_list=[st]+st.structures
        for (loc_c,loc_name),(tar_c,tar_name) in self.conn_dict.items():
            if st is  tar_c:
                pin_list.append((loc_c,loc_name))
        return  pin_list
        
    def get_in_from(self,st):
        pin_list=[]
        loc_list=[self]+self.structures
        for (source_c,source_name),(loc_c,loc_name) in st.conn_dict.items():
            if loc_c in  loc_list:
                pin_list.append((loc_c,loc_name))
        return  pin_list

    def join(self,st):
        #safety checks
        if st in self.structures:
            raise Exception('Cannot add structure: already containded')
        for st1 in st.structures:
            if st1 in self.structures:
                raise Exception('Cannot add structure: already containded')
        #retriving list of pins
        new_st=Structure()
        loc_out=self.get_out_to(st)
        tar_in =st.get_in_from(self)
        #print(loc_out)
        #print(tar_in)
        #Checking connectivity
        if len(loc_out)!=len(tar_in):
            raise Exception('Connectivity not symmetric')             
        for pin1 in loc_out:
            if pin1!=st.conn_dict[self.conn_dict[pin1]]:
                raise Exception('Connectivity not symmetric') 
        #Correct ordering of connection pins
        tar_in=[]
        for pin in loc_out:
            tar_in.append(self.conn_dict[pin])

        #getting list of pins for local and target
        print(loc_out)
        self.sel_output(loc_out)
        st.sel_input(tar_in)
        print('Structure:',self)
        print('In pins:',self.in_list)
        print('Out pins:',self.out_list)

        print('Structure:',st)
        print('In pins:',st.in_list)
        print('Out pins:',st.out_list)


        #Splittig scattering matrces of local and target
        self.split_in_out(self.in_list,self.out_list)
        st.split_in_out(st.in_list,st.out_list)

        print(self.Sproc)
        self.Sproc.S_print()     

        #Joining S matrices for new one
        new_st.Sproc=self.Sproc.add(st.Sproc)
        new_st.in_pins=self.in_pins
        new_st.out_pins=st.out_pins
        new_st.get_S_back()
    

        #Generating new structures list
        if len(self.structures)==0:
            new_st.structures.append(self)
            self.gone_to=new_st
        else:
            for st1 in self.structures:    
                new_st.structures.append(st1)
                st1.gone_to=new_st            
        if len(st.structures)==0:
            new_st.structures.append(st)
            st.gone_to=new_st
        else:            
            for st1 in st.structures:    
                new_st.structures.append(st1)
                st1.gone_to=new_st            
        #generating new pin lists
        add_pins=self.pin_list+st.pin_list
        for pin in loc_out+tar_in:
            add_pins.remove(pin)
        for pin in add_pins:
            new_st.add_pin(pin)

        #generating new connected to
        for st1 in self.connected_to+st.connected_to:
            if (st1 not in new_st.connected_to) and (st1 not in new_st.structures):
                new_st.connected_to.append(st1)


        return new_st


    

        

    



    
