import numpy as np
from solver.scattering import S_matrix
from copy import deepcopy
from copy import copy
import solver.model as mod


class Structure:
    def __init__(self,pin_list=[],model=None,solver=None,param_mapping={}):
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
        if solver is not None:
            for pin in solver.pin_mapping:
                self.pin_list.append((self,pin))
        self.solver=solver
        self.param_mapping={}
        for oldname,newname in param_mapping.items():
            self.param_mapping[newname]=oldname

    def __str__(self):
        if self.model is not None:
            return f'Structure (id={id(self)}) containing {str(self.model)}'
        elif self.solver is not None:
            return f'Structure (id={id(self)}) containing {str(self.solver)}'
        else:
            return f"{self.__class__.__name__} instance at {id(self)}>" 

    @property
    def pin(self):
        dic={pin:(self,pin) for sl,pin in self.pin_list}
        return dic

    def update_params(self,param_dic):
        #print(self,param_dic,self.param_mapping)
        update_dic=deepcopy(param_dic)
        for newname,oldname in self.param_mapping.items():
            if newname in param_dic:
                update_dic[oldname]=update_dic.pop(newname)
        if self.model is not None:
            self.model.param_dic.update(update_dic)
        if self.solver is not None:
            self.solver.param_dic.update(update_dic)
            #self.solver.update_params(update_dic)

    
                


    def createS(self):
        if self.model is not None:
            self.Smatrix=self.model.create_S()
        if self.solver is not None:
            self.model=self.solver.solve()
            for pin,i in self.model.pin_dic.items():
                self.pin_dic[(self,pin)]=i
            self.Smatrix=self.model.create_S()
        self.N=np.shape(self.Smatrix)[0]

    def print_pindic(self):
        for (st,pin),i in self.pin_dic.items():
            print ('(%50s, %5s) : %3i' % (st,pin,i) )

    def print_conn(self):
        for (st1,pin1),(st2,pin2) in self.conn_dict.items():
            print ('(%50s, %5s) --> (%50s, %5s)' % (st1,pin1,st2,pin2) )

    def reset(self):
        self.gone_to=self


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
        #print(self.in_list)
        for pin in pin_list:
            self.out_list.append(pin)
            self.in_list.remove(pin)



    def split_in_out(self,in_pins,out_pins):
        #print('Splitting structure: ',self)
        N=len(in_pins)
        M=len(out_pins)
        self.Sproc=S_matrix(N,M)
        for i,p in enumerate(in_pins): 
            self.in_pins[p]=i
            for j,q in enumerate(in_pins): 
                self.Sproc.S21[i,j]=self.Smatrix[self.pin_dic[p],self.pin_dic[q]]
        for i,p in enumerate(in_pins): 
            for j,q in enumerate(out_pins): 
                self.Sproc.S22[i,j]=self.Smatrix[self.pin_dic[p],self.pin_dic[q]]
        for i,p in enumerate(out_pins): 
            self.out_pins[p]=i
            for j,q in enumerate(in_pins): 
                self.Sproc.S11[i,j]=self.Smatrix[self.pin_dic[p],self.pin_dic[q]]
        for i,p in enumerate(out_pins): 
            for j,q in enumerate(out_pins): 
                self.Sproc.S12[i,j]=self.Smatrix[self.pin_dic[p],self.pin_dic[q]]


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

    def remove_pin(self,pin):
        if (self,pin) in self.conn_dict:
            self.conn_dict.pop((self,pin))
        else:
            raise Exception(f'Pin {pin} not in conn_dict')
        if (self,pin) in self.pin_list:
            self.pin_list.remove((self,pin))
            self.pin_dic.pop((self,pin))
        else:
            raise Exception(f'Pin {pin} not in conn_dict')


    def remove_connections(self,target):
        if target not in self.connected_to:
            raise Exception(f'Structure {target} is not connected to {self}. Impossible to remove.')
        self.connected_to.remove(target)
        copy_dic=copy(self.conn_dict)
        for (s,pin),(t,tpin) in copy_dic.items():
            if t is target:
                self.remove_pin(pin)

    def get_out_to(self,st):
        pin_list=[]
        target_list=[st]+st.structures
        for (loc_c,loc_name),(tar_c,tar_name) in self.conn_dict.items():
            if tar_c in target_list:
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
        self.createS()
        st.createS()
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
            raise Exception('Connectivity problem: Different number of pins')             
        for pin1 in loc_out:
            if pin1!=st.conn_dict[self.conn_dict[pin1]]:
                raise Exception('Connectivity problem: Not Symmetric') 
        #Correct ordering of connection pins
        tar_in=[]
        for pin in loc_out:
            tar_in.append(self.conn_dict[pin])

        #getting list of pins for local and target
        #print(loc_out)
        self.sel_output(loc_out)
        st.sel_input(tar_in)
        #print('Structure:',self)
        #print('In pins:',self.in_list)
        #print('Out pins:',self.out_list)

        #print('Structure:',st)
        #print('In pins:',st.in_list)
        #print('Out pins:',st.out_list)


        #Splittig scattering matrces of local and target
        self.split_in_out(self.in_list,self.out_list)
        st.split_in_out(st.in_list,st.out_list)

        #print(self.Sproc)
        #self.Sproc.S_print()     

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

        #upgrading connection_dic
        for (st_source,pin_source),(st_target,pin_target) in {**self.conn_dict,**st.conn_dict}.items():
            if not ((st_source in new_st.structures) and (st_target in new_st.structures)):
                new_st.conn_dict[(st_source,pin_source)]=(st_target,pin_target)

        return new_st


    def get_model(self,pin_mapping=None):
        Smod=np.zeros((self.N,self.N),complex)
        if pin_mapping is None:
            pin_mapping=self.solver.pin_mapping
        if len(pin_mapping)!=len(self.pin_dic):
            print(self.solver)
            print('')
            for t in pin_mapping.items():
                print(t)
            print('')
            for t in self.pin_dic.items():
                print(t)

            raise Exception('Not all pins mapped correctly')
        pin_dic={}
        for i,pin_name in enumerate(pin_mapping):
            pin_dic[pin_name]=i
            #print(i,pin_name,pin_mapping[pin_name])
            for j,pin_namej in enumerate(pin_mapping):
                Smod[i,j]=self.Smatrix[self.pin_dic[pin_mapping[pin_name]],self.pin_dic[pin_mapping[pin_namej]]]
        MOD=mod.model(pin_list=list(pin_dic.keys()))
        MOD.pin_dic=pin_dic
        MOD.N=len(pin_dic)
        MOD.S=Smod
        return MOD    

    def return_model(self):
        return self.model


    def get_T(self,pin1,pin2):
        self.createS()
        return np.abs(self.Smatrix[self.pin_dic[(self,pin1)],self.pin_dic[(self,pin2)]])**2.0

    def get_output(self,input_dic,power=True):
        try: 
            ret=self.model.get_output(input_dic,power=power)
            return ret    
        except AttributeError:
            raise ValueError('This structure does not have a model')
            
        
    



    
