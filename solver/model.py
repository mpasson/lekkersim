#-------------------------------------------
#
# GenSol Package
#
# Python tool for simulation of abstract photonic circuits
#
# @author: Marco Passoni
#
#-------------------------------------------


"""File containing the model calls and related methods
"""


import numpy as np
from solver.scattering import S_matrix
import solver.structure
from solver import sol_list
from solver import logger
from copy import deepcopy
from copy import copy
import pandas as pd
import warnings
from scipy.interpolate import interp1d
import io


def diag_blocks(array_list):
    """Function building a block diagonal array for list of array

    Args:
        array_list (list): list of ndarrays. Each array is a 2D square array.

    Returns:
         ndarray: 2D array with the arrays in array_list as blocks in the diagonal
    """
    for A in array_list:
        if np.shape(A)[0]!=np.shape(A)[1]: raise ValueError('Matrix is not square')
    Nl=[np.shape(A)[0] for A in array_list]
    N=sum(Nl)
    M=np.zeros((N,N), dtype=complex)
    m=0
    for n,A in zip(Nl,array_list):
        M[m:m+n,m:m+n]=A
        m+=n
    return M


class Model:
    """Class Model
        
        It contains the model (definition of the scattering matrix) for each base photonic building block
        This is the general template for the class, no model is defined here. 
        Each model is a separate class based on the main one.
        The only use case for using this class dicrectrly is to call it without arguments to create an empy model that will be eliminated if prune is called

        Args:
            pin_list (list): list of strings containing the model's pin names 
            param_dic (dictionary): dcitionary {'param_name':param_value} containing the definition of the model's parameters.
            Smatrix (ndarray) : Fixed S_matrix of the model

    """
    def __init__(self, pin_dic=None, param_dic=None, Smatrix=None):
        """Creator of the class
        """
        self.pin_dic={} if pin_dic is None else pin_dic
        self.N=len(self.pin_dic)
        if Smatrix is not None:
            if self.N!=np.shape(Smatrix)[-1]:
                self.N = np.shape(Smatrix)[-1]
        self.S=np.identity(self.N,complex) if Smatrix is None else Smatrix
        self.param_dic = {} if param_dic is None else param_dic
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S

    def _expand_S(self):
        self.N=self.N//self.np
        S=self._create_S()
        self.N=self.N*self.np
        #self.S= diag_blocks(self.np*[S])
        return diag_blocks(self.np*[S])

    def inspect(self):
        """Function that print self. It mocks the same function of Solver
        """
        print(f'{self}')

    def _create_S(self):
        """Function for returning the scattering matrix of the model

        Returns:
            ndarray: Scattering matrix of the model
        """
        return self.S

    def print_S(self, func=np.abs):
        """Function for nice printing of scattering matrix in agreement with pins

        Args:
            func (callable): function to be applied at the scatterinf matrix before returning.
                if None (default), the raw complex coefficients are printed
        """
        func = (lambda x:x) if func is None else func
        a=list(self.pin_dic.keys())
        ind=list(self.pin_dic.values())
        indsort = np.argsort(a)
        a=[a[i] for i in indsort]
        indsort = np.array(ind)[indsort]
        for pin,i in self.pin_dic.items():
            print(pin,i)
        print(a)
        print(indsort)
        S = self.create_S()
        I,J = np.meshgrid(indsort, indsort, indexing='ij')
        S = S[0,I,J] if len(np.shape(S))==3 else S[I,J]
        S = func(S) if func is not None else S
        st='            '
        for p in a:
            st+=f' {p:8} '
        st+='\n'
        for i,pi in enumerate(a):
            st+=f' {pi:8} '
            for j,pj in enumerate(a):
                pr= S[i,j]
                st+=f' {pr:8.4f} '
            st+='\n'
        print(st)



    def S2PD(self, func=None):
        """Function for returning the the Scattering Matrix as a PD Dataframe

        Args:
            func (callable): function to be applied at the scatterinf matrix before returning.
                if None (default), the raw complex coefficients are returned

        Returns:
            Pandas DataFrame: Scattering Matrix with name of pins  
        """
        a=list(self.pin_dic.keys())
        ind=list(self.pin_dic.values())
        indsort = np.argsort(a)
        a=[a[i] for i in indsort]
        indsort = np.array(ind)[indsort]
        S = self.create_S()
        I,J = np.meshgrid(indsort, indsort, indexing='ij')
        S = S[0,I,J] if len(np.shape(S))==3 else S[I,J]
        S = func(S) if func is not None else S
        data = pd.DataFrame(data=S, index=a, columns=a)
        return data           

    def get_T(self,pin1,pin2):
        """Function for returning the energy transmission between two ports

        If the two ports are the same the returned value has the meaning of relection

        Args:
            pin1 (str): Name of input pin
            pin2 (str): Name of output pin

        Returns:
            float: Energy transmission between the ports            
        """
        if np.shape(self.S)[0] > 1: logger.warning(f'{self}:Using get_T on a sweep solve. Consider using get_data')
        return np.abs(self.S[0,self.pin_dic[pin1],self.pin_dic[pin2]])**2.0

    def get_PH(self,pin1,pin2):
        """Function for returning the phase of the transmission between two ports

        If the two ports are the same the returned value has the meaning of relection

        Args:
            pin1 (str): Name of input pin
            pin2 (str): Name of output pin

        Returns:
            float: Phase of the transmission between the ports            
        """
        if np.shape(self.S)[0] > 1: logger.warning(f'{self}:Using get_PH on a sweep solve. Consider using get_data')
        return np.angle(self.S[0,self.pin_dic[pin1],self.pin_dic[pin2]])

    def get_A(self,pin1,pin2):
        """Function for returning complex amplitude of the transmission between two ports
            
        Args:
            pin1 (str): Name of input pin
            pin2 (str): Name of output pin
        
        Returns:
            float: Complex amplitude of the transmission between the ports            
        """
        if np.shape(self.S)[0] > 1: logger.warning(f'{self}:Using get_A on a sweep solve. Consider using get_data')
        return self.S[0,self.pin_dic[pin1],self.pin_dic[pin2]]


    def expand_mode(self,mode_list):
        """This function expands the model by adding additional modes.
        
        For each pin a number of pins equal the the length of mode_list will be created. The pin names will be "{pinname}_{modename}}".
        Each mode will have the same behavior.
        
        Args:
            mode_list (list) : list of strings containing the modenames

        Returns:
            Model : new model with expanded modes 
        """
        self.np=len(mode_list)
        self.mode_list=mode_list
        new_pin_dic={}
        for name,n in self.pin_dic.items():
            for i,mode in enumerate(self.mode_list):
                new_pin_dic[f'{name}_{mode}']=i*self.N+n
        self.pin_dic=new_pin_dic
        self.create_S=self._expand_S
        self.N=self.N*self.np
        return self

    def get_output(self,input_dic,power=True):
        """Returns the outputs from all ports of the model given the inputs amplitudes

        Args:
            input_dic (dict): dictionary {pin_name (str) : input_amplitude (complex)}. Dictionary containing the complex amplitudes at each input port. Missing port are assumed wiht amplitude 0.0
            power (bool): If True, returned values are power transmissions. If False, complex amplitudes are instead returned. Default is True

        Returns:
            dict: Dictionary containing the outputs in the form {pin_name (str) : output (float or complex)}
        """
        if np.shape(self.S)[0] > 1: logger.warning(f'{self}:Using get_output on a sweep solve. Consider using get_full_output')
        l1=list(self.pin_dic.keys())
        l2=list(input_dic.keys())
        for pin in l2:
            l1.remove(pin)
        if l1!=[]:
            for pin in l1:
                input_dic[pin]=0.0+0.0j
        u=np.zeros(self.N,complex)
        for pin,i in self.pin_dic.items():
            u[i]=input_dic[pin]
        d=np.dot(self.S[0,:,:],u)
        out_dic={}
        for pin,i in self.pin_dic.items():
            if power:
                out_dic[pin]=np.abs(d[i])**2.0
            else:
                out_dic[pin]=d[i]
        return out_dic

    def put(self,pins=None,pint=None,param_mapping={}):
        """Function for putting a model in a Solver object, and eventually specify connections
        
        This function creates a Structure object for the model and place it in the current active Solver
        If both pins and pint are provided, the connection also is made. 

       Args:
            pin (str): pin of model to be connected
            pint (tuple): tuple (structure (Structure) , pin (str)) existing structure and pin to which to connect pins of model
            param_mapping (dict): dictionary of {oldname (str) : newname (str)} containning the mapping of the names of the parameters

        Returns:
            Str  ucture: the Structure instance created from the model
        """
        ST=solver.structure.Structure(model=deepcopy(self),param_mapping=param_mapping)
        sol_list[-1].add_structure(ST)
        if (pins is not None) and (pint is not None):
            sol_list[-1].connect(ST,pins,pint[0],pint[1])
        return ST


    def solve(self,**kargs):
        """Function for returning the solved model

        This function is to align the behavior of the Model and Solver class.

        Args: 
            kwargs: dictionary of the parameters {param_name (str) : param_value (usually float)}

        Returns:
            model: solved model of self
        """
        self.param_dic.update(self.default_params)
        ns=1
        for name in kargs:
            kargs[name]=np.reshape(kargs[name],-1)
            if len(kargs[name])==1: continue
            if ns==1:
                ns=len(kargs[name])
            else:
                if ns!=len(kargs[name]): raise Exception('Different lengths between parameter arrays')
        up_dic={}
        S_list=[]
        for i in range(ns):
            for name,values in kargs.items():
                up_dic[name]=values[0] if len(values)==1 else values[i]
            self.param_dic.update(up_dic)
            S_list.append(self.create_S())
        return SolvedModel(pin_dic=self.pin_dic,param_dic=kargs,Smatrix=np.array(S_list))

    def show_free_pins(self):
        """Funciton for printing pins of model
        """
        print(f'Pins of model {self} (id={id(self)})')
        for pin,n in self.pin_dic.items():
            print(f'{pin:5s}:{n:5}')
        print('')

    def pin_mapping(self,pin_mapping):
        """Function for changing the names of the pins of a model

        Args:
            pin_mapping (dict): Dictionary containing the mapping of the pin names. 
                Format is {'oldname' : 'newname'}

        Returns:
            model: model of self with updated pin names
        """
        for pin in copy(self.pin_dic):
            if pin in pin_mapping:
                n=self.pin_dic.pop(pin)
                self.pin_dic[pin_mapping[pin]]=n
        return self

    def update_params(self,update_dic):
        """Update the parameters of model, setting defaults when value is not provides

        Args:
            update_dic (dict) : dictionary of parameters in the from {param_name (str) : param_value (usually float)}
        """
        self.param_dic.update(self.default_params)
        self.param_dic.update(update_dic)

    def prune(self):
        """Check if the model is empty

        Returns:
            bool: True if the model is empty, False otherwise
        """
        return self.pin_dic == {}

    def get_pin_modes(self, pin):
        """Parse the pins for locating the pins with the same base name

        Assumes for the pins a name in the form pinname_modename.

        Args:
            pin (str): base name of the pin
     
        Returns:
            list: list of modenames for wich pinname==pin
        """
        li = []
        for pin in self.pin_dic:
            try:
                pinname, modename = pin.split('_')
            except ValueError:
                pinname, modename = pin, ''
            li.append(modename)
        return li

    def __str__(self):
        """Formatter function for printing
        """
        return f'Model object (id={id(self)}) with pins: {list(self.pin_dic)}'
                                       
               
class SolvedModel(Model):
    """Class for storing data of a solver mode. 

    Do not use this class directly. It is retuned from all solve methods. It is convinient for extracting data
    """
    def __init__(self, pin_dic=None, param_dic=None, Smatrix=None, int_func = None, monitor_mapping=None):
        super().__init__(pin_dic = pin_dic, param_dic = param_dic, Smatrix = Smatrix)
        self.solved_params=deepcopy(param_dic)
        self.ns=np.shape(Smatrix)[0]
        self.int_func = int_func
        self.monitor_mapping = {} if monitor_mapping is None else monitor_mapping

    def set_intermediate(self, int_func, monitor_mapping):
        """Methods for setting the function and mapping for monitors

        Args:
            int_func (callable): Function linking the the modal amplitudes at the monitor port to the inputs aplitudes
            monitor_mapping (dict): Dictionary connectinge the name of the monitor ports with the index in the aplitude arrays
        """
        self.int_func = int_func
        self.monitor_mapping = monitor_mapping
        

    def get_data(self,pin1,pin2):
        """Function for returning transmission data between two ports

        Args:
            pin1 (str): name of the input pin
            pin2 (str): name of the output pin
            
        Returns:
            pandas DataFrame: Dataframe containging the data. It contains one columns per parameter given to solve, plus the following:
                'T'         : Transmisison in absolute units
                'dB'        : Transmission in dB units 
                'Phase'     : Phase of the transmision 
                'Amplitude' : Complex amplitude of the transission
    
        """
        params={}
        if self.ns==1:
            params=deepcopy(self.solved_params)
        else:
            for name,values in self.solved_params.items():
                if len(values)==1:
                    params[name]=np.array([values[0] for i in range(self.ns)])
                elif len(values)==self.ns:
                    params[name]=values
                else:
                    raise Exception('Not able to convert to pandas')
        params['T']=np.abs(self.S[:,self.pin_dic[pin1],self.pin_dic[pin2]])**2.0
        params['dB']=20.0*np.log10(np.abs(self.S[:,self.pin_dic[pin1],self.pin_dic[pin2]]))
        params['Phase']=np.angle(self.S[:,self.pin_dic[pin1],self.pin_dic[pin2]])
        params['Amplitude']=self.S[:,self.pin_dic[pin1],self.pin_dic[pin2]]
        pan=pd.DataFrame.from_dict(params)
        return pan


    def get_full_output(self, input_dic, power=True):
        """Function for getting the output do the system given the inputs

        Args:
            input_dic (dict): Dictionary of the input amplitudes. Format is {'pinname' : amplitude (float or complex)}. Missin pins assume 0 amplitde.
            power (bool): if True, power (in absolute units) between the ports is returned, otherwise the complex amplitude is returned. Default is True

        Returns:
            pandas DataFrame: DataFrame with the outputs. It has one column for each parameter given to solve plus one columns for each pin.
        """    
        params={}
        if self.ns==1:
            params=deepcopy(self.solved_params)
        else:
            for name,values in self.solved_params.items():
                if len(values)==1:
                    params[name]=np.array([values[0] for i in range(self.ns)])
                elif len(values)==self.ns:
                    params[name]=values
                else:
                    raise Exception('Not able to convert to pandas')

        l1=list(self.pin_dic.keys())
        l2=list(input_dic.keys())
        for pin in l2:
            l1.remove(pin)
        if l1!=[]:
            for pin in l1:
                input_dic[pin]=0.0+0.0j
        u=np.zeros(self.N,complex)
        for pin,i in self.pin_dic.items():
            u[i]=input_dic[pin]

        output=np.matmul(self.S,u)

        for pin,i in self.pin_dic.items():
            params[pin]=np.abs(output[:,i])**2.0 if power else output[:,i] 
        pan=pd.DataFrame.from_dict(params)
        return pan


    def get_monitor(self, input_dic, power = True):
        """Function for returning data from monitors

        This function returs the mode coefficients if inputs and outputs of every structure selected as monitors

        Args:
            input_dic (dict): Dictionary of the input amplitudes. Format is {'pinname' : amplitude (float or complex)}. Missin pins assume 0 amplitde.

        Returns:
            pandas DataFrame: DataFrame with the amplitude at the ports. It has one column for each parameter given to solve plus two columns for monitor port.
        """    
        params={}
        if self.ns==1:
            params=deepcopy(self.solved_params)
        else:
            for name,values in self.solved_params.items():
                if len(values)==1:
                    params[name]=np.array([values[0] for i in range(self.ns)])
                elif len(values)==self.ns:
                    params[name]=values
                else:
                    raise Exception('Not able to convert to pandas')

        u,d = self.int_func(input_dic)
        for pin,i in self.monitor_mapping.items():
            params[f'{pin}_i']=np.abs(u[:,i])**2.0 if power else u[:,i] 
            params[f'{pin}_o']=np.abs(d[:,i])**2.0 if power else d[:,i] 

        pan=pd.DataFrame.from_dict(params)
        return pan


        return u,d
        
 
class Waveguide(Model):
    """Model of a simple waveguide

        Args:
            L (float) : length of the waveguide
            n (float or complex): effective index of the waveguide
            wl (float) : default wavelength of the waveguide
    """
    def __init__(self,L,n=1.0,wl=1.0):
        """Creator
        """
        self.pin_dic={'a0':0,'b0':1}        
        self.N=2
        self.S=np.identity(self.N,complex)
        self.L=L        
        self.param_dic={}
        self.param_dic['wl']=wl
        self.n=n
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S


    def _create_S(self):
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        wl=self.param_dic['wl']
        n=self.n
        self.S=np.zeros((self.N,self.N),complex)
        self.S[0,1]=np.exp(2.0j*np.pi*n/wl*self.L)
        self.S[1,0]=np.exp(2.0j*np.pi*n/wl*self.L)
        return self.S

    def __str__(self):
        """Formatter function for printing
        """
        return f'Model of waveguide of lenght {self.L:.3f} and index {self.n:.3f} (id={id(self)})'  

class UserWaveguide(Model):
    """Template for a user defined waveguide

    Args:
        L (float): length of the waveguide
        func (function): index function of the waveguide
        param_dic (dict): dictionary of the default parameters to be used (common to all modes)
        allowedmodes (dict): Dict of allowed modes and settins. Form is name:extra.
            extra is a dictionary containing the extra parameters to be passed to func
            Default is for 1 mode, with no name and no parameters
    
    """
    def __init__(self, L , func, param_dic, allowedmodes=None):
        self.allowed = {'' : {}} if allowedmodes is None else allowedmodes
        self.pin_dic = {}
        
        for i,mode in enumerate(self.allowed):
            if mode=='':
                self.pin_dic[f'a0'] = 2*i
                self.pin_dic[f'b0'] = 2*i+1
            else:
                self.pin_dic[f'a0_{mode}'] = 2*i
                self.pin_dic[f'b0_{mode}'] = 2*i+1
        
        self.N = len(self.pin_dic)
        self.S=np.identity(self.N,complex)
        self.param_dic=deepcopy(param_dic)
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S
        self.index_func=func
        self.L=L

    def _create_S(self):
        """Created the scattering Matrix
        """
        wl=self.param_dic['wl']
        S_list=[]
        for mode, extra in self.allowed.items():
            self.param_dic.update(extra)
            n=self.index_func(**self.param_dic)
            S=np.zeros((2,2),complex)
            S[0,1]=np.exp(2.0j*np.pi*n/wl*self.L)
            S[1,0]=np.exp(2.0j*np.pi*n/wl*self.L)
            S_list.append(S)
        self.S = diag_blocks(S_list)
        return self.S


class GeneralWaveguide(Model):
    """Model of dispersive waveguide

        Args:
            L (float) : length of the waveguide
            Neff (function): function returning the effecive index of the wavegude. It must be function of wl,R,w, and pol
            wl (float) : default wavelength of the waveguide
            w  (float) : default width of the waveguide
            R  (float) : default bending radius of the waveguide
            pol (int)  : default mode of the waveguide

    """
    def __init__(self,L,Neff,R=None,w=None, wl=None, pol=None):
        """Creator
        """
        self.pin_dic={'a0':0,'b0':1}        
        self.N=2
        self.Neff=Neff
        self.L=L
        self.param_dic={}
        self.param_dic['R']=R
        self.param_dic['w']=w
        self.param_dic['wl']=wl
        if pol is None:
            self.param_dic['pol']=0
        else:
            self.param_dic['pol']=pol
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S
        
    def _create_S(self):
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        wl=self.param_dic['wl']
        n=self.Neff(**self.param_dic)
        self.S=np.zeros((self.N,self.N),complex)
        self.S[0,1]=np.exp(2.0j*np.pi*n/wl*self.L)
        self.S[1,0]=np.exp(2.0j*np.pi*n/wl*self.L)
        return self.S

    def __str__(self):
        """Formatter function for printing
        """
        return f'Model of waveguide of lenght {self.L:.3} (id={id(self)})'        

class MultiModeWave(Model):
    """Model of multimode dispersive waveguide

        Args:
            L (float) : length of the waveguide
            Neff (function)     : function returning the effecive index of the wavegude. It must be function of wl,R,w, and pol
            wl (float)          : default wavelength of the waveguide
            w  (float)          : default width of the waveguide
            R  (float)          : default bending radius of the waveguide
            allowedmodes (dict) : dict of allowed modes. Structure is name:extra
                name is the name of the allowed mode, extra is a tuple of the parameters to be fittend in the Neff function 
    """
    def __init__(self,L,Neff,pol_list=[0],R=None,w=None, wl=None):
        """Creator
        """
        self.pin_dic={}
        self.pol_list=pol_list
        self.mp=len(pol_list)
        for i,pol in enumerate(pol_list):
            self.pin_dic[f'a0_pol{i}']=i        
            self.pin_dic[f'b0_pol{i}']=i+self.mp      

        self.N=2*self.mp
        self.Neff=Neff
        self.L=L
        self.param_dic={}
        self.param_dic['R']=R
        self.param_dic['w']=w
        self.param_dic['wl']=wl
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S

    def _create_S(self):
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        R=self.param_dic['R']
        w=self.param_dic['w']
        wl=self.param_dic['wl']
        mp=self.mp
        St=np.zeros((mp,mp),complex)
        for i,pol in enumerate(self.pol_list):
            n=self.Neff(wl=wl,w=w,R=R,pol=pol)
            St[i,i]=np.exp(2.0j*np.pi*n/wl*self.L)
        self.S=np.zeros((self.N,self.N),complex)
        self.S[:mp,-mp:]=St
        self.S[-mp:,:mp]=St
        return self.S
    


class BeamSplitter(Model):
    """Model of 50/50 beam splitter

        Args:
            phase (float) : phase shift of the coupled ray (in unit of pi)
    """
    def __init__(self,phase=0.5):
        """Creator
        """
        self.pin_dic={'a0':0,'a1':1,'b0':2,'b1':3}        
        self.N=4
        self.S=np.zeros((self.N,self.N),complex)
        self.phase=phase
        self.param_dic={}
        p1=np.pi*self.phase
        p2=np.pi*(1.0-self.phase)
        #self.S[:2,2:]=1.0/np.sqrt(2.0)*np.array([[1.0,np.exp(1.0j*p1)],[-np.exp(-1.0j*p1),1.0]])
        #self.S[2:,:2]=1.0/np.sqrt(2.0)*np.array([[1.0,-np.exp(1.0j*p1)],[np.exp(-1.0j*p1),1.0]])
        self.S[:2,2:]=1.0/np.sqrt(2.0)*np.array([[np.exp(1.0j*p1),1.0],[-1.0,np.exp(-1.0j*p1)]])
        self.S[2:,:2]=1.0/np.sqrt(2.0)*np.array([[np.exp(-1.0j*p1),1.0],[-1.0,np.exp(1.0j*p1)]])
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S



class GeneralBeamSplitter(Model):
    """Model of variable ration beam splitter

        Args:
            ratio (float) : Power coupling coefficient. It is also the splitting ratio if t is not provided.
            t (float): Power transmission coefficent. If None (defalut) it is calculated from the ratio assiming no loss in the component.
            phase (float) : phase shift of the coupled ray (in unit of pi). Defauls is 0.5
    """
    def __init__(self,ratio=0.5,t=None,phase=0.5):
        """Creator
        """
        self.pin_dic={'a0':0,'a1':1,'b0':2,'b1':3}        
        self.N=4
        self.ratio=ratio
        self.phase=phase
        p1=np.pi*self.phase
        c=np.sqrt(self.ratio)
        t=np.sqrt(1.0-self.ratio) if t is None else np.sqrt(t)
        self.S=np.zeros((self.N,self.N),complex)
        self.param_dic={}
        self.default_params=deepcopy(self.param_dic)
        #self.S[:2,2:]=np.array([[t,c],[c,-t]])
        #self.S[2:,:2]=np.array([[t,c],[c,-t]])
        self.S[:2,2:]=np.array([[t,c*np.exp(-1.0j*p1)],[c*np.exp(1.0j*p1),-t]])
        self.S[2:,:2]=np.array([[t,c*np.exp(-1.0j*p1)],[c*np.exp(1.0j*p1),-t]])
        self.create_S=self._create_S

    def __str__(self):
        """Formatter function for printing
        """
        return f'Model of beam-splitter with ratio {self.ratio:.3} (id={id(self)})'

    
class Splitter1x2(Model):
    """Model of 1x2 Splitter
    """
    def __init__(self):
        """Creator
        """
        self.pin_dic={'a0':0,'b0':1,'b1':2}        
        self.N=3
        self.S=1.0/np.sqrt(2.0)*np.array([[0.0,1.0,1.0],[1.0,0.0,0.0],[1.0,0.0,0.0]],complex)
        self.param_dic={}
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S


    def __str__(self):
        return f'Model of 1x2 splitter (id={id(self)})'      

class Splitter1x2Gen(Model):
    """Model of 1x2 Splitter with possible reflection between the 2 port side. TODO: verify this model makes sense

        Args:
            cross (float) : ratio of reflection (power ratio)
            phase (float) : phase shift of the reflected ray (in unit of pi)
    """
    def __init__(self,cross=0.0,phase=0.0):
        """Creator
        """
        self.pin_dic={'a0':0,'b0':1,'b1':2}        
        self.N=3
        self.param_dic={}
        self.default_params=deepcopy(self.param_dic)
        c=np.sqrt(cross)
        t=np.sqrt(0.5-cross)
        p1=np.pi*phase
        self.S=np.array([[0.0,t,t],[t,0.0,c*np.exp(1.0j*p1)],[t,c*np.exp(-1.0j*p1),0.0]],complex)
        self.create_S=self._create_S


class PhaseShifter(Model):
    """Model of multimode variable phase shifter

        Args:
            param_name (str)       : name of the parameter of the Phase Shifter
            param_default (float)  : default value of the Phase Shift in pi units
    """
    def __init__(self,param_name='PS', param_default=0.0):
        """Creator
        """
        self.param_dic={}
        self.pin_dic={'a0':0,'b0':1}
        self.N=2
        self.pn=param_name
        self.param_dic={}
        self.param_dic[param_name]=param_default    
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S


    def _create_S(self):
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        S=np.zeros((self.N,self.N),complex)
        S[0,1]=np.exp(1.0j*np.pi*self.param_dic[self.pn])
        S[1,0]=np.exp(1.0j*np.pi*self.param_dic[self.pn])
        self.S=S
        return self.S

class PushPullPhaseShifter(Model):
    """Model of multimode variable phase shifter
    """
    def __init__(self,param_name='PS'):
        """Creator

        Args:
            param_name (str) : name of the parameter of the Phase Shifter
        """
        self.param_dic={}
        self.pin_dic={'a0':0,'b0':1,'a1':2,'b1': 3}
        self.N=4
        self.pn=param_name
        self.param_dic={}
        self.param_dic[param_name]=0.0    
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S


    def _create_S(self):
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        S1=np.zeros((2,2),complex)
        S1[0,1]=np.exp(0.5j*np.pi*self.param_dic[self.pn])
        S1[1,0]=np.exp(0.5j*np.pi*self.param_dic[self.pn])
        S2=np.zeros((2,2),complex)
        S2[0,1]=np.exp(-0.5j*np.pi*self.param_dic[self.pn])
        S2[1,0]=np.exp(-0.5j*np.pi*self.param_dic[self.pn])
        self.S=diag_blocks([S1,S2])
        return self.S

    def __str__(self):
        """Formatter function for printing
        """
        return f'Model of variable Push-Pull phase shifter (id={id(self)})'  

class PolRot(Model):
    """Model of a 2 modes polarization rotator

        If angle is provided the rotation is fixed to that value. If not, the rotation is assumed variable and the angle will be fetched form the parameter dictionary.

        Args:
            angle (float) : fixed value of the rotation angle (in pi units). Default is None
            angle_name (str) : name of the angle parameter
    """
    def __init__(self,angle=None,angle_name='angle'):
        """Creator:
        """
        self.pin_dic={'a0_pol0':0, 'a0_pol1':1, 'b0_pol0':2, 'b0_pol1':3}        
        self.N=4
        self.param_dic={}
        if angle is None:
            self.fixed=False
            self.angle_name=angle_name
            self.param_dic={angle_name: 0.0}
        else:
            self.fixed=True
            c=np.cos(np.pi*angle)
            s=np.sin(np.pi*angle)
            self.S=np.zeros((self.N,self.N),complex)
            self.S[:2,2:]=np.array([[c,s],[-s,c]])
            self.S[2:,:2]=np.array([[c,-s],[s,c]])
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S


    def _create_S(self):
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        if self.fixed:
            return self.S
        else:
            angle=self.param_dic[self.angle_name]
            c=np.cos(np.pi*angle)
            s=np.sin(np.pi*angle)
            S=np.zeros((self.N,self.N),complex)
            S[:2,2:]=np.array([[c,s],[-s,c]])
            S[2:,:2]=np.array([[c,-s],[s,c]])
            return S

class Attenuator(Model):
    """Model of attenuator

        Args:
            loss: value of the loss (in dB)
    """
    def __init__(self,loss=0.0):
        """Creator
        """
        self.param_dic={}
        self.pin_dic={'a0':0,'b0':1}        
        self.N=2
        self.loss=loss
        self.S=np.zeros((self.N,self.N),complex)
        self.S[0,1]=10.0**(-0.05*loss)
        self.S[1,0]=10.0**(-0.05*loss)
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S


class Mirror(Model):
    """Model of partilly reflected Mirror

        Args:
            ref (float) : ratio of reflected power
            phase (float): phase shift of the reflected ray (in pi units)
    """
    def __init__(self,ref=0.5,phase=0.0):
        """Creator
        """
        self.pin_dic={'a0':0,'b0':1}        
        self.param_dic={}
        self.default_params=deepcopy(self.param_dic)
        self.N=2
        self.ref=ref
        self.phase=phase
        t=np.sqrt(self.ref)
        c=np.sqrt(1.0-self.ref)
        p1=np.pi*self.phase
        self.S=np.array([[t*np.exp(1.0j*p1),c],[-c,t*np.exp(-1.0j*p1)]],complex)
        self.create_S=self._create_S



class PerfectMirror(Model):
    """Model of perfect mirror (only one port), 100% reflection

    Args:
        phase (float): phase of the reflected ray (in pi unit)
    """
    def __init__(self,phase=0.0):
        """Creator
        """
        self.pin_dic={'a0':0}      
        self.param_dic={}  
        self.default_params=deepcopy(self.param_dic)
        self.N=1
        self.phase=phase
        p1=np.pi*self.phase
        self.S=np.array([[np.exp(1.0j*p1)]],complex)
        self.create_S=self._create_S


class FPR_NxM(Model):   
    """Model of Free Propagation Region. TODO: check this model makes sense

    Args:
        N (int) : number of input ports
        M (int) : number of output ports
        phi (float) : phase difference between adjacent ports
    """
    def __init__(self,N,M,phi=0.1):
        """Creator
        """
        self.param_dic={}  
        self.default_params=deepcopy(self.param_dic)
        self.pin_dic={f'a{i}':i for i in range(N)}
        self.pin_dic.update({f'b{i}': N+i for i in range(M)}) 
        Sint=np.zeros((N,M),complex)
        for i in range(N):
            for j in range(M):     
                Sint[i,j]=np.exp(-1.0j*np.pi*phi*(i-0.5*N+0.5)*(j-0.5*M+0.5))
        Sint2=np.conj(np.transpose(Sint))
        self.S=np.concatenate([np.concatenate([np.zeros((N,N),complex),Sint/np.sqrt(M)],axis=1),np.concatenate([Sint2/np.sqrt(N),np.zeros((M,M),complex)],axis=1)],axis=0)      
        self.create_S=self._create_S
  

class Ring(Model):
    """Model of ring resonator filter

    Args:
        R (float) : radius of the ring
        n (float) : effective index of the waveguide in the ring
        alpha (float) : one trip loss coefficient (remaingin complex amplitude)
        t (float) : transission of the beam splitter (complex amplitude)
    """
    def __init__(self,R,n,alpha,t):
        """Creator
        """
        self.pin_dic={'a0':0,'b0':1}        
        self.N=2
        self.S=np.identity(self.N,complex)
        self.R=R        
        self.n=n
        self.alpha=alpha
        self.t=t

        self.param_dic={}
        self.param_dic['wl']=None
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S



    def _create_S(self):
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        wl=self.param_dic['wl']
        n=self.n
        t=self.t
        ex=np.exp(-4.0j*np.pi**2.0/wl*n*self.R)
        b=(-self.alpha+t*ex)/(-self.alpha*t+ex)
        self.S=np.zeros((self.N,self.N),complex)
        self.S[0,1]=b
        self.S[1,0]=b
        return self.S

class TH_PhaseShifter(Model):
    """Model of thermal phase shifter (dispersive waveguide + phase shifter)

    Args:
        L (float) : length of the waveguide
        Neff (function): function returning the effecive index of the wavegude. It must be function of wl,R,w, and pol
        wl (float) : default wavelength of the waveguide
        w  (float) : default width of the waveguide
        R  (float) : default bending radius of the waveguide
        pol (int)  : default mode of the waveguide
        param_name (str) : name of the parameter of the Phase Shifter
    """
    def __init__(self,L,Neff,R=None,w=None, wl=None, pol=None, param_name='PS'):
        """Creator
        """
        self.pin_dic={'a0':0,'b0':1}        
        self.N=2
        self.Neff=Neff
        self.L=L
        self.pn=param_name
        self.param_dic={}
        self.param_dic['R']=R
        self.param_dic['w']=w
        self.param_dic['wl']=wl
        self.param_dic['pol']=pol
        self.param_dic[param_name]=0.0
        self.default_params=deepcopy(self.param_dic)
        self.create_S=self._create_S


    def _create_S(self):
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        wl=self.param_dic['wl']
        n=self.Neff(**self.param_dic)
        self.S=np.zeros((self.N,self.N),complex)
        self.S[0,1]=np.exp(1.0j*np.pi*(2.0*n/wl*self.L+self.param_dic[self.pn]))
        self.S[1,0]=np.exp(1.0j*np.pi*(2.0*n/wl*self.L+self.param_dic[self.pn]))
        return self.S


class AWGfromVPI(Model):
    def __init__(self,filename,fsr=1.1):
        with open(filename) as f:
            data=f.read()

            data=data.split('\n\n')[2:]

        coeff={}
        pins=[]
        for t in data:
            if 'NullTransferFunction' in t: continue
            p=t.split('\n')
            if len(p)==1: continue 
            pin_in=p[0]
            pin_out=p[1]
            if 'TM' in pin_in: continue
            if 'TM' in pin_out: continue
            pin_in=pin_in.split(' ')[2][0]+pin_in.split(' ')[2][-1]
            pin_out=pin_out.split(' ')[2][0]+pin_out.split(' ')[2][-1]
            if pin_in not in pins: pins.append(pin_in)
            if pin_out not in pins: pins.append(pin_out)
            dd=io.StringIO(t)
            ar=np.loadtxt(dd)
            LAM=ar[:,0]
            coeff[(pin_in,pin_out)]=np.sqrt(ar[:,1])*np.exp(1.0j*np.pi/180*0*ar[:,2])

        pins.sort()
        S=np.zeros((len(LAM),len(pins),len(pins)),dtype=complex)
        for i,ipin in enumerate(pins):
            for j,jpin in enumerate(pins):
                if (ipin,jpin) in coeff: S[:,i,j]=coeff[(ipin,jpin)] 


        self.pin_dic={pin : i for i,pin in enumerate(pins)}
        self.N=len(pins)
        self.param_dic={}
        self.default_params={}
        self.create_S=self._create_S

        self.S_func=interp1d(LAM,S,axis=0)

        self.fsr=(LAM[-1]-LAM[0])/1.1
        self.lamc=(LAM[-1]+LAM[0])/2.0


    def _create_S(self):
        lam=self.param_dic['wl']
        self.S=self.S_func(self.lamc-0.5*self.fsr+np.mod(lam-self.lamc+0.5*self.fsr,self.fsr))
        return self.S


class Model_from_NazcaCM(Model):
    """Class for model from a nazca cell with compact models 
    """

    def __init__(self, cell, tracker, allowed=None):
        """Creator of the class
        
        Args:
            cell (Nazca Cell): it expects a Nazca cell with some compact model defined.
            tracker (str): tracker to use to define the compact models
            allowed (dict): mapping {Mode:extra}. The allowed mode in the cell and thee extra information to pass to the compact model to build the optical length. 
        """
        self.pin_dic = {}
        self.param_dic={}
        self.default_params={}
        opt_conn = {}
        n=0
        for name, pin in cell.pin.items():       
            opt = list(pin.path_nb_iter(tracker))
            if len(opt)!=0:
                opt_conn[pin] = opt
                for mode in allowed:
                    if mode=='':
                        self.pin_dic[name] = n
                    else:
                        self.pin_dic['_'.join([name,mode])] = n
                    n+=1
        self.N = len(self.pin_dic)
        self.CM = {}
        for pin, conn in opt_conn.items():
            for stuff in conn:
                target = stuff[0]
                CM = stuff[1]
                for mode, extra in allowed.items():
                    tup = (pin.name, target.name) if mode=='' else ('_'.join([pin.name,mode]), '_'.join([target.name,mode]))
                    if callable(CM):
                        self.CM[tup] = CM(extra)
                    else:
                        self.CM[tup] = CM                
        self.create_S=self._create_S

    @classmethod
    def check_init(cls, cell, tracker, allowed=None):
        try:
            obj = cls(cell=cell, tracker=tracker, allowed=allowed)
            obj.solve(wl=1.55)
            return obj
        except AttributeError:
            return None



    def _create_S(self):
        """Creates the scattering matrix

        Returns:
            ndarray: Scattering Matrix
        """
        self.S = np.zeros((self.N,self.N), dtype='complex')
        wl = self.param_dic['wl']
        for (pin1, pin2), CM in self.CM.items():
            if callable(CM):
                self.S[self.pin_dic[pin1], self.pin_dic[pin2]] = np.exp(2.0j*np.pi/wl*CM(**self.param_dic))
            else:
                self.S[self.pin_dic[pin1], self.pin_dic[pin2]] = np.exp(2.0j*np.pi/wl*CM)

        return self.S 
                    
            
                

