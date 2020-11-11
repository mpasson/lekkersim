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


class model:
    """Class model
        It contains the model (definition of the scattering matrix) for each base photonic building block
        This is the general template for the class, no model is defined here. 
        Each model is a separate class based on the main one.

    """
    def __init__(self,pin_dic={},param_dic={},Smatrix=None):
        """Creator of the class
        Args:
            pin_list (list): list of strings containing the model's pin names 
            param_dic (dictionary): dcitionary {'param_name':param_value} containing the definition of the model's parameters.
            Smatrix (ndarray) : Fixed S_matrix of the model
        """
        self.pin_dic=pin_dic
        self.N=len(pin_dic)
        self.S=np.identity(self.N,complex) if Smatrix is None else Smatrix
        self.param_dic=param_dic
        self.default_params=deepcopy(param_dic)
        self.create_S=self._create_S

    def _expand_S(self):
        self.N=self.N//self.np
        S=self._create_S()
        self.N=self.N*self.np
        #self.S= diag_blocks(self.np*[S])
        return diag_blocks(self.np*[S])


    def _create_S(self):
        """Function for returning the scattering matrix of the model
        Returns:
            ndarray: Scattering matrix of the model
        """
        return self.S

    def print_S(self, func=None):
        """Function for nice printing of scattering matrix in agreement with pins
        """
        func = (lambda x:x) if func is None else func
        a=list(self.pin_dic.keys())
        a.sort()
        S=self.create_S()[0,:,:]
        st='            '
        for p in a:
            st+=f' {p:8} '
        st+='\n'
        for pi in a:
            st+=f' {pi:8} '
            for pj in a:
                pr=func(S[self.pin_dic[pi],self.pin_dic[pj]])
                st+=f' {pr:8.4f} '
            st+='\n'
        print(st)
           

    def get_T(self,pin1,pin2):
        """Function for returning the energy transmission between two ports
        If the two ports are the same the returned value has the meaning of relection
        Args:
            pin1 (str): Name of input pin
            pin2 (str): Name of output pin
        Returns:
            float: Energy transmission between the ports            
        """
        if np.shape(self.S)[0] > 1: warnings.warn('You are using get_T on a parametric solve. First value is returned, nut get_data should be used instead')
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
        if np.shape(self.S)[0] > 1: warnings.warn('You are using get_PH on a parametric solve. First value is returned, nut get_data should be used instead')
        return np.angle(self.S[0,self.pin_dic[pin1],self.pin_dic[pin2]])

    def get_A(self,pin1,pin2):
        """Function for returning complex amplitude of the transmission between two ports
            
        Args:
         - pin1 (str): Name of input pin
         - pin2 (str): Name of output pin
        Returns:
         - float: Complex amplitude of the transmission between the ports            
        """
        if np.shape(self.S)[0] > 1: warnings.warn('You are using get_A on a parametric solve. First value is returned, nut get_data should be used instead')
        return self.S[0,self.pin_dic[pin1],self.pin_dic[pin2]]


    def expand_pol(self,pol_list=[0]):
        """This function expands the model by adding additional modes based on pol_list.
        
        For each pin a number of pins equal the the length of pol_list will be created the name will be "{pinname}_pol{pol}" for pol in pol_list
        
        Args:
            pol_list (list) : list of integers wiht the indexing of modes to be considered. Default is [0]
        Returns:
            Model : new model with expanded polarization 
        """
        self.np=len(pol_list)
        self.pol_list=pol_list
        new_pin_dic={}
        for i,pol in enumerate(self.pol_list):
            for name,n in self.pin_dic.items():
                new_pin_dic[f'{name}_pol{pol}']=i*self.N+n
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
        l1=list(self.pin_dic.keys())
        l2=list(input_dic.keys())
        for pin in l2:
            l1.remove(pin)
        if l1!=[]:
            #print('WARNING: Not all input pin provided, assumed 0')
            pass
            for pin in l1:
                input_dic[pin]=0.0+0.0j
        u=np.zeros(self.N,complex)
        for pin,i in self.pin_dic.items():
            u[i]=input_dic[pin]
        #for pin,i in self.pin_dic.items():
        #    print(pin,i,u[i])
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
            pins (str): pin of model to be connected
            pint (tuple): tuple (structure (Structure) , pin (str)) existing structure and pin to which to connect pins of model
            param_mapping (dict): dictionary of {oldname (str) : newname (str)} containning the mapping of the names of the parameters
        Returns:
            Structure: the Structure instance created from the model
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
            Model: solved model of self
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

    def __str__(self):
        """Formatter function for printing
        """
        return f'Model object (id={id(self)}) with pins: {list(self.pin_dic)}'
                                       
               
class SolvedModel(model):
    def __init__(self,pin_dic={},param_dic={},Smatrix=None):
        self.pin_dic=pin_dic
        self.N=len(pin_dic)
        self.S= Smatrix
        self.solved_params=deepcopy(param_dic)
        self.ns=np.shape(Smatrix)[0]
        self.create_S=self._create_S


    def get_data(self,pin1,pin2):
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
            #print('WARNING: Not all input pin provided, assumed 0')
            pass
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


 
class Waveguide(model):
    """Model of a simple waveguide
    """
    def __init__(self,L,n=1.0,wl=1.0):
        """Creator
        Args:
            L (float) : length of the waveguide
            n (float or complex): effective index of the waveguide
            wl (float) : default wavelength of the waveguide
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

class UserWaveguide(model):
    """Template for a user defined waveguide
    Args:
        L (float): length of the waveguide
        func (function): index function of the waveguide
        param_dic (dict): dictionary of the default parameters to be used
        pol_list (list): list of integers representing the analyzed modes
    
    Note for nazca:
        This model is the one used in the building of the circit model for cell in naza. 
        If the user wants to create a personal funciton to be used with that, the function must have at least the following arguments as a keywork arguments:
            wl (float) : the wavelegth (in um)
            W (float) : waveguide width (in um)
            R (float) : waveguide bending radius (in um)
            pol (int) : index of the mode

    """
    def __init__(self, L , func, param_dic, pol_list = None):
        if pol_list is None:
            self.pin_dic={'a0':0,'b0':1}        
            self.N=2
        else:
            #self.pin_dic = {f'a0_pol{p}' : 2*i for i, p in enumerate(pol_list)}.update({ f'b0_pol{p}' : 2*i+1 for i, p in enumerate(pol_list)})        
            self.pin_dic = {}
            for i, p in enumerate(pol_list):
                self.pin_dic[f'a0_pol{p}'] = 2*i
                self.pin_dic[f'b0_pol{p}'] = 2*i+1
            self.N = 2*len(pol_list)
        
        self.pol_list = pol_list
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
        if self.pol_list is None:
            n=self.index_func(**self.param_dic)
            S=np.zeros((2,2),complex)
            S[0,1]=np.exp(2.0j*np.pi*n/wl*self.L)
            S[1,0]=np.exp(2.0j*np.pi*n/wl*self.L)
            S_list=[S]
        else:
            S_list=[]
            for p in self.pol_list:
                self.param_dic.update({'pol' : p})
                n=self.index_func(**self.param_dic)
                S=np.zeros((2,2),complex)
                S[0,1]=np.exp(2.0j*np.pi*n/wl*self.L)
                S[1,0]=np.exp(2.0j*np.pi*n/wl*self.L)
                S_list.append(S)
        self.S = diag_blocks(S_list)
        return self.S


class GeneralWaveguide(model):
    """Model of dispersive waveguide
    """
    def __init__(self,L,Neff,R=None,w=None, wl=None, pol=None):
        """Creator
        Args:
            L (float) : length of the waveguide
            Neff (function): function returning the effecive index of the wavegude. It must be function of wl,R,w, and pol
            wl (float) : default wavelength of the waveguide
            w  (float) : default width of the waveguide
            R  (float) : default bending radius of the waveguide
            pol (int)  : default mode of the waveguide
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

class MultiPolWave(model):
    """Model of multimode dispersive waveguide
    """
    def __init__(self,L,Neff,pol_list=[0],R=None,w=None, wl=None):
        """Creator
        Args:
            L (float) : length of the waveguide
            Neff (function) : function returning the effecive index of the wavegude. It must be function of wl,R,w, and pol
            pol_list (list)  : list of int cotaining the relevant modes of the waveguide
            wl (float) : default wavelength of the waveguide
            w  (float) : default width of the waveguide
            R  (float) : default bending radius of the waveguide
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
    


class BeamSplitter(model):
    """Model of 50/50 beam splitter
    """
    def __init__(self,phase=0.5):
        """Creator
        Args:
            phase (float) : phase shift of the coupled ray (in unit of py)
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



class GeneralBeamSplitter(model):
    """Model of variable ration beam splitter
    """
    def __init__(self,ratio=0.5,t=None,phase=0.5):
        """Creator
        Args:
            ratio (float) : splitting ratio of beam-splitter (ratio of the coupled power)
            phase (float) : phase shift of the coupled ray (in unit of py)
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
        #self.S[:2,2:]=np.array([[t,c*np.exp(1.0j*p1)],[-c*np.exp(-1.0j*p1),t]])
        #self.S[2:,:2]=np.array([[t,-c*np.exp(1.0j*p1)],[c*np.exp(-1.0j*p1),t]])
        self.S[:2,2:]=np.array([[t*np.exp(1.0j*p1),c],[c,-t*np.exp(-1.0j*p1)]])
        self.S[2:,:2]=np.array([[t*np.exp(-1.0j*p1),c],[c,-t*np.exp(1.0j*p1)]])
        self.create_S=self._create_S

    def __str__(self):
        """Formatter function for printing
        """
        return f'Model of beam-splitter with ratio {self.ratio:.3} (id={id(self)})'

    
class Splitter1x2(model):
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

class Splitter1x2Gen(model):
    """Model of 1x2 Splitter with possible reflection between the 2 port side. TODO: verify this model makes sense
    """
    def __init__(self,cross=0.0,phase=0.0):
        """Creator
        Args:
            cross (float) : ratio of reflection (power ratio)
            phase (float) : phase shift of the reflected ray (in unit of py)
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


class PhaseShifter(model):
    """Model of multimode variable phase shifter
    """
    def __init__(self,param_name='PS', param_default=0.0):
        """Creator
        Args:
            param_name (str)       : name of the parameter of the Phase Shifter
            param_default (float)  : default value of the Phase Shift in pi units
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

class PushPullPhaseShifter(model):
    """Model of multimode variable phase shifter
    """
    def __init__(self,param_name='PS'):
        """Creator
        Args:
            param_name (str) : name of the parameter of the Phase Shifter
            pol_list (list)  : list of int cotaining the relevant modes
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

class PolRot(model):
    """Model of a 2 modes polarization rotator
    """
    def __init__(self,angle=None,angle_name='angle'):
        """Creator:
        If angle is provided the rotation is fixed to that value. If not, the rotation is assumed variable and the angle will be fetched form the parameter dictionary.
        Args:
            angle (float) : fixed value of the rotation angle (in pi units). Default is None
            angle_name (str) : name of the angle parameter
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

class Attenuator(model):
    """Model of attenuator
    """
    def __init__(self,loss=0.0):
        """Creator
        Args:
            loss: value of the loss (in dB)
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


class Mirror(model):
    """Model of partilly reflected Mirror
    """
    def __init__(self,ref=0.5,phase=0.0):
        """Creator
        Args:
            ref (float) : ratio of reflected power
            phase (float): phase shift of the reflected ray (in pi units)
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



class PerfectMirror(model):
    """Model of perfect mirror (only one port), 100% reflection
    """
    def __init__(self,phase=0.0):
        """Creator
        Args:
            phase (float): phase of the reflected ray (in pi unit)
        """
        self.pin_dic={'a0':0}      
        self.param_dic={}  
        self.default_params=deepcopy(self.param_dic)
        self.N=1
        self.phase=phase
        p1=np.pi*self.phase
        self.S=np.array([[np.exp(1.0j*p1)]],complex)
        self.create_S=self._create_S


class FPR_NxM(model):   
    """Model of Free Propagation Region. TODO: check this model makes sense
    """
    def __init__(self,N,M,phi=0.1):
        """Creator
        Args:
            N (int) : number of input ports
            M (int) : number of output ports
            phi (float) : phase difference between adjacent ports
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
  

class Ring(model):
    """Model of ring resonator filter
    """
    def __init__(self,R,n,alpha,t):
        """Creator
        Args:
            R (float) : radius of the ring
            n (float) : effective index of the waveguide in the ring
            alpha (float) : one trip loss coefficient (remaingin complex amplitude)
            t (float) : transission of the beam splitter (complex amplitude)
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

class TH_PhaseShifter(model):
    """Model of thermal phase shifter (dispersive waveguide + phase shifter)
    """
    def __init__(self,L,Neff,R=None,w=None, wl=None, pol=None, param_name='PS'):
        """Creator
        Args:
            L (float) : length of the waveguide
            Neff (function): function returning the effecive index of the wavegude. It must be function of wl,R,w, and pol
            wl (float) : default wavelength of the waveguide
            w  (float) : default width of the waveguide
            R  (float) : default bending radius of the waveguide
            pol (int)  : default mode of the waveguide
            param_name (str) : name of the parameter of the Phase Shifter
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


class AWGfromVPI(model):
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
            #print(pin_in,pin_out)
            coeff[(pin_in,pin_out)]=np.sqrt(ar[:,1])*np.exp(1.0j*np.pi/180*0*ar[:,2])

        pins.sort()
        #print(pins)
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




