import numpy as np
from scattering import S_matrix


class model:
    def __init__(self,pin_list=[],param_dic={}):
        self.pin_dic={}
        for i,pin in enumerate(pin_list):
            self.pin_dic[pin]=i
        self.N=len(pin_list)
        self.S=np.identity(self.N,complex)
        self.param_dic=param_dic

    def create_S(self):
        return self.S
        
                
class waveguide(model):
    def __init__(self,L,lam=1.0,n=1.0):
        self.pin_dic={'a0':0,'b0':1}        
        self.N=2
        self.S=np.identity(self.N,complex)
        self.L=L        
        self.param_dic={}
        self.param_dic['Lam']=lam
        self.param_dic['n']=n


    def create_S(self):
        lam=self.param_dic['Lam']
        n=self.param_dic['n']
        self.S=np.zeros((self.N,self.N),complex)
        self.S[0,1]=np.exp(2.0j*np.pi*n/lam*self.L)
        self.S[1,0]=np.exp(2.0j*np.pi*n/lam*self.L)
        return self.S

class BeamSplitter(model):
    def __init__(self):
        self.pin_dic={'a0':0,'a1':1,'b0':2,'b1':3}        
        self.N=4
        self.S=np.identity(self.N,complex)

    def create_S(self):
        self.S=np.zeros((self.N,self.N),complex)
        self.S[:2,2:]=1.0/np.sqrt(2.0)*np.array([[1.0,1.0],[1.0,-1.0]])
        self.S[2:,:2]=1.0/np.sqrt(2.0)*np.array([[1.0,1.0],[1.0,-1.0]])
        return self.S

class GeneralBeamSplitter(model):
    def __init__(self,ratio=0.5):
        self.pin_dic={'a0':0,'a1':1,'b0':2,'b1':3}        
        self.N=4
        self.S=np.identity(self.N,complex)
        self.ratio=ratio
    
    def create_S(self):
        c=np.sqrt(self.ratio)
        t=np.sqrt(1.0-self.ratio)
        self.S=np.zeros((self.N,self.N),complex)
        self.S[:2,2:]=np.array([[t,c],[c,-t]])
        self.S[2:,:2]=np.array([[t,c],[c,-t]])
        return self.S





