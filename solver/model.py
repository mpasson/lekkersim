import numpy as np
from solver.scattering import S_matrix
import solver.structure
from solver import sol_list
from copy import deepcopy


def diag_blocks(array_list):
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
    def __init__(self,pin_list=[],param_dic={}):
        self.pin_dic={}
        for i,pin in enumerate(pin_list):
            self.pin_dic[pin]=i
        self.N=len(pin_list)
        self.S=np.identity(self.N,complex)
        self.param_dic=param_dic

    def create_S(self):
        return self.S

    def get_T(self,pin1,pin2):
        return np.abs(self.S[self.pin_dic[pin1],self.pin_dic[pin2]])**2.0

    def get_PH(self,pin1,pin2):
        return np.angle(self.S[self.pin_dic[pin1],self.pin_dic[pin2]])

    def get_A(self,pin1,pin2):
        return self.S[self.pin_dic[pin1],self.pin_dic[pin2]]


    def get_output(self,input_dic,power=True):
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
        d=np.dot(self.S,u)
        out_dic={}
        for pin,i in self.pin_dic.items():
            if power:
                out_dic[pin]=np.abs(d[i])**2.0
            else:
                out_dic[pin]=d[i]
        return out_dic

    def put(self,pins=None,pint=None,param_mapping={}):
        ST=solver.structure.Structure(model=deepcopy(self),param_mapping=param_mapping)
        sol_list[-1].add_structure(ST)
        if (pins is not None) and (pint is not None):
            sol_list[-1].connect(ST,pins,pint[0],pint[1])
        return ST

    def __str__(self):
        return f'Model object (id={id(self)}) with pins: {list(self.pin_dic)}'
                                       
                
class Waveguide(model):
    def __init__(self,L,wl=1.0,n=1.0):
        self.pin_dic={'a0':0,'b0':1}        
        self.N=2
        self.S=np.identity(self.N,complex)
        self.L=L        
        self.param_dic={}
        self.param_dic['wl']=wl
        self.n=n


    def create_S(self):
        wl=self.param_dic['wl']
        n=self.n
        self.S=np.zeros((self.N,self.N),complex)
        self.S[0,1]=np.exp(2.0j*np.pi*n/wl*self.L)
        self.S[1,0]=np.exp(2.0j*np.pi*n/wl*self.L)
        return self.S

    def __str__(self):
        return f'Model of waveguide of lenght {self.L:.3f} and index {self.n:.3f} (id={id(self)})'  

class GeneralWaveguide(model):
    def __init__(self,L,Neff,R=None,w=None, wl=None, pol=None):
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
        
    def create_S(self):
        wl=self.param_dic['wl']
        n=self.Neff(**self.param_dic)
        self.S=np.zeros((self.N,self.N),complex)
        self.S[0,1]=np.exp(2.0j*np.pi*n/wl*self.L)
        self.S[1,0]=np.exp(2.0j*np.pi*n/wl*self.L)
        return self.S

    def __str__(self):
        return f'Model of waveguide of lenght {self.L:.3} (id={id(self)})'        

class MultiPolWave(model):
    def __init__(self,L,Neff,pol_list=[0],R=None,w=None, wl=None):
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

    def create_S(self):
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
    def __init__(self,phase=0.5):
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



class GeneralBeamSplitter(model):
    def __init__(self,ratio=0.5,phase=0.5):
        self.pin_dic={'a0':0,'a1':1,'b0':2,'b1':3}        
        self.N=4
        self.ratio=ratio
        self.phase=phase
        p1=np.pi*self.phase
        c=np.sqrt(self.ratio)
        t=np.sqrt(1.0-self.ratio)
        self.S=np.zeros((self.N,self.N),complex)
        self.param_dic={}
        #self.S[:2,2:]=np.array([[t,c],[c,-t]])
        #self.S[2:,:2]=np.array([[t,c],[c,-t]])
        #self.S[:2,2:]=np.array([[t,c*np.exp(1.0j*p1)],[-c*np.exp(-1.0j*p1),t]])
        #self.S[2:,:2]=np.array([[t,-c*np.exp(1.0j*p1)],[c*np.exp(-1.0j*p1),t]])
        self.S[:2,2:]=np.array([[t*np.exp(1.0j*p1),c],[-c,t*np.exp(-1.0j*p1)]])
        self.S[2:,:2]=np.array([[t*np.exp(-1.0j*p1),c],[-c,t*np.exp(1.0j*p1)]])

    def __str__(self):
        return f'Model of beam-splitter with ratio {self.ratio:.3} (id={id(self)})'


class GeneralBeamSplitterMultiPol(model):
    def __init__(self,pol_list=[0],ratio=0.5,phase=0.5):
        self.n_pol=len(pol_list)
        self.pol_list=pol_list
        #self.pin_dic={'a0':0,'a1':1,'b0':2,'b1':3}
        self.pin_dic={}
        for i,pol in enumerate(pol_list):
            self.pin_dic[f'a0_pol{pol}']=4*i
            self.pin_dic[f'a1_pol{pol}']=4*i+1
            self.pin_dic[f'b0_pol{pol}']=4*i+2
            self.pin_dic[f'b1_pol{pol}']=4*i+3

        self.N=4*self.n_pol
        self.ratio=ratio
        self.phase=phase
        self.param_dic={}
        p1=np.pi*self.phase
        c=np.sqrt(self.ratio)
        t=np.sqrt(1.0-self.ratio)
        S=np.zeros((4,4),complex)
        S[:2,2:]=np.array([[t*np.exp(1.0j*p1),c],[-c,t*np.exp(-1.0j*p1)]])
        S[2:,:2]=np.array([[t*np.exp(-1.0j*p1),c],[-c,t*np.exp(1.0j*p1)]])
        self.S=diag_blocks(self.n_pol*[S])

    def __str__(self):
        return f'Model of beam-splitter ratio {self.ratio:.3}, pol_list={self.pol_list} (id={id(self)})'
    
class Splitter1x2(model):
    def __init__(self):
        self.pin_dic={'a0':0,'b0':1,'b1':2}        
        self.N=3
        self.S=1.0/np.sqrt(2.0)*np.array([[0.0,1.0,1.0],[1.0,0.0,0.0],[1.0,0.0,0.0]],complex)
        self.param_dic={}

    def __str__(self):
        return f'Model of 1x2 splitter (id={id(self)})'      

class Splitter1x2Gen(model):
    def __init__(self,cross=0.0,phase=0.0):
        self.pin_dic={'a0':0,'b0':1,'b1':2}        
        self.N=3
        self.param_dic={}
        c=np.sqrt(cross)
        t=np.sqrt(0.5-cross)
        p1=np.pi*phase
        self.S=np.array([[0.0,t,t],[t,0.0,c*np.exp(1.0j*p1)],[t,c*np.exp(-1.0j*p1),0.0]],complex)

class PhaseShifter(model):
    def __init__(self,param_name='PS',pol_list=None):
        self.param_dic={}
        self.pol_list=pol_list
        if pol_list is None:
            self.pin_dic={'a0':0,'b0':1}
        else:
            self.pin_dic={}
            for i,pol in enumerate(pol_list):
                self.pin_dic[f'a0_pol{pol}']=2*i
                self.pin_dic[f'b0_pol{pol}']=2*i+1

        self.N=2
        self.pn=param_name
        self.param_dic={}
        self.param_dic[param_name]=0.0    


    def create_S(self):
        S=np.zeros((self.N,self.N),complex)
        S[0,1]=np.exp(1.0j*np.pi*self.param_dic[self.pn])
        S[1,0]=np.exp(1.0j*np.pi*self.param_dic[self.pn])
        if self.pol_list is not None:
            self.S=diag_blocks(len(self.pol_list)*[S])
        else:
            self.S=S
        return self.S

    def __str__(self):
        return f'Model of variable phase shifter (id={id(self)})'  

class PolRot(model):
    def __init__(self,angle=None,angle_name='angle'):
        self.pin_dic={'a0_TE':0, 'a0_TM':1, 'b0_TE':2, 'b0_TM':3}        
        self.N=4
        self.param_dic={}
        if angle is None:
            self.fixed=False
            self.angle_name=angle_name
            self.param_dic={angle_name: 0.0}
        else:
            self.fixed=True
            c=np.cos(np.pi*angle/180.0)
            s=np.sin(np.pi*angle/180.0)
            self.S=np.zeros((self.N,self.N),complex)
            self.S[:2,2:]=np.array([[c,s],[-s,c]])
            self.S[2:,:2]=np.array([[c,-s],[s,c]])

    def create_S(self):
        if self.fixed:
            return self.S
        else:
            angle=self.param_dic[self.angle_name]
            c=np.cos(np.pi*angle/180.0)
            s=np.sin(np.pi*angle/180.0)
            S=np.zeros((self.N,self.N),complex)
            S[:2,2:]=np.array([[c,s],[-s,c]])
            S[2:,:2]=np.array([[c,-s],[s,c]])
            return S

class Attenuator(model):
    def __init__(self,loss=0.0):
        self.param_dic={}
        self.pin_dic={'a0':0,'b0':1}        
        self.N=2
        self.loss=loss
        self.S=np.zeros((self.N,self.N),complex)
        self.S[0,1]=10.0**(-0.1*loss)
        self.S[1,0]=10.0**(-0.1*loss)

class Mirror(model):
    def __init__(self,ref=0.5,phase=0.0):
        self.pin_dic={'a0':0,'b0':1}        
        self.param_dic={}
        self.N=2
        self.ref=ref
        self.phase=phase
        t=np.sqrt(self.ref)
        c=np.sqrt(1.0-self.ref)
        p1=np.pi*self.phase
        self.S=np.array([[t*np.exp(1.0j*p1),c],[-c,t*np.exp(-1.0j*p1)]],complex)


class PerfectMirror(model):
    def __init__(self,phase=0.0):
        self.pin_dic={'a0':0}      
        self.param_dic={}  
        self.N=1
        self.phase=phase
        p1=np.pi*self.phase
        self.S=np.array([[np.exp(1.0j*p1)]],complex)

class FPR_NxM(model):
    def __init__(self,N,M,phi=0.1):
        self.param_dic={}  
        self.pin_dic={f'a{i}':i for i in range(N)}
        self.pin_dic.update({f'b{i}': N+i for i in range(M)}) 
        Sint=np.zeros((N,M),complex)
        for i in range(N):
            for j in range(M):     
                Sint[i,j]=np.exp(-1.0j*np.pi*phi*(i-0.5*N+0.5)*(j-0.5*M+0.5))
        Sint2=np.conj(np.transpose(Sint))
        self.S=np.concatenate([np.concatenate([np.zeros((N,N),complex),Sint/np.sqrt(M)],axis=1),np.concatenate([Sint2/np.sqrt(N),np.zeros((M,M),complex)],axis=1)],axis=0)        

class Ring(model):
    def __init__(self,R,n,alpha,t):
        self.pin_dic={'a0':0,'b0':1}        
        self.N=2
        self.S=np.identity(self.N,complex)
        self.R=R        
        self.n=n
        self.alpha=alpha
        self.t=t

        self.param_dic={}
        self.param_dic['wl']=None


    def create_S(self):
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
    def __init__(self,L,Neff,R=None,w=None, wl=None, pol=None, param_name='PS'):
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

    def create_S(self):
        wl=self.param_dic['wl']
        n=self.Neff(**self.param_dic)
        self.S=np.zeros((self.N,self.N),complex)
        self.S[0,1]=np.exp(1.0j*np.pi*(2.0*n/wl*self.L+self.param_dic[self.pn]))
        self.S[1,0]=np.exp(1.0j*np.pi*(2.0*n/wl*self.L+self.param_dic[self.pn]))
        return self.S

