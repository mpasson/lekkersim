#-------------------------------------------
#
# GenSol Package
#
# Python tool for simulation of abstract photonic circuits
#
# @author: Marco Passoni
#
#-------------------------------------------

import numpy as np
import numpy.linalg as linalg

class S_matrix:
    """Class implmenting the scattering matrix object and recursion alghoritms for joining two of them
    """
    def __init__(self,N,M,ns=None):
        """Creator
        Args:
            N (int) : number of "left" ports
            M (int) : number of "right" ports
        """
        self.N=N
        self.M=M
        if ns is None:
            self.S11=np.zeros((M,N),complex)
            self.S22=np.zeros((N,M),complex)
            self.S12=np.zeros((M,M),complex)
            self.S21=np.zeros((N,N),complex)
        else:
            self.ns=ns
            self.S11=np.zeros((ns,M,N),complex)
            self.S22=np.zeros((ns,N,M),complex)
            self.S12=np.zeros((ns,M,M),complex)
            self.S21=np.zeros((ns,N,N),complex)

    #OLD RECURSION VERSION
    #def add(self,s):
    #    T1=np.matmul(linalg.inv(np.identity(self.N,complex)-np.matmul(self.S12,s.S21)),self.S11)
    #    T2=np.matmul(linalg.inv(np.identity(self.N,complex)-np.matmul(s.S21,self.S12)),s.S22)
    #    self.S11=np.matmul(s.S11,T1)
    #    self.S12=s.S12+np.matmul(np.matmul(s.S11,self.S12),T2)
    #    self.S21=self.S21+np.matmul(np.matmul(self.S22,s.S21),T1)
    #    self.S22=np.matmul(self.S22,T2)

    #NEW RECURSION VERSION
    def add(self,s):
        """Recursion algorith for joining two matrices
        Args:
            s (S_matrix) : target S_matrix to join to self
        Returns:
            S_matrix : joined scattering matrix
        """
        if self.M!=s.N:
            raise Exception('Trying to concatenate matrices with different intermediate dimension')
        I=np.identity(self.M,complex)
        T1=np.matmul(s.S11,linalg.inv(I-np.matmul(self.S12,s.S21)))
        T2=np.matmul(self.S22,linalg.inv(I-np.matmul(s.S21,self.S12)))
        S=S_matrix(self.N,s.M)
        S.S21=self.S21+np.matmul(np.matmul(T2,s.S21),self.S11)
        S.S11=np.matmul(T1,self.S11)
        S.S12=s.S12   +np.matmul(np.matmul(T1,self.S12),s.S22)             
        S.S22=np.matmul(T2,s.S22)
        return S
  



    def S_print(self,i=None,j=None):
        """Print scattering matrix as numpy array
        Args:
            i,j (int) : number of ports to print. Default is None (all matrix is printed)
        """
        if i==None:
            S=np.vstack([np.hstack([self.S11,self.S12]),np.hstack([self.S21,self.S22])])
        else:
            j=i if j==None else j
            S=np.vstack([np.hstack([self.S11[i,j],self.S12[i,j]]),np.hstack([self.S21[i,j],self.S22[i,j]])])
        print(S)

    def det(self):
        """Calculated determinat of scattering matrix
        """
        return linalg.det(np.vstack([np.hstack([self.S11,self.S12]),np.hstack([self.S21,self.S22])]))

#    def S_modes(self):
#        ID=np.identity(self.N)
#        Z=np.zeros((self.N,self.N))
#        S1=np.vstack([np.hstack([self.S11,Z]),np.hstack([self.S21,-ID])])
#        S2=np.vstack([np.hstack([ID,-self.S12]),np.hstack([Z,-self.S22])])
#        [W,V]=linalg.eig(S1,b=S2)
#        return [W,V]

#    def det_modes(self,kz,d):
#        ID=np.identity(self.N)
#        Z=np.zeros((self.N,self.N))
#        S1=np.vstack([np.hstack([self.S11,Z]),np.hstack([self.S21,-ID])])
#        S2=np.vstack([np.hstack([ID,-self.S12]),np.hstack([Z,-self.S22])])
#        return linalg.det(S1-np.exp((0.0+1.0j)*kz*d)*S2)        

#    def der(self,Sm,Sp,h=0.01):
#        S=np.vstack([np.hstack([self.S11,self.S12]),np.hstack([self.S21,self.S22])])
#        S_m=np.vstack([np.hstack([Sm.S11,Sm.S12]),np.hstack([Sm.S21,Sm.S22])])
#        S_p=np.vstack([np.hstack([Sp.S11,Sp.S12]),np.hstack([Sp.S21,Sp.S22])])
#        S1=(S_p-S_m)/(2.0*h)
#        S2=(S_p+S_m-2.0*S)/(h*h)
#        return (S1,S2)

    def matrix(self):
        """Return scattering matrix as ndarray
        Returns:
            ndarray : scattering matrix
        """
        return np.vstack([np.hstack([self.S11,self.S12]),np.hstack([self.S21,self.S22])])

#    def output(self,u1,d2):
#        u2=np.add(np.matmul(self.S11,u1),np.matmul(self.S12,d2))
#        d1=np.add(np.matmul(self.S21,u1),np.matmul(self.S22,d2))
#        return (u2,d1)

#    def left(self,u1,d1):
#        d2=linalg.solve(self.S22,d1-np.matmul(self.S21,u1))
#        u2=np.add(np.matmul(self.S11,u1),np.matmul(self.S21,d2))
#        return (u2,d2)

#    def int_f(self,S2,u):
#        ID=np.identity(self.N)
#        ut=np.matmul(self.S11,u)
#        uo=linalg.solve(ID-np.matmul(self.S12,S2.S21),ut)
#        do=linalg.solve(ID-np.matmul(S2.S21,self.S12),np.matmul(S2.S21,ut))
#        return (uo,do)

#    def int_f_tot(self,S2,u,d):
#        ID=np.identity(self.N)
#        ut=np.matmul(self.S11,u)
#        dt=np.matmul(S2.S22,d)
#        uo=linalg.solve(ID-np.matmul(self.S12,S2.S21),np.add(ut,np.matmul(self.S12,dt)))
#        do=linalg.solve(ID-np.matmul(S2.S21,self.S12),np.add(np.matmul(S2.S21,ut),dt))
#        return (uo,do)


#    def int_complete(self,S2,u,d):
#        ID=np.identity(self.N)
#        ut=np.matmul(self.S11,u)
#        dt=np.matmul(S2.S22,d)
#        uo=linalg.solve(ID-np.matmul(self.S12,S2.S21),ut+np.matmul(self.S12,dt))
#        do=linalg.solve(ID-np.matmul(S2.S21,self.S12),dt+np.matmul(S2.S21,ut))
#        return (uo,do)

            




