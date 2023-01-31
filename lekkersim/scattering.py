# -------------------------------------------
#
# GenSol Package
#
# Python tool for simulation of abstract photonic circuits
#
# @author: Marco Passoni
#
# -------------------------------------------
from __future__ import annotations
import numpy as np
import numpy.linalg as linalg

import lekkersim


class S_matrix:
    """Class implmenting the scattering matrix object and recursion alghoritms for joining two of them"""

    def __init__(self, N: int, M: int, ns: int = None) -> None:
        """Creator

        Args:
            N (int) : number of "left" ports
            M (int) : number of "right" ports
        """
        self.N = N
        self.M = M
        if ns is None:
            self.S11 = np.zeros((M, N), complex)
            self.S22 = np.zeros((N, M), complex)
            self.S12 = np.zeros((M, M), complex)
            self.S21 = np.zeros((N, N), complex)
        else:
            self.ns = ns
            self.S11 = np.zeros((ns, M, N), complex)
            self.S22 = np.zeros((ns, N, M), complex)
            self.S12 = np.zeros((ns, M, M), complex)
            self.S21 = np.zeros((ns, N, N), complex)

    # OLD RECURSION VERSION
    # def add(self,s):
    #    T1=np.matmul(linalg.inv(np.identity(self.N,complex)-np.matmul(self.S12,s.S21)),self.S11)
    #    T2=np.matmul(linalg.inv(np.identity(self.N,complex)-np.matmul(s.S21,self.S12)),s.S22)
    #    self.S11=np.matmul(s.S11,T1)
    #    self.S12=s.S12+np.matmul(np.matmul(s.S11,self.S12),T2)
    #    self.S21=self.S21+np.matmul(np.matmul(self.S22,s.S21),T1)
    #    self.S22=np.matmul(self.S22,T2)

    # NEW RECURSION VERSION
    def add(self, s: S_matrix) -> S_matrix:
        """Recursion algorith for joining two matrices

        Args:
            s (S_matrix) : target S_matrix to join to self

        Returns:
            S_matrix : joined scattering matrix
        """
        if self.M != s.N:
            raise Exception(
                "Trying to concatenate matrices with different intermediate dimension"
            )
        I = np.identity(self.M, complex)
        T1 = np.matmul(s.S11, linalg.inv(I - np.matmul(self.S12, s.S21)))
        T2 = np.matmul(self.S22, linalg.inv(I - np.matmul(s.S21, self.S12)))
        S = S_matrix(self.N, s.M)
        S.S21 = self.S21 + np.matmul(np.matmul(T2, s.S21), self.S11)
        S.S11 = np.matmul(T1, self.S11)
        S.S12 = s.S12 + np.matmul(np.matmul(T1, self.S12), s.S22)
        S.S22 = np.matmul(T2, s.S22)
        return S

    def S_print(self, i: int = None, j: int = None) -> None:
        """Print scattering matrix as numpy array

        Args:
            i,j (int) : number of ports to print. Default is None (all matrix is printed)
        """
        if i == None:
            S = np.vstack(
                [np.hstack([self.S11, self.S12]), np.hstack([self.S21, self.S22])]
            )
        else:
            j = i if j == None else j
            S = np.vstack(
                [
                    np.hstack([self.S11[i, j], self.S12[i, j]]),
                    np.hstack([self.S21[i, j], self.S22[i, j]]),
                ]
            )
        print(S)

    def det(self) -> float:
        """Calculated determinat of scattering matrix"""
        return linalg.det(
            np.vstack(
                [np.hstack([self.S11, self.S12]), np.hstack([self.S21, self.S22])]
            )
        )

    def matrix(self) -> np.ndarray:
        """Return scattering matrix as ndarray

        Returns:
            ndarray : scattering matrix
        """
        return np.vstack(
            [np.hstack([self.S11, self.S12]), np.hstack([self.S21, self.S22])]
        )

    def int_complete(self, S2: S_matrix, u: np.ndarray, d: np.ndarray):
        """Function for computing the internal modal coefficients between two structures

        Args:
            S2 (S_matrix): Second Scattering Matrix
            u (numpy array): coefficients of input of first scattering matrix
            d (numpy array): coefficients of input of second scattering matrix

        Returns:
            numpy array: coefficients of intermediate modes (first to second)
            numpy array: coefficients of intermediate modes (second to first)
        """
        ID = np.identity(self.M)
        ut = (
            np.matmul(self.S11, np.expand_dims(u, -1))
            if len(u) > 0
            else np.zeros((self.M, 1))
        )
        dt = (
            np.matmul(S2.S22, np.expand_dims(d, -1))
            if len(d) > 0
            else np.zeros((self.M, 1))
        )

        uo = linalg.solve(
            ID - np.matmul(self.S12, S2.S21), ut + np.matmul(self.S12, dt)
        )
        do = linalg.solve(ID - np.matmul(S2.S21, self.S12), dt + np.matmul(S2.S21, ut))

        return (np.squeeze(uo, -1), np.squeeze(do, -1))
