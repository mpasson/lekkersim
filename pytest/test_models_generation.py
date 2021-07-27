#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:24:25 2021

@author: marco
"""

import numpy as np
import pytest as pt

import nazca as nd



def test_ampl_lev0():
    """Generation of compact model from amplitude with fixed numbers
    """
    with nd.Cell(name='test_ampl_lev0') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
    
        nd.connect_path(T.pin['a0'], T.pin['b0'],  0.9, 'Ampl')
        nd.connect_path(T.pin['a0'], T.pin['a0'],  0.1, 'Ampl')
        nd.connect_path(T.pin['b0'], T.pin['b0'],  0.1, 'Ampl')
    
    S = nd.get_solver(T)
    assert np.allclose(S.solve(wl=1.55).S, np.array([[[0.1+0.0j, 0.9+0.0j],[0.9+0.0j, 0.1+0.0j]]]))
    
    
    
def test_ampl_lev1():
    """Generation of compact model from amplitude with models returning numbers
    """
    with nd.Cell(name='test_ampl_lev1') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        def MOD(pol=0, mode=0):
            return 1.0-0.1*pol
        nd.connect_path(T.pin['a0'], T.pin['b0'],  MOD, 'Ampl')

    S = nd.get_solver(T, allowed={'TE':dict(pol=0, mode=0), 'TM':dict(pol=1, mode=0)})
    ref = np.array([[[0. +0.0j, 0. +0.0j, 1. +0.0j, 0. +0.0j],
                     [0. +0.0j, 0. +0.0j, 0. +0.0j, 0.9+0.0j],
                     [1. +0.0j, 0. +0.0j, 0. +0.0j, 0. +0.0j],
                     [0. +0.0j, 0.9+0.0j, 0. +0.0j, 0. +0.0j]]])
    assert np.allclose(S.solve(wl=1.5).S, ref)
    
    
def test_ampl_lev2():
    """Generation of compact model from amplitude with models returning functions
    """
    with nd.Cell(name='test_ampl_lev2') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        def MOD(pol=0, mode=0):
            def InMod(wl):
                return 1.0-0.1*pol-0.1*wl
            return InMod
        nd.connect_path(T.pin['a0'], T.pin['b0'],  MOD, 'Ampl')

    S = nd.get_solver(T, allowed={'TE':dict(pol=0, mode=0), 'TM':dict(pol=1, mode=0)})
    ref = np.array([[[0. +0.0j, 0. +0.0j, 0.9+0.0j, 0. +0.0j],
                     [0. +0.0j, 0. +0.0j, 0. +0.0j, 0.8+0.0j],
                     [0.9+0.0j, 0. +0.0j, 0. +0.0j, 0. +0.0j],
                     [0. +0.0j, 0.8+0.0j, 0. +0.0j, 0. +0.0j]]])
    assert np.allclose(S.solve(wl=1.0).S, ref)
    ref = np.array([[[0. +0.0j, 0. +0.0j, 0.8+0.0j, 0. +0.0j],
                     [0. +0.0j, 0. +0.0j, 0. +0.0j, 0.7+0.0j],
                     [0.8+0.0j, 0. +0.0j, 0. +0.0j, 0. +0.0j],
                     [0. +0.0j, 0.7+0.0j, 0. +0.0j, 0. +0.0j]]])
    assert np.allclose(S.solve(wl=2.0).S, ref)
    
    
def test_ampl_multiport():
    with nd.Cell(name='test_ampl_multiport') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        nd.Pin('a1').put(0.0, 10.0, 180.0)
        nd.Pin('b1').put(0.0, 10.0, 0.0)
        
    nd.connect_path(T.pin['a0'], T.pin['b0'],  1.0/np.sqrt(2.0), 'Ampl')
    nd.connect_path(T.pin['a0'], T.pin['b1'],  1.0j/np.sqrt(2.0), 'Ampl')
    nd.connect_path(T.pin['a1'], T.pin['b0'],  -1.0j/np.sqrt(2.0), 'Ampl')
    nd.connect_path(T.pin['a1'], T.pin['b1'],  -1.0/np.sqrt(2.0), 'Ampl')
    
    S = nd.get_solver(T)
    ref = np.array([[ 0.        +0.j        ,  0.        +0.j        ,
         0.70710678+0.j        ,  0.        +0.70710678j],
       [ 0.        +0.j        ,  0.        +0.j        ,
        -0.        -0.70710678j, -0.70710678+0.j        ],
       [ 0.70710678+0.j        , -0.        -0.70710678j,
         0.        +0.j        ,  0.        +0.j        ],
       [ 0.        +0.70710678j, -0.70710678+0.j        ,
         0.        +0.j        ,  0.        +0.j        ]])
    assert np.allclose(S.solve(wl=1.0).S2PD().values, ref)
   
def test_optloss_lev0():
    with nd.Cell(name='test_optloss_lev0') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        nd.connect_path(T.pin['a0'], T.pin['b0'],  -3.0, 'OptLoss')
        
    S = nd.get_solver(T)
    ref = np.array([[[0.        +0.j, 0.70794578+0.j],
        [0.70794578+0.j, 0.        +0.j]]])
        
    assert np.allclose(S.solve(wl=1.0).S, ref)
    
    
def test_optlen_lev0():    
    with nd.Cell(name='test_optlen_lev0') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        nd.connect_path(T.pin['a0'], T.pin['b0'],  1.55, 'OptLen')
        
    S = nd.get_solver(T)
    ref = np.array([[[ 0.        +0.j        , -0.95105652-0.30901699j],
        [-0.95105652-0.30901699j,  0.        +0.j        ]]])
    
    assert np.allclose(S.solve(wl=1.0).S, ref)

def test_both_lev0():    
    with nd.Cell(name='test_both_lev0') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        nd.connect_path(T.pin['a0'], T.pin['b0'],  1.55, 'OptLen')
        nd.connect_path(T.pin['a0'], T.pin['b0'],  -3.0, 'OptLoss')
    
    S = nd.get_solver(T)
    ref = np.array([[[ 0.        +0.j        , -0.67329645-0.21876728j],
        [-0.67329645-0.21876728j,  0.        +0.j        ]]])
    assert np.allclose(S.solve(wl=1.0).S, ref)
    

def test_optloss_lev1():
    with nd.Cell(name='test_optloss_lev1') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        nd.connect_path(T.pin['a0'], T.pin['b0'],  1.55, 'OptLen')
        def optloss(pol=0, mode=0):
            return -3.0-pol
        nd.connect_path(T.pin['a0'], T.pin['b0'],  optloss, 'OptLoss')

    S = nd.get_solver(T, allowed={'TE':dict(pol=0, mode=0), 'TM':dict(pol=1, mode=0)})
    ref = np.array([[[ 0.        +0.j        ,  0.        +0.j        ,
         -0.67329645-0.21876728j,  0.        +0.j        ],
        [ 0.        +0.j        ,  0.        +0.j        ,
          0.        +0.j        , -0.60007609-0.19497654j],
        [-0.67329645-0.21876728j,  0.        +0.j        ,
          0.        +0.j        ,  0.        +0.j        ],
        [ 0.        +0.j        , -0.60007609-0.19497654j,
          0.        +0.j        ,  0.        +0.j        ]]])
    assert np.allclose(S.solve(wl=1.0).S, ref)
    
def test_optloss_lev2():
    with nd.Cell(name='test_optloss_lev2') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        nd.connect_path(T.pin['a0'], T.pin['b0'],  1.55, 'OptLen')
        def optloss(pol=0, mode=0):
            def innerloss(wl):
                return -3.0-pol-wl
            return innerloss
        nd.connect_path(T.pin['a0'], T.pin['b0'],  optloss, 'OptLoss')

    S = nd.get_solver(T, allowed={'TE':dict(pol=0, mode=0), 'TM':dict(pol=1, mode=0)})
    
    ref = np.array([[[ 0.        +0.j        ,  0.        +0.j        ,
         -0.60007609-0.19497654j,  0.        +0.j        ],
        [ 0.        +0.j        ,  0.        +0.j        ,
          0.        +0.j        , -0.53481838-0.17377303j],
        [-0.60007609-0.19497654j,  0.        +0.j        ,
          0.        +0.j        ,  0.        +0.j        ],
        [ 0.        +0.j        , -0.53481838-0.17377303j,
          0.        +0.j        ,  0.        +0.j        ]]])
    assert np.allclose(S.solve(wl=1.0).S, ref)
    
    ref = np.array([[[0.        +0.j        , 0.        +0.j        ,
         0.08796956-0.55541797j, 0.        +0.j        ],
        [0.        +0.j        , 0.        +0.j        ,
         0.        +0.j        , 0.07840296-0.49501679j],
        [0.08796956-0.55541797j, 0.        +0.j        ,
         0.        +0.j        , 0.        +0.j        ],
        [0.        +0.j        , 0.07840296-0.49501679j,
         0.        +0.j        , 0.        +0.j        ]]])
    assert np.allclose(S.solve(wl=2.0).S, ref)
    



    
def test_optlen_lev1():
    with nd.Cell(name='test_optlen_lev1') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        def optlen(pol=0, mode=0):
            return 1.55 + 0.5*pol

        nd.connect_path(T.pin['a0'], T.pin['b0'],  optlen, 'OptLen')
        nd.connect_path(T.pin['a0'], T.pin['b0'],  -3.0, 'OptLoss')
    
    S = nd.get_solver(T, allowed={'TE':dict(pol=0, mode=0), 'TM':dict(pol=1, mode=0)})
    
    ref = np.array([[[ 0.        +0.j        ,  0.        +0.j        ,
          0.25104103+0.66194081j,  0.        +0.j        ],
        [ 0.        +0.j        ,  0.        +0.j        ,
          0.        +0.j        , -0.62685486-0.32899881j],
        [ 0.25104103+0.66194081j,  0.        +0.j        ,
          0.        +0.j        ,  0.        +0.j        ],
        [ 0.        +0.j        , -0.62685486-0.32899881j,
          0.        +0.j        ,  0.        +0.j        ]]])
    assert np.allclose(S.solve(wl=1.3).S, ref)


def test_optlen_lev2():
    with nd.Cell(name='test_optlen_lev2') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        def optlen(pol=0, mode=0):
            def len(wl):
                return 1.55 + 0.5*pol - wl
            return len

        nd.connect_path(T.pin['a0'], T.pin['b0'],  optlen, 'OptLen')
        nd.connect_path(T.pin['a0'], T.pin['b0'],  -3.0, 'OptLoss')
    
    S = nd.get_solver(T, allowed={'TE':dict(pol=0, mode=0), 'TM':dict(pol=1, mode=0)})

    ref = np.array([[[ 0.        +0.j        ,  0.        +0.j        ,
          0.25104103+0.66194081j,  0.        +0.j        ],
        [ 0.        +0.j        ,  0.        +0.j        ,
          0.        +0.j        , -0.62685486-0.32899881j],
        [ 0.25104103+0.66194081j,  0.        +0.j        ,
          0.        +0.j        ,  0.        +0.j        ],
        [ 0.        +0.j        , -0.62685486-0.32899881j,
          0.        +0.j        ,  0.        +0.j        ]]])
    assert np.allclose(S.solve(wl=1.3).S, ref)

    ref = np.array([[[ 0.        +0.j        ,  0.        +0.j        ,
          0.5534943 +0.44139698j,  0.        +0.j        ],
        [ 0.        +0.j        ,  0.        +0.j        ,
          0.        +0.j        , -0.69019611+0.15753276j],
        [ 0.5534943 +0.44139698j,  0.        +0.j        ,
          0.        +0.j        ,  0.        +0.j        ],
        [ 0.        +0.j        , -0.69019611+0.15753276j,
          0.        +0.j        ,  0.        +0.j        ]]])

    assert np.allclose(S.solve(wl=1.4).S, ref)
    
    
def test_ampl_allowed1():    
    with nd.Cell(name='test_ampl_allowed1') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        nd.connect_path(T.pin['a0'], T.pin['b0'],  1.0, 'Ampl', allowed_in = dict(pol=0, mode=0), allowed_out = dict(pol=1, mode=0))

    S=nd.get_solver(T, infolevel=2, allowed = {'TE' : dict(pol=0, mode=0), 'TM' : dict(pol=1, mode=0)})  
    ref = np.array([[[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]])
    assert np.allclose(S.solve(wl=1.0).S2PD().values, ref)
    
def test_ampl_allowed2():    
    with nd.Cell(name='test_ampl_allowed2') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        nd.connect_path(T.pin['a0'], T.pin['b0'],  0.0, 'Ampl')
        nd.connect_path(T.pin['a0'], T.pin['b0'],  1.0, 'Ampl', allowed_in = dict(pol=0, mode=0), allowed_out = dict(pol=1, mode=0))

    S=nd.get_solver(T, infolevel=2, allowed = {'TE' : dict(pol=0, mode=0), 'TM' : dict(pol=1, mode=0)})  
    ref = np.array([[[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]])
    assert np.allclose(S.solve(wl=1.0).S2PD().values, ref)
    
 
def test_loss_allowed1():    
    with nd.Cell(name='test_loss_allowed3') as T:
        nd.Pin('a0').put(0.0, 0.0, 180.0)
        nd.Pin('b0').put(0.0, 0.0, 0.0)
        
        nd.connect_path(T.pin['a0'], T.pin['b0'],  0.0, 'OptLoss')
        nd.connect_path(T.pin['a0'], T.pin['b0'],  -1e6, 'OptLoss', allowed_in = dict(pol=0, mode=0), allowed_out = dict(pol=0, mode=0))

    S=nd.get_solver(T, infolevel=2, allowed = {'TE' : dict(pol=0, mode=0), 'TM' : dict(pol=1, mode=0), 'TE1' : dict(pol=0, mode=1)})
    ref = np.array([[[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]])
    print('')
    print(repr(S.solve(wl=1.0).S2PD()))

    assert np.allclose(S.solve(wl=1.0).S2PD().values, ref)

        
    
if __name__ == "__main__":
    pt.main([__file__, '-s', '-vv']) # -s: show print output