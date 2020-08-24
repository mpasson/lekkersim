import numpy as np
import solver
from copy import deepcopy

def wave(L0=100.0,L1=300.0):
    with solver.Solver() as wave:
        WG1=solver.Waveguide(L0,n=2.45)
        WG2=solver.Waveguide(L1,n=2.45)
        M1=solver.Mirror(ref=0.05)
        M2=solver.Mirror(ref=0.05)
        
        e1=M1.put()
        t_=WG1.put('a0',e1.pin['b0'])
        t_=M2.put('a0',t_.pin['b0'])
        e2=WG2.put('a0',t_.pin['b0'])

        solver.putpin('a0',e1.pin['a0'])
        solver.putpin('b0',e2.pin['b0'])
    return wave

with solver.Solver() as sol:
    BS=solver.Splitter1x2Gen().put()
    
    I0=wave(L0=100.0).put('a0',BS.pin['a0'])
    solver.putpin('a0',I0.pin['b0'])

    I0=wave(L0=101.0,L1=300.0).put('a0',BS.pin['b0'])
    solver.putpin('b0',I0.pin['b0'])

    I0=wave(L0=102.0).put('a0',BS.pin['b1'])
    solver.putpin('b1',I0.pin['b0'])

for lam in np.linspace(1.53,1.58,1001):
    sol.set_param('Lam',value=lam)
    D=sol.solve()
    print(4*'%15.8f' % (lam,D.get_T('a0','a0'),D.get_T('a0','b0'),D.get_T('a0','b1')))
