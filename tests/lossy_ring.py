import numpy as np
import solver
from copy import deepcopy

def ring(R,k=0.0):
    with solver.Solver() as sol:
        loss=1.0-np.exp(-8.0*np.pi**2.0*R*k/1.55)
        BS=solver.GeneralBeamSplitter(phase=0.5,ratio=loss)
        WG=solver.waveguide(2.0*np.pi*R,n=1.0+1.0j*k)

        BS1=BS.put()
        WG1=WG.put('a0',(BS1,'b1'))

        solver.connect((WG1,'b0'),(BS1,'a1'))

        solver.putpin('a0',(BS1,'a0'))
        solver.putpin('b0',(BS1,'b0'))

    return sol
    
R20_1=ring(20.0,k=0.0001)
R30_1=ring(24.0)


for Lam in np.linspace(1.5,1.6,1001):
    R20_1.set_param('Lam',value=Lam)
    D=R20_1.solve()
    print('%15.8f %15.8f' % (Lam,D.get_T('a0','b0')))
    #print('%15.8f %15.8f'  % (Lam,D.get_T('a0','a1'))















