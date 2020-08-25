import numpy as np
import solver
from copy import deepcopy

def ring(R):
    with solver.Solver() as sol:
        BS=solver.GeneralBeamSplitter(phase=0.5,ratio=0.1)
        WG=solver.waveguide(np.pi*R,n=1.0+0.001j)

        BS1=BS.put()
        WG1=WG.put('a0',(BS1,'b1'))
        BS2=BS.put('a0',(WG1,'b0'))
        WG2=WG.put('a0',(BS2,'b0'))

        solver.connect((WG2,'b0'),(BS1,'a1'))

        solver.putpin('a0',(BS1,'a0'))
        solver.putpin('b0',(BS1,'b0'))
        solver.putpin('b1',(BS2,'a1'))
        solver.putpin('a1',(BS2,'b1'))

    return sol
    
R20_1=ring(20.0)
R30_1=ring(24.0)
WG=solver.waveguide(100.0)


with solver.Solver() as DOUBLE:
    R=R20_1.put()
    W=WG.put('a0',(R,'b0'))
    solver.connect((W,'b0'),(R,'b1'))
    solver.putpin('a0',(R,'a0'))
    solver.putpin('a1',(R,'a1'))


for Lam in np.linspace(1.5,1.6,1001):
    #R20_1.set_param('Lam',value=Lam)
    DOUBLE.set_param('Lam',value=Lam)
    D=DOUBLE.solve()
    print('%15.8f %15.8f' % (Lam,D.get_T('a0','a1')))
    #print('%15.8f %15.8f'  % (Lam,D.get_T('a0','a1'))















