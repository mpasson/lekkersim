import numpy as np
import solver
from copy import deepcopy

def ring(R):
    with solver.Solver() as sol:
        #print('inside with:',solver.sol_list)
        #sol.show_structures()
        #sol.show_connections()
        BS=solver.GeneralBeamSplitter(phase=0.5,ratio=0.1)
        WG=solver.waveguide(np.pi*R)

        BS1=BS.put()
        WG1=WG.put('a0',(BS1,'b1'))
        BS2=BS.put('a0',(WG1,'b0'))
        WG2=WG.put('a0',(BS2,'b0'))

        solver.connect((WG2,'b0'),(BS1,'a1'))

        solver.putpin('a0',(BS1,'a0'))
        solver.putpin('b0',(BS1,'b0'))
        solver.putpin('b1',(BS2,'a1'))
        solver.putpin('a1',(BS2,'b1'))
        #sol.show_structures()
        #sol.show_connections()

    return sol
    

#BS=solver.BeamSplitter(phase=0.5)
#BS1=BS.put()

#sol.show_connections()
#sol.show_free_pins()

R20_1=ring(20.0)
R30_1=ring(24.0)

#R20_1.show_free_pins()
#print(R20_1.pin_mapping)


with solver.Solver() as DOUBLE:
    R1=R20_1.put()
    R2=R30_1.put('b0',(R1,'a1'))

    solver.putpin('a0',(R1,'a0'))
    solver.putpin('a1',(R2,'a0'))
    solver.putpin('a2',(R2,'a1'))
    solver.putpin('b0',(R1,'b0'))
    solver.putpin('b1',(R1,'b1'))
    solver.putpin('b2',(R2,'b1'))




for Lam in np.linspace(1.5,1.6,2001):
    
    R20_1.set_param('Lam',value=Lam)
    R30_1.set_param('Lam',value=Lam)
    DOUBLE.set_param('Lam',value=Lam)


    R20=R20_1.solve()
    R30=R30_1.solve()
    D=DOUBLE.solve()


    print('%15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f' % (Lam,R20.get_T('a0','b0'),R20.get_T('a0','a1'),R30.get_T('a0','b0'),R30.get_T('a0','a1'),D.get_T('a0','b0'),D.get_T('a0','b2')))
