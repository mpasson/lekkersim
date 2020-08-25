import numpy as np
import solver


with solver.Solver() as sol:
    BS=solver.BeamSplitter(phase=0.5)
    WG=solver.waveguide(0.0)

    BS1=BS.put()
    BS2=BS.put('a0',(BS1,'b0'))

    solver.connect((BS2,'a1'),(BS1,'b1'))

    solver.putpin('a0',(BS1,'a0'))
    solver.putpin('a1',(BS1,'a1'))
    solver.putpin('b0',(BS2,'b0'))
    solver.putpin('b1',(BS2,'b1'))

sol.show_connections()

new=sol.solve()
print(4*'%15.8f' % (new.get_T('a0','b0'),new.get_T('a1','b1'),new.get_T('a0','b1'),new.get_T('a1','b0')))

