import numpy as np
import solver



with solver.Solver() as MZI1:
    BS1=solver.GeneralBeamSplitter().put()
    BS2=solver.GeneralBeamSplitter().put('a0',(BS1,'b0'))

    PS=solver.PhaseShifter().put('a0',(BS1,'b1'))

    solver.connect((PS,'b0'),(BS2,'a1'))

    solver.putpin('a0',(BS1,'a0'))
    solver.putpin('a1',(BS1,'a1'))
    solver.putpin('b0',(BS2,'b0'))
    solver.putpin('b1',(BS2,'b1'))

with solver.Solver() as MZI2:
    BS1=solver.Splitter1x2().put()
    BS2=solver.GeneralBeamSplitter().put('a0',(BS1,'b0'))

    PS=solver.PhaseShifter().put('a0',(BS1,'b1'))

    solver.connect((PS,'b0'),(BS2,'a1'))

    solver.putpin('a0',(BS1,'a0'))
    solver.putpin('b0',(BS2,'b0'))
    solver.putpin('b1',(BS2,'b1'))

for PS in np.linspace(0.0,1.0,201):

    MZI1.set_param('PS',value=PS)
    MZI2.set_param('PS',value=PS)
    R=MZI2.solve()
    print(3*'%15.6f' % (PS,R.get_T('a0','b0'),R.get_T('a0','b1')))
