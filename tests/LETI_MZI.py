import numpy as np
import solver



with solver.Solver() as MZI1:
    BS1=solver.GeneralBeamSplitter().put()

    AT1=solver.Attenuator(loss=7.0).put('a0',(BS1,'b0'))
    AT2=solver.Attenuator(loss=7.0).put('a0',(BS1,'b1'))

    BS2=solver.GeneralBeamSplitter().put('a0',(AT1,'b0'))


    PS=solver.PhaseShifter().put('a0',(AT2,'b0'))
    

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

for V in np.linspace(0.0,8.0,201):

    MZI1.set_param('PS',value=V*2.0/16.0)
    MZI2.set_param('PS',value=V*2.0/16.0)
    R=MZI1.solve()
    print(3*'%15.6f' % (V,R.get_T('a0','b0'),R.get_T('a0','b1')))
