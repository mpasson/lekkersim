import numpy as np
import solver

with solver.Solver() as S:
    BM=solver.GeneralBeamSplitter(phase=0.5)
    WG=solver.Waveguide(20.0,n=2.5)

    BM1=BM.put()
    WG1=WG.put('a0',BM1.pin['b0'])
        
    solver.connect(BM1.pin['b1'],WG1.pin['b0'])

    solver.putpin('a0',BM1.pin['a0'])
    solver.putpin('a1',BM1.pin['a1'])

S.set_param('Lam',value=1.55)
R=S.solve()
print(3*'%12.6f' % (1.55,R.get_T('a0','a0'),R.get_T('a0','a1')))



