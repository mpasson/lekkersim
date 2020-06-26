import numpy as np
import solver

with solver.Solver() as FP:
    MIR=solver.Mirror(ref=0.2)
    WG=solver.Waveguide(100.0,n=2.45)
    
    mir1=MIR.put()
    wg1=WG.put('a0',mir1.pin['b0'])
    mir2=MIR.put('a0',wg1.pin['b0'])

    solver.putpin('a0',mir1.pin['a0'])
    solver.putpin('b0',mir2.pin['b0'])


with solver.Solver() as MMI_cav:
    BM=solver.GeneralBeamSplitter()
    WG=solver.Waveguide(10.0,n=2.45)
    MIR=solver.PerfectMirror()

    bm=BM.put()
    w=WG.put('a0',bm.pin['b1'])
    MIR.put('a0',w.pin['b0'])

    w=WG.put('a0',bm.pin['a1'])
    MIR.put('a0',w.pin['b0'])

    solver.putpin('a0',bm.pin['a0'])
    solver.putpin('b0',bm.pin['b0'])


with solver.Solver() as FP2:
    MIR1=solver.Mirror(ref=0.01)
    MIR2=solver.Mirror(ref=0.01)
    WG=solver.Waveguide(20.0,n=2.45)
    SP=solver.Splitter1x2Gen(cross=0.1,phase=0.5)

    sp=SP.put()

    w=WG.put('a0',(sp,'a0'))
    m=MIR1.put('a0',(w,'b0'))
    solver.putpin('a0',(m,'b0'))

    w=WG.put('a0',(sp,'b0'))
    m=MIR1.put('a0',(w,'b0'))
    solver.putpin('b0',(m,'b0'))

    w=WG.put('a0',(sp,'b1'))
    m=MIR2.put('a0',(w,'b0'))
    solver.putpin('b1',(m,'b0'))


for Lam in np.linspace(1.52,1.59,1401):
    
    MMI_cav.set_param('Lam',value=Lam)
    D=MMI_cav.solve()
    #print(4*'%10.6f' % (Lam,D.get_T('a0','a0'),D.get_T('a0','b0'),D.get_T('a0','b1')))
    print(3*'%10.6f' % (Lam,D.get_T('a0','a0'),D.get_T('a0','b0')))
