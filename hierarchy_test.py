import numpy as np
import solver

with solver.Solver() as sol:
    BS=solver.BeamSplitter(phase=0.5)
    WG=solver.waveguide(20.0)

    BS1=BS.put()
    WG1=WG.put('a0',(BS1,'b1'))
    BS2=BS.put('a0',(WG1,'b0'))
    WG2=WG.put('a0',(BS2,'b0'))

    solver.connect((WG2,'b0'),(BS1,'a1'))

    solver.putpin('a0',(BS1,'a0'))
    solver.putpin('b0',(BS1,'b0'))
    solver.putpin('b1',(BS2,'a1'))
    solver.putpin('a1',(BS2,'b1'))

#sol.show_connections()
#sol.show_free_pins()

for Lam in np.linspace(1.4,1.6,1001):
    
    sol.set_param('Lam',value=Lam)
    st=solver.Structure(solver=sol)    
    st.createS()


    model=st.return_model()

    model=sol.solve()
    print('%15.8f %15.8f %15.8f' % (Lam,model.get_T('a0','b0'),model.get_T('a0','a1')))
