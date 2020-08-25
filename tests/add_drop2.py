import numpy as np
import solver
import sys 


r=0.1

BS1=solver.Structure(model=solver.GeneralBeamSplitter(ratio=r,phase=0.5))
BS2=solver.Structure(model=solver.GeneralBeamSplitter(ratio=r,phase=0.5))

#BS1=solver.Structure(model=solver.GeneralBeamSplitter(ratio=r))
#BS2=solver.Structure(model=solver.GeneralBeamSplitter(ratio=r))


WG1=solver.Structure(model=solver.waveguide(20.0))
WG2=solver.Structure(model=solver.waveguide(20.0))

#print(BS1.model.S)
#quit()

Sol=solver.Solver(structures=[BS1,BS2,WG1,WG2])


Sol.connect(BS1,'b1',WG1,'a0')  
Sol.connect(BS1,'a1',WG2,'a0')
Sol.connect(BS2,'b0',WG1,'b0')
Sol.connect(BS2,'a0',WG2,'b0')

pin_mapping={
    'a0': (BS1,'a0'),
    'b0': (BS1,'b0'),
    'a1': (BS2,'a1'),
    'b1': (BS2,'b1'),
}

Sol.map_pins(pin_mapping)


for Lam in np.linspace(1.4,1.6,1001):
    
    Sol.set_param('Lam',value=Lam)
    full=Sol.solve()
    model=full.get_model(pin_mapping)
    print('%15.8f %15.8f %15.8f' % (Lam,model.get_T('a0','b0'),model.get_T('a0','a1')))




