import numpy as np
import solver
import sys 


r=0.5

BS1=solver.Structure(model=solver.Splitter1x2())
BS2=solver.Structure(model=solver.Splitter1x2())

WG1=solver.Structure(model=solver.waveguide(40.0))
WG2=solver.Structure(model=solver.waveguide(20.0))

Sol=solver.Solver(structures=[BS1,BS2,WG1,WG2])


Sol.connect(BS1,'b0',WG1,'a0')  
Sol.connect(BS1,'b1',WG2,'a0')
Sol.connect(BS2,'b0',WG1,'b0')
Sol.connect(BS2,'b1',WG2,'b0')

pin_mapping={
    'a0': (BS1,'a0'),
    'b0': (BS2,'a0'),
}

Sol.map_pins(pin_mapping)


for Lam in np.linspace(1.4,1.6,1001):
    
    Sol.set_param('Lam',value=Lam)
    full=Sol.solve()
    new=solver.Structure(model=full.get_model(pin_mapping))
    print(2*'%15.8f' % (Lam,new.get_T('a0','b0')))




