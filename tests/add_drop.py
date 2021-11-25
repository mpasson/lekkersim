import numpy as np
import matplotlib.pyplot as plt

import solver as sv
 




def add_drop(R,n,l,c1=0.1,c2=None):
    """Generate the solver for a simple add drop.
    
    The port are the following:
        a0 : in-port
        a1 : drop port
        b0 : trough port
        b1 : add port
    

    Args:
        R (float): Radius of the ring.
        n (float): effective index if waveguide.
        l (float): loss of waveguide (dB/um).
        c1 (float, optional): ring-waveguide coupling 1. Defaults to 0.1.
        c2 (float, optional): ring-waveguide coupling 1. Defaults to None (c1 will be assumed).

    Returns:
        S (Solver): DESCRIPTION.

    """
    c2 = c1 if c2 is None else c2
    with sv.Solver(name='add_drop') as S:
        WG = sv.Waveguide(np.pi*R, n=n)
        AT = sv.Attenuator(loss = l*np.pi*R)
        
        bm1 = sv.GeneralBeamSplitter(ratio = c1).put()
        bm2 = sv.GeneralBeamSplitter(ratio = c2).put()
        
        wg1 = WG.put()
        at1 = AT.put('a0', wg1.pin['b0'])
        wg2 = WG.put()
        at2 = AT.put('a0', wg2.pin['b0'])
        
        sv.connect(wg1.pin['a0'], bm1.pin['b1'])
        sv.connect(at1.pin['b0'], bm2.pin['b0'])
        
        sv.connect(wg2.pin['a0'], bm2.pin['a0'])
        sv.connect(at2.pin['b0'], bm1.pin['a1'])
        
        sv.raise_pins()
    return S


ring = add_drop(100, 1.5, 1e-3, 0.1)
mod = ring.solve(wl=np.linspace(1.545, 1.555, 501))
for pin in ['a0', 'b0', 'a1', 'b1']:
    data = mod.get_data('a0', pin)
    plt.plot(data['wl'], data['T'], label = 'a0 -> {pin}')

