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

    a1 ------------- b1
            __
           /  \
          |    |
           \__/
    a0 ------------- b0

    

    Args:
        R (float): Radius of the ring.
        n (float): effective index if waveguide.
        l (float): loss of waveguide (dB/um).
        c1 (float, optional): ring-waveguide coupling 1. Defaults to 0.1.
        c2 (float, optional): ring-waveguide coupling 1. Defaults to None (c1 will be assumed).

    Returns:
        S (Solver)

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


def coupled_add_drop(R1, n, l, R2 = None, c1=0.1 , c2=None, c3 = None):
    """Generate the solver of add-drop with 2 coupled rings.
    
    The port are the following:
        a0 : in-port
        a1 : add port
        b0 : trough port
        b1 : drop port

    a1 ------------- b1
            __
           /  \
          |    |
           \__/
            __
           /  \
          |    |
           \__/
    a0 ------------- b0
    

    Args:
        R1 (float): Radius of bottom ring/
        n (float): effective index if waveguide.
        l (float): loss of waveguide (dB/um).
        R2 (float): Radius of top ring. Default to None (same radius as bottom )
        c1 (float, optional): ring-waveguide coupling (waveguide to bottom ring). Defaults to 0.1.
        c2 (float, optional): ring-ring coupling. Defaults to None (c1).
        c3 (float, optional): ring-waveguide coupling (waveguide to top ring). Defaults to None (c1).

    Returns:
        S (Solver)

    """
    c2 = c1 if c2 is None else c2
    c3 = c1 if c3 is None else c3
    R2 = R1 if R2 is None else R2
    with sv.Solver(name='add_drop_2') as S:

        WG1 = sv.Waveguide(np.pi*R1, n=n)
        AT1 = sv.Attenuator(loss = l*np.pi*R1)

        WG2 = sv.Waveguide(np.pi*R2, n=n)
        AT2 = sv.Attenuator(loss = l*np.pi*R2)
        
        PS = sv.PhaseShifter()
        
        
        c1 = sv.GeneralBeamSplitter(ratio=c1).put()
        c2 = sv.GeneralBeamSplitter(ratio=c2).put()
        c3 = sv.GeneralBeamSplitter(ratio=c3).put()
        
        _ = WG1.put('a0', c1.pin['b1'])
        _ = AT1.put('a0', _.pin['b0'])
        _ = PS.put('a0', _.pin['b0'], param_mapping={'PS' : 'PS1'})
        sv.connect(_.pin['b0'], c2.pin['b0'])

        _ = WG1.put('a0', c2.pin['a0'])
        sv.connect(_.pin['b0'], c1.pin['a1'])
        
        _ = WG2.put('a0', c2.pin['b1'])
        _ = AT2.put('a0', _.pin['b0'])
        _ = PS.put('a0', _.pin['b0'], param_mapping={'PS' : 'PS2'})
        sv.connect(_.pin['b0'], c3.pin['b0'])
        
        _ = WG2.put('a0', c3.pin['a0'])
        sv.connect(_.pin['b0'], c2.pin['a1'])
        
        sv.raise_pins()
    return S
        


ring = coupled_add_drop(150, 1.5, 1e-4, c1 = 0.1, R2 = 160, c2 = 0.001)
mod = ring.solve(wl=np.linspace(1.540, 1.545, 5001), PS1 = 0.062)
#for pin in ['a0', 'b0', 'a1', 'b1']:
for pin in ['b0']:
    data = mod.get_data('a0', pin)
    plt.plot(data['wl'], data['dB'], label = 'a0 -> {pin}')

