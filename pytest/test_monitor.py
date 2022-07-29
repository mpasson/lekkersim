import numpy as np
import pytest
import importlib.util
import matplotlib.pyplot as plt
import solver as sv


def Ring(r,L):
    with sv.Solver() as S:
        bm = sv.BeamSplitter(ratio=r).put()
        wg = sv.Waveguide(L).put('a0', bm.pin['b1'])
        sv.connect(wg.pin['b0'], bm.pin['a1'])
        sv.raise_pins()
        sv.add_structure_to_monitors(wg)
    return S



def test_singlemonitor():
    ring=Ring(0.1,100.0)
    wll = np.linspace(1.54,1.56,11)
    I = ring.solve(wl=wll).get_monitor({'a0' : 1.0})['Monitor_a0_i'].to_numpy()
    c = -np.exp(2.0j*np.pi*100.0/wll)
    P = 0.1/np.abs(1+np.sqrt(0.9)*c)**2.0
    assert np.allclose(I,P)    


def test_doublesolver():
    ring1=Ring(0.1,100.0)
    ring2=Ring(0.1,120.0)
    with sv.Solver() as S:
        r1 = ring1.put()
        r2 = ring2.put()
        sv.connect(r1.pin['b0'], r2.pin['a0'])
        sv.raise_pins()
    S.solve()
    assert True

if __name__ == '__main__':
    pytest.main([__file__, '-s', '-vv'])  # -s: show print output



