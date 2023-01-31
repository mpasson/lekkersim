import numpy as np
import pytest
import importlib.util
import matplotlib.pyplot as plt
import lekkersim as lk


def Ring(r, L):
    with lk.Solver() as S:
        bm = lk.BeamSplitter(ratio=r).put()
        wg = lk.Waveguide(L).put("a0", bm.pin["b1"])
        lk.connect(wg.pin["b0"], bm.pin["a1"])
        lk.raise_pins()
        lk.add_structure_to_monitors(wg)
    return S


def test_singlemonitor():
    ring = Ring(0.1, 100.0)
    wll = np.linspace(1.54, 1.56, 11)
    I = ring.solve(wl=wll).get_monitor({"a0": 1.0})["Monitor_a0_i"].to_numpy()
    c = -np.exp(2.0j * np.pi * 100.0 / wll)
    P = 0.1 / np.abs(1 + np.sqrt(0.9) * c) ** 2.0
    assert np.allclose(I, P)


def test_doublesolver():
    ring1 = Ring(0.1, 100.0)
    ring2 = Ring(0.1, 120.0)
    with lk.Solver() as S:
        r1 = ring1.put()
        r2 = ring2.put()
        lk.connect(r1.pin["b0"], r2.pin["a0"])
        lk.raise_pins()
    S.solve()
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-vv"])  # -s: show print output
