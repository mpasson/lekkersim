import numpy as np
import pytest
import importlib.util
spec = importlib.util.spec_from_file_location("solver", "../solver/__init__.py")
sv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sv)


def test_waveguide():
    with sv.Solver() as S:
        WG=sv.Waveguide(10.0, 2.5)
        wg=WG.put()
        sv.Pin('a0').put(wg.pin['a0'])
        sv.Pin('b0').put(wg.pin['b0'])

    T=S.solve(wl=1.55).get_T('a0','b0')
    assert T == pytest.approx(1.0, 1e-8)


def test_single_composition():
    with sv.Solver() as S:
        WG=sv.Waveguide(0.05,2.5)
        for i in range(10):
            wg=WG.put()
            sv.Pin(f'a{i}').put(wg.pin['a0'])
            sv.Pin(f'b{i}').put(wg.pin['b0'])

    M=S.solve(wl=1.55)
    for i in range(10):
        for j in range(10):
            T=M.get_A(f'a{i}',f'b{j}')
            if i!=j:
                assert T.real == pytest.approx(0.0, 1e-8)
                assert T.imag == pytest.approx(0.0, 1e-8)
            else:
                assert T.real == pytest.approx(np.cos(2.0*np.pi/1.55*0.05), 1e-8)
                assert T.imag == pytest.approx(np.sin(2.0*np.pi/1.55*0.05), 1e-8)


def test_composition():
    with sv.Solver() as S:
        WG=sv.Waveguide(0.05,2.5)
        for i in range(10):
            wg=WG.put()
            sv.Pin(f'a{i}').put(wg.pin['a0'])
            for j in range(i):
                wg=WG.put('a0',wg.pin['b0'])
            sv.Pin(f'b{i}').put(wg.pin['b0'])

    M=S.solve(wl=1.55)
    for i in range(10):
        for j in range(10):
            T=M.get_A(f'a{i}',f'b{j}')
            if i!=j:
                assert T.real == pytest.approx(0.0, 1e-8)
                assert T.imag == pytest.approx(0.0, 1e-8)
            else:
                assert T.real == pytest.approx(np.cos(2.0*np.pi/1.55*0.05*(j+1)), 1e-8)
                assert T.imag == pytest.approx(np.sin(2.0*np.pi/1.55*0.05*(j+1)), 1e-8)
