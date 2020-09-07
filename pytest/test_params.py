import numpy as np
import pytest
import importlib.util
spec = importlib.util.spec_from_file_location("solver", "../solver/__init__.py")
sv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sv)


def test_single():
    ps=sv.PhaseShifter()

    with sv.Solver() as S1:
        ps.put()
        sv.raise_pins()

    with sv.Solver() as S2:
        ps.put(param_mapping={'PS':'PS2'})
        sv.raise_pins()

    with sv.Solver() as S3:
        S1.put(param_mapping={'PS':'PS2'})
        sv.raise_pins()

    assert S1.solve().get_PH('a0','b0') == pytest.approx(0.0, 1e-8)
    assert S1.solve(PS=0.5).get_PH('a0','b0') == pytest.approx(0.5*np.pi, 1e-8)
    assert S1.solve().get_PH('a0','b0') == pytest.approx(0.0, 1e-8)

    assert S2.solve().get_PH('a0','b0') == pytest.approx(0.0, 1e-8)
    assert S2.solve(PS=0.5).get_PH('a0','b0') == pytest.approx(0.0, 1e-8)
    assert S2.solve(PS2=0.5).get_PH('a0','b0') == pytest.approx(0.5*np.pi, 1e-8)
    assert S2.solve().get_PH('a0','b0') == pytest.approx(0.0, 1e-8)

    assert S3.solve().get_PH('a0','b0') == pytest.approx(0.0, 1e-8)
    assert S3.solve(PS=0.5).get_PH('a0','b0') == pytest.approx(0.0, 1e-8)
    assert S3.solve(PS2=0.5).get_PH('a0','b0') == pytest.approx(0.5*np.pi, 1e-8)
    assert S3.solve().get_PH('a0','b0') == pytest.approx(0.0, 1e-8)


def test_params():
    with sv.Solver(name='MZM_BB') as MZM_BB_sol:
        BM=sv.GeneralBeamSplitter()
        WG=sv.Waveguide(L=500,n=2.5)
        PS=sv.PhaseShifter()
        AT=sv.Attenuator(loss=0.0)

        bm1=BM.put()
        t_=WG.put('a0',bm1.pin['b0'])
        t_=PS.put('a0',t_.pin['b0'],param_mapping={'PS':'PS1'})
        t_=PS.put('a0',t_.pin['b0'],param_mapping={'PS':'DP'})
        t_=AT.put('a0',t_.pin['b0'])
        bm2=BM.put('a0',t_.pin['b0'])
        t_=WG.put('a0',bm1.pin['b1'])
        t_=PS.put('a0',t_.pin['b0'],param_mapping={'PS':'PS2'})
        t_=AT.put('a0',t_.pin['b0'])
        sv.connect(t_.pin['b0'],bm2.pin['a1'])

        sv.Pin('a0').put(bm1.pin['a0'])
        sv.Pin('a1').put(bm1.pin['a1'])
        sv.Pin('b0').put(bm2.pin['b0'])
        sv.Pin('b1').put(bm2.pin['b1'])

        sv.set_default_params({'PS1':0.0, 'PS2': 0.5, 'DP' : 0.0})

        psl=np.linspace(0.0,1.0,5)

        T=[MZM_BB_sol.solve(wl=1.55, DP=ps, PS1=0.0, PS2=0.0).get_T('a0','b0') for ps in psl]
        assert np.allclose(T,np.cos(0.5*np.pi*psl)**2.0)
        T=[MZM_BB_sol.solve(wl=1.55, DP=ps, PS1=0.0, PS2=0.5).get_T('a0','b0') for ps in psl]
        assert np.allclose(T,np.cos((0.5*psl-0.25)*np.pi)**2.0)
        T=[MZM_BB_sol.solve(wl=1.55, DP=ps).get_T('a0','b0') for ps in psl]
        assert np.allclose(T,np.cos((0.5*psl-0.25)*np.pi)**2.0)

