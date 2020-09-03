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
                assert T.real == pytest.approx(np.cos(2.0*np.pi/1.55*2.5*0.05), 1e-8)
                assert T.imag == pytest.approx(np.sin(2.0*np.pi/1.55*2.5*0.05), 1e-8)


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
                assert T.real == pytest.approx(np.cos(2.0*np.pi/1.55*2.5*0.05*(j+1)), 1e-8)
                assert T.imag == pytest.approx(np.sin(2.0*np.pi/1.55*2.5*0.05*(j+1)), 1e-8)


def test_get_output():
    BST=sv.Structure(model=sv.BeamSplitter(phase=0.5))
    BSB=sv.Structure(model=sv.BeamSplitter(phase=0.5))
    BSC=sv.Structure(model=sv.BeamSplitter(phase=0.5))
    SPC=sv.Structure(model=sv.Splitter1x2())



    Sol=sv.Solver(structures=[SPC,BSB,BSC,BST])

    Sol.connect(SPC,'b0',BSB,'b1')  
    Sol.connect(BSB,'b0',BSC,'b1')
    Sol.connect(BSC,'b0',BST,'b1')
    Sol.connect(BST,'b0',SPC,'b1')
      

    pin_mapping={
        'a0': (SPC,'a0'),
        'r0': (BSC,'a0'),
        'r1': (BSC,'a1'),
        't0': (BST,'a0'),
        't1': (BST,'a1'),
        'b0': (BSB,'a0'),
        'b1': (BSB,'a1'),
    }

    Sol.map_pins(pin_mapping)

    Sol.set_param('Lam',value=1.55)
    new=Sol.solve()


    input_dic={'r0':0.0+0.0j,'a0': 1.0}
    out=new.get_output(input_dic)
    assert out['t0'] == pytest.approx(0.25, 1e-8)
    assert out['t1'] == pytest.approx(0.25, 1e-8)
    assert out['b0'] == pytest.approx(0.25, 1e-8)
    assert out['b1'] == pytest.approx(0.25, 1e-8)

    input_dic={'r0':1.0+0.0j,'a0': 1.0}
    out=new.get_output(input_dic)
    assert out['t0'] == pytest.approx(0.0, 1e-8)
    assert out['t1'] == pytest.approx(1.0, 1e-8)
    assert out['b0'] == pytest.approx(0.5, 1e-8)
    assert out['b1'] == pytest.approx(0.5, 1e-8)

    input_dic={'r0':0.0+1.0j,'a0': 1.0}
    out=new.get_output(input_dic)
    assert out['t0'] == pytest.approx(0.5, 1e-8)
    assert out['t1'] == pytest.approx(0.5, 1e-8)
    assert out['b0'] == pytest.approx(1.0, 1e-8)
    assert out['b1'] == pytest.approx(0.0, 1e-8)

    input_dic={'r0':1.0+1.0j,'a0': 1.0}
    out=new.get_output(input_dic)
    assert out['t0'] == pytest.approx(0.25, 1e-8)
    assert out['t1'] == pytest.approx(1.25, 1e-8)
    assert out['b0'] == pytest.approx(1.25, 1e-8)
    assert out['b1'] == pytest.approx(0.25, 1e-8)


if __name__=='__main__':
    test_get_output()
