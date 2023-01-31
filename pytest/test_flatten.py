import numpy as np
import pytest
import importlib.util
import lekkersim as lk

import matplotlib.pyplot as plt


def test_basic():
    """Test the basic flatten ruotine, not involving any parameters"""
    with lk.Solver() as S1:
        wg1 = lk.Waveguide(10.0).put()
        lk.raise_pins()

    with lk.Solver() as S2:
        wg2 = lk.Waveguide(10.0).put()
        lk.raise_pins()

    with lk.Solver() as S3:
        s1 = S1.put()
        s2 = S2.put("a0", s1.pin["b0"])
        lk.raise_pins()

    wg1 = S3.structures[0].solver.structures[0]
    wg2 = S3.structures[1].solver.structures[0]

    S3.flatten()

    assert S3.structures[0].model == wg1.model
    assert S3.structures[1].model == wg2.model

    # assert S3.connections == {(wg2,'a0'):(wg1,'b0')}
    # assert S3.pin_mapping == {'a0': (wg1,'a0'), 'b0': (wg2,'b0')}


def test_multiple_placing():
    with lk.Solver() as S1:
        WG = lk.Waveguide(10.0)
        wgs = [WG.put() for i in range(4)]
        for wg1, wg2 in zip(wgs[:-1], wgs[1:]):
            lk.connect(wg1.pin["b0"], wg2.pin["a0"])
        lk.raise_pins()

    with lk.Solver() as S2:
        wgs = [S1.put() for i in range(4)]
        for wg1, wg2 in zip(wgs[:-1], wgs[1:]):
            lk.connect(wg1.pin["b0"], wg2.pin["a0"])
        lk.raise_pins()

    wl = np.linspace(1.5, 1.6, 5)
    ph_ref = S2.solve(wl=wl).get_data("a0", "b0")["Phase"].to_numpy()
    S2.flatten()


def test_params_structure_singletree_top():
    """ """

    with lk.Solver() as S1:
        ps = lk.PhaseShifter().put()
        lk.raise_pins()

    with lk.Solver() as S2:
        S1.put(param_mapping={"PS": "PS1"})
        lk.raise_pins()

    psl = np.linspace(0.0, 1.0, 5)
    ref = S2.solve(wl=1.55, PS1=psl).get_data("a0", "b0")

    S2.flatten()
    out = S2.solve(wl=1.55, PS1=psl).get_data("a0", "b0")

    assert (ref["Amplitude"].to_numpy() == out["Amplitude"].to_numpy()).all()


def test_params_structure_singletree_bottom():
    """ """

    with lk.Solver() as S1:
        ps = lk.PhaseShifter().put(param_mapping={"PS": "PS1"})
        lk.raise_pins()

    with lk.Solver() as S2:
        S1.put()
        lk.raise_pins()

    psl = np.linspace(0.0, 1.0, 5)
    ref = S2.solve(wl=1.55, PS1=psl).get_data("a0", "b0")
    print(ref)

    S2.flatten()
    out = S2.solve(wl=1.55, PS1=psl).get_data("a0", "b0")
    print(out)

    print("")
    print(ps.model)
    print(ps.param_mapping)
    print(S2.structures[0].model)
    print(S2.structures[0].param_mapping)

    # assert (ref['Amplitude'].to_numpy() == out['Amplitude'].to_numpy()).all()


def test_params_structure_singletree_both():
    """ """

    with lk.Solver() as S1:
        ps = lk.PhaseShifter().put(param_mapping={"PS": "PS2"})
        lk.raise_pins()

    with lk.Solver() as S2:
        S1.put(param_mapping={"PS2": "PS1"})
        lk.raise_pins()

    psl = np.linspace(0.0, 1.0, 5)
    ref = S2.solve(wl=1.55, PS1=psl).get_data("a0", "b0")

    S2.flatten()
    out = S2.solve(wl=1.55, PS1=psl).get_data("a0", "b0")

    assert (ref["Amplitude"].to_numpy() == out["Amplitude"].to_numpy()).all()


def test_params_structure_multipletree():
    """ """
    with lk.Solver() as S1:
        ps = lk.PhaseShifter().put(param_mapping={"PS": "PS1"})
        lk.raise_pins()

    with lk.Solver() as S2:
        ps = lk.PhaseShifter().put(param_mapping={"PS": "PS1"})
        S1.put("a0", ps.pin["b0"], param_mapping={"PS1": "PS2"})
        lk.raise_pins()

    with lk.Solver() as S3:
        ps = lk.PhaseShifter().put(param_mapping={"PS": "PS1"})
        S2.put("a0", ps.pin["b0"], param_mapping={"PS1": "PS2", "PS2": "PS3"})
        lk.raise_pins()

    psl = np.linspace(0.0, 1.0, 20)
    ref1 = S3.solve(wl=1.55, PS1=psl).get_data("a0", "b0")["Amplitude"].to_numpy()
    ref2 = S3.solve(wl=1.55, PS2=psl).get_data("a0", "b0")["Amplitude"].to_numpy()
    ref3 = S3.solve(wl=1.55, PS3=psl).get_data("a0", "b0")["Amplitude"].to_numpy()
    ref4 = (
        S3.solve(wl=1.55, PS1=psl, PS2=psl, PS3=psl)
        .get_data("a0", "b0")["Amplitude"]
        .to_numpy()
    )

    assert np.allclose(
        S3.solve(wl=1.55, PS1=psl).get_data("a0", "b0")["Amplitude"].to_numpy(), ref1
    )
    assert np.allclose(
        S3.solve(wl=1.55, PS2=psl).get_data("a0", "b0")["Amplitude"].to_numpy(), ref2
    )
    assert np.allclose(
        S3.solve(wl=1.55, PS3=psl).get_data("a0", "b0")["Amplitude"].to_numpy(), ref3
    )
    assert np.allclose(
        S3.solve(wl=1.55, PS1=psl, PS2=psl, PS3=psl)
        .get_data("a0", "b0")["Amplitude"]
        .to_numpy(),
        ref4,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-vv"])  # -s: show print output
