import numpy as np
import pytest
import importlib.util
import lekkersim as lk

import matplotlib.pyplot as plt


def test_split_simple():

    with lk.Solver() as S:
        g1 = []
        g1.append(lk.Waveguide(1.0).put())
        g1.append(lk.Waveguide(1.0).put("a0", g1[-1].pin["b0"]))
        g1.append(lk.Waveguide(1.0).put("a0", g1[-1].pin["b0"]))
        lk.Pin("a0").put(g1[0].pin["a0"])
        lk.Pin("b0").put(g1[-1].pin["b0"])

        g2 = []
        g2.append(lk.Waveguide(1.0).put())
        g2.append(lk.Waveguide(1.0).put("a0", g2[-1].pin["b0"]))
        lk.Pin("a1").put(g2[0].pin["a0"])
        lk.Pin("b2").put(g2[-1].pin["b0"])

        g3 = []
        g3.append(lk.Waveguide(1.0).put())
        lk.Pin("a1").put(g3[0].pin["a0"])
        lk.Pin("b2").put(g3[-1].pin["b0"])

    solvers = S.split()

    assert set(solvers[0].structures) == set(g1)
    assert set(solvers[1].structures) == set(g2)
    assert set(solvers[2].structures) == set(g3)


def test_split_parametric():
    with lk.Solver() as S:
        ps = lk.PhaseShifter()
        ps1 = ps.put(param_mapping={"PS": "PS1"})
        ps2 = ps.put(param_mapping={"PS": "PS2"})
        ps1.raise_pins()
        ps2.raise_pins(["a0", "b0"], ["a1", "b1"])

    solvers = S.split()

    ps = np.linspace(0.0, 1.0, 5)

    data_original = S.solve(wl=1.55, PS1=ps).get_data("a0", "b0")
    data_split = solvers[0].solve(wl=1.55, PS1=ps).get_data("a0", "b0")
    assert np.allclose(data_original["Phase"], data_split["Phase"])

    data_original = S.solve(wl=1.55, PS2=ps).get_data("a1", "b1")
    data_split = solvers[1].solve(wl=1.55, PS2=ps).get_data("a1", "b1")
    assert np.allclose(data_original["Phase"], data_split["Phase"])


def test_split_parametric_mapped():
    with lk.Solver() as S:
        ps = lk.PhaseShifter()
        ps1 = ps.put(param_mapping={"PS": "PS1"})
        ps2 = ps.put(param_mapping={"PS": "PS2"})
        ps1.raise_pins()
        ps2.raise_pins(["a0", "b0"], ["a1", "b1"])
        lk.add_param("PS1", lambda V1=0.0: V1**2.0)
        lk.add_param("PS2", lambda V2=0.0: V2**2.0)

    solvers = S.split()

    V = np.linspace(0.0, 1.0, 5)

    data_original = S.solve(wl=1.55, V1=V).get_data("a0", "b0")
    data_split = solvers[0].solve(wl=1.55, V1=V).get_data("a0", "b0")
    assert np.allclose(data_original["Phase"], data_split["Phase"])

    data_original = S.solve(wl=1.55, V2=V).get_data("a1", "b1")
    data_split = solvers[1].solve(wl=1.55, V2=V).get_data("a1", "b1")
    assert np.allclose(data_original["Phase"], data_split["Phase"])


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-vv"])  # -s: show print output
