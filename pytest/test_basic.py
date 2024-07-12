import numpy as np
import pytest
import importlib.util
import lekkersim as lk


def test_waveguide():
    with lk.Solver() as S:
        WG = lk.Waveguide(10.0, 2.5)
        wg = WG.put()
        lk.Pin("a0").put(wg.pin["a0"])
        lk.Pin("b0").put(wg.pin["b0"])

    T = S.solve(wl=1.55).get_T("a0", "b0")
    assert T == pytest.approx(1.0, 1e-8)


def test_single_composition():
    with lk.Solver() as S:
        WG = lk.Waveguide(0.05, 2.5)
        for i in range(10):
            wg = WG.put()
            lk.Pin(f"a{i}").put(wg.pin["a0"])
            lk.Pin(f"b{i}").put(wg.pin["b0"])

    M = S.solve(wl=1.55)
    for i in range(10):
        for j in range(10):
            T = M.get_A(f"a{i}", f"b{j}")
            if i != j:
                assert T.real == pytest.approx(0.0, 1e-8)
                assert T.imag == pytest.approx(0.0, 1e-8)
            else:
                assert T.real == pytest.approx(
                    np.cos(2.0 * np.pi / 1.55 * 2.5 * 0.05), 1e-8
                )
                assert T.imag == pytest.approx(
                    np.sin(2.0 * np.pi / 1.55 * 2.5 * 0.05), 1e-8
                )


def test_composition():
    with lk.Solver() as S:
        WG = lk.Waveguide(0.05, 2.5)
        for i in range(10):
            wg = WG.put()
            lk.Pin(f"a{i}").put(wg.pin["a0"])
            for j in range(i):
                wg = WG.put("a0", wg.pin["b0"])
            lk.Pin(f"b{i}").put(wg.pin["b0"])

    M = S.solve(wl=1.55)
    for i in range(10):
        for j in range(10):
            T = M.get_A(f"a{i}", f"b{j}")
            if i != j:
                assert T.real == pytest.approx(0.0, 1e-8)
                assert T.imag == pytest.approx(0.0, 1e-8)
            else:
                assert T.real == pytest.approx(
                    np.cos(2.0 * np.pi / 1.55 * 2.5 * 0.05 * (j + 1)), 1e-8
                )
                assert T.imag == pytest.approx(
                    np.sin(2.0 * np.pi / 1.55 * 2.5 * 0.05 * (j + 1)), 1e-8
                )


def test_get_output_by_pin():
    BST = lk.Structure(model=lk.BeamSplitter())
    BSB = lk.Structure(model=lk.BeamSplitter())
    BSC = lk.Structure(model=lk.BeamSplitter())
    SPC = lk.Structure(model=lk.Splitter1x2())

    Sol = lk.Solver(structures=[SPC, BSB, BSC, BST])

    Sol.connect(SPC, lk.Pin("b0"), BSB, lk.Pin("b1"))
    Sol.connect(BSB, lk.Pin("b0"), BSC, lk.Pin("b1"))
    Sol.connect(BSC, lk.Pin("b0"), BST, lk.Pin("b1"))
    Sol.connect(BST, lk.Pin("b0"), SPC, lk.Pin("b1"))

    pin_mapping = {
        lk.Pin("a0"): (SPC, lk.Pin("a0")),
        lk.Pin("r0"): (BSC, lk.Pin("a0")),
        lk.Pin("r1"): (BSC, lk.Pin("a1")),
        lk.Pin("t0"): (BST, lk.Pin("a0")),
        lk.Pin("t1"): (BST, lk.Pin("a1")),
        lk.Pin("b0"): (BSB, lk.Pin("a0")),
        lk.Pin("b1"): (BSB, lk.Pin("a1")),
    }

    Sol.map_pins(pin_mapping)

    Sol.set_param("Lam", value=1.55)
    new = Sol.solve()

    input_dic = {"r0": 0.0 + 0.0j, "a0": 1.0}
    out = new.get_output(input_dic)
    assert out["t0"] == pytest.approx(0.25, 1e-8)
    assert out["t1"] == pytest.approx(0.25, 1e-8)
    assert out["b0"] == pytest.approx(0.25, 1e-8)
    assert out["b1"] == pytest.approx(0.25, 1e-8)

    input_dic = {"r0": 1.0 + 0.0j, "a0": 1.0}
    out = new.get_output(input_dic)
    assert out["t0"] == pytest.approx(0.5, 1e-8)
    assert out["t1"] == pytest.approx(0.5, 1e-8)
    assert out["b0"] == pytest.approx(1.0, 1e-8)
    assert out["b1"] == pytest.approx(0.0, 1e-8)

    input_dic = {"r0": 0.0 + 1.0j, "a0": 1.0}
    out = new.get_output(input_dic)
    assert out["t0"] == pytest.approx(0.0, 1e-8)
    assert out["t1"] == pytest.approx(1.0, 1e-8)
    assert out["b0"] == pytest.approx(0.5, 1e-8)
    assert out["b1"] == pytest.approx(0.5, 1e-8)

    input_dic = {"r0": 1.0 + 1.0j, "a0": 1.0}
    out = new.get_output(input_dic)
    assert out["t0"] == pytest.approx(0.25, 1e-8)
    assert out["t1"] == pytest.approx(1.25, 1e-8)
    assert out["b0"] == pytest.approx(1.25, 1e-8)
    assert out["b1"] == pytest.approx(0.25, 1e-8)


def test_get_output_by_name():
    BST = lk.Structure(model=lk.BeamSplitter())
    BSB = lk.Structure(model=lk.BeamSplitter())
    BSC = lk.Structure(model=lk.BeamSplitter())
    SPC = lk.Structure(model=lk.Splitter1x2())

    Sol = lk.Solver(structures=[SPC, BSB, BSC, BST])

    Sol.connect(SPC, "b0", BSB, "b1")
    Sol.connect(BSB, "b0", BSC, "b1")
    Sol.connect(BSC, "b0", BST, "b1")
    Sol.connect(BST, "b0", SPC, "b1")

    pin_mapping = {
        "a0": (SPC, "a0"),
        "r0": (BSC, "a0"),
        "r1": (BSC, "a1"),
        "t0": (BST, "a0"),
        "t1": (BST, "a1"),
        "b0": (BSB, "a0"),
        "b1": (BSB, "a1"),
    }

    Sol.map_pins(pin_mapping)

    Sol.set_param("Lam", value=1.55)
    new = Sol.solve()

    input_dic = {"r0": 0.0 + 0.0j, "a0": 1.0}
    out = new.get_output(input_dic)
    assert out["t0"] == pytest.approx(0.25, 1e-8)
    assert out["t1"] == pytest.approx(0.25, 1e-8)
    assert out["b0"] == pytest.approx(0.25, 1e-8)
    assert out["b1"] == pytest.approx(0.25, 1e-8)

    input_dic = {"r0": 1.0 + 0.0j, "a0": 1.0}
    out = new.get_output(input_dic)
    assert out["t0"] == pytest.approx(0.5, 1e-8)
    assert out["t1"] == pytest.approx(0.5, 1e-8)
    assert out["b0"] == pytest.approx(1.0, 1e-8)
    assert out["b1"] == pytest.approx(0.0, 1e-8)

    input_dic = {"r0": 0.0 + 1.0j, "a0": 1.0}
    out = new.get_output(input_dic)
    assert out["t0"] == pytest.approx(0.0, 1e-8)
    assert out["t1"] == pytest.approx(1.0, 1e-8)
    assert out["b0"] == pytest.approx(0.5, 1e-8)
    assert out["b1"] == pytest.approx(0.5, 1e-8)

    input_dic = {"r0": 1.0 + 1.0j, "a0": 1.0}
    out = new.get_output(input_dic)
    assert out["t0"] == pytest.approx(0.25, 1e-8)
    assert out["t1"] == pytest.approx(1.25, 1e-8)
    assert out["b0"] == pytest.approx(1.25, 1e-8)
    assert out["b1"] == pytest.approx(0.25, 1e-8)


if __name__ == "__main__":
    test_get_output_by_name()
