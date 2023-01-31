import numpy as np
import pytest
import os

import lekkersim as lk


def test_save_and_load_1_paramter():
    sol = lk.Waveguide(10.0).expand_mode(["TE", "TM"])
    mod = sol.solve(wl=[1.001, 1.002, 1.003])
    data = mod.export_InPulse(
        parameter_name_mapping={"wl": "wavelength"}, units={"wavelength": "um"}
    )
    out1 = sol.solve(wl=1.002).S2PD()
    test = lk.Model_from_InPulse(
        "exported_model.csvy",
        parameter_name_mapping={"wavelength": "wl"},
        # mode_mapping={"TE": "TEE"},
    )

    out2 = test.solve(wl=1.002).S2PD()
    os.remove("exported_model.csvy")

    assert np.allclose(out1, out2)


def test_save_and_load_2_paramter():

    with lk.Solver("test") as sol:
        _ = lk.Waveguide(10.0).put()
        lk.PhaseShifter().put("a0", _.pin["b0"])
        lk.raise_pins()

    wl = np.linspace(1.0, 1.02, 3)
    ps = np.linspace(0.0, 0.2, 3)

    wl, ps = np.meshgrid(wl, ps, indexing="ij")

    mod = sol.solve(wl=wl, PS=ps)
    mod.export_InPulse(
        parameter_name_mapping={"wl": "wavelength"},
        units={"wavelength": "um", "PS": "unit of pi"},
    )
    out1 = sol.solve(wl=1.01, PS=0.1).S2PD()
    test = lk.Model_from_InPulse(
        "exported_model.csvy",
        parameter_name_mapping={"wavelength": "wl"},
    )

    out2 = test.solve(wl=1.01, PS=0.1).S2PD()
    os.remove("exported_model.csvy")

    assert np.allclose(out1.values, out2.values)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-vv"])  # -s: show print output
