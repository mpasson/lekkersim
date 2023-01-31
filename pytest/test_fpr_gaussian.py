import os
import pytest as pt
import numpy as np
from lekkersim import FPRGaussian
import pickle

file_dir = os.path.dirname(os.path.abspath(__file__))


@pt.fixture
def fpr1():
    """Fixture symmetric FPR for testing"""
    return FPRGaussian(
        n=5,
        m=5,
        R=20,
        d1=1.4,
        d2=1.4,
        w1=1,
        w2=1,
        n_slab=1.6,
    )


@pt.fixture
def fpr2():
    """Fixture asymmetric FPR for testing"""
    return FPRGaussian(n=3, m=8, R=30, d1=1.4, d2=3, w1=1, w2=1, n_slab=1.6, Ri=10)


def test_init1(fpr1):
    """Tests the initialization of the class."""
    assert np.allclose([-0.14, -0.07, 0.0, 0.07, 0.14], fpr1.t1)
    assert np.allclose([-0.14, -0.07, 0.0, 0.07, 0.14], fpr1.t2)
    assert np.allclose(
        [
            (0.19568007574725543, -2.7908622928847295),
            (0.0489799949344083, -1.398856946750655),
            (0.0, 0.0),
            (0.0489799949344083, 1.398856946750655),
            (0.19568007574725543, 2.7908622928847295),
        ],
        fpr1.pos1,
    )
    assert np.allclose(
        [
            (19.804319924252745, -2.7908622928847295),
            (19.95102000506559, -1.398856946750655),
            (20.0, 0.0),
            (19.95102000506559, 1.398856946750655),
            (19.804319924252745, 2.7908622928847295),
        ],
        fpr1.pos2,
    )


def test_init2(fpr2):
    """Tests the initialization of the class."""
    assert np.allclose([-0.14, 0.0, 0.14], fpr2.t1)
    assert np.allclose([-0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35], fpr2.t2)
    assert np.allclose(
        [
            (0.09784003787362772, -1.3954311464423648),
            (0.0, 0.0),
            (0.09784003787362772, 1.3954311464423648),
        ],
        fpr2.pos1,
    )
    assert np.allclose(
        [
            (28.181181385421368, -10.286934223663541),
            (29.067372651319342, -7.4221187776356885),
            (29.66313233808127, -4.4831439742079775),
            (29.962507811848987, -1.4993750781203499),
            (29.962507811848987, 1.4993750781203499),
            (29.66313233808127, 4.4831439742079775),
            (29.067372651319342, 7.4221187776356885),
            (28.181181385421368, 10.286934223663541),
        ],
        fpr2.pos2,
    )


def test_s_matrix_1(fpr1):
    """Tests the S-matrix generation for the symmetric FPR"""
    file = os.path.join(file_dir, "references", "s_matrix_fpr_gaussian.pkl")
    with open(file, "rb") as f:
        ref = pickle.load(f)

    mod = fpr1.solve(wl=1.55)

    assert np.allclose(ref, np.squeeze(mod.S))


def test_s_matrix_2(fpr2):
    """Tests the S-matrix generation for the asymmetric FPR"""
    file = os.path.join(file_dir, "references", "s_matrix_fpr_gaussian_asym.pkl")
    with open(file, "rb") as f:
        ref = pickle.load(f)

    mod = fpr2.solve(wl=1.55)

    assert np.allclose(ref, np.squeeze(mod.S))


if __name__ == "__main__":
    pt.main([__file__, "-s", "-vv"])
