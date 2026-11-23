"""Regression tests: values quoted in the paper text (Sections 3 and 5)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pistla import propagation as pr, scintillation as sc  # noqa: E402


def test_p838_coefficients():
    k, a = pr.p838_coeffs(2.4)
    assert abs(k - 1.2e-4) / 1.2e-4 < 0.05 and abs(a - 1.107) < 0.002
    k, a = pr.p838_coeffs(5.8)
    assert abs(k - 5.6e-4) / 5.6e-4 < 0.01 and abs(a - 1.614) < 0.002


def test_rain_specific_attenuation_145mmh():
    assert abs(pr.gamma_rain(2.4, 145) - 0.031) < 0.001
    assert abs(pr.gamma_rain(5.8, 145) - 1.73) < 0.01
    assert abs(pr.gamma_rain(10, 145) - 6.3) < 0.1
    assert abs(pr.gamma_rain(28, 145) - 25.3) < 0.1


def test_plos_values():
    np.testing.assert_allclose([pr.p_los(30, e) for e in ("urban", "dense urban", "high-rise")],
                               [0.73, 0.37, 0.04], atol=0.01)


def test_loss_of_lock():
    assert sc.p_loss_of_lock(35, 0.7) > 0.02
    assert abs(sc.p_loss_of_lock(35, 1.0) - 0.12) < 0.005
