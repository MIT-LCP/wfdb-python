import numpy as np
import pytest
from numpy.testing import assert_allclose

from wfdb import processing


@pytest.mark.parametrize("fs", [250, 1000])
@pytest.mark.parametrize("fraction", [0, 0.1234, 0.4999, 0.5])
@pytest.mark.parametrize(
    "b,a,expected",
    [
        ([1], [1], lambda w: 1),
        ([0.5, 0.5], [1], lambda w: abs(np.cos(w / 2))),
        ([1, -1], [1], lambda w: 2 * abs(np.sin(w / 2))),
        ([1], [1, -0.5], lambda w: 1 / np.sqrt(1.25 - np.cos(w))),
    ],
)
def test_filter_gain_at_requested_frequency(fs, fraction, b, a, expected):
    """Measure analytic FIR/IIR responses via the public processing API."""
    frequency = fraction * fs
    omega = 2 * np.pi * fraction
    original_b, original_a = b.copy(), a.copy()

    gain = processing.get_filter_gain(b, a, frequency, fs)

    assert_allclose(gain, expected(omega), rtol=1e-12, atol=1e-14)
    assert b == original_b
    assert a == original_a
