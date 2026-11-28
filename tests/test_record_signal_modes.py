"""Loaded signal value layouts for single-segment reads."""

import numpy as np
import pytest

import wfdb


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("expanded", [False, True])
def test_multifrequency_record_uses_one_signal_field(physical, expanded):
    """Unit and frame choices select one field with the expected shape."""
    record = wfdb.rdrecord(
        "sample-data/mixedsignals",
        sampto=3,
        physical=physical,
        smooth_frames=not expanded,
        return_res=32,
    )
    field = ("e_" if expanded else "") + ("p_signal" if physical else "d_signal")
    fields = ("p_signal", "d_signal", "e_p_signal", "e_d_signal")

    assert (record.sig_len, record.n_sig) == (3, 6)
    assert [name for name in fields if getattr(record, name) is not None] == [field]
    values = getattr(record, field)
    if expanded:
        assert isinstance(values, list)
        assert [len(signal) for signal in values] == [12, 12, 12, 6, 6, 3]
        assert all(
            signal.dtype == f"{'float' if physical else 'int'}32" for signal in values
        )
    else:
        assert values.shape == (3, 6)
        assert values.dtype == f"{'float' if physical else 'int'}32"


@pytest.mark.parametrize("smooth_frames", [False, True])
def test_physical_values_keep_missing_samples_after_skew(smooth_frames):
    """Digital missing markers become NaNs while other values are calibrated."""
    path = "sample-data/100skew"
    options = {"sampfrom": 649995, "sampto": 650000, "smooth_frames": smooth_frames}
    digital = wfdb.rdrecord(path, physical=False, **options)
    physical = wfdb.rdrecord(path, physical=True, **options)
    digital_values = digital.d_signal[:, 1] if smooth_frames else digital.e_d_signal[1]
    physical_values = (
        physical.p_signal[:, 1] if smooth_frames else physical.e_p_signal[1]
    )

    np.testing.assert_allclose(
        physical_values[:-3],
        (digital_values[:-3] - digital.baseline[1]) / digital.adc_gain[1],
    )
    assert np.isnan(physical_values[-3:]).all()
