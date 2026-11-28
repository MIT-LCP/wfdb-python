"""Loaded signal value layouts for single and multi-segment reads."""

import numpy as np
import pytest

import wfdb
from wfdb.io.record import _LoadedSignalValues


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


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("expanded", [False, True])
def test_flattened_multisegment_record_uses_one_signal_field(physical, expanded):
    """Combining segments preserves the selected value mode and frame count."""
    record = wfdb.rdrecord(
        "sample-data/multi-segment/fixed1/v102s",
        sampfrom=74999,
        sampto=75001,
        physical=physical,
        smooth_frames=not expanded,
        return_res=32,
    )
    field = ("e_" if expanded else "") + ("p_signal" if physical else "d_signal")
    fields = ("p_signal", "d_signal", "e_p_signal", "e_d_signal")

    assert (record.sig_len, record.n_sig) == (2, 4)
    assert [name for name in fields if getattr(record, name) is not None] == [field]
    values = getattr(record, field)
    if expanded:
        assert [len(signal) for signal in values] == [2] * 4
        assert all(
            signal.dtype == f"{'float' if physical else 'int'}32" for signal in values
        )
    else:
        assert values.shape == (2, 4)
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


def test_uniform_mode_rejects_wrong_channel_count():
    """A uniform array must match both the frame and channel counts."""
    record = wfdb.Record(sig_len=3, n_sig=2)
    values = _LoadedSignalValues(np.zeros((3, 1), dtype=np.int16), False, False)

    with pytest.raises(ValueError, match="Uniform signal shape"):
        values.set_on_record(record)
    assert record.d_signal is None


def test_expanded_mode_rejects_wrong_sample_count():
    """Expanded channels must contain their frame count times their rate."""
    record = wfdb.Record(sig_len=3, n_sig=1, samps_per_frame=[2])
    values = _LoadedSignalValues([np.zeros(2, dtype=np.int16)], False, True)

    with pytest.raises(ValueError, match="expected 6"):
        values.set_on_record(record)
    assert record.e_d_signal is None
