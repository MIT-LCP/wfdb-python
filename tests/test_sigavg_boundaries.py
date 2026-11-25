import numpy as np
import pytest
from numpy.testing import assert_allclose

import wfdb
from wfdb import processing


def write_record(tmp_path, name, values, sample):
    wfdb.wrsamp(
        name,
        fs=10,
        units=["mV"],
        sig_name=["ECG"],
        d_signal=np.asarray(values, dtype=np.int64).reshape(-1, 1),
        fmt=["16"],
        adc_gain=[1],
        baseline=[0],
        write_dir=str(tmp_path),
    )
    wfdb.wrann(
        name,
        "atr",
        sample=np.array([sample]),
        symbol=["N"],
        fs=10,
        write_dir=str(tmp_path),
    )
    return str(tmp_path / name)


@pytest.mark.parametrize("sample", [0, 1])
@pytest.mark.parametrize("tail", [9, 9000])
def test_sigavg_pads_before_start_without_wrapping(tmp_path, sample, tail):
    values = list(range(10))
    values[-1] = tail
    record = write_record(tmp_path, "start", values, sample)

    result = processing.sigavg(
        record,
        "atr",
        return_df=True,
        start_range=-0.2,
        stop_range=0.2,
        stop_time=0.5,
    )

    # Compute expected padding without indexing negative positions.
    expected = [
        values[sample + offset] if sample + offset >= 0 else 0
        for offset in [-2, -1, 0, 1]
    ]
    assert_allclose(result["ECG"], expected)


@pytest.mark.parametrize("sample", [5, 9])
def test_sigavg_interior_and_end_padding_unchanged(tmp_path, sample):
    values = list(range(10))
    record = write_record(tmp_path, "control", values, sample)

    result = processing.sigavg(
        record,
        "atr",
        return_df=True,
        start_range=-0.2,
        stop_range=0.2,
        stop_time=1,
    )

    expected = [
        values[sample + offset] if sample + offset < 10 else 0
        for offset in [-2, -1, 0, 1]
    ]
    assert_allclose(result["ECG"], expected)


@pytest.mark.parametrize("samples", [[0], [9], [0, 5, 9]])
@pytest.mark.parametrize("baseline,gain", [(100, 2), (-120, 4)])
def test_sigavg_padding_is_zero_in_physical_units(tmp_path, samples, baseline, gain):
    physical = np.array([3, -2, 0, 1, 4, -1, 2, 0, -3, 5])
    digital = baseline + gain * physical
    wfdb.wrsamp(
        "calibrated",
        fs=10,
        units=["mV"],
        sig_name=["ECG"],
        d_signal=digital[:, None],
        fmt=["16"],
        adc_gain=[gain],
        baseline=[baseline],
        write_dir=str(tmp_path),
    )
    wfdb.wrann(
        "calibrated",
        "atr",
        sample=np.array(samples),
        symbol=["N"] * len(samples),
        fs=10,
        write_dir=str(tmp_path),
    )
    result = processing.sigavg(
        str(tmp_path / "calibrated"),
        "atr",
        return_df=True,
        start_range=-0.2,
        stop_range=0.2,
        stop_time=1,
    )
    expected = [
        sum(
            physical[sample + offset] if 0 <= sample + offset < len(physical) else 0
            for sample in samples
        )
        / len(samples)
        for offset in [-2, -1, 0, 1]
    ]
    assert_allclose(result["ECG"], np.round(expected, 5))


@pytest.mark.parametrize("sample", [0, 9])
def test_sigavg_flat_zero_physical_signal_stays_zero_at_boundaries(tmp_path, sample):
    wfdb.wrsamp(
        "flat",
        fs=10,
        units=["mV"],
        sig_name=["ECG"],
        d_signal=np.full((10, 1), 100, dtype=np.int64),
        fmt=["16"],
        adc_gain=[2],
        baseline=[100],
        write_dir=str(tmp_path),
    )
    wfdb.wrann(
        "flat",
        "atr",
        sample=np.array([sample]),
        symbol=["N"],
        fs=10,
        write_dir=str(tmp_path),
    )
    result = processing.sigavg(
        str(tmp_path / "flat"),
        "atr",
        return_df=True,
        start_range=-0.2,
        stop_range=0.2,
        stop_time=1,
    )
    assert_allclose(result["ECG"], 0)
