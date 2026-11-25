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
