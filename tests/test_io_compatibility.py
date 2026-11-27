"""Public I/O behavior to preserve while restructuring record objects."""

import pickle

import numpy as np

import wfdb


def test_header_only_and_loaded_record_contract():
    """A header read returns the same type, without loaded signal values."""
    header = wfdb.rdheader("sample-data/100")
    record = wfdb.rdrecord("sample-data/100", sampfrom=2, sampto=6, channels=[1])

    assert isinstance(header, wfdb.Record)
    assert header.sig_len == 650000
    assert header.n_sig == 2
    assert all(
        getattr(header, field) is None
        for field in ("p_signal", "d_signal", "e_p_signal", "e_d_signal")
    )
    assert isinstance(record, wfdb.Record)
    assert record.sig_len == 4
    assert record.n_sig == 1
    assert record.sig_name == [header.sig_name[1]]
    assert record.p_signal.shape == (4, 1)


def test_expanded_values_use_frame_count_and_per_channel_arrays():
    """Expanded channels may hold more samples than the record has frames."""
    record = wfdb.rdrecord(
        "sample-data/mixedsignals", sampto=3, physical=False, smooth_frames=False
    )

    assert isinstance(record, wfdb.Record)
    assert record.sig_len == 3
    assert record.samps_per_frame == [4, 4, 4, 2, 2, 1]
    assert [len(signal) for signal in record.e_d_signal] == [12, 12, 12, 6, 6, 3]
    assert record.p_signal is None
    assert record.d_signal is None
    assert record.e_p_signal is None


def test_multisegment_read_type_depends_on_combining_segments():
    """The default combines segments, while m2s=False preserves them."""
    path = "sample-data/multi-segment/fixed1/v102s"
    header = wfdb.rdheader(path)
    combined = wfdb.rdrecord(path, sampto=8)
    segmented = wfdb.rdrecord(path, sampto=8, m2s=False)

    assert isinstance(header, wfdb.MultiRecord)
    assert header.segments is None
    assert isinstance(combined, wfdb.Record)
    assert combined.p_signal.shape == (8, header.n_sig)
    assert isinstance(segmented, wfdb.MultiRecord)
    assert len(segmented.segments) == segmented.n_seg


def test_rdsamp_legacy_tuple_and_field_names():
    """The simplified reader returns an array and its established field dict."""
    signals, fields = wfdb.rdsamp("sample-data/100", sampfrom=2, sampto=6, channels=[1])
    record = wfdb.rdrecord("sample-data/100", sampfrom=2, sampto=6, channels=[1])

    np.testing.assert_array_equal(signals, record.p_signal)
    assert set(fields) == {
        "fs",
        "sig_len",
        "n_sig",
        "base_date",
        "base_time",
        "units",
        "sig_name",
        "comments",
    }
    assert fields["sig_len"] == 4
    assert fields["n_sig"] == 1
    assert fields["sig_name"] == record.sig_name


def test_positional_read_arguments_keep_their_current_order():
    """Existing positional calls select the same digital frames and channel."""
    positional = wfdb.rdrecord("sample-data/100", 2, 6, [1], False)
    keyword = wfdb.rdrecord(
        "sample-data/100", sampfrom=2, sampto=6, channels=[1], physical=False
    )

    np.testing.assert_array_equal(positional.d_signal, keyword.d_signal)
    assert positional.sig_name == keyword.sig_name


def test_in_place_channel_edit_survives_write_and_read(tmp_path):
    """Callers can edit a channel name inside the existing list before writing."""
    record = wfdb.rdrecord("sample-data/100", sampto=8, physical=False)
    record.record_name = "copy"
    record.file_name = ["copy.dat"] * record.n_sig
    record.sig_name[0] = "ECG"

    record.wrsamp(write_dir=str(tmp_path))
    written = wfdb.rdrecord(str(tmp_path / "copy"), physical=False)

    assert written.sig_name == ["ECG", record.sig_name[1]]
    np.testing.assert_array_equal(written.d_signal, record.d_signal)


def test_record_and_annotation_remain_pickleable():
    """Public record and annotation objects survive ordinary pickle use."""
    objects = [
        wfdb.rdheader("sample-data/100"),
        wfdb.rdrecord("sample-data/100", sampto=8),
        wfdb.rdann("sample-data/100", "atr", sampto=8),
    ]

    for original in objects:
        restored = pickle.loads(pickle.dumps(original))
        assert type(restored) is type(original)
        assert restored == original


def test_multirecord_pickle_preserves_loaded_segments():
    """MultiRecord serialization preserves its segment metadata and values."""
    original = wfdb.rdrecord(
        "sample-data/multi-segment/fixed1/v102s", sampto=8, m2s=False
    )
    restored = pickle.loads(pickle.dumps(original))

    assert isinstance(restored, wfdb.MultiRecord)
    assert restored.seg_name == original.seg_name
    assert restored.seg_len == original.seg_len
    assert len(restored.segments) == len(original.segments)
    for first, second in zip(restored.segments, original.segments):
        if second is None:
            assert first is None
        else:
            assert first == second
