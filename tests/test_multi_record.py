import copy
import os
import stat

import numpy as np
import pytest

import wfdb


class TestMultiRecordRanges:
    """Test logic that deduces relevant segments/ranges for given signals."""

    def test_flattened_variable_ranges_include_clipped_empty_segment(self):
        """Preserve explicit gaps separately from missing channel samples."""
        path = "sample-data/multi-segment/s25047/s25047-2704-05-04-10-44"
        record = wfdb.rdrecord(path, sampfrom=1000, sampto=26000)

        assert record.source_segments == [
            ("~", 0, 24740),
            ("3234460_0001", 24740, 25000),
        ]
        assert np.isnan(record.p_signal[:24740]).all()

    def test_source_segments_are_named_tuples(self):
        """Ranges can be used by field name or unpacked as plain tuples."""
        path = "sample-data/multi-segment/s25047/s25047-2704-05-04-10-44"
        record = wfdb.rdrecord(path, sampfrom=1000, sampto=26000)

        gap, data = record.source_segments
        assert isinstance(gap, wfdb.io.SourceSegment)
        assert (gap.name, gap.start, gap.stop) == ("~", 0, 24740)
        name, start, stop = data
        assert (name, start, stop) == ("3234460_0001", 24740, 25000)
        assert np.isnan(record.p_signal[gap.start : gap.stop]).all()

    def test_flattened_range_inside_empty_segment(self):
        """A digital read retains the empty segment's clipped boundaries."""
        path = "sample-data/multi-segment/s25047/s25047-2704-05-04-10-44"
        record = wfdb.rdrecord(path, sampfrom=1000, sampto=2000, physical=False)

        assert record.source_segments == [("~", 0, 1000)]

    def test_flattened_fixed_ranges_cover_requested_window(self):
        """Fixed-layout segment ranges use the returned frame coordinates."""
        record = wfdb.rdrecord(
            "sample-data/multi-segment/041s/041s",
            sampfrom=500,
            sampto=1500,
            physical=False,
            smooth_frames=False,
        )

        assert record.source_segments == [
            ("041s01", 0, 500),
            ("041s02", 500, 1000),
        ]

    def test_single_segment_has_no_source_ranges(self):
        """Ordinary records have no multi-segment provenance."""
        record = wfdb.rdrecord("sample-data/100", sampto=100)

        assert record.source_segments is None

    def test_flattened_in_memory_record_uses_segment_names(self):
        """Direct conversion works when a MultiRecord has no seg_name list."""
        segment = wfdb.Record(
            record_name="segment",
            n_sig=1,
            fs=1,
            sig_len=2,
            p_signal=np.array([[1.0], [2.0]]),
            fmt=["16"],
            adc_gain=[1.0],
            baseline=[0],
            units=["mV"],
            sig_name=["I"],
            samps_per_frame=[1],
        )
        multi = wfdb.MultiRecord(
            segments=[segment], layout="fixed", n_sig=1, fs=1, sig_len=2
        )

        record = multi.multi_to_single(physical=True)

        assert record.source_segments == [("segment", 0, 2)]

    def test_contained_ranges_simple_cases(self):
        record = wfdb.MultiRecord(
            segments=[
                wfdb.Record(sig_name=["I", "II"], sig_len=5),
                wfdb.Record(sig_name=["I", "III"], sig_len=10),
            ],
        )

        assert record.contained_ranges("I") == [(0, 15)]
        assert record.contained_ranges("II") == [(0, 5)]
        assert record.contained_ranges("III") == [(5, 15)]

    def test_contained_ranges_variable_layout(self):
        record = wfdb.rdheader(
            "sample-data/multi-segment/s00001/s00001-2896-10-10-00-31",
            rd_segments=True,
        )

        assert record.contained_ranges("II") == [
            (3261, 10136),
            (4610865, 10370865),
            (10528365, 14518365),
        ]
        assert record.contained_ranges("V") == [
            (3261, 918261),
            (920865, 4438365),
            (4610865, 10370865),
            (10528365, 14518365),
        ]
        assert record.contained_ranges("MCL1") == [
            (10136, 918261),
            (920865, 4438365),
        ]
        assert record.contained_ranges("ABP") == [
            (14428365, 14450865),
            (14458365, 14495865),
        ]

    def test_contained_ranges_fixed_layout(self):
        record = wfdb.rdheader(
            "sample-data/multi-segment/041s/041s",
            rd_segments=True,
        )

        for sig_name in record.sig_name:
            assert record.contained_ranges(sig_name) == [(0, 2000)]

    def test_contained_combined_ranges_simple_cases(self):
        record = wfdb.MultiRecord(
            segments=[
                wfdb.Record(sig_name=["I", "II", "V"], sig_len=5),
                wfdb.Record(sig_name=["I", "III", "V"], sig_len=10),
                wfdb.Record(sig_name=["I", "II", "V"], sig_len=20),
            ],
        )

        assert record.contained_combined_ranges(["I", "II"]) == [
            (0, 5),
            (15, 35),
        ]
        assert record.contained_combined_ranges(["II", "III"]) == []
        assert record.contained_combined_ranges(["I", "III"]) == [(5, 15)]
        assert record.contained_combined_ranges(["I", "II", "V"]) == [
            (0, 5),
            (15, 35),
        ]

    def test_contained_combined_ranges_variable_layout(self):
        record = wfdb.rdheader(
            "sample-data/multi-segment/s00001/s00001-2896-10-10-00-31",
            rd_segments=True,
        )

        assert record.contained_combined_ranges(["II", "V"]) == [
            (3261, 10136),
            (4610865, 10370865),
            (10528365, 14518365),
        ]
        assert record.contained_combined_ranges(["II", "MCL1"]) == []
        assert record.contained_combined_ranges(["II", "ABP"]) == [
            (14428365, 14450865),
            (14458365, 14495865),
        ]
        assert record.contained_combined_ranges(["II", "V", "ABP"]) == [
            (14428365, 14450865),
            (14458365, 14495865),
        ]
        assert record.contained_combined_ranges(["II", "V", "MCL1", "ABP"]) == []

    def test_contained_combined_ranges_all_pairs(self):
        record = wfdb.rdheader(
            "sample-data/multi-segment/041s/041s",
            rd_segments=True,
        )

        for sig_1 in record.sig_name:
            for sig_2 in record.sig_name:
                if sig_1 == sig_2:
                    continue

                assert record.contained_combined_ranges([sig_1, sig_2]) == [(0, 2000)]


class TestMultiRecordFields:
    """Check the types used for multi-segment header fields."""

    def test_segment_fields_accept_lists(self):
        record = wfdb.MultiRecord(seg_name=["layout", "segment"], seg_len=[0, 100])

        record.check_field("seg_name")
        record.check_field("seg_len")

    @pytest.mark.parametrize(
        ("field", "value"),
        [("seg_name", "segment"), ("seg_len", 100)],
    )
    def test_segment_fields_require_lists(self, field, value):
        record = wfdb.MultiRecord(seg_name=["segment"], seg_len=[100])
        setattr(record, field, value)

        with pytest.raises(TypeError, match=f"Field `{field}` must be a list"):
            record.check_field(field)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("seg_name", ["layout", 1]), ("seg_len", [0, "100"])],
    )
    def test_segment_fields_check_list_elements(self, field, value):
        record = wfdb.MultiRecord(seg_name=["layout", "segment"], seg_len=[0, 100])
        setattr(record, field, value)

        with pytest.raises(TypeError, match=f"Channel 1 of field `{field}`"):
            record.check_field(field)


class TestMultiRecordWriting:
    """Test writing multi-segment records."""

    record_path = "sample-data/multi-segment/041s/041s"

    def test_round_trip(self, tmp_path):
        record = wfdb.rdrecord(self.record_path, m2s=False, physical=False)
        expected_signals = [segment.d_signal.copy() for segment in record.segments]

        record.wrsamp(write_dir=str(tmp_path))

        record_write = wfdb.rdrecord(str(tmp_path / "041s"), m2s=False, physical=False)
        assert record_write.seg_len == record.seg_len
        assert record_write.sig_name == record.sig_name
        for segment, expected_signal in zip(record_write.segments, expected_signals):
            np.testing.assert_array_equal(segment.d_signal, expected_signal)

    @pytest.mark.parametrize("physical", [True, False])
    @pytest.mark.parametrize("expanded", [True, False])
    def test_write_flattened_fixed_record(self, tmp_path, physical, expanded):
        """A fixed-layout record can be written after flattening."""
        record = wfdb.rdrecord(
            "sample-data/multi-segment/fixed1/v102s",
            sampfrom=74990,
            sampto=75010,
            physical=physical,
            smooth_frames=not expanded,
        )
        record.record_name = "flat"
        record.file_name = ["flat.dat"] * record.n_sig
        original = copy.deepcopy(record)

        record.wrsamp(write_dir=str(tmp_path), expanded=expanded)
        reread = wfdb.rdrecord(
            str(tmp_path / "flat"), physical=physical, smooth_frames=not expanded
        )

        if physical:
            field = "e_p_signal" if expanded else "p_signal"
        else:
            field = "e_d_signal" if expanded else "d_signal"
        np.testing.assert_equal(getattr(reread, field), getattr(record, field))
        assert record == original
        assert record.source_segments == original.source_segments

    @pytest.mark.parametrize("expanded", [True, False])
    def test_write_flattened_variable_record(self, tmp_path, expanded):
        """Missing channels and an empty segment survive physical writing."""
        path = "sample-data/multi-segment/p000878/p000878-2137-10-26-16-57"
        record = wfdb.rdrecord(
            path,
            sampfrom=3700,
            sampto=3800,
            smooth_frames=not expanded,
        )
        record.record_name = "flat"
        record.file_name = ["flat.dat"] * record.n_sig
        original_record = copy.deepcopy(record)

        record.wrsamp(write_dir=str(tmp_path), expanded=expanded)
        reread = wfdb.rdrecord(str(tmp_path / "flat"), smooth_frames=not expanded)

        assert reread.fmt == ["32"] * record.n_sig
        original = record.e_p_signal if expanded else record.p_signal.T
        written = reread.e_p_signal if expanded else reread.p_signal.T
        for expected, actual in zip(original, written):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
        assert record == original_record
        assert record.source_segments == original_record.source_segments

    @pytest.mark.parametrize("expanded", [True, False])
    def test_write_flattened_variable_digital_record(self, tmp_path, expanded):
        """Unknown-format missing samples get the new format's marker."""
        path = "sample-data/multi-segment/p000878/p000878-2137-10-26-16-57"
        record = wfdb.rdrecord(
            path,
            sampfrom=3700,
            sampto=3800,
            physical=False,
            smooth_frames=not expanded,
        )
        original = (
            record.e_d_signal[0].copy() if expanded else record.d_signal[:, 0].copy()
        )
        record.record_name = "flat"
        record.file_name = ["flat.dat"] * record.n_sig
        original_record = copy.deepcopy(record)

        record.wrsamp(write_dir=str(tmp_path), expanded=expanded)
        reread = wfdb.rdrecord(
            str(tmp_path / "flat"), physical=False, smooth_frames=not expanded
        )

        assert reread.fmt == ["32"] * record.n_sig
        output = reread.e_d_signal if expanded else reread.d_signal.T
        valid = original != -128
        np.testing.assert_array_equal(output[0][valid], original[valid])
        np.testing.assert_array_equal(output[0][~valid], -(2**31))
        for channel in output[1:]:
            np.testing.assert_array_equal(channel, -(2**31))
        assert record == original_record
        assert record.source_segments == original_record.source_segments

    def test_write_flattened_record_requires_resolved_units(self, tmp_path):
        """Do not silently label a channel whose source units conflict."""
        path = "sample-data/multi-segment/p000878/p000878-2137-10-26-16-57"
        record = wfdb.rdrecord(path, sampfrom=3700, sampto=3800)
        record.record_name = "flat"
        record.file_name = ["flat.dat"] * record.n_sig
        record.units = None

        with pytest.raises(ValueError, match="inconsistent units"):
            record.wrsamp(write_dir=str(tmp_path))

        assert list(tmp_path.iterdir()) == []

    def test_segment_length_mismatch_writes_no_files(self, tmp_path):
        record = wfdb.rdrecord(self.record_path, m2s=False, physical=False)
        record.seg_len[1] -= 1

        with pytest.raises(ValueError, match="signal length of segment 1"):
            record.wrsamp(write_dir=str(tmp_path))

        assert list(tmp_path.iterdir()) == []

    def test_short_seg_len_writes_no_files(self, tmp_path):
        record = wfdb.rdrecord(self.record_path, m2s=False, physical=False)
        record.seg_len.pop()

        with pytest.raises(
            ValueError, match="Length of seg_len must match the 'n_seg' field"
        ):
            record.wrsamp(write_dir=str(tmp_path))

        assert list(tmp_path.iterdir()) == []

    def test_outdated_expanded_signal_is_ignored_when_not_expanded(self, tmp_path):
        record = wfdb.rdrecord(self.record_path, m2s=False, physical=False)
        expected_signal = record.segments[0].d_signal.copy()
        record.segments[0].e_d_signal = [[] for _ in range(record.segments[0].n_sig)]

        record.wrsamp(write_dir=str(tmp_path), expanded=False)

        record_write = wfdb.rdrecord(str(tmp_path / "041s"), m2s=False, physical=False)
        np.testing.assert_array_equal(
            record_write.segments[0].d_signal, expected_signal
        )

    def test_unspecified_samps_per_frame_is_accepted_when_expanded(self, tmp_path):
        record = wfdb.rdrecord(
            self.record_path,
            m2s=False,
            physical=False,
            smooth_frames=False,
            channels=[6],
        )
        record.segments[0].samps_per_frame[0] = None

        record.wrsamp(write_dir=str(tmp_path), expanded=True)

        record_write = wfdb.rdrecord(
            str(tmp_path / "041s"),
            m2s=False,
            physical=False,
            smooth_frames=False,
        )
        assert record_write.segments[0].samps_per_frame == [1]

    @pytest.mark.parametrize("expanded", [False, True])
    def test_invalid_later_segment_writes_no_files(self, tmp_path, expanded):
        """A segment that fails to be written leaves no partial record (#557)."""
        record = wfdb.rdrecord(
            self.record_path, m2s=False, physical=False, smooth_frames=not expanded
        )
        segment = record.segments[1]
        if expanded:
            segment.e_d_signal[0] = segment.e_d_signal[0][:-3]
        else:
            segment.fmt = ["999"] * segment.n_sig

        with pytest.raises(ValueError):
            record.wrsamp(write_dir=str(tmp_path), expanded=expanded)

        assert list(tmp_path.iterdir()) == []

    def test_segment_without_samps_per_frame_when_expanded(self, tmp_path):
        record = wfdb.rdrecord(
            self.record_path, m2s=False, physical=False, smooth_frames=False
        )
        record.segments[1].samps_per_frame = None

        with pytest.raises(ValueError, match=r"Segment 1 \(041s02\) has no samps"):
            record.wrsamp(write_dir=str(tmp_path), expanded=True)

        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("expanded", [False, True])
    def test_segment_without_a_signal(self, tmp_path, expanded):
        record = wfdb.rdrecord(
            self.record_path, m2s=False, physical=False, smooth_frames=not expanded
        )
        segment = record.segments[1]
        segment.d_signal = segment.p_signal = None
        segment.e_d_signal = segment.e_p_signal = None

        with pytest.raises(ValueError, match=r"Segment 1 \(041s02\) has no"):
            record.wrsamp(write_dir=str(tmp_path), expanded=expanded)

        assert list(tmp_path.iterdir()) == []

    def test_write_to_the_current_directory(self, tmp_path, monkeypatch):
        """The default write_dir is the current directory, with no files left over."""
        record = wfdb.rdrecord(self.record_path, m2s=False, physical=False)
        monkeypatch.chdir(tmp_path)

        record.wrsamp()

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "041s.hea",
            "041s01.dat",
            "041s01.hea",
            "041s02.dat",
            "041s02.hea",
        ]

    def test_archive_path_is_relative_to_write_dir(self, tmp_path, monkeypatch):
        record = wfdb.rdrecord(self.record_path, m2s=False, physical=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "out").mkdir()

        record.wrsamp(write_dir="out", wfdb_archive="041s.wfdb")

        assert [p.name for p in (tmp_path / "out").iterdir()] == ["041s.wfdb"]
        assert not (tmp_path / "041s.wfdb").exists()
        assert wfdb.rdrecord("out/041s.wfdb").sig_len == record.sig_len

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
    def test_overwriting_keeps_file_permissions(self, tmp_path):
        record = wfdb.rdrecord(self.record_path, m2s=False, physical=False)
        record.wrsamp(write_dir=str(tmp_path))
        (tmp_path / "041s01.dat").chmod(0o600)

        record.wrsamp(write_dir=str(tmp_path))

        mode = stat.S_IMODE((tmp_path / "041s01.dat").stat().st_mode)
        assert mode == 0o600

    @staticmethod
    def fail_second_move(monkeypatch, target):
        """Make the second move of a new file into `target` fail."""
        real_replace = os.replace
        moves = []

        def replace(src, dst):
            into_target = os.path.dirname(os.path.abspath(dst)) == str(target)
            if into_target and ".replaced" not in str(src):
                moves.append(dst)
                if len(moves) == 2:
                    raise OSError("disk full")
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", replace)

    def test_failed_move_leaves_no_files(self, tmp_path, monkeypatch):
        record = wfdb.rdrecord(self.record_path, m2s=False, physical=False)
        self.fail_second_move(monkeypatch, tmp_path)

        with pytest.raises(OSError, match="disk full"):
            record.wrsamp(write_dir=str(tmp_path))

        assert list(tmp_path.iterdir()) == []

    def test_failed_move_restores_the_record_it_replaces(self, tmp_path, monkeypatch):
        record = wfdb.rdrecord(self.record_path, m2s=False, physical=False)
        record.wrsamp(write_dir=str(tmp_path))
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        for segment in record.segments:
            segment.d_signal = segment.d_signal // 2
        self.fail_second_move(monkeypatch, tmp_path)

        with pytest.raises(OSError, match="disk full"):
            record.wrsamp(write_dir=str(tmp_path))

        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
