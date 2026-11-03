import numpy as np
import pytest

import wfdb


class TestMultiRecordRanges:
    """Test logic that deduces relevant segments/ranges for given signals."""

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

    def test_segment_length_mismatch_writes_no_files(self, tmp_path):
        record = wfdb.rdrecord(self.record_path, m2s=False, physical=False)
        record.seg_len[1] -= 1

        with pytest.raises(ValueError, match="signal length of segment 1"):
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
