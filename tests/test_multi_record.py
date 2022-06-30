import wfdb


class TestMultiRecord:
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
