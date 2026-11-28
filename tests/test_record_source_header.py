"""Keep source metadata separate from selected single-segment records."""

import numpy as np

from wfdb.io import record


def test_selected_records_do_not_change_the_source_header():
    """Legacy edits to one loaded view leave source fields and other views alone."""
    source = record._read_header_fields("sample-data/100", pn_dir=None)

    def read_channel(channel):
        result = record._SingleSegmentReadResult(
            source_header=source,
            values=[np.array([1, 2, 3], dtype=np.int16)],
            channels=(channel,),
            sampfrom=2,
            physical=False,
            smooth_frames=True,
            ignore_skew=False,
            return_res=64,
        )
        return record._record_from_single_segment_read(result)

    first = read_channel(1)
    first.sig_name[0] = "changed"
    second = read_channel(0)

    assert source.record_fields["sig_len"] == 650000
    assert source.signal_fields["sig_name"] == ["MLII", "V5"]
    assert (first.sig_len, first.n_sig) == (3, 1)
    assert second.sig_name == ["MLII"]
    np.testing.assert_array_equal(second.d_signal[:, 0], [1, 2, 3])
