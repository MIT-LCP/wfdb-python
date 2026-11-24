"""Regression tests for FLAC seek failures in bundled libFLAC libraries."""

import pytest
import requests
import soundfile


def test_flac_read_ending_at_reported_sample():
    """Read the file from issue 486 through its formerly failing position."""
    # https://github.com/MIT-LCP/wfdb-python/issues/486
    with soundfile.SoundFile("sample-data/issue-486-flac.dat") as file:
        samples = file.read(78419)
        assert samples.shape == (78419, 3)
        assert file.tell() == 78419


@pytest.mark.network
def test_flac_seek_to_reported_sample(tmp_path):
    """Seek to the position reported in issue 488."""
    # https://github.com/MIT-LCP/wfdb-python/issues/488
    url = (
        "https://physionet.org/files/mimic4wdb/0.1.0/waves/"
        "p179/p17973277/83801243/83801243_0010r.dat"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    file_path = tmp_path / "83801243_0010r.dat"
    file_path.write_bytes(response.content)

    with soundfile.SoundFile(file_path) as file:
        assert file.seek(267184) == 267184
        assert file.read(1).shape == (1,)
