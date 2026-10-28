import pytest


@pytest.fixture(scope="class")
def temp_path(tmp_path_factory):
    """Temporary directory (as a string) that is shared by all tests in a class."""
    return str(tmp_path_factory.mktemp("wfdb"))
