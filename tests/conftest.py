import pytest
import requests

# Errors that indicate that PhysioNet is unreachable or too slow (and not a test failure)
NETWORK_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.Timeout,
)


@pytest.fixture(scope="class")
def temp_path(tmp_path_factory):
    """Temporary directory (as a string) that is shared by all tests in a class."""
    return str(tmp_path_factory.mktemp("wfdb"))


@pytest.fixture(params=["local", pytest.param("physionet", marks=pytest.mark.network)])
def source(request):
    """Run a test with local files, and (as a "network" test) with PhysioNet."""
    return request.param


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Report tests marked as "network" as skipped if the network fails."""
    report = yield
    if (
        call.when == "call"
        and report.failed
        and call.excinfo is not None
        and call.excinfo.errisinstance(NETWORK_ERRORS)
        and item.get_closest_marker("network")
    ):
        report.outcome = "skipped"
        report.longrepr = (
            str(item.path),
            item.location[1] + 1,
            f"Skipped: network error: {call.excinfo.value}",
        )
    return report
