import pytest
import requests

from wfdb.io._url import NetFileError

# Errors that mean PhysioNet is unreachable or too slow (not that a test failed)
NETWORK_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.Timeout,
)


def is_network_error(exc):
    """Whether an exception means that PhysioNet is not available right now."""
    if isinstance(exc, NETWORK_ERRORS):
        return True
    # Server errors (for example 502 Bad Gateway) and rate limiting, but not errors
    # like 404 or 403 which indicate a real problem
    return (
        isinstance(exc, NetFileError)
        and exc.status_code is not None
        and (exc.status_code >= 500 or exc.status_code == 429)
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
        and is_network_error(call.excinfo.value)
        and item.get_closest_marker("network")
    ):
        report.outcome = "skipped"
        report.longrepr = (
            str(item.path),
            item.location[1] + 1,
            f"Skipped: network error: {call.excinfo.value}",
        )
    return report
