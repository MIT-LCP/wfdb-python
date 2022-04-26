import numpy as np


_np_error_state = {}


def setup_module():
    # Raise exceptions for arithmetic errors, except underflow
    global _np_error_state
    _np_error_state = np.seterr()
    np.seterr("raise", under="ignore")


def teardown_module():
    # Restore original error handling state
    np.seterr(**_np_error_state)
