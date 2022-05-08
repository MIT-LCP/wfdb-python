
from wfdb.io.record import rdheader


class RecordStream:
    """
    Class representing a WFDB record, with signal values that can be read/written as a stream.
    """

    def __init__(self, record_name: str):
        pass


def read_record_stream(record_name: str) -> RecordStream:

    # Read the header
    record = rdheader(record_name)


class Record:
    def __init__(self, metadata: Metadata):
        self.metadata = metadata
