import numpy as np


class Record:
    """
    Top-level class for users to interact with.

    """

    def __init__(self, info):
        self.info = info  # Contains all header fields.
        self._signal_handler = SignalHandler(
            info
        )  # Encapsulates signal read/write logic

    # Can define a bunch of properties for users to access record info fields directly.

    def open_signals(self, stream: bool = False):
        self._signal_handler.open(stream=stream)

    def read(self, samp_from: int, samp_to: int):
        return self._signal_handler.read(samp_from=samp_from, samp_to=samp_to)


class SignalHandler:
    """
    Class for reading, writing, and manipulating signal data.

    What is a handler anyway?

    class SignalData

    Question: Should the signals (aside from a buffer) ever be stored in the object? We could also keep it as an
    attribute in a SignalData object, and return it.

    - Is it useful for the program logic during reading/writing?
    - Is it useful for the user?

    If we don't have it as an object attribute, we no longer
    need to care about the attribute names.
    """

    def __init__(self, record_info):
        self.record_info = record_info
        self.mode = None  # Read or write
        self.stream = None  # Stream or not
        self.files = {}  # Dict of dat file pointers
        self.buffer = None  # Signal bytes buffer
        self.sample_buffer = None  # Digital samples buffer

        # Read options
        self.channels = None

    # TODO: Close when out of scope.
    def open(self, stream: bool = False):
        """
        Initializes reading
        """
        if self.mode is not None:
            raise Exception("Close the signals first")

        self.mode = "read"
        self.stream = stream

        # Is this necessary here?
        for file_name in self.record_info.file_name:
            if file_name not in self.files:
                self.files[file_name] = open(file_name, "rb")

    def close(self):
        self.mode = None
        self.stream = None
        self.files = {}
        self.buffer = None
        self.sample_buffer = None

    def read(
        self,
        samp_from: int,
        samp_to: int,
        smooth_frames: bool = False,
        return_dims: int = 2,
    ):
        self.open(stream=False)
        read_len = samp_to - samp_from
        # TODO: Selected channels

        if return_dims not in (1, 2):
            raise ValueError("return_dims must be 1 or 2")

        # Allocate memory
        if smooth_frames and return_dims == 2:
            signals = np.empty((read_len, self.info.n_sig))
        elif smooth_frames and return_dims == 1:
            signals = [np.empty((read_len,)) for _ in range(self.info.n_sig)]
        elif not smooth_frames and return_dims == 1:
            signals = [
                np.empty((read_len * self.record_info.samps_per_frame[ch],))
                for ch in range(self.info.n_sig)
            ]
        else:
            raise ValueError(
                "Illegal combination: smooth_frames == False and return_dims == 2"
            )

        return signals

    def write(self, signals, samp_from: int, samp_to: int):
        pass

    @classmethod
    def smooth_frames(expanded_signals, target_signals, ind_start: int):
        """
        Given a list of 1D Numpy arrays,

        Also requires the output signal array to avoid allocating
        intermediate twice.
        """

    def stream_read(self, samples: int):
        """


        Q: Should you be able to specify whether data is stored
        in the object?


        """
        # In streaming mode and non-streaming mode, data is stored in the object?

        # stream,
        # false, : output stored in attribute array
        # if not stream:

        #     # Allocate
        #     # I
        #     pass
        pass

    def stream_write(self, samples: int):
        pass


def open_record(record_name: str):
    """
    Returns a Record object containing the info read from the header file.

    Users can then use it to read/write the signals in the record.

    """
    info = read_header(record_name=record_name)
    record = Record(info=info)
    return record


def read_record_data(record_name: str, samp_from: int, samp_to: int):
    """
    The new rdsamp. No streaming. That would be too complicated.
    """
    record = open_record(record_name)  # Does NOT read the signals.
    signals = record.read(samp_from=samp_from, samp_to=samp_to)
    return record.info, signals


def example():
    record = open_record("recname")

    # Non-streaming mode
    signals = record.read(samp_from=100, samp_to=200)
    other_signals = record.read(samp_from=300, samp_to=400)

    # Streaming mode
    with record.open() as record_stream:
        record_stream.seek(samp_from=100)
        data = record_stream.stream_read(samples=100)
