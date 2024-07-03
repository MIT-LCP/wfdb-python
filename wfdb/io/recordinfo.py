from dataclasses import dataclass
import datetime
from typing import List, Optional, Union


@dataclass
class SignalInfo:
    """
    Signal specification fields for one signal
    """

    file_name: Optional[str] = None
    fmt: Optional[str] = None
    samps_per_frame: Optional[int] = None
    skew: Optional[int] = None
    byte_offset: Optional[int] = None
    adc_gain: Optional[float] = None
    baseline: Optional[int] = None
    units: Optional[str] = None
    adc_res: Optional[int] = None
    adc_zero: Optional[int] = None
    init_value: Optional[int] = None
    checksum: Optional[int] = None
    block_size: Optional[int] = None
    sig_name: Optional[str] = None


class SignalSet:
    """
    Wrapper for a set of signal information. Provides useful access/modify methods.
    """

    def __init__(self, signals: List[SignalInfo]):
        self._signal_info = signals
        try:
            self._generate_name_map()
        except ValueError:
            pass

    def _generate_name_map(self):
        """
        Generate mapping of channel names to channel indices to allow
        for access by both index and name.

        Raises
        ------
        ValueError
            Raises unless all channel names are present and unique.
        """
        self._channel_inds = None
        channel_inds = {}

        for ch, signal in enumerate(self._signal_info):
            sig_name = signal.sig_name
            if not sig_name or sig_name in channel_inds:
                raise ValueError(
                    "Cannot generate name map: channel names are not unique"
                )
            channel_inds[sig_name] = ch

        self._channel_inds = channel_inds

    def __getitem__(self, key: Union[int, str]):
        if isinstance(key, str):
            if not self._channel_inds:
                raise KeyError("Channel name mapping not available")

        return self._signal_info[key]


@dataclass
class SegmentFields:
    """
    Segment specification fields for one segment.
    """

    seg_name: Optional[str] = None
    seg_len: Optional[int] = None


@dataclass
class RecordInfo:
    """
    Encapsulates WFDB metadata for a single or multi-segment record.
    """

    record_name: Optional[str] = None
    n_seg: Optional[int] = None
    n_sig: Optional[int] = None
    fs: Optional[float] = None
    counter_freq: Optional[float] = None
    base_counter: Optional[float] = None
    sig_len: Optional[int] = None
    base_time: Optional[datetime.time] = None
    base_date: Optional[datetime.date] = None

    # All signal fields are encapsulated under this field
    signals: Optional[SignalSet] = None

    # Only present for multi-segment headers
    segments: List[SegmentFields] = None
    comments: List[str] = None

    @property
    def is_multi(self):
        return self.n_seg is not None

    @property
    def base_datetime(self):
        if self.base_date is None or self.base_time is None:
            return None
        else:
            return datetime.datetime.combine(
                date=self.base_date, time=self.base_time
            )
