"""Immutable descriptions of WFDB header files."""

import datetime
from dataclasses import dataclass

from wfdb.io import _header


@dataclass(frozen=True)
class SignalHeader:
    """Describe one signal line in a single-segment WFDB header.

    The values reflect WFDB's interpreted header defaults. They describe the
    stored signal, including its file layout and calibration, not a selected
    or aligned array returned by ``rdrecord``.
    """

    name: str | None
    file_name: str | None
    format: str | None
    samples_per_frame: int | None
    skew: int | None
    byte_offset: int | None
    adc_gain: float | None
    baseline: int | None
    units: str | None
    adc_res: int | None
    adc_zero: int | None
    init_value: int | None
    checksum: int | None
    block_size: int | None


@dataclass(frozen=True)
class SegmentHeader:
    """Describe one segment line in a multi-segment WFDB header."""

    name: str
    n_frames: int


@dataclass(frozen=True)
class _HeaderBase:
    """Hold the record-line fields shared by both header forms."""

    record_name: str
    n_signals: int
    fs: float | int
    n_frames: int | None
    counter_frequency: float | None
    base_counter: float | None
    base_time: datetime.time | None
    base_date: datetime.date | None
    comments: tuple[str, ...]

    @property
    def base_datetime(self) -> datetime.datetime | None:
        """Return the header's date and time when both are present."""
        if self.base_date is None or self.base_time is None:
            return None
        return datetime.datetime.combine(self.base_date, self.base_time)


@dataclass(frozen=True)
class SingleSegmentHeader(_HeaderBase):
    """Describe a header containing signal lines and no segment list."""

    signals: tuple[SignalHeader, ...]


@dataclass(frozen=True)
class MultiSegmentHeader(_HeaderBase):
    """Describe a header containing segment names and lengths.

    This describes the multi-segment header itself. Reading it does not load
    the referenced segment headers or their signal samples.
    """

    n_segments: int
    segments: tuple[SegmentHeader, ...]

    @property
    def layout(self) -> str | None:
        """Return ``"variable"`` or ``"fixed"`` when segments are listed."""
        if not self.segments:
            return None
        return "variable" if self.segments[0].n_frames == 0 else "fixed"


def _from_parsed_header(
    parsed: _header._ParsedSingleSegmentHeader | _header._ParsedMultiSegmentHeader,
) -> SingleSegmentHeader | MultiSegmentHeader:
    """Build a public immutable description from a parsed header."""
    record_line = parsed.record_fields
    common = dict(
        record_name=record_line.record_name,
        n_signals=record_line.n_sig,
        fs=record_line.fs,
        n_frames=record_line.sig_len,
        counter_frequency=record_line.counter_freq,
        base_counter=record_line.base_counter,
        base_time=record_line.base_time,
        base_date=record_line.base_date,
        comments=parsed.comments,
    )
    if isinstance(parsed, _header._ParsedSingleSegmentHeader):
        signals = tuple(
            SignalHeader(
                name=signal.sig_name,
                file_name=signal.file_name,
                format=signal.fmt,
                samples_per_frame=signal.samps_per_frame,
                skew=signal.skew,
                byte_offset=signal.byte_offset,
                adc_gain=signal.adc_gain,
                baseline=signal.baseline,
                units=signal.units,
                adc_res=signal.adc_res,
                adc_zero=signal.adc_zero,
                init_value=signal.init_value,
                checksum=signal.checksum,
                block_size=signal.block_size,
            )
            for signal in parsed.signals or ()
        )
        return SingleSegmentHeader(**common, signals=signals)

    segments = tuple(
        SegmentHeader(name=segment.seg_name, n_frames=segment.seg_len)
        for segment in parsed.segments
    )
    return MultiSegmentHeader(**common, n_segments=record_line.n_seg, segments=segments)
