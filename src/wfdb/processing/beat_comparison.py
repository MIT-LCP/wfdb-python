"""Beat-by-beat comparison of annotation files, as done by the WFDB bxb program.

.. warning::

    This is a development feature. Its interface may change, and its results
    should be checked against the WFDB ``bxb`` program before they are used
    for reporting.

The comparison follows the beat-by-beat method of ANSI/AAMI EC57 (and
EC38), as implemented by ``bxb`` in the WFDB Software Package. This code was
written from the descriptions in those standards and in the ``bxb``
documentation, and by comparing its results with those of ``bxb`` on many
generated and real annotation files. It does not use or derive from the
``bxb`` source code, which is licensed under the GPL.

"""

import bisect
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["BxbResult", "bxb", "bxb_record", "sumstats"]

# AAMI beat classes of the WFDB annotation codes (label_store values) that
# are beats. Other codes are not beats.
_AAMI_CLASSES = {
    "N": (1, 2, 3, 25),  # NORMAL, LBBB, RBBB, BBB
    "S": (4, 7, 8, 9, 11, 34, 35),  # ABERR, NPC, APC, SVPB, NESC, AESC, SVESC
    "V": (5, 10, 41),  # PVC, VESC, RONT
    "F": (6,),  # FUSION
    "Q": (12, 13, 30, 38),  # PACE, UNKNOWN, LEARN, PFUS
}
_CLASS_OF_CODE = {code: c for c, codes in _AAMI_CLASSES.items() for code in codes}
_NOISE, _VFON, _VFOFF = 14, 32, 33

# Rows are reference classes (O: no reference beat; X: no reference beat,
# during a reference shutdown), and columns are test classes (o: no test
# beat; x: no test beat, during a test shutdown)
ROWS = "NSVFQOX"
COLUMNS = "nsvfqox"


@dataclass(frozen=True)
class _Beat:
    time: int
    cls: str
    index: int  # position in the annotation file


@dataclass(frozen=True)
class _Interval:
    start: float
    end: float
    index: int  # position of the annotation that starts it
    resets: bool = True  # whether it resets the previous beat for RR intervals


def _beats_and_intervals(sample, label_store, subtype, window):
    """Read the beats, VF intervals and shutdown intervals of one file.

    Beats between a VF onset (``[``) and offset (``]``), in file order, are
    not beats. A shutdown starts at a NOISE annotation whose subtype has both
    the 16 and 32 bits set, and ends at the next NOISE annotation. If the
    next annotation is not a NOISE annotation, the shutdown marks the middle
    of an unreadable interval (the AHA database convention), which is taken
    to start `window` after the previous annotation and to end `window`
    before the next one.

    """
    beats, vf, shutdowns = [], [], []
    in_vf, vf_start = False, None
    n = len(sample)
    for i in range(n):
        code, t = int(label_store[i]), int(sample[i])
        if code == _VFON:
            if not in_vf:
                in_vf, vf_start = True, i
        elif code == _VFOFF:
            if in_vf:
                vf.append(_Interval(int(sample[vf_start]), t, vf_start))
                in_vf = False
        elif code in _CLASS_OF_CODE and not in_vf:
            beats.append(_Beat(t, _CLASS_OF_CODE[code], i))
        if code == _NOISE and (int(subtype[i]) & 48) == 48:
            if (
                i + 1 < n
                and int(label_store[i + 1]) == _NOISE
                and (int(subtype[i + 1]) & 48) != 48
            ):
                shutdowns.append(_Interval(t, int(sample[i + 1]), i))
            elif i + 1 < n:
                # (taking the previous annotation to be at 0 if there is none)
                start = (int(sample[i - 1]) if i > 0 else 0) + window
                end = int(sample[i + 1]) - window
                # The previous beat is forgotten only if the next annotation
                # comes after the start of the shutdown
                shutdowns.append(
                    _Interval(start, max(start, end), i, int(sample[i + 1]) > start)
                )
            else:
                shutdowns.append(_Interval(t, math.inf, i))
    if in_vf:
        vf.append(_Interval(int(sample[vf_start]), math.inf, vf_start))
    # A shutdown that starts during VF in the same file is ignored
    vf_index = _Intervals(vf)
    shutdowns = [s for s in shutdowns if not vf_index.contains(s.start)]
    return beats, vf, shutdowns


class _Intervals:
    """A set of intervals, for finding quickly whether a time is in one.

    The intervals are sorted by start, with the latest end of those starting
    at or before each one, so that a lookup takes logarithmic time even if
    the intervals overlap.

    """

    def __init__(self, intervals):
        intervals = sorted(intervals, key=lambda iv: iv.start)
        self.starts = [iv.start for iv in intervals]
        self.max_ends = list(np.maximum.accumulate([iv.end for iv in intervals]))

    def contains(self, t, closed=False):
        """Whether `t` is in one of the intervals (with its end, if `closed`)."""
        k = bisect.bisect_right(self.starts, t)
        if k == 0:
            return False
        return self.max_ends[k - 1] >= t if closed else self.max_ends[k - 1] > t


def _label_store(annotation):
    """The annotation codes of an Annotation, from its symbols if needed."""
    if annotation.label_store is not None:
        return np.asarray(annotation.label_store)
    codes = annotation.convert_label_attribute("symbol", "label_store", inplace=False)
    return np.asarray(codes)


@dataclass
class BxbResult:
    """The result of a beat-by-beat comparison of two annotation files.

    Attributes
    ----------
    counts : dict
        For each row of the comparison matrix (reference class: N, S, V, F,
        Q, or O and X for test beats with no reference beat, outside and
        during a reference shutdown), the counts in each column (test class:
        n, s, v, f, q, or o and x for reference beats with no test beat,
        outside and during a test shutdown). Rows O and X have no o and x
        columns.
    rr_errors : list of float
        The differences between the reference and test RR intervals, in
        samples, that are used for the RMS RR interval error.
    shutdown : float
        The total time in test shutdown, in samples.
    fs : float
        The sampling frequency.
    record_name : str
        The name of the record, used in reports.
    reference : str
        The name of the reference annotator, used in reports.
    test : str
        The name of the test annotator, used in reports.

    """

    counts: dict
    rr_errors: list
    shutdown: float
    fs: float
    record_name: str = ""
    reference: str = ""
    test: str = ""
    _events: list = field(default_factory=list, repr=False)

    def count(self, rows, columns):
        """The sum of the counts in the given rows and columns."""
        total = 0
        for r in rows:
            for c in columns:
                j = COLUMNS.index(c)
                if j < len(self.counts[r]):
                    total += self.counts[r][j]
        return total

    @property
    def matrix(self):
        """The comparison matrix, as a DataFrame (rows O and X have no o, x)."""
        data = {
            c: [self.counts[r][j] if j < len(self.counts[r]) else pd.NA for r in ROWS]
            for j, c in enumerate(COLUMNS)
        }
        return pd.DataFrame(data, index=list(ROWS), dtype="Int64")

    def statistics(self):
        """The statistics of the standard report, as (numerator, denominator).

        Returns
        -------
        dict
            The QRS, VEB and SVEB sensitivity and positive predictivity, and
            the beats missed in shutdown, each as a (numerator, denominator)
            pair of beat counts.

        """
        g = self.count
        tp = g("NSVFQ", "nsvfq")
        return {
            "QRS sensitivity": (tp, tp + g("NSVFQ", "ox")),
            "QRS positive predictivity": (tp, tp + g("OX", "nsvfq")),
            "VEB sensitivity": (g("V", "v"), g("V", COLUMNS)),
            "VEB positive predictivity": (g("V", "v"), g("NSVOX", "v")),
            "SVEB sensitivity": (g("S", "s"), g("S", COLUMNS)),
            "SVEB positive predictivity": (g("S", "s"), g("NSVFOX", "s")),
            "Beats missed in shutdown": (g("NVFQ", "x"), g("NSVFQ", COLUMNS)),
            "N missed in shutdown": (g("N", "x"), g("N", COLUMNS)),
            "S missed in shutdown": (g("S", "x"), g("S", COLUMNS)),
            "V missed in shutdown": (g("V", "x"), g("V", COLUMNS)),
            "F missed in shutdown": (g("F", "x"), g("F", COLUMNS)),
        }

    @property
    def rms_rr_error(self):
        """The RMS RR interval error in milliseconds, or None if undefined."""
        if not self.rr_errors:
            return None
        ms = [e / self.fs * 1000 for e in self.rr_errors]
        return math.sqrt(sum(x * x for x in ms) / len(ms))

    @property
    def shutdown_seconds(self):
        """The total shutdown time in whole seconds, as bxb reports it."""
        return int((round(self.shutdown) + math.ceil(self.fs / 2)) // self.fs)

    def report(self):
        """The standard report, in the format of ``bxb -S``.

        Returns
        -------
        str
            The comparison matrix and statistics.

        """
        lines = [
            f"Beat-by-beat comparison results for record {self.record_name}",
            f"Reference annotator: {self.reference}",
            f"     Test annotator: {self.test}",
            "",
            "               Algorithm",
            "        n    s    v    f    q    o    x",
            "   ____________________________________",
        ]
        for r in ROWS:
            lines.append(f" {r} |" + "".join(f" {x:4d}" for x in self.counts[r]))
        lines.append("")
        stats = self.statistics()

        def stat(label, key):
            num, den = stats[key]
            if den:
                return f"{label:>26}: {100 * num / den:6.2f}% ({num}/{den})"
            return f"{label:>26}:      -  ({num}/{den})"

        for key in (
            "QRS sensitivity",
            "QRS positive predictivity",
            "VEB sensitivity",
            "VEB positive predictivity",
            "SVEB sensitivity",
            "SVEB positive predictivity",
        ):
            lines.append(stat(key, key))
        rr = self.rms_rr_error
        lines.append(
            f"{'RMS RR interval error':>26}: "
            + (f"{rr:6.2f} ms" if rr is not None else "     -")
        )
        lines.append("")
        for key in (
            "Beats missed in shutdown",
            "N missed in shutdown",
            "S missed in shutdown",
            "V missed in shutdown",
            "F missed in shutdown",
        ):
            lines.append(stat(key, key))
        lines.append(f"{'Total shutdown time':>26}: {self.shutdown_seconds:5d} seconds")
        return "\n".join(lines) + "\n"

    def line_columns(self):
        """The beat counts of the line-format report, as in ``bxb -L``.

        Returns
        -------
        dict
            The counts named as in the report's heading: the primed columns
            combine F with Q, O with X, and o with x.

        """
        g = self.count
        return {
            "Nn'": g("N", "nfq"),
            "Sn'": g("S", "nfq"),
            "Vn'": g("V", "nfq"),
            "Fn'": g("FQ", "nfq"),
            "On'": g("OX", "nfq"),
            "Ns": g("N", "s"),
            "Ss": g("S", "s"),
            "Vs": g("V", "s"),
            "Fs'": g("FQ", "s"),
            "Os'": g("OX", "s"),
            "Nv": g("N", "v"),
            "Sv": g("S", "v"),
            "Vv": g("V", "v"),
            "Fv'": g("FQ", "v"),
            "Ov'": g("OX", "v"),
            "No'": g("N", "ox"),
            "So'": g("S", "ox"),
            "Vo'": g("V", "ox"),
            "Fo'": g("FQ", "ox"),
        }

    def lines(self):
        """The two lines of the line-format reports, as in ``bxb -L``.

        Returns
        -------
        tuple of str
            The line of beat counts and statistics (for the first file,
            often named ``bxb.out``), and the line of shutdown statistics
            (for the second, ``sd.out``). The column headings of each file
            are in ``wfdb.processing.beat_comparison.LINE_HEADINGS``.

        """
        cols = self.line_columns()
        return _format_line1(
            self.record_name, cols, self._line_stats(), self.rms_rr_error
        ), (_format_line2(self.record_name, self))

    def _line_stats(self):
        """The six statistics of the first line-format report."""
        stats = self.statistics()
        return [
            stats[key]
            for key in (
                "QRS sensitivity",
                "QRS positive predictivity",
                "VEB sensitivity",
                "VEB positive predictivity",
                "SVEB sensitivity",
                "SVEB positive predictivity",
            )
        ]


LINE_HEADINGS = (
    "Record Nn' Sn' Vn' Fn' On'  Ns  Ss  Vs  Fs' Os' Nv  Sv   Vv  Fv' Ov' No' "
    "So' Vo' Fo'  Q Se   Q +P   V Se   V +P   S Se   S +P RR err\n",
    "Record Nx   Sx   Vx   Fx   Qx  % beats  % N    % S    % V    % F   Total "
    "Shutdown\n                                missed missed missed missed "
    "missed      Time\n",
)


def _pct(num, den, width=7):
    """A percentage in a column of `width` (a space and the number), or "-"."""
    return f" {100 * num / den:{width - 1}.2f}" if den else f" {'-':>{width - 1}}"


def _gross_stats(c):
    """The gross statistics of sumstats, from the summed line-format counts."""
    tp = sum(
        c[k]
        for k in (
            "Nn'",
            "Sn'",
            "Vn'",
            "Fn'",
            "Ns",
            "Ss",
            "Vs",
            "Fs'",
            "Nv",
            "Sv",
            "Vv",
            "Fv'",
        )
    )
    fn = c["No'"] + c["So'"] + c["Vo'"] + c["Fo'"]
    fp = c["On'"] + c["Os'"] + c["Ov'"]
    veb = c["Vn'"] + c["Vs"] + c["Vv"] + c["Vo'"]
    sveb = c["Sn'"] + c["Ss"] + c["Sv"] + c["So'"]
    return [
        (tp, tp + fn),
        (tp, tp + fp),
        (c["Vv"], veb),
        (c["Vv"], c["Nv"] + c["Sv"] + c["Vv"] + c["Ov'"]),
        (c["Ss"], sveb),
        # F and Q beats are combined in the line format, so unlike each
        # record's statistic, this leaves out F beats called S
        (c["Ss"], c["Ns"] + c["Ss"] + c["Vs"] + c["Os'"]),
    ]


def _counts_text(c):
    widths = [6] + [4] * 11 + [5] + [4] * 6
    return "".join(f" {c[k]:{w - 1}d}" for k, w in zip(_LINE1_KEYS, widths))


_LINE1_KEYS = (
    "Nn'",
    "Sn'",
    "Vn'",
    "Fn'",
    "On'",
    "Ns",
    "Ss",
    "Vs",
    "Fs'",
    "Os'",
    "Nv",
    "Sv",
    "Vv",
    "Fv'",
    "Ov'",
    "No'",
    "So'",
    "Vo'",
    "Fo'",
)


def _format_line1(record_name, c, stats, rr):
    stats = "".join(_pct(n, d) for n, d in stats)
    rr_text = f" {rr:6.2f}" if rr is not None else f" {'-':>5}"
    return f"{record_name:>4}" + _counts_text(c) + stats + rr_text


def _format_line2(record_name, result):
    g = result.count
    x = [g(r, "x") for r in "NSVFQ"]
    totals = [g(r, COLUMNS) for r in "NSVFQ"]
    missed = _pct(g("NVFQ", "x"), sum(totals), 9)
    per_class = "".join(_pct(x[k], totals[k]) for k in range(4))
    return (
        f"{record_name:>4}"
        + "".join(f" {v:4d}" for v in x)
        + missed
        + per_class
        + f" {result.shutdown_seconds:4d} seconds "
        + " ".join(str(t) for t in totals)
    )


def bxb(
    reference,
    test,
    fs=None,
    sig_len=None,
    start=None,
    stop=None,
    window=0.15,
    record_name=None,
):
    """Compare two annotation files beat by beat, as the WFDB bxb program does.

    .. warning::

        This is a development feature. It was written without reference
        to the ``bxb`` source code (see ``wfdb.processing.beat_comparison``),
        its interface may change, and its results should be checked against
        ``bxb`` before they are used for reporting.

    The reference and test beats are matched if they are within `window` of
    each other, and tallied by their ANSI/AAMI EC57 classes (N, S, V, F and
    Q). Beats in ventricular flutter or fibrillation (between ``[`` and
    ``]`` annotations) and test shutdowns (marked by NOISE annotations) are
    handled as by ``bxb``.

    Parameters
    ----------
    reference : Annotation
        The reference annotations.
    test : Annotation
        The test annotations.
    fs : float, optional
        The sampling frequency. By default, the reference annotations' `fs`.
    sig_len : int, optional
        The length of the record, in samples. If given, and `stop` is not,
        the comparison ends at the end of the record. Otherwise it ends at
        the last reference beat.
    start : int, optional
        The sample at which to start the comparison. By default, 5 minutes
        after the start of the record, as EC57 requires.
    stop : int, optional
        The sample at which to stop the comparison (inclusive). If 0, the
        comparison stops at the last beat of whichever file ends first, as
        with ``bxb -t 0``. By default, it stops at the end of the record if
        `sig_len` is given, and at the last reference beat otherwise.
    window : float, optional
        The match window, in seconds. The default, 0.15, is the one that
        EC57 requires.
    record_name : str, optional
        The record name to use in reports. By default, that of `reference`.

    Returns
    -------
    BxbResult
        The comparison matrix and the data for its statistics.

    Examples
    --------
    >>> ref = wfdb.rdann('sample-data/100', 'atr')
    >>> test = wfdb.rdann('sample-data/100', 'qrs')
    >>> result = wfdb.processing.bxb(ref, test, sig_len=650000)
    >>> print(result.report())

    """
    fs = fs if fs is not None else reference.fs
    if not fs:
        raise ValueError("The sampling frequency is needed: pass fs")
    w = int(window * fs + 0.5)
    start = int(300 * fs + 0.5) if start is None else int(start)
    files, samples = [], []
    for ann in (reference, test):
        sample = np.asarray(ann.sample)
        # Annotation times are converted to the sampling frequency of the
        # comparison, as the WFDB library does, if the file has another one
        if ann.fs and ann.fs != fs:
            sample = np.floor(sample * (fs / ann.fs) + 0.5).astype(np.int64)
        samples.append(sample)
        codes = _label_store(ann)
        subtype = (
            np.zeros(len(sample), dtype=int)
            if ann.subtype is None
            else np.asarray(ann.subtype)
        )
        files.append(_beats_and_intervals(sample, codes, subtype, w))
    (R, rvf, rshut), (A, tvf, tshut) = files
    whole_files = stop == 0
    end_known = bool(sig_len) or stop is not None
    rvf_index, rshut_index, tshut_index = (
        _Intervals(rvf),
        _Intervals(rshut),
        _Intervals(tshut),
    )
    if stop is None:
        if sig_len:
            stop = int(sig_len)
        else:
            # Without a record length, the comparison ends at the last
            # reference beat
            stop = R[-1].time if R else 0
    elif stop == 0:
        # As with bxb -t 0, the comparison ends when either file has no more
        # beats
        stop = min(R[-1].time, A[-1].time) if R and A else 0
    else:
        stop = int(stop)

    events = _match(R, A, w)
    counts = {r: [0] * (5 if r in "OX" else 7) for r in ROWS}
    rr_errors = []

    # Each side forgets its previous beat (for RR intervals) at the start of
    # a shutdown or VF interval in its own file
    r_resets = sorted(
        [(iv.start, iv.index) for iv in rshut if iv.resets]
        + [(iv.start, iv.index) for iv in rvf]
    )
    a_resets = sorted(
        [(iv.start, iv.index) for iv in tshut if iv.resets]
        + [(iv.start, iv.index) for iv in tvf]
    )

    # The first test beat after the start is not counted if it is within the
    # match window of the start, matches no reference beat, and the next test
    # beat is closer than it to the reference beat it was compared with
    k0 = bisect.bisect_left([a.time for a in A], start)
    skipped = None
    if k0 + 1 < len(A) and A[k0].time - start <= w:
        for kind, _, a, other in events:
            if a is A[k0]:
                if (
                    kind == "fp"
                    and other is not None
                    and abs(A[k0 + 1].time - other.time) < abs(a.time - other.time)
                ):
                    skipped = a
                break

    prev_r = prev_a = None
    ri = ai = 0
    counted_events = []
    for kind, r, a, other in events:
        t = r.time if r else a.time
        while ri < len(r_resets) and _before(r_resets[ri], r, t):
            prev_r = None
            ri += 1
        while ai < len(a_resets) and _before(a_resets[ai], a, t):
            prev_a = None
            ai += 1
        if kind == "match":
            counted = start <= r.time and min(r.time, a.time) <= stop
        elif kind == "fn":
            counted = start <= r.time <= stop
        else:
            # A test beat during reference VF is not counted (but is still
            # the previous test beat)
            counted = (
                start <= a.time <= stop
                and a is not skipped
                and not rvf_index.contains(a.time, closed=True)
            )
        if counted:
            counted_events.append((kind, r, a))
            if kind == "match":
                counts[r.cls][COLUMNS.index(a.cls.lower())] += 1
            elif kind == "fn":
                counts[r.cls][
                    6 if tshut_index.contains(r.time, closed=True) else 5
                ] += 1
            else:
                row = "X" if rshut_index.contains(a.time, closed=True) else "O"
                counts[row]["nsvfq".index(a.cls.lower())] += 1
            if prev_r is not None and prev_a is not None:
                # For an unmatched beat, the other side's interval is the one
                # ending at the beat it was compared with
                nr = r.time if r else _next(other, r_resets, ri)
                na = a.time if a else _next(other, a_resets, ai)
                if nr is not None and na is not None and nr > prev_r and na > prev_a:
                    rr_errors.append((nr - prev_r) - (na - prev_a))
        if r:
            prev_r = r.time
        if a:
            prev_a = a.time

    # Test annotations are read until the first reference beat after the end
    # of the comparison, so a shutdown that starts later is not counted
    rt = [b.time for b in R]
    k = bisect.bisect_right(rt, stop)
    cutoff = rt[k] if k < len(rt) else math.inf
    shutdown = sum(
        # A shutdown with no annotation after it lasts until the end of the
        # comparison, if that is known (from the record length or `stop`)
        (
            max(0, iv.end - iv.start)
            if iv.end != math.inf
            else (max(0, stop - iv.start) if end_known else 0)
        )
        for iv in tshut
        # With stop=0, the test file is read to its end
        if whole_files
        or iv.start <= stop
        or int(samples[1][iv.index]) <= stop
        or int(samples[1][iv.index]) < cutoff
    )
    return BxbResult(
        counts=counts,
        rr_errors=rr_errors,
        shutdown=shutdown,
        fs=fs,
        record_name=record_name
        if record_name is not None
        else (reference.record_name or ""),
        reference=reference.extension or "",
        test=test.extension or "",
        _events=counted_events,
    )


def _match(R, A, w):
    """Match reference and test beats, in time order.

    Returns a list of (kind, reference beat, test beat, other) events, where
    kind is "match", "fn" (a reference beat with no test beat) or "fp" (a
    test beat with no reference beat), and `other` is the beat on the other
    side that an unmatched beat was compared with.

    """
    events = []
    i = j = 0
    while i < len(R) or j < len(A):
        cur_r = R[i] if i < len(R) else None
        cur_a = A[j] if j < len(A) else None
        if cur_a is None or (cur_r is not None and cur_a.time > cur_r.time + w):
            events.append(("fn", cur_r, None, cur_a))
            i += 1
            continue
        if cur_r is None or cur_a.time < cur_r.time - w:
            events.append(("fp", None, cur_a, cur_r))
            j += 1
            continue
        r, a = cur_r, cur_a
        # The next test beat is preferred if it is closer to the reference
        # beat, or as close and this test beat comes before it
        if j + 1 < len(A):
            d, d_next = abs(a.time - r.time), abs(A[j + 1].time - r.time)
            if d_next < d or (d_next == d and a.time < r.time):
                events.append(("fp", None, a, r))
                j += 1
                continue
        # The next reference beat takes this test beat if it is closer to
        # it, unless the next test beat is closer still to that reference beat
        if i + 1 < len(R) and abs(R[i + 1].time - a.time) < abs(r.time - a.time):
            if not (
                j + 1 < len(A)
                and abs(A[j + 1].time - R[i + 1].time) < abs(a.time - R[i + 1].time)
            ):
                events.append(("fn", r, None, a))
                i += 1
                continue
        events.append(("match", r, a, None))
        i += 1
        j += 1
    return events


def _before(reset, beat, t):
    """Whether a reset comes before a side's beat, or by time `t` if it has none.

    Resets and beats are ordered by time, and by file order at the same time.

    """
    time, index = reset
    if beat is not None:
        return (time, index) < (beat.time, beat.index)
    return time <= t


def _next(beat, resets, ri):
    """The time of `beat`, unless the next of `resets` comes before it."""
    if beat is None:
        return None
    if ri < len(resets) and resets[ri] < (beat.time, beat.index):
        return None
    return beat.time


def bxb_record(
    record_name,
    reference_extension,
    test_extension,
    pn_dir=None,
    start=None,
    stop=None,
    window=0.15,
):
    """Compare two annotation files of a record beat by beat, as bxb does.

    .. warning::

        This is a development feature: see `bxb`.

    The sampling frequency and length of the record are read from its
    header, as by ``bxb -r record -a reference test``.

    Parameters
    ----------
    record_name : str
        The name of the record, which may be a path or a URI.
    reference_extension : str
        The annotator of the reference annotations, such as 'atr'.
    test_extension : str
        The annotator of the test annotations.
    pn_dir : str, optional
        The PhysioNet database directory of the record, if it is on PhysioNet.
    start : int, optional
        The sample at which to start the comparison (by default, 5 minutes).
    stop : int, optional
        The sample at which to stop the comparison (by default, the end of
        the record). If 0, it stops when either file has no more beats.
    window : float, optional
        The match window, in seconds (by default, 0.15).

    Returns
    -------
    BxbResult
        The comparison matrix and the data for its statistics.

    """
    from wfdb.io.annotation import rdann
    from wfdb.io.record import rdheader

    header = rdheader(record_name, pn_dir=pn_dir)
    kwargs = dict(pn_dir=pn_dir, return_label_elements=["label_store"])
    reference = rdann(record_name, reference_extension, **kwargs)
    test = rdann(record_name, test_extension, **kwargs)
    return bxb(
        reference,
        test,
        fs=header.fs,
        sig_len=header.sig_len,
        start=start,
        stop=stop,
        window=window,
        record_name=header.record_name,
    )


def sumstats(results):
    """Combine the results of several records, as WFDB sumstats does.

    .. warning::

        This is a development feature: see `bxb`.

    Parameters
    ----------
    results : list of BxbResult
        The results of each record.

    Returns
    -------
    tuple of str
        The summaries of the two line-format reports (``bxb.out`` and
        ``sd.out``): the line of each record, the sums of their counts, and
        for the first, the gross and average statistics.

    """
    lines1 = [r.lines()[0] for r in results]
    lines2 = [r.lines()[1] for r in results]
    cols = [r.line_columns() for r in results]
    total = {k: sum(c[k] for c in cols) for k in _LINE1_KEYS}

    # Averages are of each record's statistics, recomputed (as sumstats does)
    # from its line-format counts, ignoring undefined ones. RR errors are
    # averaged as printed.
    per_record = [_gross_stats(c) for c in cols]
    averages = []
    for k in range(6):
        values = [100 * n / d for n, d in (s[k] for s in per_record) if d]
        averages.append(
            f" {sum(values) / len(values):6.2f}" if values else f" {'-':>6}"
        )
    rrs = [round(r.rms_rr_error, 2) for r in results if r.rms_rr_error is not None]
    rr_avg = f" {sum(rrs) / len(rrs):6.2f}" if rrs else f" {'-':>6}"
    gross = "".join(_pct(n, d) for n, d in _gross_stats(total))
    tp = _gross_stats(total)
    pad = " " * (4 + 6 + 4 * 17 + 5)
    out1 = (
        LINE_HEADINGS[0]
        + "".join(line + "\n" for line in lines1)
        + "_" * 132
        + "\n"
        + "Sum "
        + _counts_text(total)
        + "\n"
        + "Gross"
        + pad[5:]
        + gross
        + "\n"
        + "Average"
        + pad[7:]
        + "".join(averages)
        + rr_avg
        + "\n"
        + f"Total QRS complexes: {tp[0][1]}  Total VEBs: {tp[2][1]}"
        + f"  Total SVEBs: {tp[4][1]}\n"
        + f"\nSummary of results from {len(results)} records\n"
    )
    x_sums = [sum(r.count(c, "x") for r in results) for c in "NSVFQ"]
    out2 = (
        LINE_HEADINGS[1]
        + "".join(line + "\n" for line in lines2)
        + "_" * 72
        + "\n"
        + "Sum "
        + "".join(f" {v:4d}" for v in x_sums)
        + " " * 37
        + f" {sum(r.shutdown_seconds for r in results):4d} seconds\n"
        + f"\nSummary of results from {len(results)} records\n"
    )
    return out1, out2
