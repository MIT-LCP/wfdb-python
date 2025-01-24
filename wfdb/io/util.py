"""
A module for general utility functions
"""

import io
import math
import os

import numpy as np

from typing import List, Sequence, Tuple


def lines_to_file(file_name: str, write_dir: str, lines: Sequence[str]):
    """
    Write each line in a list of strings to a text file.

    Parameters
    ----------
    file_name: str
        The base name of the file
    write_dir : str
        The output directory in which the file is to be written.
    lines : list
        The lines to be written to the text file.

    Returns
    -------
    N/A

    """
    with open(os.path.join(write_dir, file_name), "w", encoding="utf-8") as f:
        for l in lines:
            f.write(f"{l}\n")


def is_monotonic(items: Sequence) -> bool:
    """
    Determine whether elements in a list are monotonic. ie. unique
    elements are clustered together.

    ie. [5,5,3,4] is, [5,3,5] is not.

    Parameters
    ----------
    items : Sequence
        The input elements to be checked.

    Returns
    -------
    bool
        Whether the elements are monotonic (True) or not (False).

    """
    prev_elements = set({items[0]})
    prev_item = items[0]

    for item in items:
        if item != prev_item:
            if item in prev_elements:
                return False
            prev_item = item
            prev_elements.add(item)

    return True


def downround(x, base):
    """
    Round <x> down to nearest <base>.

    Parameters
    ---------
    x : str, int, float
        The number that will be rounded down.
    base : int, float
        The base to be rounded down to.

    Returns
    -------
    float
        The rounded down result of <x> down to nearest <base>.

    """
    return base * math.floor(float(x) / base)


def upround(x, base):
    """
    Round <x> up to nearest <base>.

    Parameters
    ---------
    x : str, int, float
        The number that will be rounded up.
    base : int, float
        The base to be rounded up to.

    Returns
    -------
    float
        The rounded up result of <x> up to nearest <base>.

    """
    return base * math.ceil(float(x) / base)


def overlapping_ranges(
    ranges_1: Sequence[Tuple[int, int]],
    ranges_2: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Given two collections of integer ranges, return a list of ranges
    in which both input inputs overlap.

    From: https://stackoverflow.com/q/40367461

    Slightly modified so that if the end of one range exactly equals
    the start of the other range, no overlap would be returned.
    """
    return [
        (max(first[0], second[0]), min(first[1], second[1]))
        for first in ranges_1
        for second in ranges_2
        if max(first[0], second[0]) < min(first[1], second[1])
    ]


def fromfile(fileobj, dtype, count=-1):
    """
    Detect if the object will work with numpy.fromfile - if so, use it. If not, read the object into a numpy array and
    calculate the number of elements (if not provided) - this is needed for fsspec objects.
    """
    if isinstance(fileobj, io.FileIO) or (
        isinstance(fileobj, (io.BufferedReader, io.BufferedRandom))
        and isinstance(fileobj.raw, io.FileIO)
    ):
        return np.fromfile(fileobj, dtype=dtype, count=count)
    else:
        dtype = np.dtype(dtype)
        if count < 0:
            start = fileobj.tell()
            fileobj.seek(0, os.SEEK_END)
            end = fileobj.tell()
            fileobj.seek(start, os.SEEK_SET)
            count = (end - start) // dtype.itemsize
        array = np.empty(count, dtype)
        size = fileobj.readinto(array)
        array.resize(size // dtype.itemsize)
        return array
