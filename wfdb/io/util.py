"""
A module for general utility functions
"""
import math
import os

from typing import Sequence


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
