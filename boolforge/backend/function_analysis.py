#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ._numba import njit

@njit
def _is_degenerate_numba(f : np.ndarray, n : int) -> bool:
    """
    Check whether a Boolean function contains a non-essential variable.

    This Numba-accelerated helper determines whether there exists at least
    one input variable whose value can be flipped without affecting the
    output of the Boolean function.

    Parameters
    ----------
    f : np.ndarray
        Truth table of the Boolean function, of length ``2**n``.
    n : int
        Number of input variables.

    Returns
    -------
    bool
        ``True`` if the function contains at least one non-essential
        variable, ``False`` otherwise.
    """
    N = 1 << n  # 2**n
    for i in range(n):
        stride = 1 << (n - 1 - i)
        step = stride << 1  # 2 * stride
        depends_on_i = False
        # Iterate in blocks that differ only in bit i
        for base in range(0, N, step):
            for offset in range(stride):
                if f[base + offset] != f[base + offset + stride]:
                    depends_on_i = True
                    break
            if depends_on_i:
                break
        if not depends_on_i:
            return True  # found non-essential variable
    return False

def _get_essential_variables_numba(f : np.ndarray, n : int) -> bool:
    """
    Check whether a Boolean function contains a non-essential variable.

    This Numba-accelerated helper determines all input variables whose value 
    cannot be flipped without affecting the output of the Boolean function.

    Parameters
    ----------
    f : np.ndarray
        Truth table of the Boolean function, of length ``2**n``.
    n : int
        Number of input variables.

    Returns
    -------
    np.ndarray[bool]
        Array of length n. ``True`` if the variable at position i is essential.
    """
    
    N = 1 << n  # 2**n
    is_essential = np.zeros(n, dtype=bool)
    for i in range(n):
        stride = 1 << (n - 1 - i)
        step = stride << 1  # 2 * stride
        depends_on_i = False
        # Iterate in blocks that differ only in bit i
        for base in range(0, N, step):
            for offset in range(stride):
                if f[base + offset] != f[base + offset + stride]:
                    depends_on_i = True
                    is_essential[i] = True
                    break
            if depends_on_i:
                break
    return is_essential