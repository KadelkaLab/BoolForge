#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides functions for generating random Boolean functions and
Boolean networks with specified structural and dynamical properties.

The :mod:`~boolforge.generate` module enables the systematic creation of
Boolean functions and networks that satisfy particular constraints, such as
specified canalization depth, sensitivity range, bias, or connectivity.
Generated instances can be used for statistical analysis, benchmarking, or
simulation studies.

Several generation routines leverage Numba acceleration for efficient sampling
and evaluation of large function spaces. While Numba is **recommended** to
achieve near-native performance, it is **not required** for functionality; all
functions have pure Python fallbacks.

This module complements :mod:`~boolforge.boolean_function` and
:mod:`~boolforge.boolean_network` by facilitating reproducible generation of
synthetic test cases and large ensembles of random networks.

Example
-------
>>> import boolforge
>>> boolforge.random_function(n=3)
>>> boolforge.random_network(N=5, n=2)
"""

import numpy as np

from ..boolean_function import BooleanFunction
from .. import utils

def random_function_with_bias(
    n: int,
    bias: float = 0.5,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random Boolean function with a specified bias.

    The Boolean function is represented by its truth table of length
    ``2**n``, where each entry is independently set to 1 with probability
    ``bias`` and to 0 otherwise.

    Parameters
    ----------
    n : int
        Number of Boolean variables.
    bias : float, optional
        Probability that a given truth-table entry equals 1. Default is 0.5.
    rng : int, np.random.Generator, np.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        Random Boolean function with the specified bias.
    """
    rng = utils._coerce_rng(rng)
    return BooleanFunction._from_f_unchecked(
        np.array(rng.random(2**n) < bias, dtype=int)
    )


def random_function_with_exact_hamming_weight(
    n: int,
    hamming_weight: int,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random Boolean function with a fixed Hamming weight.

    The Boolean function is represented by its truth table of length
    ``2**n``, containing exactly ``hamming_weight`` entries equal to 1.
    All such functions are sampled uniformly at random.

    Parameters
    ----------
    n : int
        Number of Boolean variables.
    hamming_weight : int
        Number of truth-table entries equal to 1.
    rng : int, np.random.Generator, np.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        Random Boolean function with exactly ``hamming_weight`` ones in its
        truth table.

    Raises
    ------
    TypeError
        If ``hamming_weight`` is not an integer.
    ValueError
        If ``hamming_weight`` is not in the range ``[0, 2**n]``.
    """
    rng = utils._coerce_rng(rng)

    if not isinstance(hamming_weight, (int, np.integer)):
        raise TypeError("hamming_weight must be an integer")

    if not (0 <= hamming_weight <= 2**n):
        raise ValueError("hamming_weight must satisfy 0 <= hamming_weight <= 2**n")

    one_indices = rng.choice(2**n, hamming_weight, replace=False)
    f = np.zeros(2**n, dtype=int)
    f[one_indices] = 1

    return BooleanFunction._from_f_unchecked(f)


def random_parity_function(
    n: int,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random parity Boolean function.

    A parity Boolean function evaluates to the parity (sum modulo 2) of all
    input variables, optionally shifted by a constant. This function returns
    either the parity function or its complement, chosen uniformly at random.

    Parameters
    ----------
    n : int
        Number of Boolean variables.
    rng : int, np.random.Generator, np.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        Random parity Boolean function on ``n`` variables.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.

    Notes
    -----
    - The returned function is either
      ``x1 XOR x2 XOR ... XOR xn`` or its complement.
    - All variables are included symmetrically.
    - Parity functions are never canalizing. All variables must always be known
      to determine the output; they have maximal average sensitivity.

    Examples
    --------
    >>> f = random_parity_function(3)
    >>> sum(f)
    4
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    rng = utils._coerce_rng(rng)

    # choose parity or its complement
    val = rng.integers(2)

    f = np.zeros(2**n, dtype=np.uint8)
    for i in range(2**n):
        if i.bit_count() % 2 == val:
            f[i] = 1

    return BooleanFunction._from_f_unchecked(f)



def random_non_degenerate_function(
    n: int,
    bias: float = 0.5,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random non-degenerate Boolean function.

    A Boolean function is non-degenerate if every variable is essential, i.e.,
    the function depends on all ``n`` input variables. Functions are sampled
    repeatedly from the Bernoulli(bias) ensemble until a non-degenerate
    function is obtained.

    Parameters
    ----------
    n : int
        Number of Boolean variables.
    bias : float, optional
        Probability that a truth-table entry equals 1. Default is 0.5.
    rng : int, np.random.Generator, np.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        Random non-degenerate Boolean function.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.
    ValueError
        If ``bias`` is not strictly between 0 and 1.

    Notes
    -----
    - For moderate bias values, almost all Boolean functions are non-degenerate.
    - Extremely biased functions are very likely to be degenerate, which may
      lead to long rejection-sampling times.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(bias, (float, np.floating)) or not (0.0 < bias < 1.0):
        raise ValueError("bias must be a float strictly between 0 and 1")

    rng = utils._coerce_rng(rng)

    # Rejection sampling; almost all Boolean functions are non-degenerate
    while True:
        f = random_function_with_bias(n, bias=bias, rng=rng)
        if not f.is_degenerate():
            return f


def random_degenerate_function(
    n: int,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random degenerate Boolean function uniformly at random.
    
    A Boolean function is degenerate if at least one variable is
    non-essential, i.e., the function does not depend on that variable.
    By using appropriate acceptance weights, the resulting distribution
    is function-uniform over all degenerate Boolean functions on ``n``
    variables.
    
    Parameters
    ----------
    n : int
        Number of Boolean variables.
    rng : int, np.random.Generator, np.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.
    
    Returns
    -------
    BooleanFunction
        Random degenerate Boolean function on ``n`` variables, sampled
        uniformly over all degenerate Boolean functions.
    
    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.
    
    Notes
    -----
    - A non-essential variable is forced by construction, but additional
      variables may also be non-essential by chance.
    - The forced non-essential variable is chosen uniformly at random
      from all ``n`` variables.
    - Function-uniformity is achieved by accepting a candidate function
      with ``k`` non-essential variables with probability ``1/k``,
      correcting for the fact that such functions are ``k`` times more
      likely to be proposed than functions with exactly one non-essential
      variable.
    - At bias != 0.5 this acceptance correction would no longer be valid,
      so bias is fixed at 0.5 internally.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    rng = utils._coerce_rng(rng)
    while True:
        # Choose forced non-essential variable uniformly at random
        index_non_essential_variable = rng.integers(n)
        
        # Generate an (n-1)-variable Boolean function
        f_original = random_function_with_bias(n - 1, bias=0.5, rng=rng)
        
        # Copy the (n-1)-variable function across both values of the non-essential variable
        block = 2 ** index_non_essential_variable
        indices = (np.arange(2**n) // block) % 2 == 1
        f = np.zeros(2**n, dtype=np.uint8)
        f[indices] = f_original.f
        f[~indices] = f_original.f
        candidate = BooleanFunction._from_f_unchecked(f)
        
        k = n - candidate.get_number_of_essential_variables()  # number of non-essential variables
        if rng.random() < 1.0 / k:
            return candidate
