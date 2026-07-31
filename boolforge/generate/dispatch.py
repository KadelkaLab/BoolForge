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

from .canalization import random_k_canalizing_function_with_specific_layer_structure
from .canalization import random_k_canalizing_function
from .canalization import random_non_canalizing_function
from .canalization import random_non_canalizing_non_degenerate_function
from .functions import random_function_with_bias
from .functions import random_function_with_exact_hamming_weight
from .functions import random_parity_function
from .functions import random_non_degenerate_function


def _validate_bias(bias: float) -> None:
    if not isinstance(bias, (float, int, np.floating)):
        raise TypeError("bias must be a float")
    if not (0.0 <= bias <= 1.0):
        raise ValueError("bias must be in [0, 1]")

def _validate_absolute_bias(absolute_bias: float) -> None:
    if not isinstance(absolute_bias, (float, int, np.floating)):
        raise TypeError("absolute_bias must be a float")
    if not (0.0 <= absolute_bias <= 1.0):
        raise ValueError("absolute_bias must be in [0, 1]")

def _validate_hamming_weight(
    n: int,
    hamming_weight: int,
    *,
    exact_depth: bool,
) -> None:
    if not isinstance(hamming_weight, (int, np.integer)):
        raise TypeError("hamming_weight must be an integer")
    if not (0 <= hamming_weight <= 2**n):
        raise ValueError("hamming_weight must satisfy 0 <= hamming_weight <= 2**n")

    if exact_depth and not (1 < hamming_weight < 2**n - 1):
        raise ValueError(
            "If exact_depth=True and depth=0, hamming_weight must be in "
            "{2, 3, ..., 2**n - 2}. "
            "Functions with weights 0, 1, 2**n-1, 2**n are canalizing."
        )


def random_function(
    n: int,
    depth: int = 0,
    exact_depth: bool = False,
    uniform_over_functions: bool = True,
    layer_structure: list[int] | None = None,
    parity: bool = False,
    allow_degenerate_functions: bool = False,
    bias: float = 0.5,
    absolute_bias: float = 0,
    use_absolute_bias: bool = False,
    hamming_weight: int | None = None,
    *,
    rng = None,
) -> BooleanFunction:
    """
    Generate a random Boolean function under flexible structural constraints.

    This function acts as a high-level generator that unifies several common
    ensembles of Boolean functions, including parity functions, canalizing
    functions of specified depth or layer structure, functions with fixed
    Hamming weight, and biased random functions. The first applicable
    generation rule (in the order described below) is applied.

    Selection logic (first applicable rule is used)

    1. If ``parity`` is True, return a random parity function
       (see ``random_parity_function``).

    2. Else, if ``layer_structure`` is provided, return a Boolean function
       with the specified canalizing layer structure using
       ``random_k_canalizing_function_with_specific_layer_structure``.
       Exactness of the canalizing depth is controlled by ``exact_depth``.

    3. Else, if ``depth > 0``, return a k-canalizing function with
       ``k = min(depth, n)`` using ``random_k_canalizing_function``.
       If ``exact_depth`` is True, the function has exactly this depth;
       otherwise, its canalizing depth is at least ``k``.

       If ``uniform_over_functions`` is True, canalizing layer structures are
       sampled uniformly at random (up to the imposed constraints).
       If False, canalized outputs are sampled independently and uniformly
       as bitstrings, which biases the distribution toward more symmetric
       layer structures.

    4. Else, if ``hamming_weight`` is provided, repeatedly sample Boolean
       functions with the specified Hamming weight until additional
       constraints implied by ``exact_depth`` and
       ``allow_degenerate_functions`` are satisfied.

    5. Else, generate a random Boolean function using a Bernoulli model with
       either:

       - fixed bias ``bias``, or
       - an automatically chosen bias determined by ``absolute_bias`` if
         ``use_absolute_bias`` is True.

       Additional constraints on canalization and degeneracy are enforced
       depending on ``exact_depth`` and ``allow_degenerate_functions``.

    Parameters
    ----------
    n : int
        Number of input variables. Must be a positive integer.
    depth : int, optional
        Requested canalizing depth. Used only if ``layer_structure`` is None
        and ``depth > 0``. If ``exact_depth`` is True, the function has exactly
        this canalizing depth (clipped at ``n``); otherwise, its depth is at
        least ``depth``. Default is 0.
    exact_depth : bool, optional
        Enforce exact canalizing depth where applicable. If ``depth == 0``,
        setting ``exact_depth=True`` enforces that the function is
        non-canalizing. Default is False.
    uniform_over_functions : bool, optional
        If True (default), canalizing layer structures are sampled uniformly
        at random in canalizing-function branches. If False, canalized outputs
        are sampled independently as bitstrings, inducing a bias toward more
        symmetric structures.
    layer_structure : list[int] or None, optional
        Explicit canalizing layer structure ``[k1, ..., kr]``. If provided,
        this takes precedence over ``depth``. Default is None.
    parity : bool, optional
        If True, ignore all other options and return a random parity function.
        Default is False.
    allow_degenerate_functions : bool, optional
        If True, functions with non-essential variables may be returned in
        random-generation branches. If False, non-degenerate functions are
        enforced whenever possible. Default is False.
    bias : float, optional
        Probability of a 1 when sampling truth-table entries independently.
        Used only if ``use_absolute_bias`` is False and no other branch applies.
        Must lie in ``[0, 1]``. Default is 0.5.
    absolute_bias : float, optional
        Absolute deviation from 0.5 used to determine the bias when
        ``use_absolute_bias`` is True. The bias is chosen uniformly from
        ``{0.5*(1 - absolute_bias), 0.5*(1 + absolute_bias)}``. Must lie in
        ``[0, 1]``. Default is 0.
    use_absolute_bias : bool, optional
        If True, ignore ``bias`` and determine the bias using
        ``absolute_bias``. Default is False.
    hamming_weight : int or None, optional
        If provided, enforce that the Boolean function has exactly this many
        ones in its truth table. Additional constraints are enforced depending
        on ``exact_depth`` and ``allow_degenerate_functions``. Default is None.
    rng : int, numpy.random.Generator, numpy.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        A randomly generated Boolean function of arity ``n``.

    Raises
    ------
    TypeError
        If parameters have invalid types.
    ValueError
        If parameter values or combinations are invalid.

    Notes
    -----
    For any fixed combination of parameters, this function samples **uniformly
    at random** from the set of Boolean functions satisfying the corresponding
    constraints. Non-uniformity arises only when explicitly requested via
    ``uniform_over_functions=False``.

    Extremely biased functions are often degenerate or highly canalizing;
    under restrictive parameter choices, some branches may reject repeatedly
    before returning a valid function.

    Examples
    --------
    >>> # Unbiased, non-degenerate random function
    >>> f = random_function(n=3)

    >>> # Function with canalizing depth at least 2
    >>> f = random_function(n=5, depth=2)

    >>> # Function with exact canalizing depth 2
    >>> f = random_function(n=5, depth=2, exact_depth=True)

    >>> # Function with a specific canalizing layer structure
    >>> f = random_function(n=6, layer_structure=[2, 1])

    >>> # Parity function
    >>> f = random_function(n=4, parity=True)

    >>> # Fixed Hamming weight with non-canalizing and non-degenerate constraints
    >>> f = random_function(
    ...     n=5,
    ...     hamming_weight=10,
    ...     exact_depth=True,
    ...     allow_degenerate_functions=False
    ... )
    """

    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    rng = utils._coerce_rng(rng)

    # ------------------------------------------------------------
    # Parity branch (highest priority)
    # ------------------------------------------------------------
    if parity:
        return random_parity_function(n, rng=rng)

    # ------------------------------------------------------------
    # Layer structure branch
    # ------------------------------------------------------------
    if layer_structure is not None:
        return random_k_canalizing_function_with_specific_layer_structure(
            n,
            layer_structure,
            exact_depth=exact_depth,
            rng=rng,
        )

    # ------------------------------------------------------------
    # Canalizing depth branch
    # ------------------------------------------------------------
    if not isinstance(depth, (int, np.integer)) or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    
    if depth > 0:
        return random_k_canalizing_function(
            n,
            min(depth, n),
            exact_depth=exact_depth,
            uniform_over_functions=uniform_over_functions,
            rng=rng,
        )

    # ------------------------------------------------------------
    # Fixed Hamming weight branch
    # ------------------------------------------------------------
    if hamming_weight is not None:
        _validate_hamming_weight(n, hamming_weight, exact_depth=exact_depth)

        while True:
            f = random_function_with_exact_hamming_weight(
                n, hamming_weight, rng=rng
            )

            if allow_degenerate_functions and exact_depth:
                if not f.is_canalizing():
                    return f

            elif allow_degenerate_functions:
                return f

            elif exact_depth:
                if not f.is_canalizing() and not f.is_degenerate():
                    return f

            else:
                if not f.is_degenerate():
                    return f

    # ------------------------------------------------------------
    # Bias-based random generation
    # ------------------------------------------------------------
    if use_absolute_bias:
        _validate_absolute_bias(absolute_bias)
        bias_of_function = rng.choice(
            [0.5 * (1 - absolute_bias), 0.5 * (1 + absolute_bias)]
        )
    else:
        _validate_bias(bias)
        bias_of_function = bias

    if allow_degenerate_functions:
        if exact_depth:
            return random_non_canalizing_function(
                n, bias_of_function, rng=rng
            )
        else:
            return random_function_with_bias(
                n, bias_of_function, rng=rng
            )

    else:
        if exact_depth:
            return random_non_canalizing_non_degenerate_function(
                n, bias_of_function, rng=rng
            )
        else:
            return random_non_degenerate_function(
                n, bias_of_function, rng=rng
            )