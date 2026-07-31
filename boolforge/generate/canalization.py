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

import math
import numpy as np

from ..boolean_function import BooleanFunction
from .. import utils

from .functions import random_function_with_bias
from .functions import random_non_degenerate_function

def random_non_canalizing_function(
    n: int,
    bias: float = 0.5,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random non-canalizing Boolean function.

    A Boolean function is canalizing if there exists at least one variable
    and a value of that variable such that fixing it forces the output of
    the function. This function samples Boolean functions from the
    Bernoulli(bias) ensemble until a non-canalizing function is obtained.

    Parameters
    ----------
    n : int
        Number of Boolean variables. Must satisfy ``n > 1``.
    bias : float, optional
        Probability that a truth-table entry equals 1. Default is 0.5.
    rng : int, np.random.Generator, np.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        Random non-canalizing Boolean function on ``n`` variables.

    Raises
    ------
    ValueError
        If ``n`` is not an integer greater than 1.
    ValueError
        If ``bias`` is not strictly between 0 and 1.

    Notes
    -----
    - This function uses rejection sampling.
    - For moderate bias values, almost all Boolean functions are
      non-canalizing.
    - Extremely biased functions are more likely to be canalizing and may
      lead to longer sampling times.

    References
    ----------
    C. Kadelka, J. Kuipers, and R. Laubenbacher (2017).
    The influence of canalization on the robustness of Boolean networks.
    Physica D: Nonlinear Phenomena, 353, 39–47.
    """
    if not isinstance(n, (int, np.integer)) or n <= 1:
        raise ValueError("n must be an integer greater than 1")

    if not isinstance(bias, (float, np.floating)) or not (0.0 < bias < 1.0):
        raise ValueError("bias must be a float strictly between 0 and 1")

    rng = utils._coerce_rng(rng)

    # Rejection sampling; most Boolean functions are non-canalizing
    while True:
        f = random_function_with_bias(n, bias=bias, rng=rng)
        if not f.is_canalizing():
            return f


def random_non_canalizing_non_degenerate_function(
    n: int,
    bias: float = 0.5,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random Boolean function that is both non-canalizing and
    non-degenerate.

    A Boolean function is non-canalizing if no variable can force the output
    when fixed, and non-degenerate if every variable is essential. This
    function samples Boolean functions from the Bernoulli(bias) ensemble
    until both properties are satisfied.

    Parameters
    ----------
    n : int
        Number of Boolean variables. Must satisfy ``n > 1``.
    bias : float, optional
        Probability that a truth-table entry equals 1. Default is 0.5.
    rng : int, np.random.Generator, np.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        Random Boolean function on ``n`` variables that is both non-canalizing
        and non-degenerate.

    Raises
    ------
    ValueError
        If ``n`` is not an integer greater than 1.
    ValueError
        If ``bias`` is not strictly between 0 and 1.

    Notes
    -----
    - This function uses rejection sampling.
    - For moderate bias values and sufficiently large ``n``, almost all
      Boolean functions are both non-canalizing and non-degenerate.
    - Extremely biased functions are more likely to be canalizing or
      degenerate and may lead to longer sampling times.

    References
    ----------
    C. Kadelka, J. Kuipers, and R. Laubenbacher (2017).
    The influence of canalization on the robustness of Boolean networks.
    Physica D: Nonlinear Phenomena, 353, 39–47.
    """
    if not isinstance(n, (int, np.integer)) or n <= 1:
        raise ValueError("n must be an integer greater than 1")

    if not isinstance(bias, (float, np.floating)) or not (0.0 < bias < 1.0):
        raise ValueError("bias must be a float strictly between 0 and 1")

    rng = utils._coerce_rng(rng)

    # Rejection sampling; almost all Boolean functions satisfy both properties
    while True:
        f = random_function_with_bias(n, bias=bias, rng=rng)
        if not f.is_canalizing() and not f.is_degenerate():
            return f

_uniform_over_functions_weights = {}

def _get_uniform_over_functions_weights(max_n, is_ncf=True):
    """
    Compute dynamic-programming weights for uniform canalized layer structures.

    This function constructs a dynamic-programming table ``W`` used to sample
    canalized output bitstrings with probability proportional to the inverse
    factorials of their layer sizes. The table encodes the total weight of all
    valid completions of a partially constructed canalized structure.

    Parameters
    ----------
    max_n : int
        Maximum total length of the canalized output string.
    is_ncf : bool, optional
        If True (default), enforce the nested canalizing function (NCF)
        constraint that the final canalizing layer has size at least 2.
        If False, no constraint is imposed on the final layer size.

    Returns
    -------
    W : ndarray of shape (max_n + 1, max_n + 2)
        Dynamic-programming weight table. ``W[m, s]`` gives the total weight of
        all valid completions with ``m`` positions remaining and current layer
        size ``s``.

    Notes
    -----
    The recursion is given by

        W[m, s] = W[m - 1, s + 1] + (1 / s!) * W[m - 1, 1],

    corresponding to either extending the current canalizing layer or
    terminating it and starting a new layer of size 1. The base case ``m = 0``
    accounts for the weight of the final layer and enforces the optional NCF
    constraint.

    Results are cached internally to avoid recomputation for repeated calls
    with the same ``max_n`` and ``NCF`` values.
    """
    if (max_n,is_ncf) in _uniform_over_functions_weights:
        return _uniform_over_functions_weights[(max_n,is_ncf)]
    
    W = np.zeros((max_n + 1, max_n + 2), dtype=float)

    # Base case: no positions left -> close final run
    inv_factorial_of_s = 1.0
    for s in range(1, max_n + 2):
        inv_factorial_of_s /= s
        if (not is_ncf) or (s >= 2):
            W[0, s] = inv_factorial_of_s
        else:
            W[0, s] = 0.0

    # DP
    for m in range(1, max_n + 1):
        inv_factorial_of_s = 1.0
        for s in range(1, max_n + 1):
            inv_factorial_of_s /= s
            W[m, s] = (
                W[m - 1, s + 1] +
                inv_factorial_of_s * W[m - 1, 1]
            )
    
    _uniform_over_functions_weights[(max_n,is_ncf)] = W    
    return W


def sample_canalized_outputs_uniform_over_functions(n, W, *, rng):
    """
    Sample a canalized output bitstring yielding uniform layer-structure weighting.

    This function samples a binary vector ``b`` of length ``n`` representing
    canalized output values, where consecutive equal values are part of the same
    canalizing layer. The probability of a given layer structure
    ``(k_1, k_2, ..., k_r)`` is proportional to

        1 / (k_1! k_2! ... k_r!).
        
    Sampling is performed sequentially using precomputed dynamic-programming
    weights ``W``, stored in _uniform_over_functions_weights.

    Parameters
    ----------
    n : int
        Length of the output bitstring to sample.
    W : ndarray of shape (n+1, n+1)
        Dynamic-programming weight table, where ``W[m, s]`` gives the total
        weight of all valid completions with ``m`` positions remaining and
        current layer size ``s``.
    rng : numpy.random.Generator
        Random number generator used for sampling.

    Returns
    -------
    b : ndarray of shape (n,), dtype int
        Sampled binary canalized output vector.

    Notes
    -----
    The bitstring is generated left-to-right. At each step, the algorithm
    probabilistically chooses whether to extend the current layer or start a
    new one, using the weights in ``W`` to ensure correct global sampling
    probabilities. The first bit is chosen uniformly at random.
    """
    rng = utils._coerce_rng(rng)

    b = np.zeros(n, dtype=int)

    # Randomly pick first canalized output
    b[0] = rng.integers(2)

    s = 1              # current layer size
    inv_factorial_of_s = 1.0
    m = n - 1          # positions remaining to fill

    for i in range(1, n):

        # DP-weighted decision
        w_extend = W[m - 1, s + 1]
        w_split  = inv_factorial_of_s * W[m - 1, 1]

        p_extend = w_extend / (w_extend + w_split)
        if rng.random() < p_extend:
            b[i] = b[i - 1]
            s += 1
            inv_factorial_of_s /= s
        else:
            b[i] = 1 - b[i - 1]
            s = 1
            inv_factorial_of_s = 1.0

        m -= 1

    return b

def random_k_canalizing_function(
    n: int,
    k: int,
    exact_depth: bool = False,
    uniform_over_functions: bool = True,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random k-canalizing Boolean function in n variables.

    A Boolean function is k-canalizing if it has at least k conditionally 
    canalizing variables. If ``exact_depth`` is True, the function has exactly
    k conditionally canalizing variables; otherwise, its canalizing depth 
    may exceed k.

    Parameters
    ----------
    n : int
        Number of Boolean variables.
    k : int
        Requested canalizing depth. Must satisfy ``0 <= k <= n``.
        Setting ``k = n`` generates a nested canalizing function.
    exact_depth : bool, optional
        If True, enforce that the canalizing depth is exactly ``k``.
        If False (default), the depth is at least ``k``.
    uniform_over_functions : bool, optional
        If True (default), the function is sampled uniformly at random
        from the set of Boolean functions consistent with the specified
        constraints (n, k, exact_depth).
        
        Internally, this is achieved by a rejection sampling scheme that
        compensates for all combinatorial multiplicities arising from
        symmetric canalizing layers and possible merges between the
        outer canalizing layers and the core function.
    
        If False, canalized outputs are sampled independently as a bitstring,
        and no rejection correction is applied. In this case, the resulting
        distribution is biased toward Boolean functions with more symmetric
        canalizing structures.
    rng : int, numpy.random.Generator, numpy.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        A Boolean function on ``n`` variables with canalizing depth at
        least ``k`` (or exactly ``k`` if ``exact_depth=True``).

    Raises
    ------
    AssertionError
        If ``n`` is not a positive integer.
    AssertionError
        If ``k`` does not satisfy ``0 <= k <= n``.
    AssertionError
        If ``exact_depth=True`` and ``k = n-1`` (no such functions exist).

    Notes
    -----
    When uniform_over_functions=True, this function samples uniformly from the
    space of Boolean functions with the specified canalizing properties.
    As a consequence, different canalizing layer structures generally
    occur with different frequencies, reflecting the fact that they
    support different numbers of Boolean functions.
    
    Uniformity over canalizing layer structures is not enforced and is
    not expected.

    The construction follows the standard decomposition of a k-canalizing
    function into canalizing variables, canalizing inputs and outputs, and
    a residual core function on ``n-k`` variables.

    References
    ----------
    He, Q., and Macauley, M. (2016).
        Stratification and enumeration of Boolean functions by canalizing depth.
        Physica D: Nonlinear Phenomena, 314, 1–8.

    Dimitrova, E., Stigler, B., Kadelka, C., and Murrugarra, D. (2022).
        Revealing the canalizing structure of Boolean functions: Algorithms
        and applications. Automatica, 146, 110630.
    """
    rng = utils._coerce_rng(rng)

    assert isinstance(n, (int, np.integer)) and n > 0, "n must be a positive integer"
    assert n - k != 1 or not exact_depth, (
        "There are no functions of exact canalizing depth n-1.\nEither set exact_depth=False or ensure k != n-1"
    )
    assert isinstance(k, (int, np.integer)) and 0 <= k and k <= n, (
        "k, the canalizing depth, must satisfy 0 <= k <= n."
    )
    
    if k==0:
        if exact_depth:
            if n == 1:
                raise ValueError(
                    "No Boolean functions with canalizing depth 0 exist for n = 1."
                )
            return random_non_canalizing_non_degenerate_function(n, rng=rng)
        else:
            return random_non_degenerate_function(n, rng=rng)
    elif k==n-1 and n>1: #canalizing functions with depth n-1>0 really have depth n 
        k=n

    # Step 1: canalizing inputs and variables
    aas = rng.integers(2, size=k)
    can_vars = rng.choice(n, k, replace=False)

    while True: 
        #include the generation of canalized outputs in the rejection sampling scheme
        #because some output vectors `bbs` may give rise to more distinct functions than others
        
        # Step 2: canalized outputs, determining layers
        if uniform_over_functions:
            bbs = sample_canalized_outputs_uniform_over_functions(
                k,
                _get_uniform_over_functions_weights(k, is_ncf=(k >= n)),
                rng=rng,
            )
        else:
            bbs = rng.integers(2, size=k)
            
        # Step 3: sample core function using efficient rejection sampling
        if k < n:
            core_function = random_k_canalizing_function(
                n=n - k,
                k=0,
                exact_depth=exact_depth,#True if exact_depth or uniform_over_functions else False,
                rng=rng,
            )
            
            if exact_depth or k==0 or not uniform_over_functions:
                break
            else:
                #check if the core function is canalizing and correct for combinatorial
                #bias in the selection of the final functions
                
                #compute canalizing info of core_function, stored in f.properties
                core_function.get_layer_structure() 
                if core_function.properties['CanalizingDepth'] == 0:
                    break
                else: #rejection sampling is efficient because it happens with probability <= 50%
                    bbs_core = core_function.properties['CanalizedOutputs']
                    if bbs_core[0] != bbs[-1]:  
                        #no combinatorial explosion, the core varaibles start a new layer
                        break
                    else: 
                        # merge occurs: last layer grows
                        
                        # determine size of last outer layer in bbs
                        s = 1
                        for i in range(k - 1, 0, -1):
                            if bbs[i] == bbs[i - 1]:
                                s += 1
                            else:
                                break
                            
                        # determine number of additional variables in the same layer
                        s_core = 1
                        for i in range(core_function.properties['CanalizingDepth'] - 1):
                            if bbs_core[i] == bbs_core[i + 1]:
                                s_core += 1
                            else:
                                break                    
                        
                        accept_prob = 1 / math.comb(s + s_core, s)
                        
                        if rng.random() <= accept_prob:
                            break
        else:
            core_function = [1 - bbs[-1]]
            break
        
    # Step 4: build truth table and return canalizing Boolean function
    left_side_of_truth_table = utils.get_left_side_of_truth_table(n)
    f = np.full(2**n, -1, dtype=np.int8)
    
    for j in range(k):
        mask = (left_side_of_truth_table[:, can_vars[j]] == aas[j]) & (f < 0)
        f[mask] = bbs[j]
        
    # fill remaining with core truth table
    f[f < 0] = np.asarray(core_function, dtype=np.int8)

    return BooleanFunction._from_f_unchecked(f)


def random_k_canalizing_function_with_specific_layer_structure(
    n: int,
    layer_structure: list,
    exact_depth: bool = False,
    *,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random Boolean function with a specified canalizing layer structure.

    The canalizing layer structure is given as a list
    ``[k_1, ..., k_r]``, where each ``k_i`` specifies the number of
    canalizing variables in the i-th layer. The total canalizing depth is
    ``sum(layer_structure)``.

    If ``sum(layer_structure) == n`` and ``n > 1``, the function is a nested
    canalizing function and the final layer is required to have size at
    least 2.

    Parameters
    ----------
    n : int
        Total number of Boolean variables.
    layer_structure : list of int
        Canalizing layer structure ``[k_1, ..., k_r]``. Each entry must be
        at least 1. If ``sum(layer_structure) == n`` and ``n > 1``, the final
        entry must satisfy ``layer_structure[-1] >= 2``.
    exact_depth : bool, optional
        If True, enforce that the canalizing depth is exactly
        ``sum(layer_structure)``. If False (default), additional canalizing
        variables may occur in the core function.
    rng : int, numpy.random.Generator, numpy.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        A Boolean function on ``n`` variables with the prescribed canalizing
        layer structure, plus potentially more canalizing variables (if exact_depth=True).

    Raises
    ------
    AssertionError
        If ``n`` is not a positive integer.
    AssertionError
        If ``sum(layer_structure)`` does not satisfy ``0 <= sum(layer_structure) <= n``.
    AssertionError
        If ``exact_depth=True`` and ``sum(layer_structure) = n - 1``.
    AssertionError
        If ``sum(layer_structure) = n > 1`` and the final layer has size less
        than 2.
    AssertionError
        If any entry of ``layer_structure`` is less than 1.

    Notes
    -----
    For fixed parameter values, this function samples uniformly at random
    from the ensemble of Boolean functions consistent with the specified
    canalizing layer structure and additional constraints.

    The construction follows the standard decomposition of a canalizing
    function into ordered canalizing layers and a residual core function on
    the remaining variables.

    References
    ----------
    He, Q., and Macauley, M. (2016).
        Stratification and enumeration of Boolean functions by canalizing depth.
        Physica D: Nonlinear Phenomena, 314, 1–8.

    Kadelka, C., Kuipers, J., and Laubenbacher, R. (2017).
        The influence of canalization on the robustness of Boolean networks.
        Physica D: Nonlinear Phenomena, 353, 39–47.
    """
    rng = utils._coerce_rng(rng)
    depth = sum(layer_structure)  # canalizing depth
    if depth == 0:
        layer_structure = [0]

    assert isinstance(n, (int, np.integer)) and n > 0, "n must be an integer > 0"
    assert n - depth != 1 or not exact_depth, (
        "There are no functions of exact canalizing depth n-1.\nEither set exact_depth=False or ensure depth=sum(layer_structure)!=n-1."
    )
    assert 0 <= depth and depth <= n, "Ensure 0 <= depth = sum(layer_structure) <= n."
    assert depth < n or layer_structure[-1] > 1 or n == 1, (
        "The last layer of an NCF (i.e., an n-canalizing function) has to have size >= 2 whenever n > 1.\nIf depth=sum(layer_structure)=n, ensure that layer_structure[-1]>=2."
    )
    assert min(layer_structure) >= 1, (
        "Each layer must have at least one variable (each element of layer_structure must be >= 1)."
    )

    size_state_space = 2**n
    aas = rng.integers(2, size=depth)  # canalizing inputs
    b0 = rng.integers(2)
    bbs = [b0] * layer_structure[0]  # canalized outputs for first layer
    for i in range(1, len(layer_structure)):
        if i % 2 == 0:
            bbs.extend([b0] * layer_structure[i])
        else:
            bbs.extend([1 - b0] * layer_structure[i])
    can_vars = rng.choice(n, depth, replace=False)
    f = np.zeros(size_state_space, dtype=int)
    if depth < n:
        core_function = random_k_canalizing_function(
            n=n - depth,
            k=0,
            exact_depth=exact_depth,
            rng=rng,
        )
    else:
        core_function = [1 - bbs[-1]]

    left_side_of_truth_table = utils.get_left_side_of_truth_table(n)
    f = np.full(2**n, -1, dtype=np.int8)
    for j in range(depth):
        mask = (left_side_of_truth_table[:, can_vars[j]] == aas[j]) & (f < 0)
        f[mask] = bbs[j]
    # fill remaining with core truth table
    f[f < 0] = np.asarray(core_function, dtype=np.int8)

    return BooleanFunction._from_f_unchecked(f)


def random_NCF(
    n: int,
    uniform_over_functions: bool = True,
    layer_structure: list | None = None,
    *,
    rng=None
) -> BooleanFunction:
    """
    Generate a random nested canalizing Boolean function in n variables.

    A nested canalizing function (NCF) is an n-canalizing Boolean function,
    i.e., a function whose canalizing depth equals the number of variables.
    Optionally, a specific canalizing layer structure may be prescribed.

    Parameters
    ----------
    n : int
        Total number of Boolean variables.
    uniform_over_functions : bool, optional
        If True (default) and ``layer_structure`` is None, canalizing layer
        structures are sampled uniformly at random, removing the bias toward
        symmetric structures induced by independent sampling of canalized
        outputs. If False, canalized outputs are sampled independently and
        uniformly as a bitstring, which biases the distribution toward more
        symmetric layer structures.
        This parameter is ignored if ``layer_structure`` is provided.
    layer_structure : list of int or None, optional
        Canalizing layer structure ``[k_1, ..., k_r]``. Each entry must be at
        least 1. If provided, it must satisfy ``sum(layer_structure) == n``.
        If ``n > 1``, the final entry must satisfy
        ``layer_structure[-1] >= 2``. If None (default), the layer structure
        is sampled at random.
    rng : int, numpy.random.Generator, numpy.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        A nested canalizing Boolean function on ``n`` variables.

    Raises
    ------
    AssertionError
        If ``n`` is not a positive integer.
    AssertionError
        If ``layer_structure`` is provided but does not satisfy
        ``sum(layer_structure) == n``.
    AssertionError
        If ``n > 1`` and the final layer has size less than 2.

    Notes
    -----
    For fixed parameter values, this function samples uniformly at random
    from the ensemble of nested canalizing Boolean functions consistent with
    the specified constraints. Non-uniformity arises only when
    ``uniform_over_functions=False``.

    This function is a convenience wrapper around
    ``random_k_canalizing_function`` and
    ``random_k_canalizing_function_with_specific_layer_structure``.

    References
    ----------
    He, Q., and Macauley, M. (2016).
        Stratification and enumeration of Boolean functions by canalizing depth.
        Physica D: Nonlinear Phenomena, 314, 1–8.

    Kadelka, C., Kuipers, J., and Laubenbacher, R. (2017).
        The influence of canalization on the robustness of Boolean networks.
        Physica D: Nonlinear Phenomena, 353, 39–47.
    """
    rng = utils._coerce_rng(rng)
    if layer_structure is None:
        return random_k_canalizing_function(n, 
                                            n, 
                                            uniform_over_functions=uniform_over_functions,
                                            exact_depth=False, 
                                            rng=rng)
    else:
        assert sum(layer_structure) == n, "Ensure sum(layer_structure) == n."
        assert layer_structure[-1] > 1 or n == 1, (
            "The last layer of an NCF has to have size >= 2 whenever n > 1.\nEnsure that layer_structure[-1]>=2."
        )
        return random_k_canalizing_function_with_specific_layer_structure(
            n, 
            layer_structure, 
            exact_depth=False, 
            rng=rng
        )
