#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random function generators with canalization constraints.

Provides generators for Boolean and multistate functions with
specified canalizing properties, including canalizing depth,
layer structure, and non-canalization.
"""

import math
import numpy as np
from collections.abc import Sequence

from ..function import Function, BooleanFunction
from .. import utils, utils_multistate

from .functions import (
    _validate_n,
    _resolve_output_bias,
    _sample_function_unchecked,
    random_function_with_bias,
    random_non_degenerate_function,
)


def random_non_canalizing_function(
        n: int,
        bias: float | None = None,
        *,
        r: int = 2,
        r_inputs: Sequence[int] | None = None,
        output_bias: Sequence[float] | None = None,
        strong: bool = False,
        constant_functions_canalizing: bool = True,
        rng=None,
) -> Function:
    """
    Generate a random non-canalizing finite-state function.

    Functions are sampled independently from the specified
    output distribution until no canalizing variable exists
    under the selected definition.

    Parameters
    ----------
    n : int
        Number of input variables (n >= 2).
    bias : float | None, optional
        Probability of output 1 for binary-valued functions.
        Default is None.
    r : int, optional
        Number of output states. Default is 2.
    r_inputs : Sequence[int] | None, optional
        Input cardinalities. If None, all inputs have r states.
    output_bias : Sequence[float] | None, optional
        Probabilities for output states 0, ..., r-1.
        If None, outputs are sampled uniformly.
    strong : bool, optional
        If False (default), exclude all weakly canalizing
        functions. If True, exclude only strongly canalizing functions.
    constant_functions_canalizing : bool, optional
        Whether constant functions are considered canalizing.
        Default is True.
    rng : optional
        Random number generator or seed specification.

    Returns
    -------
    Function
        Random non-canalizing function. Returns BooleanFunction
        when all input and output cardinalities are 2.

    Notes
    -----
    Rejection sampling preserves the underlying product
    distribution conditioned on non-canalization.
    """
    n = _validate_n(n, allow_zero=False)

    if n < 2:
        raise ValueError("n must be at least 2")

    r, r_inputs, _ = (
        utils_multistate.normalize_and_validate_state_specs(
            n, r, r_inputs
        )
    )

    output_bias = _resolve_output_bias(
        bias, output_bias, r
    )

    # A deterministic output distribution produces only
    # constant functions, which are canalizing by convention.
    if (
        constant_functions_canalizing
        and output_bias is not None
        and np.count_nonzero(output_bias > 0) < 2
    ):
        raise ValueError(
            "Non-canalizing generation requires at least two "
            "output states with positive probability."
        )

    rng = utils._coerce_rng(rng)

    while True:
        candidate = _sample_function_unchecked(
            r,
            r_inputs,
            rng=rng,
            output_bias=output_bias,
        )

        if not candidate.is_canalizing(strong=strong,
                                       constant_functions_canalizing=constant_functions_canalizing):
            return candidate


def random_non_canalizing_non_degenerate_function(
        n: int,
        bias: float | None = None,
        *,
        r: int = 2,
        r_inputs: Sequence[int] | None = None,
        output_bias: Sequence[float] | None = None,
        strong: bool = False,
        constant_functions_canalizing: bool = True,
        rng=None,
) -> Function:
    """
    Generate a random non-canalizing, non-degenerate function.

    Functions are sampled from the specified output distribution
    until they are non-canalizing under the selected definition
    and every input variable is essential.

    Parameters
    ----------
    n : int
        Number of input variables (n >= 2).
    bias : float | None, optional
        Probability of output 1 for binary-valued functions.
        Default is None.
    r : int, optional
        Number of output states. Default is 2.
    r_inputs : Sequence[int] | None, optional
        Input cardinalities. If None, all inputs have r states.
    output_bias : Sequence[float] | None, optional
        Probabilities for output states 0, ..., r-1.
        If None, outputs are sampled uniformly.
    strong : bool, optional
        If False (default), exclude all weakly canalizing
        functions. If True, exclude only strongly canalizing functions.
    constant_functions_canalizing : bool, optional
        Whether constant functions are considered canalizing.
        Default is True.
    rng : optional
        Random number generator or seed specification.

    Returns
    -------
    Function
        Random non-canalizing, non-degenerate function.

    Notes
    -----
    Rejection sampling preserves the underlying product
    distribution conditioned on both properties.
    """
    n = _validate_n(n, allow_zero=False)

    if n < 2:
        raise ValueError("n must be at least 2")

    r, normalized_r_inputs, _ = (
        utils_multistate.normalize_and_validate_state_specs(
            n, r, r_inputs
        )
    )

    output_bias = _resolve_output_bias(
        bias, output_bias, r
    )

    if (
        output_bias is not None
        and np.count_nonzero(output_bias > 0) < 2
    ):
        raise ValueError(
            "Non-canalizing generation requires at least two "
            "output states with positive probability."
        )

    rng = utils._coerce_rng(rng)

    while True:
        candidate = _sample_function_unchecked(
            r=r,
            r_inputs=normalized_r_inputs,
            rng=rng,
            output_bias=output_bias,
        )

        if (
            not candidate.is_canalizing(strong=strong, constant_functions_canalizing=constant_functions_canalizing)
            and not candidate.is_degenerate()
        ):
            return candidate


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
        allow_degenerate_functions: bool = False,
        *,
        r: int = 2,
        r_inputs: Sequence[int] | None = None,
        rng=None,
) -> Function:
    """
    Generate a random k-canalizing Boolean function.

    A Boolean function is k-canalizing if it has at least k
    successively canalizing variables. If exact_depth=True,
    its canalizing depth is exactly k; otherwise, its depth
    may exceed k.

    Multistate generation is not yet implemented.

    Parameters
    ----------
    n : int
        Number of input variables (n >= 1).
    k : int
        Requested canalizing depth, with 0 <= k <= n.
        Setting k=n generates a nested canalizing function.
    exact_depth : bool, optional
        If True, enforce exactly k canalizing variables.
        If False (default), enforce at least k.
    uniform_over_functions : bool, optional
        If True (default), sample uniformly over Boolean
        functions satisfying the requested constraints.
        If False, sample canalized outputs independently,
        inducing a nonuniform distribution over functions.
    allow_degenerate_functions : bool, optional
        If False (default), require all variables in the core 
        to be essential.
    r : int, optional
        Number of output states. Default is 2.
        Currently only Boolean functions are supported.
    r_inputs : Sequence[int] | None, optional
        Input cardinalities. If None, all inputs have r states.
        Currently only binary inputs are supported.
    rng : optional
        Random number generator or seed specification.

    Returns
    -------
    Function
        Random k-canalizing BooleanFunction.

    Raises
    ------
    ValueError
        If n or k is outside its allowed range.
    NotImplementedError
        If any input or output cardinality differs from 2.

    Notes
    -----
    When uniform_over_functions=True, rejection sampling
    corrects for combinatorial multiplicities associated
    with canalizing layers and possible merges with the core.

    Exact canalizing depth n-1 is impossible for Boolean
    functions with n > 1.

    References
    ----------
    He, Q., & Macauley, M. (2016).
        Stratification and enumeration of Boolean functions
        by canalizing depth.
        Physica D: Nonlinear Phenomena, 314, 1-8.

    Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022).
        Revealing the canalizing structure of Boolean functions:
        Algorithms and applications.
        Automatica, 146, 110630.
    """
    n = _validate_n(n, allow_zero=False)

    r, r_inputs, is_boolean = (
        utils_multistate.normalize_and_validate_state_specs(
            n, r, r_inputs
        )
    )

    if not is_boolean:
        raise NotImplementedError(
            "Multistate k-canalizing function generation "
            "is not yet implemented."
        )

    if (
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or not 0 <= k <= n
    ):
        raise ValueError(
            "k must be an integer satisfying 0 <= k <= n."
        )

    if exact_depth and n - k == 1:
        raise ValueError(
            "There are no Boolean functions of exact "
            "canalizing depth n-1."
        )

    rng = utils._coerce_rng(rng)

    if k == 0:
        if exact_depth:
            if allow_degenerate_functions:
                return random_non_canalizing_function(
                    n, rng=rng
                )
            return random_non_canalizing_non_degenerate_function(
                n, rng=rng
            )

        if allow_degenerate_functions:
            return random_function_with_bias(n, rng=rng)

        return random_non_degenerate_function(n, rng=rng)
    
    elif k == n - 1 and n > 1:
        # Boolean functions with depth n-1 have depth n.
        k = n

    # Step 1: Canalizing inputs and variables.
    aas = rng.integers(2, size=k)
    can_vars = rng.choice(n, k, replace=False)

    while True:
        # Step 2: Sample canalized outputs.
        if uniform_over_functions:
            bbs = sample_canalized_outputs_uniform_over_functions(
                k,
                _get_uniform_over_functions_weights(
                    k, is_ncf=(k >= n)
                ),
                rng=rng,
            )
        else:
            bbs = rng.integers(2, size=k)

        # Step 3: Sample the core function.
        if k < n:
            while True:
                core_function = random_k_canalizing_function(
                    n=n - k,
                    k=0,
                    exact_depth=exact_depth,
                    allow_degenerate_functions=allow_degenerate_functions,
                    rng=rng,
                )

                # The last prescribed canalizing variable must remain essential.
                # Thus a core function that is constant to this last output must
                # be rejected.
                if not allow_degenerate_functions or not np.all(core_function.f == bbs[-1]):
                    break

            if exact_depth or k == 0 or not uniform_over_functions:
                break

            # Correct for possible merges between the last
            # outer canalizing layer and the core function.
            core_function.get_layer_structure()

            if core_function.properties["CanalizingDepth"] == 0:
                break

            bbs_core = core_function.properties["CanalizedOutputs"]

            if bbs_core[0] != bbs[-1]:
                break

            # Determine size of the last outer layer.
            s = 1
            for i in range(k - 1, 0, -1):
                if bbs[i] == bbs[i - 1]:
                    s += 1
                else:
                    break

            # Determine size of the first core layer.
            s_core = 1
            for i in range(
                core_function.properties["CanalizingDepth"] - 1
            ):
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

    # Step 4: Construct the truth table.
    left_side = utils.get_left_side_of_truth_table(n)
    f = np.full(2**n, -1, dtype=np.int8)

    for j in range(k):
        mask = (
            (left_side[:, can_vars[j]] == aas[j])
            & (f < 0)
        )
        f[mask] = bbs[j]

    # Fill the remaining entries with the core function.
    f[f < 0] = np.asarray(core_function, dtype=np.int8)

    return BooleanFunction._from_f_unchecked(f)



def random_k_canalizing_function_with_specific_layer_structure(
    n: int,
    layer_structure: list,
    exact_depth: bool = False,
    *,
    allow_degenerate_functions: bool = False,
    r: int = 2,
    r_inputs: Sequence[int] | None = None,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random function with a specified canalizing layer structure.

    The canalizing layer structure is given as a list
    ``[k_1, ..., k_r]``, where each ``k_i`` specifies the number of
    canalizing variables in the i-th layer. The total canalizing depth is
    ``sum(layer_structure)``.

    If ``sum(layer_structure) == n`` and ``n > 1``, the function is a nested
    canalizing function and the final layer is required to have size at
    least 2.

    Multistate generation is not yet implemented.

    Parameters
    ----------
    n : int
        Total number of input variables.
    layer_structure : list of int
        Canalizing layer structure ``[k_1, ..., k_r]``. Each entry must be
        at least 1. If ``sum(layer_structure) == n`` and ``n > 1``, the final
        entry must satisfy ``layer_structure[-1] >= 2``.
    exact_depth : bool, optional
        If True, enforce that the canalizing depth is exactly
        ``sum(layer_structure)``. If False (default), additional canalizing
        variables may occur in the core function.
    allow_degenerate_functions : bool, optional
        If True, allow non-essential variables in the core function.
        If False (default), require all variables to be essential.
    r : int, optional
        Number of output states. Default is 2.
        Currently only Boolean functions are supported.
    r_inputs : Sequence[int] | None, optional
        Input cardinalities. If None, all inputs have r states.
        Currently only binary inputs are supported.
    rng : int, numpy.random.Generator, numpy.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    BooleanFunction
        A Boolean function on ``n`` variables with the prescribed canalizing
        layer structure, plus potentially more canalizing variables
        (if exact_depth=False).

    Raises
    ------
    AssertionError
        If the Boolean canalizing layer structure constraints are invalid.
    ValueError
        If n or the state cardinalities are invalid.
    NotImplementedError
        If any input or output cardinality differs from 2.

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
    n = _validate_n(n, allow_zero=False)

    r, normalized_r_inputs, is_boolean = (
        utils_multistate.normalize_and_validate_state_specs(
            n, r, r_inputs
        )
    )

    if not is_boolean:
        raise NotImplementedError(
            "Multistate canalizing layer structure generation "
            "is not yet implemented."
        )

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

    if depth < n:
        while True:
            core_function = random_k_canalizing_function(
                n=n - depth,
                k=0,
                exact_depth=exact_depth,
                allow_degenerate_functions=allow_degenerate_functions,
                rng=rng,
            )

            # The last prescribed canalizing variable must remain essential.
            # Reject a core function that is constant to its canalized output.
            if (
                not allow_degenerate_functions
                or depth == 0
                or np.any(core_function.f != bbs[-1])
            ):
                break
    else:
        core_function = [1 - bbs[-1]]

    left_side_of_truth_table = utils.get_left_side_of_truth_table(n)
    f = np.full(size_state_space, -1, dtype=np.int8)

    for j in range(depth):
        mask = (
            (left_side_of_truth_table[:, can_vars[j]] == aas[j])
            & (f < 0)
        )
        f[mask] = bbs[j]

    # Fill remaining entries with core truth table.
    f[f < 0] = np.asarray(core_function, dtype=np.int8)

    return BooleanFunction._from_f_unchecked(f)


def random_NCF(
    n: int,
    uniform_over_functions: bool = True,
    layer_structure: list | None = None,
    *,
    r: int = 2,
    r_inputs: Sequence[int] | None = None,
    rng=None,
) -> BooleanFunction:
    """
    Generate a random nested canalizing function.

    A nested canalizing function (NCF) is an n-canalizing function,
    i.e., a function whose canalizing depth equals the number of variables.
    Optionally, a specific canalizing layer structure may be prescribed.

    Multistate generation is not yet implemented.

    Parameters
    ----------
    n : int
        Total number of input variables.
    uniform_over_functions : bool, optional
        If True (default) and ``layer_structure`` is None, sample uniformly
        over nested canalizing Boolean functions. This induces a generally
        nonuniform distribution over canalizing layer structures.

        If False, canalized outputs are sampled independently and uniformly
        as bitstrings, inducing a bias toward more symmetric structures.

        This parameter is ignored if ``layer_structure`` is provided.
    layer_structure : list of int or None, optional
        Canalizing layer structure ``[k_1, ..., k_r]``. Each entry must be at
        least 1. If provided, it must satisfy ``sum(layer_structure) == n``.
        If ``n > 1``, the final entry must satisfy
        ``layer_structure[-1] >= 2``. If None (default), the layer structure
        is sampled at random.
    r : int, optional
        Number of output states. Default is 2.
        Currently only Boolean functions are supported.
    r_inputs : Sequence[int] | None, optional
        Input cardinalities. If None, all inputs have r states.
        Currently only binary inputs are supported.
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
        If the Boolean canalizing layer structure constraints are invalid.
    ValueError
        If n or the state cardinalities are invalid.
    NotImplementedError
        If any input or output cardinality differs from 2.

    Notes
    -----
    For fixed parameter values, this function samples uniformly at random
    from the ensemble of nested canalizing Boolean functions consistent with
    the specified constraints, unless ``uniform_over_functions=False``.

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
    n = _validate_n(n, allow_zero=False)

    r, normalized_r_inputs, is_boolean = (
        utils_multistate.normalize_and_validate_state_specs(
            n, r, r_inputs
        )
    )

    if not is_boolean:
        raise NotImplementedError(
            "Multistate nested canalizing function generation "
            "is not yet implemented."
        )

    rng = utils._coerce_rng(rng)

    if layer_structure is None:
        return random_k_canalizing_function(
            n,
            k=n,
            uniform_over_functions=uniform_over_functions,
            exact_depth=False,
            rng=rng,
        )

    assert sum(layer_structure) == n, "Ensure sum(layer_structure) == n."
    assert layer_structure[-1] > 1 or n == 1, (
        "The last layer of an NCF has to have size >= 2 whenever n > 1.\n"
        "Ensure that layer_structure[-1]>=2."
    )

    return random_k_canalizing_function_with_specific_layer_structure(
        n,
        layer_structure,
        exact_depth=False,
        rng=rng,
    )


