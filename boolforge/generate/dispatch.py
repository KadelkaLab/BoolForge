#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-level dispatch for random finite-state function generation.

Selects generators according to canalizing structure, parity,
Hamming weight, output distribution, and degeneracy constraints.
Some generation modes are currently restricted to Boolean functions.
"""

import numpy as np
from collections.abc import Sequence

from ..function import Function, BooleanFunction
from .. import utils, utils_multistate

from .functions import (
    _validate_n,
    _resolve_output_bias,
    random_function_with_bias,
    random_function_with_exact_hamming_weight,
    random_parity_function,
    random_non_degenerate_function,
)

from .canalization import (
    random_k_canalizing_function,
    random_k_canalizing_function_with_specific_layer_structure,
    random_non_canalizing_function,
    random_non_canalizing_non_degenerate_function,
)

def _validate_bias(bias: float) -> None:
    if not isinstance(bias, (float, int, np.floating)):
        raise TypeError("bias must be a float")
    if not (0.0 <= bias <= 1.0):
        raise ValueError("bias must be in [0, 1]")

def _validate_absolute_bias(absolute_bias: float) -> None:
    """Validate a Boolean absolute-bias parameter."""
    if (
        isinstance(absolute_bias, (bool, np.bool_))
        or not isinstance(
            absolute_bias, (int, float, np.integer, np.floating)
        )
        or not 0 <= absolute_bias <= 1
    ):
        raise ValueError("absolute_bias must be between 0 and 1.")


def _validate_hamming_weight(
        n: int,
        hamming_weight: int,
        *,
        exact_depth: bool,
) -> None:
    """Validate a prescribed Boolean Hamming weight."""
    if (
        isinstance(hamming_weight, (bool, np.bool_))
        or not isinstance(hamming_weight, (int, np.integer))
    ):
        raise TypeError("hamming_weight must be an integer")

    if not 0 <= hamming_weight <= 2**n:
        raise ValueError(
            "hamming_weight must satisfy 0 <= hamming_weight <= 2**n"
        )

    if exact_depth and not (1 < hamming_weight < 2**n - 1):
        raise ValueError(
            "If exact_depth=True and depth=0, hamming_weight "
            "must satisfy 2 <= hamming_weight <= 2**n - 2."
        )


def random_function(
        n: int,
        depth: int = 0,
        exact_depth: bool = False,
        uniform_over_functions: bool = True,
        layer_structure: list[int] | None = None,
        parity: bool = False,
        allow_degenerate_functions: bool = False,
        bias: float | None = None,
        absolute_bias: float = 0,
        use_absolute_bias: bool = False,
        hamming_weight: int | None = None,
        *,
        r: int = 2,
        r_inputs: Sequence[int] | None = None,
        output_bias: Sequence[float] | None = None,
        rng=None,
) -> Function:
    """
    Generate a random Boolean or multistate function under flexible constraints.

    This function acts as a high-level generator that unifies several common
    ensembles of finite-state functions, including parity functions, canalizing
    functions of specified depth or layer structure, functions with fixed
    Hamming weight, and functions with prescribed output distributions.

    Some generation modes are currently restricted to Boolean functions.
    The first applicable generation rule (in the order described below)
    is applied.

    By default, calling ``random_function(n)`` generates a uniformly
    random non-degenerate Boolean function on ``n`` variables, i.e.,
    a function in which every input variable is essential. To allow degenerate 
    functions, set ``allow_degenerate_functions=True``.    

    Selection logic (first applicable rule is used)
    ------------------------------------------------

    1. If ``parity`` is True, return a random Boolean parity function
    (see ``random_parity_function``). This mode requires Boolean
    inputs and outputs.

    2. Else, if ``layer_structure`` is provided, return a Boolean function
    with the specified canalizing layer structure using
    ``random_k_canalizing_function_with_specific_layer_structure``.
    Exactness of the canalizing depth is controlled by ``exact_depth``.
    Multistate generation is not yet implemented for this mode.

    3. Else, if ``depth > 0``, return a k-canalizing function with
    ``k = min(depth, n)`` using ``random_k_canalizing_function``.
    If ``exact_depth`` is True, the function has exactly this depth;
    otherwise, its canalizing depth is at least ``k``.
    Multistate generation is not yet implemented for this mode.

    If ``uniform_over_functions`` is True, functions are sampled
    uniformly from the admissible Boolean functions (up to the
    imposed constraints). This induces a generally nonuniform
    distribution over canalizing layer structures.

    If False, canalized outputs are sampled independently and uniformly
    as bitstrings, inducing a bias toward more symmetric layer
    structures.

    4. Else, if ``hamming_weight`` is provided, repeatedly sample Boolean
    functions with the specified Hamming weight until additional
    constraints implied by ``exact_depth`` and
    ``allow_degenerate_functions`` are satisfied.
    This mode requires Boolean inputs and outputs.

    5. Else, generate a random Boolean or multistate function with
    independently sampled truth-table outputs using one of the
    following distributions:

    - ``bias``: probability of output 1, available whenever r=2,
        regardless of the input cardinalities.
    - ``output_bias``: probabilities for all r output states.
    - ``use_absolute_bias=True``: randomly choose a binary-output
        bias from the two values determined by ``absolute_bias``.
    - If none of these are specified, sample outputs uniformly
        from {0, ..., r-1}.

    ``bias`` and ``output_bias`` cannot both be specified, and neither
    can be combined with ``use_absolute_bias=True``.

    Additional constraints on canalization and degeneracy are enforced
    depending on ``exact_depth`` and ``allow_degenerate_functions``.
    For ``depth=0`` and ``exact_depth=True``, the generated function
    is required to be non-canalizing.

    Parameters
    ----------
    n : int
        Number of input variables (n >= 1).
    depth : int, optional
        Requested canalizing depth. If positive, generate a function
        with at least this depth, or exactly this depth when
        exact_depth=True. Default is 0.
    exact_depth : bool, optional
        If True, require exact canalizing depth. When depth=0,
        require a non-canalizing function. Default is False.
    uniform_over_functions : bool, optional
        Whether to apply the function-uniform sampling correction
        in the Boolean canalizing-depth generator. Default is True.
    layer_structure : list[int] | None, optional
        Prescribed canalizing layer structure. Takes precedence
        over depth. Default is None.
    parity : bool, optional
        If True, generate a Boolean parity function, taking
        precedence over all other generation options.
    allow_degenerate_functions : bool, optional
        If False (default), require all variables to be essential
        in the random-sampling branches.
    bias : float | None, optional
        Probability of output 1 for functions with r=2.
        If None (default), no scalar bias is specified.
    absolute_bias : float, optional
        Boolean absolute-bias parameter in [0, 1].
        Used only when use_absolute_bias=True.
        Default is 0.
    use_absolute_bias : bool, optional
        If True, choose the scalar bias uniformly from
        (1-absolute_bias)/2 and (1+absolute_bias)/2.
        Default is False.
    hamming_weight : int | None, optional
        Prescribed number of ones in the Boolean truth table.
        Default is None.
    r : int, optional
        Number of output states. Default is 2.
    r_inputs : Sequence[int] | None, optional
        Input cardinalities. If None, all inputs have r states.
    output_bias : Sequence[float] | None, optional
        Probabilities for output states 0, ..., r-1.
        If None and no scalar bias is specified, outputs are
        sampled uniformly.
    rng : optional
        Random number generator or seed specification.

    Returns
    -------
    Function
        Generated function. Returns a BooleanFunction when all
        input and output cardinalities are 2.

    Raises
    ------
    ValueError
        If generation parameters are invalid or incompatible.
    NotImplementedError
        If a Boolean-only generation mode is requested with
        multistate cardinalities.

    Notes
    -----
    Rejection-based generation preserves the underlying
    independent output distribution conditioned on the
    requested properties.

    For multistate functions, exact_depth=True with depth=0
    currently excludes weakly canalizing functions.

    Examples
    --------
    >>> f = random_function(5)
    >>> f = random_function(5, depth=2)
    >>> f = random_function(5, bias=0.3)
    >>> f = random_function(3, r=3)
    >>> f = random_function(
    ...     2, r=3, r_inputs=[2, 3],
    ...     output_bias=[0.2, 0.5, 0.3]
    ... )
    """
    n = _validate_n(n, allow_zero=False)

    r, validated_r_inputs, is_boolean = (
        utils_multistate.normalize_and_validate_state_specs(
            n, r, r_inputs
        )
    )

    rng = utils._coerce_rng(rng)

    # --------------------------------------------------
    # 1. Parity
    # --------------------------------------------------
    if parity:
        if not is_boolean:
            raise NotImplementedError(
                "Multistate parity generation is not yet implemented."
            )

        return random_parity_function(n, rng=rng)

    # --------------------------------------------------
    # 2. Prescribed layer structure
    # --------------------------------------------------
    if layer_structure is not None:
        return random_k_canalizing_function_with_specific_layer_structure(
            n,
            layer_structure,
            exact_depth=exact_depth,
            r=r,
            r_inputs=validated_r_inputs,
            rng=rng,
        )

    # --------------------------------------------------
    # 3. Canalizing depth
    # --------------------------------------------------
    if (
        isinstance(depth, (bool, np.bool_))
        or not isinstance(depth, (int, np.integer))
        or depth < 0
    ):
        raise ValueError("depth must be a nonnegative integer.")

    if depth > 0:
        return random_k_canalizing_function(
            n,
            min(depth, n),
            exact_depth=exact_depth,
            uniform_over_functions=uniform_over_functions,
            r=r,
            r_inputs=validated_r_inputs,
            rng=rng,
        )

    # --------------------------------------------------
    # 4. Fixed Hamming weight
    # --------------------------------------------------
    if hamming_weight is not None:
        if not is_boolean:
            raise NotImplementedError(
                "Fixed Hamming weight generation is currently "
                "implemented only for Boolean functions."
            )

        _validate_hamming_weight(
            n,
            hamming_weight,
            exact_depth=exact_depth,
        )

        while True:
            f = random_function_with_exact_hamming_weight(
                n, hamming_weight, rng=rng
            )

            if allow_degenerate_functions:
                if not exact_depth or not f.is_canalizing():
                    return f
            else:
                if (
                    (not exact_depth or not f.is_canalizing())
                    and not f.is_degenerate()
                ):
                    return f

    # --------------------------------------------------
    # 5. Resolve output distribution
    # --------------------------------------------------
    if use_absolute_bias:
        if bias is not None or output_bias is not None:
            raise ValueError(
                "use_absolute_bias cannot be combined with "
                "bias or output_bias."
            )

        if r != 2:
            raise ValueError(
                "Absolute bias requires binary outputs (r=2)."
            )

        _validate_absolute_bias(absolute_bias)

        bias = float(rng.choice([
            0.5 * (1 - absolute_bias),
            0.5 * (1 + absolute_bias),
        ]))

    # Validate once before selecting a rejection-based generator.
    probabilities = _resolve_output_bias(
        bias, output_bias, r
    )

    # --------------------------------------------------
    # 6. Generate with the requested structural constraints
    # --------------------------------------------------
    kwargs = dict(
        r=r,
        r_inputs=validated_r_inputs,
        output_bias=probabilities,
        rng=rng,
    )

    if allow_degenerate_functions:
        if exact_depth:
            return random_non_canalizing_function(
                n, **kwargs
            )

        return random_function_with_bias(
            n, **kwargs
        )

    if exact_depth:
        return random_non_canalizing_non_degenerate_function(
            n, **kwargs
        )

    return random_non_degenerate_function(
        n, **kwargs
    )
