#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random discrete function generators.

Provides generators with prescribed output distributions, Hamming
weight, parity, and essential-variable constraints. Some generators
are restricted to Boolean functions.
"""

import numpy as np
from collections.abc import Sequence
from math import prod

from ..function import Function, BooleanFunction
from .. import utils, utils_multistate


def _validate_n(n: int, *, allow_zero: bool = True) -> int:
    """Validate the number of input variables."""
    if (
        isinstance(n, (bool, np.bool_))
        or not isinstance(n, (int, np.integer))
        or n < (0 if allow_zero else 1)
    ):
        raise ValueError(
            "n must be a nonnegative integer"
            if allow_zero else
            "n must be a positive integer"
        )
    return int(n)


def _validate_output_bias(
        output_bias,
        r: int,
) -> np.ndarray:
    """
    Validate an output probability distribution.

    Parameters
    ----------
    output_bias : Sequence[float]
        Probabilities for output states 0, ..., r-1.
    r : int
        Number of output states.

    Returns
    -------
    np.ndarray
        Validated probability vector of length r.
    """
    try:
        p = np.asarray(output_bias, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "output_bias must contain numeric probabilities."
        ) from exc

    if p.shape != (r,):
        raise ValueError(
            f"output_bias must contain exactly {r} probabilities."
        )

    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError(
            "output_bias must contain finite, nonnegative probabilities."
        )

    total = p.sum()

    if not np.isclose(total, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "output_bias probabilities must sum to 1."
        )

    return p / total



def _resolve_output_bias(
        bias: float | None,
        output_bias: Sequence[float] | None,
        r: int,
) -> np.ndarray | None:
    """
    Resolve scalar bias and output probabilities.

    Returns None for uniform sampling, or a validated
    probability vector of length r.
    """
    if bias is not None and output_bias is not None:
        raise ValueError(
            "Specify either bias or output_bias, not both."
        )

    if bias is not None:
        if r != 2:
            raise ValueError(
                "Scalar bias requires binary outputs (r=2)."
            )

        if (
            isinstance(bias, (bool, np.bool_))
            or not isinstance(
                bias, (int, float, np.integer, np.floating)
            )
            or not 0 <= bias <= 1
        ):
            raise ValueError("bias must be between 0 and 1.")

        return np.array([1 - bias, bias], dtype=float)

    if output_bias is not None:
        return _validate_output_bias(output_bias, r)

    return None



def _sample_function_unchecked(
        r: int,
        r_inputs: np.ndarray,
        *,
        rng,
        output_bias: np.ndarray | None = None,
) -> Function:
    """
    Sample a finite-state function from a validated distribution.

    Parameters are assumed to be validated and normalized.

    If output_bias is None, truth-table outputs are sampled
    uniformly. Otherwise, outputs are sampled independently
    according to the specified probability vector.
    """
    size = prod(r_inputs)

    if output_bias is None:
        truth = rng.integers(r, size=size, dtype=np.uint8)
    else:
        truth = rng.choice(r, size=size, p=output_bias)

    cls = (
        BooleanFunction
        if r == 2 and np.all(r_inputs == 2)
        else Function
    )

    return cls._from_f_unchecked(
        truth,
        r=r,
        r_inputs=r_inputs,
    )




def random_function_with_bias(
        n: int,
        bias: float | None = None,
        *,
        r: int = 2,
        r_inputs: Sequence[int] | None = None,
        output_bias: Sequence[float] | None = None,
        rng=None,
) -> Function:
    """
    Generate a random finite-state function with specified output probabilities.

    Truth-table entries are sampled independently. For Boolean functions,
    ``bias`` specifies the probability of output 1. For arbitrary finite-state
    functions, ``output_bias`` specifies the probabilities of all output states.

    If neither parameter is provided, outputs are sampled uniformly.

    Parameters
    ----------
    n : int
        Number of input variables (n >= 0).
    bias : float | None, optional
        Probability of output 1 for functions with binary outputs.
        If None (default), no scalar bias is specified.
    r : int, optional
        Number of output states. Default is 2.
    r_inputs : Sequence[int] | None, optional
        Input cardinalities. If None, all inputs have r states.
    output_bias : Sequence[float] | None, optional
        Probabilities for output states 0, ..., r-1.
        Must sum to 1. Default is None.
    rng : optional
        Random number generator or seed specification.

    Returns
    -------
    Function
        Random finite-state function. Returns BooleanFunction when
        all input and output cardinalities are 2.

    Raises
    ------
    ValueError
        If the probability specifications are invalid or incompatible.
    """
    n = _validate_n(n)

    r, r_inputs, _ = (
        utils_multistate.normalize_and_validate_state_specs(
            n, r, r_inputs
        )
    )

    output_bias = _resolve_output_bias(
        bias, output_bias, r
    )

    rng = utils._coerce_rng(rng)

    return _sample_function_unchecked(
        r,
        r_inputs,
        rng=rng,
        output_bias=output_bias,
    )


def random_function_with_exact_hamming_weight(
        n: int,
        hamming_weight: int,
        *,
        rng=None,
) -> BooleanFunction:
    """
    Generate a random Boolean function with a fixed Hamming weight.

    All truth tables containing exactly ``hamming_weight`` ones
    are sampled uniformly at random.

    Parameters
    ----------
    n : int
        Number of input variables (n >= 0).
    hamming_weight : int
        Number of ones in the truth table.
    rng : optional
        Random number generator or seed specification.

    Returns
    -------
    BooleanFunction
        Random Boolean function with the specified Hamming weight.
    """
    n = _validate_n(n)

    if (
        isinstance(hamming_weight, (bool, np.bool_))
        or not isinstance(hamming_weight, (int, np.integer))
    ):
        raise TypeError("hamming_weight must be an integer")

    if not 0 <= hamming_weight <= 2**n:
        raise ValueError(
            "hamming_weight must satisfy 0 <= hamming_weight <= 2**n"
        )

    rng = utils._coerce_rng(rng)

    f = np.zeros(2**n, dtype=np.uint8)
    one_indices = rng.choice(
        2**n, size=int(hamming_weight), replace=False
    )
    f[one_indices] = 1

    return BooleanFunction._from_f_unchecked(f)


def random_parity_function(
        n: int,
        *,
        rng=None,
) -> BooleanFunction:
    """
    Generate a random parity Boolean function.

    Returns either the parity function (XOR of all inputs)
    or its complement, chosen uniformly at random.

    Parameters
    ----------
    n : int
        Number of input variables (n >= 1).
    rng : optional
        Random number generator or seed specification.

    Returns
    -------
    BooleanFunction
        Random parity function on n variables.

    Notes
    -----
    Parity functions are non-degenerate, non-canalizing for
    n > 1, and have maximal average sensitivity.
    """
    n = _validate_n(n, allow_zero=False)
    rng = utils._coerce_rng(rng)

    indices = np.arange(2**n, dtype=np.uint64)
    parity = np.zeros(2**n, dtype=np.uint8)

    for i in range(n):
        parity ^= ((indices >> i) & 1).astype(np.uint8)

    if rng.integers(2):
        parity ^= 1

    return BooleanFunction._from_f_unchecked(parity)


def random_non_degenerate_function(
        n: int,
        bias: float | None = None,
        *,
        r: int = 2,
        r_inputs: Sequence[int] | None = None,
        output_bias: Sequence[float] | None = None,
        rng=None,
) -> Function:
    """
    Generate a random non-degenerate finite-state function.

    A function is non-degenerate if every input variable is
    essential. Truth-table entries are sampled independently,
    and functions are rejected until all inputs are essential.

    For binary outputs, ``bias`` specifies the probability of
    output 1. For arbitrary output cardinalities, ``output_bias``
    specifies the probabilities of all output states. If neither
    is provided, outputs are sampled uniformly.

    Parameters
    ----------
    n : int
        Number of input variables (n >= 1).
    bias : float | None, optional
        Probability of output 1 for binary-valued functions.
        Default is None.
    r : int, optional
        Number of output states. Default is 2.
    r_inputs : Sequence[int] | None, optional
        Input cardinalities. If None, all inputs have r states.
    output_bias : Sequence[float] | None, optional
        Probabilities for output states 0, ..., r-1.
        Must sum to 1. Default is None.
    rng : optional
        Random number generator or seed specification.

    Returns
    -------
    Function
        Random non-degenerate function. Returns BooleanFunction
        when all input and output cardinalities are 2.

    Raises
    ------
    ValueError
        If the parameters are invalid or the output distribution
        cannot generate a non-degenerate function.

    Notes
    -----
    Rejection sampling preserves the product distribution
    conditioned on all input variables being essential.
    """
    n = _validate_n(n, allow_zero=False)

    r, r_inputs, _ = (
        utils_multistate.normalize_and_validate_state_specs(
            n, r, r_inputs
        )
    )

    output_bias = _resolve_output_bias(
        bias, output_bias, r
    )

    # A deterministic output distribution produces only constants.
    if (
        output_bias is not None
        and np.count_nonzero(output_bias > 0) < 2
    ):
        raise ValueError(
            "Non-degenerate functions require at least two "
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

        if not candidate.is_degenerate():
            return candidate





def random_degenerate_function(
        n: int,
        *,
        r: int = 2,
        r_inputs: Sequence[int] | None = None,
        rng=None,
) -> Function:
    """
    Generate a uniformly random degenerate finite-state function.

    Samples a forced nonessential variable with probability
    proportional to the number of residual functions, then
    accepts candidates with probability 1/k, where k is the
    number of nonessential variables.

    This ensures uniform sampling over degenerate functions,
    including those with heterogeneous input cardinalities.
    """
    n = _validate_n(n, allow_zero=False)

    r, r_inputs, is_boolean = (
        utils_multistate.normalize_and_validate_state_specs(
            n, r, r_inputs
        )
    )

    rng = utils._coerce_rng(rng)
    cls = BooleanFunction if is_boolean else Function

    size = prod(r_inputs)

    # Number of possible residual functions is r**(size/r_i).
    # Use log weights to avoid overflow.
    log_weights = np.array(
        [(size // int(ri)) * np.log(r) for ri in r_inputs],
        dtype=float,
    )
    weights = np.exp(log_weights - np.max(log_weights))
    probabilities = weights / weights.sum()

    while True:
        axis = int(rng.choice(n, p=probabilities))

        reduced_radices = np.delete(r_inputs, axis)
        residual = rng.integers(
            r,
            size=prod(reduced_radices),
            dtype=np.uint8,
        )

        table = residual.reshape(tuple(reduced_radices))
        truth = np.repeat(
            np.expand_dims(table, axis=axis),
            int(r_inputs[axis]),
            axis=axis,
        ).ravel()

        candidate = cls._from_f_unchecked(
            truth,
            r=r,
            r_inputs=r_inputs,
        )

        k = n - candidate.get_number_of_essential_variables()

        if rng.random() < 1.0 / k:
            return candidate

            return candidate
