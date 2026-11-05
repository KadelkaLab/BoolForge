#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:17:28 2026

@author: ckadelka
"""

import numpy as np
from collections.abc import Sequence
from math import prod

from .utils import _is_number

def _is_boolean(n_states, n_states_inputs):
    if n_states_inputs is None:
        return n_states == 2
    return n_states == 2 and np.all(k == 2 for k in n_states_inputs)

def _is_homogeneous(n_states, n_states_inputs):
    return n_states_inputs is None or np.all(k == n_states for k in n_states_inputs)


def normalize_and_validate_state_specs(
        n: int,
        n_states: int,
        n_states_inputs=None,
) -> tuple[int, np.ndarray, bool]:
    """
    Validate and normalize state cardinalities.

    Returns the output cardinality, input cardinalities,
    and whether all cardinalities are Boolean.
    """
    if (isinstance(n_states, (bool, np.bool_))
            or not isinstance(n_states, (int, np.integer))
            or not 2 <= n_states <= 255):
        raise ValueError("n_states must be an integer between 2 and 255.")

    if n_states_inputs is None:
        n_states_inputs = np.full(n, n_states, dtype=int)
    elif isinstance(n_states_inputs, (int, np.integer)):
        n_states_inputs = np.full(n, n_states_inputs, dtype=int)
    else:
        raw = np.asarray(n_states_inputs)

        if (raw.shape != (n,)
                or (raw.size > 0 and raw.dtype.kind not in "iu")):
            raise ValueError(
                "n_states_inputs must be a sequence of n integers."
            )

        n_states_inputs = raw.astype(int, copy=False)

    if np.any((n_states_inputs < 2) | (n_states_inputs > 255)):
        raise ValueError(
            "All input cardinalities must be between 2 and 255."
        )

    n_states = int(n_states)
    is_boolean = _is_boolean(n_states, n_states_inputs)

    return n_states, n_states_inputs, is_boolean


# def mix2dec(vector : Sequence[int], radices : Sequence[int]) -> int:
#     """
#     Convert a mixed-radix vector to an integer.
    
#     Parameters
#     ----------
#     vector : Sequence of int
#         Digits ordered from least significant digit to most significant digit.
#     radices : Sequence of int
#         Radices for each digit in the vector.
    
#     Returns
#     -------
#     int
#         Integer represented by the mixed-radix vector.
#     """
#     column = 1
#     decimal = 0
#     for i, v in enumerate(vector):
#         decimal += v * column
#         column *= radices[i]
#     return decimal


def mix2dec(vector: Sequence[int], radices: Sequence[int]) -> int:
    """
    Convert a mixed-radix vector to an integer.

    Digits are ordered from most significant to least significant,
    consistent with BoolForge's lexicographic truth-table ordering.

    Parameters
    ----------
    vector : Sequence[int]
        Input digits.
    radices : Sequence[int]
        Cardinality of each digit.

    Returns
    -------
    int
        Integer represented by the mixed-radix vector.
    """
    decimal = 0
    for value, radix in zip(vector, radices):
        decimal = decimal * int(radix) + int(value)
    return decimal


# def dec2mix(decimal : int, radices : Sequence[int]) -> list[int]:
#     """
#     Convert a nonnegative integer into a mixed-radix vector.

#     Parameters
#     ----------
#     decimal : int
#         Nonnegative integer to convert.
#     radices : Sequence[int]
#         Radices for each digit in the resulting vector.

#     Returns
#     -------
#     list[int]
#         Digits ordered from least significant digit to most significant digit.
#     """
#     vector = []
#     for radix in radices:
#         vector.append(decimal % radix)
#         decimal //= radix
#     return vector


def dec2mix(decimal: int, radices: Sequence[int]) -> list[int]:
    """
    Convert a nonnegative integer to a mixed-radix vector.

    Digits are ordered from most significant to least significant,
    consistent with BoolForge's lexicographic truth-table ordering.

    Parameters
    ----------
    decimal : int
        Nonnegative integer to convert.
    radices : Sequence[int]
        Cardinality of each digit.

    Returns
    -------
    list[int]
        Mixed-radix digits in most-significant-first order.
    """
    vector = [0] * len(radices)

    for i in range(len(radices) - 1, -1, -1):
        radix = int(radices[i])
        vector[i] = int(decimal % radix)
        decimal //= radix

    return vector


def mix2bin(
        vector: Sequence[int],
        radices: Sequence[int],
) -> list[int]:
    """
    Convert a mixed-radix vector to a compact binary vector.

    The mixed-radix digits are ordered from most significant to
    least significant. The entire state is encoded as one integer,
    using the minimum number of bits required for the state space.

    Returns
    -------
    list[int]
        Binary digits in most-significant-bit-first order.
    """
    decimal = mix2dec(vector, radices)
    nbits = (prod(radices) - 1).bit_length()

    return [
        (decimal >> i) & 1
        for i in range(nbits - 1, -1, -1)
    ]


def bin2mix(
        binary_vector: Sequence[int],
        radices: Sequence[int],
) -> list[int]:
    """
    Convert a compact binary vector to a mixed-radix vector.

    Binary digits and mixed-radix digits are ordered from most
    significant to least significant.

    Returns
    -------
    list[int]
        Decoded mixed-radix vector.
    """
    decimal = 0

    for bit in binary_vector:
        decimal = (decimal << 1) | int(bit)

    if decimal >= prod(radices):
        raise ValueError(
            "Binary vector represents a state outside the "
            "mixed-radix state space."
        )

    return dec2mix(decimal, radices)


def mix2binf(
        vector: Sequence[int],
        radices: Sequence[int],
) -> list[int]:
    """
    Convert a mixed-radix vector to a formatted binary vector.

    Each digit is encoded separately using ceil(log2(radix)) bits.
    Digits and binary groups are ordered most-significant-first.

    Returns
    -------
    list[int]
        Concatenated binary representations of the input digits.
    """
    binary_vector = []

    for value, radix in zip(vector, radices):
        width = (int(radix) - 1).bit_length()

        binary_vector.extend(
            (int(value) >> i) & 1
            for i in range(width - 1, -1, -1)
        )

    return binary_vector


def binf2mix(
        binary_vector: Sequence[int],
        radices: Sequence[int],
) -> list[int] | None:
    """
    Convert a formatted binary vector to a mixed-radix vector.

    Each digit is decoded from ceil(log2(radix)) bits.
    Returns None if any decoded digit exceeds its radix.

    Returns
    -------
    list[int] | None
        Decoded mixed-radix vector, or None for an invalid
        encoded state.
    """
    vector = []
    offset = 0

    for radix in radices:
        width = (int(radix) - 1).bit_length()
        value = 0

        for bit in binary_vector[offset:offset + width]:
            value = (value << 1) | int(bit)

        if value >= radix:
            return None

        vector.append(value)
        offset += width

    return vector



left_side_of_truth_tables = {}

def get_left_side_of_truth_table_multistate(
        R: Sequence[int],
) -> np.ndarray:
    """
    Return all mixed-radix input configurations for a multistate function.

    Rows are ordered lexicographically, with the first variable
    corresponding to the most significant digit.

    Parameters
    ----------
    R : Sequence[int]
        Input cardinalities.

    Returns
    -------
    np.ndarray
        Array of shape (prod(R), len(R)) containing all
        possible input configurations.
    """
    R = tuple(int(r) for r in R)

    if R not in left_side_of_truth_tables:
        N = prod(R)
        indices = np.arange(N, dtype=np.int64)
        table = np.empty((N, len(R)), dtype=np.uint8)

        for i in range(len(R) - 1, -1, -1):
            table[:, i] = indices % R[i]
            indices //= R[i]

        left_side_of_truth_tables[R] = table

    return left_side_of_truth_tables[R]


# old style, to be reworked:
_LOGIC_MAP = {
    "AND": "&",
    "and": "&",
    "&&": "&",
    "&": "&",
    "OR": "|",
    "or": "|",
    "||": "|",
    "|": "|",
    "NOT": "~",
    "not": "~",
    "!": "~",
    "~": "~",
}
_COMPARE_OPS = {"==", "!=", ">=", "<=", ">", "<"}
_ARITH_OPS = {"+", "-", "*", "%"}


def f_from_expression(expr, R, max_degree = 16):
    expr = expr.replace('(', ' ( ').replace(')', ' ) ')
    raw_tokens = expr.split()
    tokens = []
    variables = []
    seen = set()
    
    for token in raw_tokens:
        if token in {"(", ")"}:
            tokens.append(token)
            continue
        if token in _LOGIC_MAP:
            tokens.append(_LOGIC_MAP[token])
            continue
        if token in _COMPARE_OPS:
            tokens.append(token)
            continue
        if token in _ARITH_OPS:
            tokens.append(token)
            continue
        if _is_number(token):
            tokens.append(token)
            continue
        if token not in seen:
            seen.add(token)
            variables.append(token)
        token.appnd(token)
    n = len(variables)
    if n > max_degree:
        return np.array([], np.uint8), np.array(variables)
    safe_map = { var: f"v{i}" for i, var in enumerate(variables) }
    safe_tokens = [
        safe_map[token] if token in safe_map else token for token in tokens
    ]
    expr_mod = " ".join(safe_tokens)
    truth_table = get_left_side_of_truth_table_multistate(n, R)
    local_dict = {
        safe_map[var]: truth_table[:, i].astype(np.int64) for i, var in enumerate(variables)
    }
    try:
        result = eval(expr_mod, {"__builtins__":None},local_dict)
    except Exception as e:
        raise ValueError(f"Error evaluating expression: \n{expr}\nParsed as:\n{expr_mod}\nError: {e}")
    result = np.asarray(result)
    if n == 0:
        result = np.array([int(result)], np.int64)
    else:
        result = result.astype(np.int64)
    result %= np.prod(R)
    return result.astype(np.uint8), np.array(variables)