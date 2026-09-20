#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:17:28 2026

@author: ckadelka
"""

import numpy as np
from collections.abc import Sequence

from utils import _is_number

def _is_boolean(n_states, n_states_inputs):
    if n_states_inputs is None:
        return n_states == 2
    return n_states == 2 and np.all(k == 2 for k in n_states_inputs)

def _is_homogeneous(n_states, n_states_inputs):
    return n_states_inputs is None or np.all(k == n_states for k in n_states_inputs)

def normalize_and_validate_state_specs(
    n: int,
    n_states: int,
    n_states_inputs,
) -> tuple[int, np.ndarray]:
    
    # validate output states
    if not isinstance(n_states, (int, np.integer)):
        raise TypeError("n_states must be an integer")
    if n_states < 2:
        raise ValueError("n_states must be >= 2")

    # normalize input states
    if isinstance(n_states_inputs, (int, np.integer)):
        n_states_inputs = np.full(n, n_states_inputs, dtype=int)
    else:
        n_states_inputs = np.asarray(n_states_inputs, dtype=int)

    if n_states_inputs.shape != (n,):
        raise ValueError("n_states_inputs must have length n")

    if np.any(n_states_inputs < 2):
        raise ValueError("All n_states_inputs must be >= 2")

    is_boolean = _is_boolean(n_states, n_states_inputs)

    return n_states, n_states_inputs, is_boolean

def mix2dec(vector : Sequence[int], radices : Sequence[int]) -> int:
    """
    Convert a mixed-radix vector to an integer.
    
    Parameters
    ----------
    vector : Sequence of int
        Digits ordered from least significant digit to most significant digit.
    radices : Sequence of int
        Radices for each digit in the vector.
    
    Returns
    -------
    int
        Integer represented by the mixed-radix vector.
    """
    column = 1
    decimal = 0
    for i, v in enumerate(vector):
        decimal += v * column
        column *= radices[i]
    return decimal

def dec2mix(decimal : int, radices : Sequence[int]) -> list[int]:
    """
    Convert a nonnegative integer into a mixed-radix vector.

    Parameters
    ----------
    decimal : int
        Nonnegative integer to convert.
    radices : Sequence[int]
        Radices for each digit in the resulting vector.

    Returns
    -------
    list[int]
        Digits ordered from least significant digit to most significant digit.
    """
    vector = []
    for radix in radices:
        vector.append(decimal % radix)
        decimal //= radix
    return vector

def mix2bin(vector : Sequence[int], radices : Sequence[int]) -> list[int]:
    """
    Convert a mixed-radix vector to a binary vector.
    
    Parameters
    ----------
    vector : Sequence of int
        Digits ordered from least significant digit to most significant digit.
    radices : Sequence of int
        Radices for each digit in the vector.
    
    Returns
    -------
    list of int
        Binary digits (0 or 1), ordered from most significant bit to least
        significant bit.
    """
    decimal = mix2dec(vector, radices)
    binstr = bin(decimal)[2:].zfill(sum(radices) - len(radices))
    return [ int(bit) for bit in binstr ]

def bin2mix(binary_vector : Sequence[int], radices : Sequence[int]) -> list[int]:
    """
    Convert a binary vector to a mixed-radix vector.
    
    Parameters
    ----------
    binary_vector : Sequence of int
        Binary digits (0 or 1), ordered from most significant bit to least
        significant bit.
    radices : Sequence of int
        Radices for each digit in the resulting mixed-radix vector.
    
    Returns
    -------
    list of int
        Digits ordered from least significant digit to most significant digit.
    """
    decimal = 0
    for bit in binary_vector:
        decimal = (decimal << 1) | bool(bit)
    return dec2mix(decimal, radices)

def mix2binf(vector: Sequence[int], radices: Sequence[int]) -> list[int]:
    """
    Convert a mixed-radix vector to a formatted binary vector using a
    compressed (log2) encoding: each digit is represented using
    ceil(log2(radix)) bits via standard binary representation.
    
    Parameters
    ----------
    vector : Sequence of int
        Digits ordered from least significant digit to most significant digit.
    radices : Sequence of int
        Radices for each digit in the vector.
    
    Returns
    -------
    list of int
        Binary digits (0 or 1), ordered from most significant bit to least
        significant bit.
    """
    nbits = [ int(radix - 1).bit_length() for radix in radices ]
    conversion = [ 0 for _ in range(len(radices) + 1) ]
    for i, radix in enumerate(nbits):
        conversion[i + 1] = conversion[i] + radix
    binary_vector = [ 0 for _ in range(conversion[len(conversion) - 1]) ]
    for i in range(len(vector)):
        value = vector[i]
        width = nbits[i]
        for bit_pos in range(width):
            # MSB-first: leftmost bit within the group is the highest-order bit.
            binary_vector[conversion[i] + bit_pos] = (value >> (width - 1 - bit_pos)) & 1
    return binary_vector


def binf2mix(binary_vector: Sequence[int], radices: Sequence[int]) -> list[int]:
    """
    Convert a formatted binary vector to a mixed-radix vector using a
    compressed (log2) encoding: each digit is decoded from
    ceil(log2(radix)) bits via standard binary representation.
    
    Parameters
    ----------
    binary_vector : Sequence of int
        Binary digits (0 or 1), ordered from most significant bit to least
        significant bit.
    radices : Sequence of int
        Radices for each digit in the resulting mixed-radix vector.
    
    Returns
    -------
    list of int
        Digits ordered from least significant digit to most significant digit,
        or None if any decoded digit falls outside its valid range
        [0, radix - 1] (this can occur when radix is not a power of 2, since
        ceil(log2(radix)) bits can represent more than radix values).
    """
    nbits = [ int(radix - 1).bit_length() for radix in radices ]
    conversion = [ 0 for _ in range(len(radices) + 1) ]
    for i, radix in enumerate(nbits):
        conversion[i + 1] = conversion[i] + radix
    vector = [ 0 for _ in range(len(radices)) ]
    for i in range(len(radices)):
        value = 0
        for bit_pos in range(nbits[i]):
            value = (value << 1) | (1 if binary_vector[conversion[i] + bit_pos] > 0 else 0)
        if value >= radices[i]:
            return None
        vector[i] = value
    return vector

left_side_of_truth_tables = {}

def get_left_side_of_truth_table_multistate(R : Sequence[int]) -> np.ndarray:
    """
    Return the left-hand side of a Boolean truth table.

    The left-hand side is the binary representation of all input
    combinations for ``N`` Boolean variables, ordered lexicographically.

    Parameters
    ----------
    R : Sequence[int]

    Returns
    -------
    np.ndarray
        Array of shape ``(2**N, N)`` with entries in ``{0, 1}``. Columns are
        ordered from most significant bit to least significant bit.
    """
    R = np.prod(R)
    if R in left_side_of_truth_tables:
        left_side_of_truth_table = left_side_of_truth_tables[R]
    else:
        left_side_of_truth_table = np.arange(R, dtype=np.uint64)[:, None]
        left_side_of_truth_tables[R] = left_side_of_truth_table
    return left_side_of_truth_table


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