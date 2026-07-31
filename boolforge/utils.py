#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions used throughout BoolForge.

The :mod:`~boolforge.utils` module includes low-level operations for binary and
decimal conversions, truth table manipulations, and combinatorial helper
functions. These utilities are used internally by
:class:`~boolforge.BooleanFunction` and :class:`~boolforge.BooleanNetwork`
classes to enable efficient representation and analysis of Boolean functions
and networks.

Notes
-----
Most functions in this module are intended for internal use and are not part of
the stable public API.

Examples
--------
>>> import boolforge
>>> boolforge.bin2dec([1, 0, 1])
5
>>> boolforge.dec2bin(5, 3)
array([1, 0, 1])
"""


##Imports
import numpy as np
import random as _py_random
from collections.abc import Sequence
from numpy.random import Generator as _NPGen, RandomState as _NPRandomState, SeedSequence, default_rng
import inspect
import warnings

STRICT = False

__all__ = [
    "bin2dec",
    "dec2bin",
    "hamming_weight_to_ncf_layer_structure",
    "get_left_side_of_truth_table",
    "left_side_of_truth_tables",
    'filter_kwargs',
    'allowed_keywords',
    'get_number_of_varying_nodes',
    'get_minimal_trap_space',
    'get_shannon_entropy',
    'get_number_of_varying_nodes',
]

def _require_cana():
    try:
        import cana.boolean_node
        return cana.boolean_node
    except ModuleNotFoundError as e:
        raise ImportError(
            "This functionality requires CANA. "
            "Install it with `pip install cana`."
        ) from e
        

def _coerce_rng(
    rng : int | _NPGen | _NPRandomState | _py_random.Random | None = None
) -> _NPGen:
    """
    Coerce a variety of RNG-like inputs to a NumPy ``Generator``.

    Parameters
    ----------
    rng : int | np.random.Generator | np.random.RandomState | random.Random | None, optional
        Random number generator or seed specification.

        - ``None``: return ``np.random.default_rng()``.
        - ``int``: interpreted as a seed for ``default_rng``.
        - ``np.random.Generator``: returned unchanged.
        - ``np.random.RandomState``: converted via ``SeedSequence``.
        - ``random.Random``: converted via ``SeedSequence``.

    Returns
    -------
    np.random.Generator
        A NumPy random number generator.

    Raises
    ------
    TypeError
        If ``rng`` is not one of the supported types.

    Notes
    -----
    This function provides a unified RNG interface across BoolForge by
    normalizing legacy and standard-library RNGs to the modern NumPy
    ``Generator`` API.

    Conversion from ``RandomState`` and ``random.Random`` is performed by
    extracting entropy and initializing a ``SeedSequence``. This preserves
    reproducibility while avoiding direct reliance on deprecated RNG APIs.
    """
    if rng is None:
        return default_rng()
    if isinstance(rng, _NPGen):
        return rng
    if isinstance(rng, (int, np.integer)):
        return default_rng(int(rng))
    if isinstance(rng, _NPRandomState):
        # derive robust entropy from the legacy RNG
        entropy = rng.randint(0, 2**32, size=4, dtype=np.uint32)
        return default_rng(SeedSequence(entropy))
    if isinstance(rng, _py_random.Random):
        entropy = [_py_random.Random(rng.random()).getrandbits(32) for _ in range(4)]
        # simpler: entropy = [rng.getrandbits(32) for _ in range(4)]
        return default_rng(SeedSequence(entropy))
    raise TypeError(f"Unsupported rng type: {type(rng)!r}")

def _is_number(token: str) -> bool:
    """Return True if token is a pure numeric literal."""
    try:
        float(token)
        return True
    except ValueError:
        return False
    

def is_float(element: object) -> bool:
    """
    Check whether an object can be coerced to a float.

    Parameters
    ----------
    element : object
        Object to test for float coercibility.

    Returns
    -------
    bool
        True if ``element`` can be converted to ``float`` without raising an
        exception, False otherwise.

    Notes
    -----
    This function tests coercibility, not type membership. For example,
    numeric strings and integers return True.

    Examples
    --------
    >>> is_float(3)
    True
    >>> is_float(3.14)
    True
    >>> is_float("2.7")
    True
    >>> is_float("abc")
    False
    >>> is_float(None)
    False
    """
    try:
        float(element)
        return True
    except (TypeError, ValueError):
        return False

def bin2dec(binary_vector: list[int]) -> int:
    """
    Convert a binary vector to an integer.

    Parameters
    ----------
    binary_vector : list of int
        Binary digits (0 or 1), ordered from most significant bit to least
        significant bit.

    Returns
    -------
    int
        Integer represented by the binary vector.

    Notes
    -----
    No validation is performed to ensure that entries of ``binary_vector`` are
    binary. Nonzero values are treated as 1 under bitwise conversion.

    Examples
    --------
    >>> bin2dec([1, 0, 1])
    5
    >>> bin2dec([0, 0, 1, 1])
    3
    """
    decimal = 0
    for bit in binary_vector:
        decimal = (decimal << 1) | bool(bit)
    return int(decimal)


def dec2bin(integer_value: int, num_bits: int) -> list[int]:
    """
    Convert a nonnegative integer to a binary vector.

    Parameters
    ----------
    integer_value : int
        Nonnegative integer to convert.
    num_bits : int
        Length of the binary representation.

    Returns
    -------
    list of int
        Binary digits (0 or 1), ordered from most significant bit to least
        significant bit.

    Notes
    -----
    - If ``integer_value`` requires more than ``num_bits`` bits, the most
      significant bits are truncated.
    - No validation is performed for negative inputs.

    Examples
    --------
    >>> dec2bin(5, 3)
    [1, 0, 1]
    >>> dec2bin(3, 5)
    [0, 0, 0, 1, 1]
    """
    binary_string = bin(integer_value)[2:].zfill(num_bits)
    return [int(bit) for bit in binary_string]


left_side_of_truth_tables = {}

def get_left_side_of_truth_table(N: int) -> np.ndarray:
    """
    Return the left-hand side of a Boolean truth table.

    The left-hand side is the binary representation of all ``2**N`` input
    combinations for ``N`` Boolean variables, ordered lexicographically from
    ``0`` to ``2**N - 1``.

    Parameters
    ----------
    N : int
        Number of Boolean variables.

    Returns
    -------
    np.ndarray
        Array of shape ``(2**N, N)`` with entries in ``{0, 1}``. Columns are
        ordered from most significant bit to least significant bit.

    Notes
    -----
    - The result is cached by ``N`` to avoid recomputation.
    - Row ``i`` corresponds to the binary expansion of integer ``i``.
    - The most significant bit appears in column 0.

    Examples
    --------
    >>> get_left_side_of_truth_table(2)
    array([[0, 0],
           [0, 1],
           [1, 0],
           [1, 1]], dtype=uint8)
    """
    if N in left_side_of_truth_tables:
        left_side_of_truth_table = left_side_of_truth_tables[N]
    else:
        vals = np.arange(2**N, dtype=np.uint64)[:, None]
        masks = (1 << np.arange(N-1, -1, -1, dtype=np.uint64))[None]
        left_side_of_truth_table = ((vals & masks) != 0).astype(np.uint8)
        left_side_of_truth_tables[N] = left_side_of_truth_table
    return left_side_of_truth_table


def allowed_keywords(function):
    """
    Return the keyword arguments accepted by a function.

    Parameters
    ----------
    function : callable
        Function whose signature should be inspected.

    Returns
    -------
    set of str
        Names of all parameters except ``*args`` (variable positional
        arguments). Returns an empty set if the function signature
        cannot be determined.
    """
    try:
        sig = inspect.signature(function)
    except (TypeError, ValueError):
        return set()

    return {
        k for k, p in sig.parameters.items()
        if p.kind != inspect.Parameter.VAR_POSITIONAL
    }

def filter_kwargs(function, kwargs, exclude=()):
    """
    Filter a dictionary of keyword arguments for a given function.

    Parameters
    ----------
    function : callable
        Function whose signature should be inspected.
    kwargs : dict
        Keyword arguments to filter.
    exclude : iterable of str, optional
        Keyword names to exclude even if accepted by ``function``.

    Returns
    -------
    dict
        Dictionary containing only keyword arguments accepted by
        ``function`` and not listed in ``exclude``.

    Notes
    -----
    - If ``function`` accepts ``**kwargs``, the input dictionary is
      returned unchanged.
    - Unused keyword arguments generate a warning or raise a
      ``TypeError`` depending on the value of ``STRICT``.
    """
    try:
        sig = inspect.signature(function)
    except (TypeError, ValueError):
        return kwargs

    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )

    if accepts_var_kwargs:
        return kwargs

    allowed = {
        k for k, p in sig.parameters.items()
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    } - set(exclude)

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed and k not in exclude}
    unused = set(kwargs) - allowed

    if unused:
        msg = (
            f"Attempted to pass unused keyword argument(s) "
            f"{sorted(unused)} to {function.__name__}"
        )

        if STRICT:
            raise TypeError(msg)
        else:
            warnings.warn(msg, UserWarning, stacklevel=2)

    return filtered_kwargs

def find_all_indices(arr: list, el: object) -> list[int]:
    """
    Find all indices of a given element in a sequence.

    Parameters
    ----------
    arr : list
        Sequence to search.
    el : object
        Element to locate.

    Returns
    -------
    list of int
        Indices ``i`` such that ``arr[i] == el``.

    Raises
    ------
    ValueError
        If ``el`` does not occur in ``arr``.

    Examples
    --------
    >>> find_all_indices([1, 2, 1, 3], 1)
    [0, 2]
    >>> find_all_indices(['a', 'b', 'a'], 'a')
    [0, 2]
    """
    res: list[int] = []
    for i, a in enumerate(arr):
        if a == el:
            res.append(i)

    if not res:
        raise ValueError("Element not found in sequence")

    return res

def check_if_empty(my_list: list | np.ndarray) -> bool:
    """
    Check whether a list or NumPy array is empty.

    Parameters
    ----------
    my_list : list or np.ndarray
        Sequence to check.

    Returns
    -------
    bool
        True if ``my_list`` is empty, False otherwise.

    Notes
    -----
    For NumPy arrays, emptiness is determined by ``size == 0``.
    For Python lists, emptiness is determined by equality to ``[]``.
    """
    if isinstance(my_list, np.ndarray):
        return my_list.size == 0
    return my_list == []
    
    
def is_list_or_array_of_ints(
    x: list | np.ndarray,
    required_length: int | None = None,
) -> bool:
    """
    Check whether a list or NumPy array contains only integers.

    Parameters
    ----------
    x : list or np.ndarray
        Sequence to check.
    required_length : int or None, optional
        If provided, require that ``x`` has exactly this length.

    Returns
    -------
    bool
        True if ``x`` is a list of ``int`` / ``np.integer`` or a NumPy array
        with integer dtype, and (if specified) has length ``required_length``.
        False otherwise.

    Notes
    -----
    - For Python lists, each element is checked individually.
    - For NumPy arrays, the dtype is checked using ``np.issubdtype``.
    - One-dimensional arrays are required when ``required_length`` is given.
    """
    # Case 1: Python list
    if isinstance(x, list):
        return (
            (required_length is None or len(x) == required_length)
            and all(isinstance(el, (int, np.integer)) for el in x)
        )

    # Case 2: NumPy array
    if isinstance(x, np.ndarray):
        return (
            (required_length is None or x.shape == (required_length,))
            and np.issubdtype(x.dtype, np.integer)
        )

    return False

def is_list_or_array_of_floats(
    x: list | np.ndarray,
    required_length: int | None = None,
) -> bool:
    """
    Check whether a list or NumPy array contains only floating-point numbers.

    Parameters
    ----------
    x : list or np.ndarray
        Sequence to check.
    required_length : int or None, optional
        If provided, require that ``x`` has exactly this length.

    Returns
    -------
    bool
        True if ``x`` is a list of ``float`` / ``np.floating`` or a NumPy array
        with floating-point dtype, and (if specified) has length
        ``required_length``. False otherwise.

    Notes
    -----
    - For Python lists, each element is checked individually.
    - For NumPy arrays, the dtype is checked using ``np.issubdtype``.
    - One-dimensional arrays are required when ``required_length`` is given.
    """
    # Case 1: Python list
    if isinstance(x, list):
        return (
            (required_length is None or len(x) == required_length)
            and all(isinstance(el, (float, np.floating)) for el in x)
        )

    # Case 2: NumPy array
    if isinstance(x, np.ndarray):
        return (
            (required_length is None or x.shape == (required_length,))
            and np.issubdtype(x.dtype, np.floating)
        )

    return False

def flatten(sequence: Sequence[Sequence[object]]) -> list[object]:
    """
    Flatten a sequence of sequences by one level.

    Parameters
    ----------
    sequence : list or np.ndarray
        Sequence whose elements are themselves iterable.

    Returns
    -------
    list
        A flat list containing the elements of each sub-sequence in ``l``,
        in order.

    Notes
    -----
    This function performs a single-level flattening only. Nested sequences
    deeper than one level are not recursively flattened.

    Examples
    --------
    >>> flatten([[1, 2], [3, 4]])
    [1, 2, 3, 4]
    >>> flatten(np.array([[1, 2], [3, 4]]))
    [1, 2, 3, 4]
    """
    return [item for sublist in sequence for item in sublist]


def get_number_of_varying_nodes(states: Sequence[int]):
    """
    Return the number of nodes that vary across a collection of states.

    A node is considered varying if it takes different values in at
    least two states.

    Parameters
    ----------
    states : sequence of int
        Binary vectors (states) encoded as integers.

    Returns
    -------
    int
        Number of varying nodes.
    """
    ref = states[0]
    varying = 0
    for s in states[1:]:
        varying |= (ref ^ s)
    return varying.bit_count()


def get_shannon_entropy(
    probabilities: Sequence[float]
) -> float:
    """
    Compute the Shannon entropy of a probability distribution.

    Parameters
    ----------
    probabilities : Sequence[float]
        Nonnegative weights representing a probability distribution.
        The values are normalized internally if they do not sum to one.

    Returns
    -------
    float
        Shannon entropy

        ``H = -sum(p_i * log(p_i))``,

        where ``p_i`` are the normalized probabilities.
    """
    p= np.asarray(probabilities, dtype=float)

    assert np.all(p >= 0)

    total = p.sum()
    if total == 0:
        return 0.0

    p = p / total

    return -np.sum(p[p > 0] * np.log(p[p > 0]))


def get_minimal_trap_space(states: Sequence[int], N: int) -> np.ndarray :
    """
    Compute the minimal trap space containing a collection of states.

    Nodes that take the same value in every state are assigned that
    value (0 or 1). Nodes that vary across the states are assigned -1,
    indicating a free variable.

    Parameters
    ----------
    states : sequence of int
        States encoded as integers.
    N : int
        Number of nodes in the network.

    Returns
    -------
    numpy.ndarray
        Length-N array containing 0 (fixed OFF), 1 (fixed ON), or -1 (free).
    """
    ref = states[0]

    varying = 0
    for s in states[1:]:
        varying |= (ref ^ s)

    trap_space = np.zeros(N,dtype=np.int8)
    for i in range(N):
        bit = 1 << (N-i-1)
        if varying & bit:
            trap_space[i] = -1   # free
        else:
            trap_space[i] = (ref >> i) & 1
    return trap_space



def hamming_weight_to_ncf_layer_structure(
    n: int,
    w: int,
) -> list[int]:
    """
    Compute the canalizing layer structure of a nested canalizing function (NCF)
    from its Hamming weight.

    For nested canalizing functions, there is a bijection between the (odd)
    Hamming weight ``w`` and the canalizing layer structure, with ``w`` and
    ``2**n - w`` corresponding to the same structure.

    Parameters
    ----------
    n : int
        Number of input variables of the NCF.
    w : int
        Odd Hamming weight of the NCF.

    Returns
    -------
    list of int
        Canalizing layer structure ``[k_1, ..., k_r]``.

    Raises
    ------
    TypeError
        If ``w`` is not an integer.
    ValueError
        If ``w`` is outside ``[1, 2**n - 1]`` or if ``w`` is even.

    Notes
    -----
    - All nested canalizing functions have odd Hamming weight.
    - The binary expansion of ``w`` (with ``n`` bits) determines the layer
      structure.
      
    References
    ----------
    Kadelka, C., Kuipers, J., & Laubenbacher, R. (2017). 
    The influence of canalization on the robustness of Boolean networks. 
    Physica D: Nonlinear Phenomena, 353, 39-47.
    """
    if not isinstance(w, (int, np.integer)):
        raise TypeError("Hamming weight w must be an integer")

    if not (1 <= w <= 2**n - 1):
        raise ValueError("Hamming weight w must satisfy 1 <= w <= 2**n - 1")

    if w % 2 == 0:
        raise ValueError("Hamming weight w must be odd for nested canalizing functions")

    if w == 1:
        return [n]

    w_bin = dec2bin(w, n)

    current_el = w_bin[0]
    layer_structure_NCF = [1]

    for el in w_bin[1:-1]:
        if el == current_el:
            layer_structure_NCF[-1] += 1
        else:
            layer_structure_NCF.append(1)
            current_el = el

    layer_structure_NCF[-1] += 1
    return layer_structure_NCF

