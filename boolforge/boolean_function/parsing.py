#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple

from .. import utils


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

def f_from_expression(
    expr: str,
    max_degree: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a Boolean function from a string expression.
    
    The expression is evaluated symbolically over all Boolean input
    combinations to produce the truth table of the corresponding Boolean
    function. Variables are detected automatically based on their first
    occurrence in the expression.
    
    Parameters
    ----------
    expr : str
        Boolean expression to evaluate. The expression may contain logical
        operators (``AND``, ``OR``, ``NOT`` or their lowercase equivalents),
        arithmetic operators, and comparisons.
    max_degree : int, optional
        Maximum number of variables allowed. If the number of detected
        variables exceeds ``max_degree``, an empty truth table is returned.
    
    Returns
    -------
    f : np.ndarray
        Boolean function values as an array of shape ``(2**n,)`` with entries
        in ``{0, 1}``, where ``n`` is the number of detected variables.
    variables : np.ndarray
        Variable names in the order they were first encountered in the
        expression.
    
    Notes
    -----
    - Variables are ordered by first occurrence in ``expr``.
    - Truth-table rows follow the standard lexicographic ordering with the
      most significant bit first.
    - The expression is evaluated using ``eval`` with restricted builtins.
    - No syntactic or semantic validation of ``expr`` is performed beyond
      basic parsing.
    - Arithmetic operators (+, -, `*`, %) must be surrounded by whitespace.
      This restriction avoids conflicts with biological identifiers such as Ca2+ or IL-2.
    - Whenever uncertain, use whitespace.
    
    Examples
    --------
    >>> f_from_expression('A AND NOT B')
    (array([0, 0, 1, 0], dtype=uint8), array(['A', 'B'], dtype='<U1'))
    
    >>> f_from_expression('x1 + x2 + x3 > 1')
    (array([0, 0, 0, 1, 0, 1, 1, 1], dtype=uint8),
     array(['x1', 'x2', 'x3'], dtype='<U2'))
    
    >>> f_from_expression('(x1 + x2 + x3) % 2 == 0')
    (array([1, 0, 0, 1, 0, 1, 1, 0], dtype=uint8),
     array(['x1', 'x2', 'x3'], dtype='<U2'))
    """
    
    # --------------------------------------------------
    # 1. Normalize parentheses spacing
    # --------------------------------------------------

    expr = expr.replace("(", " ( ").replace(")", " ) ")
    
    # comparisons
    for op in ["==", "!=", ">=", "<=", ">", "<"]:
        expr = expr.replace(op, f" {op} ")
        
    expr = expr.replace('> =','>=')
    expr = expr.replace('< =','<=')
    
    # logical double operators -> canonical
    expr = expr.replace("&&", " & ")
    expr = expr.replace("||", " | ")
    
    # single logical operators
    for op in ["&", "|", "!", "~"]:
        expr = expr.replace(op, f" {op} ")
    
    raw_tokens = expr.split()

    tokens = []
    variables = []
    seen = set()

    # --------------------------------------------------
    # 2. Token classification
    # --------------------------------------------------

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

        if utils._is_number(token):
            tokens.append(token)
            continue

        # Otherwise: biological identifier
        if token not in seen:
            seen.add(token)
            variables.append(token)

        tokens.append(token)

    n = len(variables)

    if n > max_degree:
        return np.array([], dtype=np.uint8), np.array(variables)

    # --------------------------------------------------
    # 3. Map biological names → safe Python names
    # --------------------------------------------------

    safe_map = {var: f"v{i}" for i, var in enumerate(variables)}

    safe_tokens = [
        safe_map[token] if token in safe_map else token
        for token in tokens
    ]

    expr_mod = " ".join(safe_tokens)

    # --------------------------------------------------
    # 4. Build evaluation environment
    # --------------------------------------------------

    truth_table = utils.get_left_side_of_truth_table(n)

    local_dict = {
        safe_map[var]: truth_table[:, i].astype(np.int64)
        for i, var in enumerate(variables)
    }

    # --------------------------------------------------
    # 5. Evaluate expression
    # --------------------------------------------------

    try:
        result = eval(expr_mod, {"__builtins__": None}, local_dict)
    except Exception as e:
        raise ValueError(
            f"Error evaluating expression:\n{expr}\nParsed as:\n{expr_mod}\nError: {e}"
        )

    # --------------------------------------------------
    # 6. Enforce Boolean semantics
    # --------------------------------------------------

    result = np.asarray(result)

    if n == 0:
        result = np.array([int(result)], dtype=np.int64)
    else:
        result = result.astype(np.int64)

    # Fix NOT and enforce {0,1}
    result = result & 1

    return result.astype(np.uint8), np.array(variables)