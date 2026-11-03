#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module defines the :class:`~boolforge.Function` class, which forms
the foundation of the BoolForge package, and its Boolean subclass
:class:`~boolforge.BooleanFunction`.

A :class:`Function` represents a finite-state mapping
:math:`f : X_1 \\times \\cdots \\times X_n \\rightarrow Y`, where each
input domain and the output domain is a finite set of states.
Boolean functions, for which all domains are :math:`\\{0,1\\}`,
are represented by the :class:`BooleanFunction` subclass.

The module provides methods for evaluating, analyzing, and transforming
finite-state functions. Supported operations include algebraic manipulation,
sensitivity and canalization analysis, truth table generation, and function
composition. Some methods are currently restricted to Boolean functions,
with multistate support being developed.

Several computationally intensive methods support optional Numba-based
just-in-time (JIT) acceleration to improve performance for large or repeated
computations. All functionality remains available without Numba where
implemented, although performance may be reduced.

This module is part of the core BoolForge library and is intended for both
direct programmatic use and integration with higher-level network classes.

Example
-------
Basic usage::

    >>> from boolforge import BooleanFunction
    >>> f = BooleanFunction("x1 | (x2 & x3)")
    >>> f([1, 0, 1])
    1
"""

from .canalization import get_layer_structure_from_canalized_outputs
from .conversions import display_truth_table
from .core import Function, BooleanFunction
from .parsing import f_from_expression

__all__ = [
    "Function",
    "BooleanFunction",
    "display_truth_table",
    "get_layer_structure_from_canalized_outputs",
    "f_from_expression",
]