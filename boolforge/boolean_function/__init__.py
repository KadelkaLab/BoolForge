#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the :class:`~boolforge.BooleanFunction` class, which forms
the foundation of the BoolForge package.

A :class:`BooleanFunction` represents a Boolean mapping
:math:`f : \\{0,1\\}^n \\rightarrow \\{0,1\\}` and provides methods for
evaluating, analyzing, and transforming Boolean functions. Supported operations
include algebraic manipulation, sensitivity and canalization analysis, truth
table generation, and function composition.

Several computationally intensive methods support optional Numba-based
just-in-time (JIT) acceleration to improve performance for large or repeated
computations. All functionality remains available without Numba, although
performance may be reduced.

This module is part of the core BoolForge library and is intended for both
direct programmatic use and integration with higher-level Boolean network
classes.

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
from .core import BooleanFunction
from .parsing import f_from_expression

__all__ = [
    "BooleanFunction",
    "display_truth_table",
    "get_layer_structure_from_canalized_outputs",
    "f_from_expression",
]