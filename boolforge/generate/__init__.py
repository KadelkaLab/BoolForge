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

from .dispatch import random_function

from .functions import (
    random_function_with_bias,
    random_function_with_exact_hamming_weight,
    random_degenerate_function,
    random_non_degenerate_function,
    random_parity_function,
)

from .canalization import (
    random_k_canalizing_function,
    random_k_canalizing_function_with_specific_layer_structure,
    random_NCF,
    random_non_canalizing_function,
    random_non_canalizing_non_degenerate_function,
)

from .wiring import (
    random_degrees,
    random_edge_list,
    random_wiring_diagram,
    rewire_wiring_diagram,
)

from .networks import (
    random_network,
    random_null_model,
)

__all__ = [
    "random_function",
    "random_function_with_bias",
    "random_function_with_exact_hamming_weight",
    "random_degenerate_function",
    "random_non_degenerate_function",
    "random_parity_function",
    "random_k_canalizing_function",
    "random_k_canalizing_function_with_specific_layer_structure",
    "random_NCF",
    "random_non_canalizing_function",
    "random_non_canalizing_non_degenerate_function",
    "random_wiring_diagram",
    "random_edge_list",
    "random_degrees",
    "rewire_wiring_diagram",
    "random_network",
    "random_null_model",
]