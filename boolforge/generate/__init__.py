#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random generation of discrete functions and networks.

This module provides routines for generating random Boolean and multistate
functions and networks with specified structural and dynamical properties.

Functions can be generated with constraints on their output distribution,
canalizing structure, essential variables, and other properties. Network
generation additionally supports constraints on connectivity and update
rules. These routines facilitate statistical analysis, benchmarking, and
simulation studies.

Boolean generation is fully supported, while multistate functionality is
being incorporated into the unified :class:`~boolforge.Function` architecture.

Selected generation routines support optional Numba-based just-in-time
(JIT) acceleration. All functionality remains available without Numba,
although performance may be reduced.

This module complements :mod:`~boolforge.function` and
:mod:`~boolforge.network` by providing reproducible generation
of synthetic functions and networks.

Examples
--------
>>> import boolforge as bf
>>> f = bf.random_function(n=3)
>>> bn = bf.random_network(N=5, n=2)
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
