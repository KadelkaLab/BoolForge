#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
This module defines the :class:`~boolforge.BooleanNetwork` class, which provides
a high-level framework for modeling, simulating, and analyzing Boolean networks.

A :class:`BooleanNetwork` represents a discrete dynamical system
:math:`F = (f_1, \ldots, f_N)` composed of multiple
:class:`~boolforge.BooleanFunction` objects as update rules. The class includes
methods for constructing state transition graphs, identifying attractors,
computing robustness and sensitivity measures, and exporting truth tables.

Several computational routines—particularly those involving state space
exploration, attractor detection, and robustness estimation—offer optional
Numba-based just-in-time (JIT) acceleration. Installing Numba is recommended
for optimal performance but not required; all features remain functional
without it.

This module serves as the central interface for dynamic Boolean network
analysis within the BoolForge package.

Example
-------
>>> from boolforge import BooleanNetwork
>>> bn = BooleanNetwork(F=[[0, 1], [0, 0, 0, 1], [0, 1]], I=[[1], [0, 2], [1]])
>>> bn.get_attractors_synchronous_exact()
"""

from .core import BooleanNetwork, dict_weights
from .robustness_sync import get_entropy_of_basin_size_distribution

__all__ = ['BooleanNetwork',
           'dict_weights',
           'get_entropy_of_basin_size_distribution',
           ]
