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

import numpy as np
import networkx as nx

from ..boolean_function import BooleanFunction
from ..boolean_network import BooleanNetwork
from ..wiring_diagram import WiringDiagram
from .. import utils

from .canalization import random_k_canalizing_function
from .dispatch import random_function
from .functions import random_non_degenerate_function
from .functions import random_function_with_exact_hamming_weight
from .wiring import random_wiring_diagram
from .wiring import rewire_wiring_diagram

def random_network(
    N: int | None = None,
    n: int | float | list | np.ndarray | None = None,
    depth: int | list | np.ndarray = 0,
    exact_depth: bool = False,
    uniform_over_functions: bool = True,
    layer_structure: list | None = None,
    allow_degenerate_functions: bool = False,
    parity: bool = False,
    bias: float | list | np.ndarray = 0.5,
    absolute_bias: float | list | np.ndarray = 0.0,
    use_absolute_bias: bool = False,
    hamming_weight: int | list | np.ndarray | None = None,
    allow_self_loops: bool = False,
    allow_indegree_zero: bool = False,
    strongly_connected: bool = False,
    indegree_distribution: str = "constant",
    min_out_degree_one: bool = False,
    max_strong_connectivity_attempts: int = 1000,
    I: list | np.ndarray | None | WiringDiagram | nx.DiGraph = None,
    *,
    rng=None,
) -> BooleanNetwork:
    """
    Construct a random Boolean network with configurable wiring and update rules.
    
    The network is built in two stages:
    
    1. Wiring diagram
       If ``I`` is provided, it is used directly as the wiring diagram, where
       ``I[v]`` lists the regulators of node ``v``. Otherwise, a wiring diagram
       for ``N`` nodes is sampled using ``random_wiring_diagram``, with in-degrees
       determined by ``n`` and ``indegree_distribution``. Self-loops may be
       disallowed and strong connectivity may be enforced.
    
    2. Update rules
       For each node ``i``, a Boolean update function with arity
       ``indegrees[i]`` is generated using ``random_function`` subject to the
       requested constraints on canalizing depth or layer structure, parity,
       bias or absolute bias, and exact Hamming weight.
    
    Parameters
    ----------
    N : int or None, optional
        Number of nodes. Required when ``I`` is not provided. Ignored if ``I`` is
        given.
    n : int, float, list of int, ndarray of int, or None, optional
        Controls the in-degree distribution when generating a wiring diagram
        (ignored if ``I`` is given). Interpretation depends on
        ``indegree_distribution``:
    
        - ``'constant'``, ``'dirac'``, ``'delta'``:
          Every node has constant in-degree ``n``.
        - ``'uniform'``:
          ``n`` is an integer upper bound; each node’s in-degree is sampled
          uniformly from ``{1, ..., n}``.
        - ``'poisson'``:
          ``n`` is a positive rate parameter lambda; in-degrees are Poisson(lambda) 
          draws truncated to ``[1, N - int(not allow_self_loops)]``.
        - If ``n`` is a length-``N`` vector of integers, it is taken as the exact
          in-degree sequence.
    depth : int, list of int, or ndarray of int, optional
        Requested canalizing depth per node. If an integer, it is broadcast to
        all nodes and clipped at each node’s in-degree. If a vector, it must have
        length ``N``. Interpreted as a minimum depth unless ``exact_depth=True``.
        Default is 0.
    exact_depth : bool, optional
        If True, each Boolean function is generated with exactly the requested
        canalizing depth (or exactly ``sum(layer_structure[i])`` if a layer
        structure is provided). If False, the canalizing depth is at least as
        large as requested. Default is False.
    uniform_over_functions : bool, optional
        Controls how canalized outputs are sampled when generating canalizing
        functions.
    
        If True (default), canalizing layer structures are sampled uniformly at
        random, i.e., proportional to the inverse factorials of layer sizes,
        removing the bias toward symmetric structures induced by independent
        sampling of canalized outputs.
    
        If False, canalized outputs are sampled independently and uniformly as
        bitstrings, which biases the distribution toward more symmetric layer
        structures.
    
        This parameter is ignored when ``layer_structure`` is explicitly provided.
    layer_structure : list, list of lists, or None, optional
        Canalizing layer structure specifications.
    
        - If None (default), rule generation is controlled by ``depth`` and
          ``exact_depth``.
        - If a single list ``[k1, ..., kr]``, the same structure is used for all
          nodes.
        - If a list of lists of length ``N``, ``layer_structure[i]`` is used for
          node ``i``.
    
        In all cases, ``sum(layer_structure[i])`` must not exceed the in-degree
        of node ``i``. When provided, ``layer_structure`` takes precedence over
        ``depth``.
    allow_degenerate_functions : bool, optional
        If True and ``depth == 0`` and ``layer_structure is None``, degenerate
        Boolean functions (with non-essential inputs) may be generated, as in
        classical NK-Kauffman models. If False, generated functions are required
        to be non-degenerate whenever possible. Default is False.
    parity : bool, optional
        If True, parity Boolean functions are generated for all nodes and all
        other rule parameters are ignored. Default is False.
    bias : float, list of float, or ndarray of float, optional
        Probability of output 1 when generating random (non-canalizing) Boolean
        functions. Used only when ``depth == 0``, ``layer_structure is None``,
        ``parity`` is False, and ``use_absolute_bias`` is False. Scalars are
        broadcast to length ``N``. Must lie in ``[0, 1]``. Default is 0.5.
    absolute_bias : float, list of float, or ndarray of float, optional
        Absolute deviation from 0.5 used when ``use_absolute_bias`` is True.
        Scalars are broadcast to length ``N``. Must lie in ``[0, 1]``. Default 0.0.
    use_absolute_bias : bool, optional
        If True, the bias of each rule is chosen at random from
        ``{0.5*(1-absolute_bias), 0.5*(1+absolute_bias)}``. If False, ``bias`` is
        used directly. Default is False.
    hamming_weight : int, list of int, ndarray of int, or None, optional
        Exact Hamming weight (number of ones) of each truth table. Scalars are
        broadcast to length ``N``. Values must lie in ``{0, ..., 2^k}`` for a
        k-input function. Additional restrictions apply when requesting exact
        depth zero. Default is None.
    allow_self_loops : bool, optional
        If True, self-loops (edges from a node to itself) are allowed.
        Default is False. Ignored if ``I`` is provided.
    allow_indegree_zero : bool, optional
        If True, some in-degrees may be ``0``.
        If False (default), all in-degrees are at least ``1``.
    strongly_connected : bool, optional
        If True, wiring generation is repeated until a strongly connected
        directed graph is obtained or the attempt limit is exceeded. Ignored if
        ``I`` is provided. Default is False.
    indegree_distribution : str, optional
        Distribution used when sampling in-degrees. Must be one of
        ``{'constant', 'dirac', 'delta', 'uniform', 'poisson'}``. Default
        ``'constant'``.
    min_out_degree_one : bool, optional
        If True, ensure that each node has at least one outgoing edge in the
        generated wiring diagram. Default is False.
    max_strong_connectivity_attempts : int, optional
        Maximum number of attempts to generate a strongly connected wiring
        diagram before raising an error. Default is 1000.
    I : list, ndarray, WiringDiagram, networkx.DiGraph, or None, optional
        Existing wiring diagram. If provided, ``N`` and ``n`` are ignored and
        in-degrees are inferred from ``I``. If I is a BooleanNetwork, its wiring
        diagram is reused and its Boolean update rules are ignored.
    rng : int, numpy.random.Generator, numpy.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.
    
    Returns
    -------
    BooleanNetwork
        A Boolean network with wiring diagram ``I`` (given or generated) and
        Boolean update functions generated according to the specified constraints.
    
    Raises
    ------
    AssertionError
        If input shapes or parameter combinations are invalid.
    RuntimeError
        If ``strongly_connected=True`` and a strongly connected wiring diagram
        cannot be generated within the specified number of attempts.
    
    Notes
    -----
    Constraint precedence for rule generation is:
    ``parity`` -> ``layer_structure`` -> ``depth`` / ``exact_depth`` -> bias or
    Hamming-weight constraints.
    
    When ``exact_depth=True`` and the requested depth is zero, Hamming weights
    ``{0, 1, 2^k - 1, 2^k}`` correspond to canalizing functions and are therefore
    disallowed.
    
    Examples
    --------
        >>> # Boolean network with only essential inputs
        >>> bn = random_network(N=10, n=2, allow_degenerate_functions=False)

        >>> # Classic NK-Kauffman network allowing degenerate rules
        >>> bn = random_network(N=10, n=3, allow_degenerate_functions=True)

        >>> # Fixed wiring: reuse an existing diagram but resample rules
        >>> bn0 = random_network(N=6, n=2)
        >>> bn  = random_network(I=bn)

        >>> # Exact canalizing depth k for all nodes
        >>> bn = random_network(N=8, n=3, depth=1, exact_depth=True)

        >>> # Nested canalizing update rules with specific layer structure (broadcast)
        >>> bn = random_network(N=5, n=3, layer_structure=[1,2])  # same for all nodes

        >>> # Parity rules
        >>> bn = random_network(N=7, n=2, parity=True)

        >>> # Poisson in-degrees (truncated), no self-regulation, request strong connectivity
        >>> bn = random_network(N=12, n=1.6, indegree_distribution='poisson',
        ...                     allow_self_loops=False, strongly_connected=True)

        >>> # Exact Hamming weights (broadcast)
        >>> bn = random_network(N=6, n=3, hamming_weight=4)

        >>> # To ensure strong connectivity, set allow_degenerate_functions=False
        >>> # and strongly_connected=True
        >>> bn = random_network(N,n,allow_degenerate_functions=False,strongly_connected=True)
    """
    rng = utils._coerce_rng(rng)
    if I is None and N is not None and n is not None:  # generate wiring diagram
        I = random_wiring_diagram(
            N,
            n,
            allow_self_loops=allow_self_loops,
            allow_indegree_zero=allow_indegree_zero,
            strongly_connected=strongly_connected,
            indegree_distribution=indegree_distribution,
            min_out_degree_one=min_out_degree_one,
            max_strong_connectivity_attempts=max_strong_connectivity_attempts,
            rng=rng,
        )

    elif I is not None:  # load wiring diagram
        assert isinstance(I, (list, np.ndarray, WiringDiagram, nx.DiGraph)), (
            "I must be an instance of WiringDiagram or a list or np.array of lists or np.arrays. Each inner list describes the regulators of node i (indexed by 0,1,...,len(I)-1)"
        )
        N = len(I)
        if isinstance(I, (list, np.ndarray)):
            for regulators in I:
                assert (
                    utils.is_list_or_array_of_ints(regulators)
                    and min(regulators) >= 0
                    and max(regulators) <= N - 1
                ), (
                    "Each element in I describes the regulators of a node (indexed by 0,1,...,len(I)-1)"
                )
            I = WiringDiagram(I)
        elif isinstance(I, nx.DiGraph):
            I = WiringDiagram.from_DiGraph( I )        
    else:
        raise AssertionError(
            "At a minimum, the wiring diagram I must be provided or the network size N and degree parameter n."
        )

    # Process the inputs, turn single inputs into vectors of length N

    # since layer_structure takes precedence over depth, this block needs to run before the depth block to ensure depth is a vector and not reset to a single value
    if layer_structure is None:
        layer_structure = [None] * N
    elif utils.is_list_or_array_of_ints(layer_structure):
        depth = sum(layer_structure)
        assert depth == 0 or (
            min(layer_structure) >= 1 and depth <= min(I.indegrees)
        ), (
            "The layer structure must be [] or a vector of positive integers with 0 <= depth = sum(layer_structure) <= N."
        )
        layer_structure = [layer_structure[:]] * N
    elif (
        np.all([utils.is_list_or_array_of_ints(el) for el in layer_structure])
        and len(layer_structure) == N
    ):
        for i, vector in enumerate(layer_structure):
            depth = sum(vector)
            assert depth == 0 or (min(vector) >= 1 and depth <= I.indegrees[i]), (
                "Ensure that layer_structure is an N-dimensional vector where each element represents a layer structure and is either [] or a vector of positive integers with 0 <= depth = sum(layer_structure[i]) <= n = indegrees[i]."
            )
    else:
        raise AssertionError(
            "Wrong input format for 'layer_structure'.\nIt must be a single vector (or N-dimensional vector of layer structures) where the sum of each element is between 0 and N."
        )

    if isinstance(depth, (int, np.integer)):
        assert depth >= 0, (
            "The canalizing depth must be an integer between 0 and min(indegrees) or an N-dimensional vector of integers must be provided to use different depths per function."
        )
        depth = [min(I.indegrees[i], depth) for i in range(N)]
    elif utils.is_list_or_array_of_ints(depth, required_length=N):
        depth = [min(I.indegrees[i], depth[i]) for i in range(N)]
        assert min(depth) >= 0, (
            "'depth' received a vector as input.\nTo use a user-defined vector, ensure that it is an N-dimensional vector where each element is a non-negative integer."
        )
    else:
        raise AssertionError(
            "Wrong input format for 'depth'.\nIt must be a single integer (or N-dimensional vector of integers) between 0 and N, specifying the minimal canalizing depth or exact canalizing depth (if exact_depth==True)."
        )

    if isinstance(bias, (float, np.floating)):
        bias = [bias] * N
    elif not utils.is_list_or_array_of_floats(bias, required_length=N):
        raise AssertionError(
            "Wrong input format for 'bias'.\nIt must be a single float (or N-dimensional vector of floats) in [0,1] , specifying the bias (probability of a 1) in the generation of the Boolean function."
        )

    if isinstance(absolute_bias, (float, np.floating)):
        absolute_bias = [absolute_bias] * N
    elif not utils.is_list_or_array_of_floats(absolute_bias, required_length=N):
        raise AssertionError(
            "Wrong input format for 'absolute_bias'.\nIt must be a single float (or N-dimensional vector of floats) in [0,1], specifying the absolute bias (divergence from the 'unbiased bias' of 0.5) in the generation of the Boolean function."
        )

    if hamming_weight == None:
        hamming_weight = [None] * N
    elif isinstance(hamming_weight, (int, np.integer)):
        hamming_weight = [hamming_weight] * N
    elif not utils.is_list_or_array_of_ints(hamming_weight, required_length=N):
        raise AssertionError(
            "Wrong input format for 'hamming_weight'.\nIf provided, it must be a single integer (or N-dimensional vector of integers) in {0,1,...,2^n}, specifying the number of 1s in the truth table of each Boolean function.\nIf exact_depth == True and depth==0, it must be in {2,3,...,2^n-2} because all functions with Hamming weight 0,1,2^n-1,2^n are canalizing."
        )

    # generate functions
    F = []
    for i in range(N):
        if I.indegrees[i]==0: #only possible if allow_indegree_zero == True
            #then turn i into an idenitity_node
            I.indegrees[i] = 1
            I.outdegrees[i] += 1
            I.I[i] = np.array([i], dtype=int)
            F.append(np.array([0,1], dtype=int))
        else:
            F.append(
                random_function(
                    n=I.indegrees[i],
                    depth=depth[i],
                    exact_depth=exact_depth,
                    uniform_over_functions=uniform_over_functions,
                    layer_structure=layer_structure[i],
                    parity=parity,
                    allow_degenerate_functions=allow_degenerate_functions,
                    bias=bias[i],
                    absolute_bias=absolute_bias[i],
                    use_absolute_bias=use_absolute_bias,
                    hamming_weight=hamming_weight[i],
                    rng=rng,
                )
            )

    return BooleanNetwork(F, I)


def random_null_model(
    bn: BooleanNetwork,
    wiring_diagram: str = "fixed",
    preserve_bias: bool = True,
    preserve_canalizing_depth: bool = True,
    *,
    rng=None,
    **kwargs,
) -> BooleanNetwork:
    """
    Generate a randomized Boolean network (null model) from an existing
    Boolean network while preserving selected structural and dynamical
    properties.
        
    The returned network has the same number of nodes as ``bn``. Depending
    on the selected options, the wiring diagram and/or the Boolean update
    rules are randomized subject to specified invariants. 
    
    To ensure that the generated null models are structurally meaningful,
    the original network ``bn`` must not contain degenerate Boolean functions
    (i.e., functions with non-essential inputs). If such functions are present,
    simplify the network first using ``bn.simplify_functions()``.
    
    Wiring diagram randomization
    ----------------------------
    The wiring diagram can be handled in one of three ways:
    
    - ``'fixed'`` (default):
      The original wiring diagram ``bn.I`` is reused unchanged.
    
    - ``'fixed_indegree'``:
      A new wiring diagram is sampled uniformly at random subject to
      preserving the in-degree of each node. This uses
      ``random_wiring_diagram`` with ``N = bn.N`` and ``n = bn.indegrees``.
    
    - ``'fixed_in_and_outdegree'``:
      The original wiring diagram is randomized via degree-preserving
      double-edge swaps using ``rewire_wiring_diagram``, preserving both
      in-degrees and out-degrees of all nodes.
    
    Rule randomization
    ------------------
    Independently of the wiring diagram, Boolean update rules are
    randomized for each node, optionally preserving properties of the
    original rules:
    
    - If ``preserve_bias`` is True, the exact Hamming weight (number of ones
      in the truth table) of each rule is preserved.
    
    - If ``preserve_canalizing_depth`` is True, the canalizing depth of each
      rule is preserved exactly.
    
    If both flags are True, both properties are preserved simultaneously.
    If neither flag is True, rules are regenerated subject only to
    non-degeneracy and the node’s in-degree.
    
    Parameters
    ----------
    bn : BooleanNetwork
        Source Boolean network.
    wiring_diagram : {'fixed', 'fixed_indegree', 'fixed_in_and_outdegree'}, optional
        Strategy for handling the wiring diagram. Default is ``'fixed'``.
    preserve_bias : bool, optional
        If True, preserve the exact Hamming weight of each Boolean rule.
        Default is True.
    preserve_canalizing_depth : bool, optional
        If True, preserve the exact canalizing depth of each Boolean rule.
        Default is True.
    rng : int, numpy.random.Generator, numpy.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.
    kwargs
        Additional keyword arguments forwarded to the wiring-diagram
        randomization routine:
    
        - For ``wiring_diagram == 'fixed_indegree'``, forwarded to
          ``random_wiring_diagram`` (e.g., ``allow_self_loops``,
          ``strongly_connected``).
    
        - For ``wiring_diagram == 'fixed_in_and_outdegree'``, forwarded to
          ``rewire_wiring_diagram`` (e.g., ``average_swaps_per_edge``,
          ``allow_new_self_loops``, ``allow_self_loop_rewiring``).

    
    Returns
    -------
    BooleanNetwork
        A randomized Boolean network satisfying the selected invariants.
    
    Raises
    ------
    AssertionError
        If invalid options are provided.
    RuntimeError
        If wiring-diagram randomization fails (e.g., strong connectivity
        cannot be achieved within the allowed number of attempts).
    ValueError
        If the Boolean network contains degenerate functions. In that case,
        simplify the network first using bn.simplify_functions(),
        then recompute the null model.
        
    Notes
    -----
    This function generates null models by selectively preserving structural
    and dynamical properties of an existing Boolean network. It is intended
    for hypothesis testing and comparative studies rather than for uniform
    sampling over all networks satisfying the given constraints.
    
    
    
    Examples
    --------
    >>> # Most restrictive use case: Preserve both wiring and rule properties (default) 
    >>> bn_null = random_null_model(bn)
    
    >>> # Preserve in-degrees only and preserve rule bias
    >>> bn_null = random_null_model(
    ...     bn,
    ...     wiring_diagram='fixed_indegree',
    ...     preserve_bias=True,
    ...     preserve_canalizing_depth=False
    ... )
    
    >>> # Preserve both in- and out-degrees via rewiring
    >>> bn_null = random_null_model(
    ...     bn,
    ...     wiring_diagram='fixed_in_and_outdegree',
    ...     average_swaps_per_edge=15
    ... )
    """
    rng = utils._coerce_rng(rng)
    if wiring_diagram == "fixed":
        I = bn.I
    elif wiring_diagram == "fixed_indegree":
        filtered_kwargs = utils.filter_kwargs(random_wiring_diagram, 
                                              kwargs,
                                              exclude=('N', 'n', 'rng'))
        I = random_wiring_diagram(N=bn.N, n=bn.indegrees, rng=rng, **filtered_kwargs)
    elif wiring_diagram == "fixed_in_and_outdegree":
        filtered_kwargs = utils.filter_kwargs(rewire_wiring_diagram, 
                                              kwargs,
                                              exclude=('I', 'rng'))
        I = rewire_wiring_diagram(I=bn.I, rng=rng, **filtered_kwargs)
    else:
        raise AssertionError(
            "There are three choices for the wiring diagram: 1. 'fixed' (i.e., as in the provided BooleanNetwork), 2. 'fixed_indegree' (i.e., edges are shuffled but the indegree is preserved), 3. 'fixed_in_and_outdegree' (i.e., edges are shuffled but both the indegree and outdegree are preserved)."
        )

    dict_identity_nodes = bn.get_identity_nodes(as_dict=True)

    F = []
    for i, f in enumerate(bn.F):
        if dict_identity_nodes[i]:  # identity nodes don't change
            F.append(np.array([0, 1], dtype=int))
            continue
        if preserve_canalizing_depth:
            depth = f.get_canalizing_depth()
            if f.n - depth == 1:
                raise ValueError(
                    f"Boolean function bn.F[{i}] is degenerate. "
                    "Simplify the network first using bn.simplify_functions(), "
                    "then recompute the null model."
                )
        else:
            if f.is_degenerate():
                raise ValueError(
                    f"Boolean function bn.F[{i}] is degenerate. "
                    "Simplify the network first using bn.simplify_functions(), "
                    "then recompute the null model."
                )            
        if preserve_bias and preserve_canalizing_depth:
            core_function = f.properties["CoreFunction"]
            can_outputs = f.properties["CanalizedOutputs"]

            can_inputs = rng.choice(2, depth, replace=True)
            can_order = rng.choice(f.n, depth, replace=False)
            if f.n - depth == 0:
                core_function = np.array([1 - can_outputs[-1]], dtype=int)
            elif f.n - depth == 2:
                core_function = rng.choice(
                    [
                        np.array([0, 1, 1, 0], dtype=int),
                        np.array([1, 0, 0, 1], dtype=int),
                    ]
                )
            else:  # if f.n-depth>=3
                hamming_weight = sum(core_function)
                while True:
                    core_function = random_function_with_exact_hamming_weight(
                        f.n - depth, hamming_weight, rng=rng
                    )
                    if not core_function.is_canalizing():
                        if not core_function.is_degenerate():
                            break
            newf = np.full(2 ** bn.indegrees[i], -1, dtype=int)
            for j in range(depth):
                newf[
                    np.where(
                        np.bitwise_and(
                            newf == -1,
                            utils.get_left_side_of_truth_table(bn.indegrees[i])[
                                :, can_order[j]
                            ]
                            == can_inputs[j],
                        )
                    )[0]
                ] = can_outputs[j]
            newf[np.where(newf == -1)[0]] = core_function
            newf = BooleanFunction(newf)
        elif preserve_bias:  # and preserve_canalizing_depth==False
            newf = random_function_with_exact_hamming_weight(
                bn.indegrees[i], f.hamming_weight, rng=rng
            )
        elif preserve_canalizing_depth:
            filtered_kwargs = utils.filter_kwargs(random_k_canalizing_function, 
                                                  kwargs,
                                                  exclude={"n", "k", "exact_depth", "rng"},)
            newf = random_k_canalizing_function(
                n=bn.indegrees[i], 
                k=depth, 
                exact_depth=True, 
                rng=rng,
                **filtered_kwargs
            )
        else:
            newf = random_non_degenerate_function(n=bn.indegrees[i], rng=rng)
        F.append(newf)
    return BooleanNetwork(F, I)
