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


##Imports

import numpy as np
import networkx as nx
from collections.abc import Sequence

from ..wiring_diagram import WiringDiagram
from .. import utils

def random_degrees(
    N: int,
    n: int | float | list | np.ndarray,
    indegree_distribution: str = "constant",
    allow_self_loops: bool = False,
    allow_indegree_zero: bool = False,
    *,
    rng=None,
) -> np.ndarray:
    """
    Draw an in-degree vector for a directed network with N nodes.

    This function either accepts a user-specified in-degree vector or
    samples in-degrees independently for each node from a specified
    distribution.

    Parameters
    ----------
    N : int
        Number of nodes in the network. Must be a positive integer.
    n : int, float, list of int, or ndarray of int
        Interpretation depends on ``indegree_distribution``:

        - If ``n`` is a length-``N`` vector of integers, it is interpreted
          as a user-specified in-degree sequence and returned after
          validation.

        - If ``indegree_distribution`` is one of
          ``{'constant', 'dirac', 'delta'}``, then ``n`` is a single integer
          specifying the in-degree of every node.

        - If ``indegree_distribution == 'uniform'``, then ``n`` is a positive
          integer upper bound, and each node independently receives an
          in-degree sampled uniformly from
          ``{1, 2, ..., n}``.

        - If ``indegree_distribution == 'poisson'``, then ``n`` is the Poisson
          rate parameter ``λ > 0``. Each node independently receives a
          Poisson(``λ``) draw, truncated to lie in
          ``[1, N - int(not allow_self_loops)]``.

    indegree_distribution : str, optional
        Distribution used to generate in-degrees when ``n`` is not a vector.
        Must be one of ``{'constant', 'dirac', 'delta', 'uniform', 'poisson'}``.
        Default is ``'constant'``.
    allow_self_loops : bool, optional
        If True, in-degrees may be as large as ``N``.
        If False (default), self-loops are disallowed in subsequent wiring
        generation. This is enforced here by capping in-degrees at ``N-1``.
    allow_indegree_zero : bool, optional
        If True, some in-degrees may be ``0``.
        If False (default), all in-degrees are at least ``1``.
    rng : int, numpy.random.Generator, numpy.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    indegrees : ndarray of int, shape (N,)
        In-degree of each node. For sampled distributions, values lie in
        ``[1, N - int(not allow_self_loops)]``.

    Raises
    ------
    AssertionError
        If inputs are malformed, out of range, or an unsupported distribution
        is requested.

    Notes
    -----
    When sampling is requested, in-degrees for different nodes are generated
    independently. No attempt is made to enforce graphicality or feasibility
    of the resulting degree sequence for a particular wiring model; such
    constraints must be handled downstream.

    Examples
    --------
    >>> random_degrees(5, n=2, indegree_distribution='constant')
    array([2, 2, 2, 2, 2])

    >>> random_degrees(4, n=2, indegree_distribution='uniform', allow_self_loops=False)
    array([2, 1, 2, 2])

    >>> random_degrees(6, n=1.7, indegree_distribution='poisson')
    array([1, 2, 1, 1, 2, 1])

    >>> random_degrees(3, n=[1, 2, 1])
    array([1, 2, 1])
    """
    rng = utils._coerce_rng(rng)

    if isinstance(n, (list, np.ndarray)):
        assert (
            utils.is_list_or_array_of_ints(n, required_length=N)
            and min(n) >= 1-int(allow_indegree_zero)
            and max(n) <= N - int(not allow_self_loops)
        ), (
            "A vector n was submitted.\nEnsure that n is an N-dimensional vector where each element is an integer between 1 and "
            + ("N-1" if not allow_self_loops else "N")
            + " representing the indegree of each nodde."
        )
        indegrees = np.array(n, dtype=int)
    elif indegree_distribution.lower() in ["constant", "dirac", "delta"]:
        assert (
            isinstance(n, (int, np.integer))
            and n >= 1-int(allow_indegree_zero)
            and n <= N - int(not allow_self_loops)
        ), (
            "n must be an integer between 1 and "
            + ("N-1" if not allow_self_loops else "N")
            + " describing the constant degree of each node."
        )
        indegrees = np.ones(N, dtype=int) * n
    elif indegree_distribution.lower() == "uniform":
        assert (
            isinstance(n, (int, np.integer))
            and n >= 1-int(allow_indegree_zero)
            and n <= N - int(not allow_self_loops)
        ), (
            "n must be an integer between 1 and "
            + ("N-1" if not allow_self_loops else "N")
            + " representing the upper bound of a uniform degree distribution (lower bound == 1)."
        )
        indegrees = rng.integers(1, n + 1, size=N)
    elif indegree_distribution.lower() == "poisson":
        assert isinstance(n, (int, float, np.integer, np.floating)) and n > 0, (
            "n must be a float > 0 representing the Poisson parameter."
        )
        indegrees = np.maximum(
            np.minimum(rng.poisson(lam=n, size=N), N - int(not allow_self_loops)), 1-int(allow_indegree_zero)
        )
    else:
        raise AssertionError(
            "None of the predefined in-degree distributions were chosen.\nTo use a user-defined in-degree vector, submit an N-dimensional vector as argument for n; each element of n must an integer between 1 and N."
        )
    return indegrees


def random_edge_list(
    N: int,
    indegrees: Sequence[int],
    allow_self_loops: bool,
    min_out_degree_one: bool = False,
    *,
    rng=None,
) -> list:
    """
    Generate a random directed edge list for a network with prescribed in-degrees.

    Each node ``i`` receives exactly ``indegrees[i]`` incoming edges, with
    regulators chosen uniformly at random from the set of admissible source
    nodes. Optionally, the construction enforces that every node regulates at
    least one other node.

    Parameters
    ----------
    N : int
        Number of nodes in the network.
    indegrees : sequence of int
        Length-``N`` sequence specifying the number of incoming edges for each
        node.
    allow_self_loops : bool
        If True, self-loops (edges from a node to itself) are allowed.
        Default is False.
    min_out_degree_one : bool, optional
        If True, enforce that every node has at least one outgoing edge.
        This is achieved by rewiring edges while preserving the prescribed
        in-degree sequence. Default is False.
    rng : int, numpy.random.Generator, numpy.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    edge_list : list of tuple of int
        List of directed edges represented as ``(source, target)`` pairs.

    Raises
    ------
    ValueError
        If ``N`` or ``indegrees`` are inconsistent.
    AssertionError
        If sampling constraints cannot be satisfied.

    Notes
    -----
    Regulators for each node are sampled uniformly at random without
    replacement from the set of admissible source nodes. If
    ``min_out_degree_one``, the algorithm post-processes
    the initially sampled edge list by replacing edges until every node
    has at least one outgoing edge, while preserving all in-degrees and
    respecting the self-regulation constraint.

    No guarantee is made that the resulting edge list is uniformly sampled
    from the space of all directed graphs satisfying the constraints.
    """

    rng = utils._coerce_rng(rng)

    # ------------------------------------------------------------
    # Step 1: generate initial edge list
    # ------------------------------------------------------------
    edge_list = []
    for i in range(N):
        if not allow_self_loops:
            candidates = np.append(np.arange(i), np.arange(i + 1, N))
        else:
            candidates = np.arange(N)

        indices = rng.choice(candidates, indegrees[i], replace=False)
        edge_list.extend(zip(indices, np.full(indegrees[i], i, dtype=int)))

    # ------------------------------------------------------------
    # Step 2: enforce at least one outgoing edge per node (optional)
    # ------------------------------------------------------------
    if min_out_degree_one:
        target_sources = [set() for _ in range(N)]
        outdegrees = np.zeros(N, dtype=int)

        for s, t in edge_list:
            target_sources[t].add(s)
            outdegrees[s] += 1

        sum_indegrees = len(edge_list)

        while np.min(outdegrees) == 0:
            index_sink = np.where(outdegrees == 0)[0][0]
            index_edge = rng.integers(sum_indegrees)

            old_source, t = edge_list[index_edge]

            if not allow_self_loops and t == index_sink:
                continue
            if index_sink in target_sources[t]:
                continue

            # perform replacement
            target_sources[t].discard(old_source)
            target_sources[t].add(index_sink)

            edge_list[index_edge] = (index_sink, t)

            outdegrees[index_sink] += 1
            outdegrees[old_source] -= 1

    return edge_list


def random_wiring_diagram(
    N: int,
    n: int | float | list | np.ndarray,
    allow_self_loops: bool = False,
    allow_indegree_zero: bool = False,
    strongly_connected: bool = False,
    indegree_distribution: str = "constant",
    min_out_degree_one: bool = False,
    max_strong_connectivity_attempts: int = 1000,
    *,
    rng=None,
) -> tuple:
    """
    Generate a random wiring diagram for a directed network with N nodes.

    A wiring diagram specifies, for each node, the set of its regulators
    (incoming neighbors). In-degrees are first generated according to the
    specified distribution, after which edges are sampled uniformly at
    random subject to the requested constraints.

    Parameters
    ----------
    N : int
        Number of nodes in the network.
    n : int, float, list of int, or ndarray of int
        Parameter determining the in-degree sequence. Interpretation depends
        on ``indegree_distribution``:

        - If a length-``N`` vector is provided, it is interpreted as the
          in-degree of each node.
        - If ``indegree_distribution`` is ``'constant'`` (or ``'dirac'`` /
          ``'delta'``), ``n`` specifies the in-degree of every node.
        - If ``indegree_distribution`` is ``'uniform'``, ``n`` specifies the
          upper bound of a discrete uniform distribution on
          ``{1, ..., n}``.
        - If ``indegree_distribution`` is ``'poisson'``, ``n`` is the Poisson
          rate parameter ``lambda > 0``.
    allow_self_loops : bool, optional
        If True, self-loops (edges from a node to itself) are allowed.
        Default is False.
    allow_indegree_zero : bool, optional
        If True, some in-degrees may be ``0``.
        If False (default), all in-degrees are at least ``1``.
    strongly_connected : bool, optional
        If True, repeatedly resample the wiring diagram until a strongly
        connected network is obtained, or until the maximum number of
        attempts is exceeded. Default is False.
    indegree_distribution : str, optional
        Distribution used to generate in-degrees. Must be one of
        ``{'constant', 'dirac', 'delta', 'uniform', 'poisson'}``.
        Default is ``'constant'``.
    min_out_degree_one : bool, optional
        If True, enforce that every node has at least one outgoing edge.
        This is achieved by rewiring edges while preserving the in-degree
        sequence. Default is False.
    max_strong_connectivity_attempts : int, optional
        Maximum number of attempts to generate a strongly connected wiring
        diagram before raising a ``RuntimeError``. Default is 1000.
    rng : int, numpy.random.Generator, numpy.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    WiringDiagram
        A wiring diagram object encoding the regulator set of each node.

    Raises
    ------
    AssertionError
        If ``strongly_connected=True`` and ``allow_indegree_zero==True``.
        This is an impossible combination.
    RuntimeError
        If ``strongly_connected=True`` and a strongly connected wiring diagram
        cannot be generated within the specified number of attempts.

    Notes
    -----
    In-degrees are generated first using ``random_degrees``, and edges are
    then sampled uniformly at random subject to the imposed constraints.
    When ``strongly_connected=True`` or
    ``min_out_degree_one=True``, the resulting distribution is
    not uniform over all wiring diagrams with the given in-degree sequence.

    This function is a high-level convenience wrapper around
    ``random_degrees`` and ``random_edge_list``.

    Examples
    --------
    >>> W = random_wiring_diagram(5, n=2)
    >>> W = random_wiring_diagram(10, n=3, strongly_connected=True)
    >>> W = random_wiring_diagram(6, n=[1, 2, 1, 2, 1, 2])
    """
    assert not strongly_connected or not allow_indegree_zero, (
        "It is impossible to create a strongly connected wiring diagram if some nodes have indegree zero."
    )    
    
    rng = utils._coerce_rng(rng)
    indegrees = random_degrees(
        N,
        n,
        indegree_distribution=indegree_distribution,
        allow_self_loops=allow_self_loops,
        allow_indegree_zero=allow_indegree_zero,
        rng=rng,
    )

    counter = 0
    while True:  
        edges_wiring_diagram = random_edge_list(
            N,
            indegrees,
            allow_self_loops,
            min_out_degree_one=min_out_degree_one,
            rng=rng,
        )
        if strongly_connected: # Keep generating until we have a strongly connected graph
            # may take a long time ("forever") if n is small and N is large
            G = nx.from_edgelist(edges_wiring_diagram, create_using=nx.MultiDiGraph())
            if not nx.is_strongly_connected(G):
                counter += 1
                if counter > max_strong_connectivity_attempts:
                    raise RuntimeError(
                        "Made "
                        + str(max_strong_connectivity_attempts)
                        + " unsuccessful attempts to generate a strongly connected wiring diagram of "
                        + str(N)
                        + " nodes and degrees "
                        + str(indegrees)
                        + ".\nYou may increase the number of attempts by modulating the parameter max_strong_connectivity_attempts."
                    )
                continue
        break
    I = [[] for _ in range(N)]
    for edge in edges_wiring_diagram:
        I[edge[1]].append(edge[0])
    for i in range(N):
        I[i] = np.sort(I[i])
    return WiringDiagram(I)


def rewire_wiring_diagram(
    I: list | np.ndarray | WiringDiagram,
    average_swaps_per_edge: float = 50,
    allow_new_self_loops: bool = False,
    allow_self_loop_rewiring: bool = False,
    *,
    rng=None,
) -> list:
    """
    Degree-preserving rewiring of a wiring diagram via double-edge swaps.

    The wiring diagram is represented in regulator form: ``I[target]`` lists
    all regulators (incoming neighbors) of ``target``. The algorithm performs
    random double-edge swaps of the form
    ``(u -> v, x -> y) -> (u -> y, x -> v)``, while preserving both the
    in-degree and out-degree of every node. Parallel edges are disallowed.

    Parameters
    ----------
    I : list of array-like or WiringDiagram
        Wiring diagram in regulator representation. For each node ``target``,
        ``I[target]`` contains the regulators of that node. Regulator indices
        must be integers in ``{0, ..., N-1}``. If a ``WiringDiagram`` is
        provided, its internal adjacency representation is used.
    average_swaps_per_edge : float, optional
        Target number of successful double-edge swaps per edge. Larger values
        typically yield better mixing but increase runtime. Default is 50.
    allow_new_self_loops : bool, optional
        If True, new self-loops may be introduced. If False (default), proposed 
        swaps that would introduce a new self-loop are rejected. 
    allow_self_loop_rewiring : bool, optional
        If True, existing self-loops may be rewired. If False (default), 
        existing self-loops are kept fixed and excluded from the pool of 
        swappable edges. 
    rng : int, numpy.random.Generator, numpy.random.RandomState, random.Random, or None, optional
        Random number generator or seed specification. Passed to
        ``utils._coerce_rng``.

    Returns
    -------
    WiringDiagram
        A new wiring diagram obtained by degree-preserving rewiring of ``I``.

    Raises
    ------
    ValueError
        If the input wiring diagram is malformed.
    AssertionError
        If rewiring constraints cannot be satisfied.

    Notes
    -----
    Both in-degrees and out-degrees of all nodes are preserved exactly.
    Duplicate edges are never introduced. Control over self-regulation is
    governed by the two Boolean flags above.

    The resulting wiring diagram is not guaranteed to be sampled uniformly
    from the space of all directed graphs with the same degree sequence; the
    procedure is intended as a practical degree-preserving randomization
    method rather than an exact uniform sampler.

    Examples
    --------
    >>> I = random_network(8,3)
    >>> J = rewire_wiring_diagram(I)
    >>> I.indegrees == J.indegrees
    True
    >>> I.get_outdegrees() == J.get_outdegrees()
    True
    """
    rng = utils._coerce_rng(rng)
    
    if isinstance(I, WiringDiagram):
        I = I.I
    
    N = len(I)

    edges = [
        (int(regulator), target)
        for target in range(N)
        for regulator in I[target]
        if regulator != target or allow_self_loop_rewiring
    ]
    n_total_edges = len(edges)

    Jset = [set(regulators) for regulators in I]

    n_rewires_before_stop = int(average_swaps_per_edge * n_total_edges)
    successes = 0
    attempts = 0
    max_attempts = 50 * n_rewires_before_stop + 100

    # Helper to check if adding edge (regulator->target) is allowed
    def edge_ok(regulator, target):
        if not allow_new_self_loops and regulator == target:
            return False
        if regulator in Jset[target]:
            return False
        return True

    while successes < n_rewires_before_stop and attempts < max_attempts:
        attempts += 1

        # Pick two distinct edges uniformly at random
        i, j = rng.choice(n_total_edges, 2, replace=False)

        (u, v) = edges[i]
        (x, y) = edges[j]

        # Swapping identical sources or identical targets is fine in principle,
        # but skip trivial cases that do nothing or re-create the same edges.
        if (u == x) or (v == y):
            continue

        # Proposed swapped edges
        a, b = u, y
        c, d = x, v

        # If the proposed edges are identical to originals, skip
        if (a, b) == (u, v) or (c, d) == (x, y):
            continue

        # Check constraints for both new edges
        if not edge_ok(a, b) or not edge_ok(c, d):
            continue

        # Perform the swap: update adjacency and edge list
        # Remove old edges
        Jset[v].discard(u)
        Jset[y].discard(x)
        # Add new edges
        Jset[b].add(a)
        Jset[d].add(c)
        # Commit edges
        edges[i] = (a, b)
        edges[j] = (c, d)

        successes += 1

    # Reconstruct J from adjacency sets
    J = [np.sort(list(Jset[target])) for target in range(N)]
    return WiringDiagram(J)
