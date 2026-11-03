#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## JUMP FLAG

import numpy as np
import networkx as nx
from collections.abc import Sequence

from . import utils

from .multistate_network_to_merge import MultistateFunction, MultistateNetwork
from .utils_multistate import f_from_expression

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

def random_non_degenerated_ms_function(
        r : int,
        R : Sequence[int],
        *,
        rng=None
    ) -> MultistateFunction:
    """
    Generate a random non-degenerate multistate function.
    
    Parameters
    ----------
    n : int
        Number of multistate variables,
    r : int
        The number of states assumable by this function.
    R : Sequence[int]
        The number of states assumable by each input variable.

    Returns
    -------
    MultistateFunction
        Random non-degerate multistate function.

    """
    rng = utils._coerce_rng(rng)
    while True:
        f = np.array(np.floor(rng.random(np.prod(R)) * r), int)
        #if not is_ms_degenerated(f, R):
        return MultistateFunction(f, r, R)

def random_linear_ms_function(
        n : int,
        r : int,
        R : Sequence[int],
        *,
        rng=None
    ) -> MultistateFunction:
    expr = '(%s) %% %i' % (' + '.join(['(%i + x%i)' % (rng.random(r), y) for y in range(n)]), r)
    f, var = f_from_expression(expr, R)[0]
    return MultistateFunction(f, r, n, var)

def random_MSN(N, in_degree = 2, base = 3, STRONGLY_CONNECTED = True, in_degree_distribution = 'const',
               UNIFORM_BASE_DISTRIBUTION = False, list_x = [], NO_SELF_REGULATION = True, edges_wiring_diagram = None, LINEAR = False):
    if in_degree_distribution in [ 'constant', 'const', 'dirac', 'delta' ]:
        if type(in_degree) in [ list, np.array ]:
            try:
                assert np.all([ type(el) in [ int, np.int64 ] for el in in_degree ])
                assert len(in_degree) == N
                assert min(in_degree) >= 1
                assert max(in_degree) <= N
                ns = np.array(in_degree[:], int)
            except AssertionError:
                print("Error: A vector was submitted for the in-degree.\nTo use a user-defined in-degree vector, ensure that it is an N-dimensional vector where each element is an integer between 1 and the N.")
                return
        else:
            try:
                assert type(in_degree) in [ int, np.int64 ]
                assert in_degree >= 1
                assert in_degree <= N
                ns = np.ones(N, int) * in_degree
            except AssertionError:
                print("Error: The in-degree must be a single integer (or N-dimensional vector of integers) between 1 and N when using a constant degree distribution.")
                return
    elif in_degree_distribution == 'uniform':
        if type(in_degree) in [ list, np.array ]:
            try:
                assert np.all([ type(el) in [ int, np.int64 ] for el in in_degree ])
                assert len(in_degree) == N
                assert min(in_degree) >= 1
                assert max(in_degree) <= N
                ns = np.array(in_degree[:])
            except AssertionError:
                print("Error: A vector was submitted for the in-degree.\nEnsure that you are providing an N-dimensional vector where each element is an integer between 1 and N representing the upper bound of a uniform degree distribution (lower bound == 1).")
                return
        else:
            try:
                assert type(in_degree) in [ int, np.int64 ]
                assert in_degree >= 1
                assert in_degree <= N
                ns = np.ones(N, int) * in_degree
            except AssertionError:
                print("Error: The in-degree must be a single integer (or N-dimensional vector of integers) between 1 and N representing the upper bound of a uniform degree distribution (lower bound == 1).")
                return
    elif in_degree_distribution == 'poisson':
        if type(in_degree) in [ list, np.array ]:
            try:
                assert np.all([ type(el) in [ int, np.int64, float, np.float64 ] for el in in_degree ])
                assert len(in_degree) == N
                assert min(in_degree) >= 1
                assert max(in_degree) <= N
                ns = np.array(in_degree[:])
            except AssertionError:
                print("Error: A vector was submitted for the in-degree.\nEnsure that the in-degree is an N-dimensional vector where each element is > 0 and represents the Poisson parameter.")
                return
        else:
            try:
                assert type(in_degree) in [ int, np.int64, float, np.float64 ]
                assert in_degree >= 1
                assert in_degree <= N
                ns = np.ones(N, int) * in_degree
            except AssertionError:
                print("Error: The in-degree must be a single number (or N-dimensional vector) > 0 representing the Poisson parameter.")
                return
    else:
        print("Error: None of the predefined in-degree distrbutions were chosen.\nTo use a user-defined in-degree vector, use the input 'in_degree' to submit an N-dimensional vector where each element must be between 1 and N.")
        return
    
    if edges_wiring_diagram is None:
        while True: # continue generation until a strongly connected graph is generated
            if in_degree_distribution == 'uniform':
                ns = 1 + np.random.randint(in_degree - 1, None, N, np.int64)
            elif in_degree_distribution == 'poisson':
                ns = np.random.poisson(in_degree, N)
                ns[ns == 0] = 1
                ns[ns > N - int(NO_SELF_REGULATION)] = N - int(NO_SELF_REGULATION)
            edges_wiring_diagram = random_edge_list(N, ns, not NO_SELF_REGULATION)
            if STRONGLY_CONNECTED:
                G = nx.from_edgelist(edges_wiring_diagram, create_using = nx.MultiDiGraph())
                if not nx.is_strongly_connected(G):
                    continue
            break
    else:
        try:
            assert len(set(np.array(edges_wiring_diagram).flatten())) == N
        except AssertionError:
            print("Number of nodes provided in edges_wiring_diagram != N")
            return
        ns = np.zeros(N, int)
        for target in np.array(edges_wiring_diagram)[:, 1]:
            ns[target] += 1
    
    if UNIFORM_BASE_DISTRIBUTION:
        if type(base) in [ list, np.array ]:
            try:
                assert np.all([ type(el) in [ int, np.int64 ] for el in base ])
                assert len(base) == N
                assert min(base) >= 2
                B = np.array([ 2 + np.random.randint(base[i] - 1, None, None) for i in range(N) ], int)
            except AssertionError:
                print("Error: A vector was submitted for the base.\nEnsure that you are providing an N-dimensional vector where each element is an integer >= 2 representing the upper bound of a uniform distribution (lower bound == 2).")
                return
        else:
            try:
                assert type(base) in [ int, np.int64 ]
                assert base >= 2
                B = 2 + np.random.randint(base - 1, None, N, np.int64)
            except AssertionError:
                print("Error: The base must be a single integer (or N-dimensional vector of integers) >= 2 representing the upper bound of a uniform distribution (lower bound == 2).")
                return
    else:
        if type(base) in [ list, np.array ]:
            try:
                assert np.all([ type(el) in [ int, np.int64 ] for el in base ])
                assert len(base) == N
                assert min(base) >= 2
                B = np.array(base[:], int)
            except AssertionError:
                print("Error: A vector was submitted for the base.\nTo use a user-defined base vector, ensure that it is an N-dimensional vector where each element is an integer >= 2.")
                return
        else:
            try:
                assert type(base) in [ int, np.int64 ]
                assert base >= 2
                B = np.ones(N, int) * base
            except AssertionError:
                print("Error: The base must be a single integer (or N-dimensional vector of integers) >= 2 when using a constant distribution.")
                return
    
    I = [ [] for _ in range(N) ]
    for edge in edges_wiring_diagram:
        I[edge[1]].append(edge[0])
    for i in range(N):
        I[i] = np.sort(I[i])
    
    F = []
    for i in range(N):
        if LINEAR:
            F.append(random_linear_ms_function(ns[i], B[i], B[I[i]]))
        else:
            F.append(random_non_degenerated_ms_function(B[i], B[I[i]]))
    
    return MultistateNetwork(F,I,B)