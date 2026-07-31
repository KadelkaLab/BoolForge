#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:46:54 2026

@author: ckadelka
"""

# ===================== #
#   Modular BoolForge   #
# ===================== #

import math

import networkx as nx
import numpy as np
from collections.abc import Sequence

from .. import utils

def merge_state_representation(x : int | Sequence[int], 
                               y : int | Sequence[int],
                               num_nodes : int | Sequence[int]
    ) -> int | Sequence[int]:
    """
    Combine two state representations into a single decimal representation.
    
    Parameters
    ----------
    x : int or sequence of int
        First state. Can be a single integer or a pair of integers.
    
    y : int or sequence of int
        Second state. Can be a single integer or a pair of integers.
    
    b : int or sequence of int
        Bit size of y. Must match the structure of y (int or pair of ints).
    
    Returns
    -------
    result : int or tuple of int
        Combined state representation. Returns an int if both x and y
        are integers. Returns a tuple of two ints if either x or y is a tuple/list.
    """

    is_pair_x = isinstance(x, Sequence)
    is_pair_y = isinstance(y, Sequence)
    if is_pair_x:
        if is_pair_y:
            return ((x[0] << num_nodes[0]) | y[0], (x[1] << num_nodes[1]) | y[1]) 
        return (x[0], (x[1] << num_nodes) | y)
    elif is_pair_y:
        return (y[0], (x << num_nodes[1]) | y[1])
    return (x << num_nodes) | y

def get_product_of_attractors(attrs_1 : Sequence[Sequence[int | Sequence[int]]],
    attrs_2 : Sequence[Sequence[int | Sequence[int]]],
    bits : int | Sequence[int]) -> list:
    """
    Compute the product of two sets of attractors by combining their states.
    
    Parameters
    ----------
    attrs_1 : sequence of sequences of int or sequence of sequences of pairs of ints
        First set of attractors. Each attractor is a list of states.
    
    attrs_2 : sequence of sequences of int or sequence of sequences of pairs of ints
        Second set of attractors. Each attractor is a list of states.
    
    bits : int or pair of ints
        Bit size of states in attrs_2. Used when merging states.
    
    Returns
    -------
    result : sequence of sequences of int or sequence of sequences of pairs of ints
        Product set of attractors obtained by merging each attractor
        from attrs_1 with each attractor from attrs_2.
    """

    attractors = []
    for attr1 in attrs_1:
        attr = []
        for attr2 in attrs_2:
            m = len(attr1)
            n = len(attr2)
            for i in range(math.lcm(*[m, n])):
                attr.append(merge_state_representation(attr1[i % m], attr2[i % n], bits))
        attractors.append(attr)
    return attractors

def compress_trajectories(trajectories : tuple[Sequence[int], int],
    num_nodes : int) -> nx.DiGraph:
    """
    Compress multiple trajectories into a single directed graph.
    
    Each trajectory is represented by a prefix (non-periodic states)
    and a cycle (periodic states). Nodes are merged when identical
    prefixes or cycles occur across trajectories.
    
    Parameters
    ----------
    trajectories : tuple of (sequence of int, int)
        List of trajectories. Each trajectory is a tuple containing
        a list of decimal states and the length of its periodic cycle.
    
    num_nodes : int
        Number of nodes in the network. Used to format node labels as binary strings.
    
    Returns
    -------
    G : networkx.DiGraph
        Directed graph representing all merged trajectories.
    """

    # Helper method: determine the 'canon' ordering of a periodic pattern.
    # The canon ordering is the phase such that the lowest states come first
    # without changing the relative ordering of the states.
    def _canon_cycle_(pattern):
        return min([ tuple(pattern[i:] + pattern[:i]) for i in range(len(pattern)) ])
    
    # Helper method: determine which offset a given pattern is from the canon
    # ordering. That is, how much the pattern has been phased relative to the
    # canon ordering.
    def _cycle_offset_(pattern, canon):
        pattern = list(pattern)
        canon = list(canon)
        len_pattern = len(pattern)
        for offset in range(len_pattern):
            if canon[offset:] + canon[:offset] == pattern:
                return offset
        raise ValueError("Pattern does not match canonical rotations")
    
    G = nx.DiGraph()
    next_id = 0
    cycle_nodes = {}
    prefix_merge = {}
    for states, period in trajectories:
        len_traj = len(states)
        # First look through the non-periodic component of the trajectory,
        # also referred to in this code as the 'prefix' of the trajectory
        len_pref = len_traj - period
        pref_ids = []
        for i in range(len_pref):
            # Determine if this prefix can be merged elsewhere into the graph
            future = states[i:]
            prefix_tail = future[:-period]
            pattern = future[-period:]
            canon = _canon_cycle_(pattern)
            entry_offset = _cycle_offset_(pattern, canon)
            signature = (tuple(prefix_tail), canon, entry_offset)
            # If so, merge the it and mark the node as initial
            if signature in prefix_merge:
                node_id = prefix_merge[signature]
                if i == 0:
                    G.nodes[node_id]["StIn"] = True
            # Otherwise, make a new initial node
            else:
                node_id = next_id
                prefix_merge[signature] = node_id
                G.add_node(next_id, StIn=(i == 0),
                    NLbl=(str(utils.dec2bin(states[i], num_nodes)).replace(' ', '').replace(',', '').replace('[', '').replace(']', '')))
                pref_ids.append(next_id)
                next_id += 1
            pref_ids.append(node_id)
        # Once prefix nodes are added, create edges
        for i in range(len(pref_ids) - 1):
            if pref_ids[i] != pref_ids[i+1]:
                G.add_edge(pref_ids[i], pref_ids[i+1])
        # Second look through the periodic component of the trajectory,
        # also referred to in this code as the 'cycle' of the trajectory
        cycle = states[-period:]
        key = _canon_cycle_(cycle)
        # If we have found a new cycle, add it to the graph
        if key not in cycle_nodes:
            ids = []
            for s in key:
                # Create nodes based off of the canon ordering to ensure
                # predictable ordering in case we need to reference
                # this cycle again for another trajectory
                G.add_node(next_id, StIn=False,
                    NLbl=(str(utils.dec2bin(s, num_nodes)).replace(' ', '').replace(',', '').replace('[', '').replace(']', '')))
                ids.append(next_id)
                next_id += 1
            # Once nodes are added, add in edges
            for a, b in zip(ids, ids[1:]):
                G.add_edge(a, b)
            G.add_edge(ids[-1], ids[0])
            cycle_nodes[key] = ids
        # For a trajectory without a prefix, mark the first state of the trajectory
        # within the cycle as an initial node
        if len_pref == 0:
            G.nodes()[cycle_nodes[key][_cycle_offset_(cycle, key)]]["StIn"] = True
        # Otherwise, we need to add an edge between the prefix and cycle
        else:
            G.add_edge(pref_ids[-1], cycle_nodes[key][_cycle_offset_(cycle, key)])
    return G

def product_of_trajectories(
        compressed_trajectory_graph_1 : nx.DiGraph,
        compressed_trajectory_graph_2 : nx.DiGraph
        ) -> nx.DiGraph:
    """
    Compute the product of two compressed trajectory graphs, following the
    premise of equal reachability.
    
    The resulting graph contains all combinations of nodes from
    the two input graphs, with edges representing all possible
    successor pairs.
    
    Parameters
    ----------
    compressed_trajectory_graph_1 : networkx.DiGraph
        First compressed trajectory graph.
    
    compressed_trajectory_graph_2 : networkx.DiGraph
        Second compressed trajectory graph.
    
    Returns
    -------
    G : networkx.DiGraph
        Directed graph representing the product of the two input graphs.
    """

    _initial_1 = []
    _initial_2 = []
    for n in compressed_trajectory_graph_1.nodes:
        if compressed_trajectory_graph_1.nodes[n]["StIn"]:
            _initial_1.append(n)
    for n in compressed_trajectory_graph_2.nodes:
        if compressed_trajectory_graph_2.nodes[n]["StIn"]:
            _initial_2.append(n)
    G = nx.DiGraph()
    starting = []
    for n1 in _initial_1:
        for n2 in _initial_2:
            starting.append((n1, n2))
            G.add_node((n1, n2), StIn=compressed_trajectory_graph_1.nodes[n1]["StIn"] and compressed_trajectory_graph_2.nodes[n2]["StIn"],
                NLbl=f"{compressed_trajectory_graph_1.nodes[n1]['NLbl']}{compressed_trajectory_graph_2.nodes[n2]['NLbl']}")
    stack = starting[:]
    visited = set(starting)
    while stack:
        u1, u2 = stack.pop()
        for v1 in compressed_trajectory_graph_1.successors(u1):
            for v2 in compressed_trajectory_graph_2.successors(u2):
                new_pair = (v1, v2)
                if new_pair not in G:
                    G.add_node(new_pair, StIn=False,
                        NLbl=f"{compressed_trajectory_graph_1.nodes[v1]['NLbl']}{compressed_trajectory_graph_2.nodes[v2]['NLbl']}")
                G.add_edge((u1, u2), new_pair)
                if new_pair not in visited:
                    visited.add(new_pair)
                    stack.append(new_pair)
    return G

