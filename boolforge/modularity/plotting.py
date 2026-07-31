#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:46:54 2026

@author: ckadelka
"""

import math
import networkx as nx
import numpy as np

def plot_trajectory(compressed_trajectory_graph : nx.DiGraph,
    show : bool = True):
    """
    Visualize a compressed trajectory graph using a layered layout.
    
    Initial states are highlighted with a box. Layers are computed
    based on weakly connected components to improve readability.
    
    Parameters
    ----------
    compressed_trajectory_graph : networkx.DiGraph
        Directed graph of compressed trajectories.
    
    show : bool, default=True
        Whether to call ``plt.show()`` at the end.
    """
    
    try:
        import matplotlib.pyplot as plt
    except:
        raise ImportError(
            "Plotting requires matplotlib. "
            "Install it with: pip install matplotlib"
        )
    
    def layout_tree(G, root, x0, y0, dx, pos, visited):
        children = [c for c in G.predecessors(root) if c not in visited]
        if not children:
            return
        width = dx * (len(children) - 1)
        xs = [x0 - width/2 + i*dx for i in range(len(children))]
        
        for child, x in zip(children, xs):
            pos[child] = (x, y0 - 1)
            visited.add(child)
            layout_tree(G, child, x, y0 - 1, dx/1.5, pos, visited)
        return
    
    G = compressed_trajectory_graph.copy()
    
    components = list(nx.weakly_connected_components(G))
    fig, axes = plt.subplots(
        nrows=len(components),
        figsize=(10, 5 * len(components)),
        squeeze=False
    )

    labels = nx.get_node_attributes(G, "NLbl")
    initial = nx.get_node_attributes(G, "StIn")

    for idx, comp in enumerate(components):
        ax = axes[idx][0]
        sub_nodes = list(comp)
        SG = G.subgraph(sub_nodes).copy()
        pos = {}

        # Find a cycle in the component
        start = sub_nodes[0]
        visited = {}
        v = start
        while v not in visited:
            visited[v] = True
            succ = list(SG.successors(v))
            if not succ:
                break
            v = succ[0]

        # Build the cycle
        cycle = [v]
        succ = list(SG.successors(v))
        if succ:
            u = succ[0]
            while u != v:
                cycle.append(u)
                u = next(iter(SG.successors(u)))

        # Place cycle nodes in a circle
        n = len(cycle)
        for i, node in enumerate(cycle):
            angle = 2 * math.pi * i / n
            pos[node] = (2 * np.cos(angle), 2 * np.sin(angle))

        # Layout trees hanging off the cycle
        visited = set(cycle)
        for node in cycle:
            layout_tree(SG, node, pos[node][0], pos[node][1], dx=1.5, pos=pos, visited=visited)
        
        nx.draw_networkx(
            SG,
            pos,
            ax=ax,
            node_size=1200,
            node_color="#00000000",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=8
        )
        
        ax.invert_yaxis() # flip, so the graph points downward
        
        normal = {}
        boxed = {}
        for n in SG.nodes():
            if initial.get(n, False):
                boxed[n] = labels[n]
            else:
                normal[n] = labels[n]
        
        nx.draw_networkx_labels(
            SG, pos, labels=normal, font_size=12,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="white", lw=1),
            ax=ax
        )
        nx.draw_networkx_labels(
            SG, pos, labels=boxed, font_size=12,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1),
            ax=ax
        )
        
        ax.axis("equal")
        ax.axis("off")

    plt.tight_layout()
    if show:
        plt.show()
    
    return fig
