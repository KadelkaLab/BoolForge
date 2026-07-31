#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .core import WiringDiagram

import networkx as nx

class WiringDiagramInteroperabilityMixin: 
    @classmethod
    def from_DiGraph(
        cls,
        nx_DiGraph: "nx.DiGraph",
    ) -> "WiringDiagram":
        """
        Construct a WiringDiagram from a NetworkX directed graph.
    
        Each node in the directed graph represents a Boolean variable, and each
        directed edge ``u -> v`` indicates that variable ``u`` regulates variable
        ``v``.
    
        Parameters
        ----------
        nx_DiGraph : nx.DiGraph
            Directed graph whose nodes represent variables and whose edges
            represent regulatory interactions.
    
            Node attributes (optional)
                name : str
                    Name of the variable. If not provided, the node label is used
                    when possible.
    
            Edge attributes (optional)
                weight : float
                    Weight of the regulatory interaction from ``u`` to ``v``.
                    If present on all edges, weights are stored in the
                    ``weights`` attribute of the resulting WiringDiagram.
    
        Returns
        -------
        WiringDiagram
            Wiring diagram representing the topology of the directed graph.
    
        Notes
        -----
        - Nodes are ordered according to the iteration order of
          ``nx_DiGraph.nodes``.
        - Regulator lists are constructed from incoming edges
          (graph predecessors).
        - Edge weights are only stored if all edges define a ``'weight'``
          attribute.
    
        Examples
        --------
        >>> import networkx as nx
        >>> from boolforge import WiringDiagram
        >>> G = nx.DiGraph()
        >>> G.add_edges_from([(0, 1), (1, 2)])
        >>> W = WiringDiagram.from_DiGraph(G)
        >>> W.I
        [array([], dtype=int64), array([0]), array([1])]
        >>> W.get_source_nodes()
        {0: True, 1: False, 2: False}
        """
        if not isinstance(nx_DiGraph, nx.DiGraph):
            raise TypeError("nx_DiGraph must be a networkx.DiGraph")
    
        # Preserve NetworkX node iteration order
        nodes = list(nx_DiGraph.nodes)
    
        # Map nodes to indices
        node_to_idx = {node: i for i, node in enumerate(nodes)}
    
        # Extract variable names
        variables = []
        for node in nodes:
            if 'name' in nx_DiGraph.nodes[node]:
                variables.append(str(nx_DiGraph.nodes[node]['name']))
            elif isinstance(node, str):
                variables.append(node)
            else:
                variables.append(f"x_{node}")
    
        # Build adjacency list I (regulators / predecessors)
        I = []
        for node in nodes:
            regulators = [node_to_idx[r] for r in nx_DiGraph.predecessors(node)]
            I.append(regulators)
    
        # Extract weights only if all edges define a weight
        weights = None
        if nx_DiGraph.number_of_edges() > 0 and all(
            'weight' in nx_DiGraph[u][v] for u, v in nx_DiGraph.edges
        ):
            weights = []
            for node in nodes:
                row = [
                    nx_DiGraph[r][node]['weight']
                    for r in nx_DiGraph.predecessors(node)
                ]
                weights.append(row)
    
        return cls(I=I, variables=variables, weights=weights)


    def to_DiGraph(self, use_variable_names: bool = True) -> nx.DiGraph:
        """
        Convert the wiring diagram into a NetworkX directed graph.
        
        A directed edge ``u -> v`` indicates that node ``u``
        regulates node ``v``.
        
        Parameters
        ----------
        use_variable_names : bool, optional
            If True (default), nodes are labeled using the variable names stored in
            ``self.variables``. If False, nodes are labeled by integer indices
            ``0, 1, ..., N-1``.
        
        Returns
        -------
        nx.DiGraph
            Directed graph representing the wiring diagram. If interaction weights
            are present, they are stored as edge attributes under the key
            ``'weight'``.
        
        Notes
        -----
        - The node ordering follows the internal ordering of the wiring diagram.
        - Incoming edges of node ``i`` correspond to the regulators listed in
          ``self.I[i]``.
        - Edge weights are included only if the wiring diagram defines weights.
        """
        G = nx.DiGraph()
    
        if use_variable_names:
            nodes = list(self.variables)
            idx_to_node = {i: self.variables[i] for i in range(self.N)}
        else:
            nodes = list(range(self.N))
            idx_to_node = {i: i for i in range(self.N)}
    
        G.add_nodes_from(nodes)
    
        for i in range(self.N):
            target = idx_to_node[i]
            for j, reg in enumerate(self.I[i]):
                source = idx_to_node[reg]
                if self.weights is not None:
                    G.add_edge(source, target, weight=self.weights[i][j])
                else:
                    G.add_edge(source, target)
    
        return G