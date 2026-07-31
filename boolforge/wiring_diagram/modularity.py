#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np

class WiringDiagramModularityMixin:
    def get_strongly_connected_components(self) -> list:
        """
        Compute the strongly connected components of the wiring diagram.
    
        A strongly connected component (SCC) is a maximal set of nodes such that
        every node in the set is reachable from every other node via directed
        paths.
    
        Returns
        -------
        list of set of int
            List of strongly connected components, where each component is
            represented as a set of node indices.
        """
        edges = [(int(reg), target) for target, regs in enumerate(self.I) for reg in regs]
        subG = nx.DiGraph(edges)
        sccs = list(nx.strongly_connected_components(subG))
        self._set_property('sccs', sccs, context=None, exact=True)
        return sccs

    def get_modular_structure(self) -> set[tuple[int, int]]:
        """
        Compute the modular (condensation) structure of the wiring diagram.
    
        The modular structure is represented as a directed acyclic graph (DAG)
        whose nodes correspond to strongly connected components (SCCs) of the
        wiring diagram. A directed edge ``(i, j)`` indicates that there exists
        at least one regulatory interaction from SCC ``i`` to SCC ``j``.
    
        Returns
        -------
        set of tuple of int
            Set of directed edges representing the condensation DAG, where each
            edge ``(i, j)`` denotes a regulation from SCC ``i`` to SCC ``j``.
            
        Notes
        -----
        SCC indices correspond to the ordering returned by
        ``get_strongly_connected_components``.
        """
        sccs = self.get_strongly_connected_components()
    
        # Map node index -> SCC index
        scc_dict = {}
        for idx, scc in enumerate(sccs):
            for node in scc:
                scc_dict[node] = idx
    
        modular_dag = set()
    
        for target, regulators in enumerate(self.I):
            for j, regulator in enumerate(regulators):
                src = scc_dict[int(regulator)]
                tgt = scc_dict[target]
    
                if src == tgt:
                    continue
    
                if self.weights is not None:
                    if np.isnan(self.weights[target][j]):
                        continue
    
                modular_dag.add((src, tgt))
        self._set_property('modular_dag', modular_dag, context=None, exact=True)
        return modular_dag