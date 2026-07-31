#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import networkx as nx
import numpy as np

class WiringDiagramMotifMixin:
    def get_ffls(
        self,
    ) -> tuple[list[list[int]], list[list[float]] | None]:
        """
        Identify feed-forward loops (FFLs) in the wiring diagram.
    
        A feed-forward loop is a triplet of nodes ``(i, j, k)`` such that
        ``i -> j``, ``j -> k``, and ``i -> k`` are all regulatory interactions.
    
        Returns
        -------
        tuple
            A tuple ``(ffls, types)`` where:
    
            - ``ffls`` is a list of feed-forward loops, each represented as
              ``[i, j, k]``.
            - ``types`` is a list of corresponding edge-weight triplets
              ``[i -> k, i -> j, j -> k]`` if interaction weights are defined,
              or ``None`` otherwise.
        """
        # Build inverted wiring diagram: regulator -> (target, index in I[target])
        I_inv = [[] for _ in range(self.N)]
        for target, regulators in enumerate(self.I):
            for idx, regulator in enumerate(regulators):
                I_inv[int(regulator)].append((target, idx))
    
        ffls: list[list[int]] = []
        types: list[list[float]] | None = [] if self.weights is not None else None
    
        for i in range(self.N):  # master regulator
            targets_i = {t for (t, _) in I_inv[i]}
    
            for j, idx_ij in I_inv[i]:  # i -> j
                if i == j:
                    continue
    
                targets_j = {t for (t, _) in I_inv[j]}
                common_targets = targets_i & targets_j
    
                for k in common_targets:
                    if k == i or k == j:
                        continue
    
                    ffls.append([i, j, k])
    
                    if types is not None:
                        # locate indices for k safely
                        idx_ik = next(idx for idx, r in enumerate(self.I[k]) if r == i)
                        idx_jk = next(idx for idx, r in enumerate(self.I[k]) if r == j)
    
                        direct = float(self.weights[k][idx_ik])
                        indirect1 = float(self.weights[j][idx_ij])
                        indirect2 = float(self.weights[k][idx_jk])
    
                        types.append([direct, indirect1, indirect2])
        
        return_dict = {'FFLs' : ffls}
        self._set_property('ffls', ffls, context=None, exact=True)
        if types is not None:
            return_dict['Types'] = types
            self._set_property('ffl_types', types, context=None, exact=True)
        return return_dict
    
        
        
    def get_fbls(self, max_length: int = 4, classify : bool = False) -> dict:
        """
        Identify feedback loops (simple directed cycles).
    
        A feedback loop is defined as a simple directed cycle in the underlying
        wiring diagram. This method enumerates elementary circuits using a
        variant of Johnson's algorithm, restricted to cycles of length at most
        ``max_length``.
    
        Parameters
        ----------
        max_length : int, optional
            Maximum length of feedback loops to consider. Cycles longer than
            ``max_length`` are not returned. Default is 4.
        classify : bool, optional
            If True and interaction weights are available, the type of each
            feedback loop is determined.
        
        Returns
        -------
        dict
            list of list of int
            List of feedback loops, where each loop is represented as a list
            of node indices forming a directed cycle. Self-loops are returned
            as single-element lists.
            
        """
        G = self.to_DiGraph(use_variable_names=False)
    
        def _unblock(thisnode, blocked, B):
            stack = set([thisnode])
            while stack:
                node = stack.pop()
                if node in blocked:
                    blocked.remove(node)
                    stack.update(B[node])
                    B[node].clear()
    
        subG = nx.DiGraph(G.edges())
        sccs = [scc for scc in nx.strongly_connected_components(subG) if len(scc) > 1]
        
        fbls = []
        
        # Yield self-cycles and remove them.
        for v in subG:
            if subG.has_edge(v, v):
                fbls.append([v])
                subG.remove_edge(v, v)
        
        while sccs:
            scc = sccs.pop()
            sccG = subG.subgraph(scc)
            startnode = scc.pop()
            path = [startnode]
            len_path = 1
            blocked = set()
            closed = set()
            blocked.add(startnode)
            B = defaultdict(set)
            stack = [(startnode, list(sccG[startnode]))]
            while stack:
                thisnode, nbrs = stack[-1]
                if nbrs and len_path <= max_length:
                    nextnode = nbrs.pop()
                    if nextnode == startnode:
                        fbls.append( path[:] )
                        closed.update(path)
                    elif nextnode not in blocked:
                        path.append(nextnode)
                        len_path += 1
                        stack.append((nextnode, list(sccG[nextnode])))
                        closed.discard(nextnode)
                        blocked.add(nextnode)
                        continue
                if not nbrs or len_path > max_length:
                    if thisnode in closed:
                        _unblock(thisnode, blocked, B)
                    else:
                        for nbr in sccG[thisnode]:
                            if thisnode not in B[nbr]:
                                B[nbr].add(thisnode)
                    stack.pop()
                    path.pop()
                    len_path -= 1
            H = subG.subgraph(scc)
            sccs.extend(scc for scc in nx.strongly_connected_components(H) if len(scc) > 1)
        
        return_dict = {'FBLs' : fbls}
        self._set_property('fbls', fbls, 
                           context=f"max_length={max_length}", exact=True)

        if self.weights is not None and classify:
            types,n_negative = self._get_types_of_fbls(fbls)
            return_dict['Types'] = types
            return_dict['NumberNegativeEdges'] = n_negative
            self._set_property('fbl_types', types, 
                               context=f"max_length={max_length}", exact=True)
            self._set_property('fbl_negative_edges', n_negative, 
                               context=f"max_length={max_length}", exact=True)
        return return_dict


    def _get_types_of_fbls(
        self,
        fbls: list[list[int]],
    ) -> tuple[list[float], list[int]] | None:
        """
        Determine the types of feedback loops based on interaction weights.
    
        For each feedback loop, this method computes the product of the
        interaction weights along the cycle and counts the number of
        negative regulations.
    
        Parameters
        ----------
        fbls : list of list of int
            List of feedback loops, where each loop is represented as a list
            of node indices forming a directed cycle.
    
        Returns
        -------
        tuple of list or None
            If interaction weights are defined, returns a tuple
            ``(types, n_negative)``, where:
    
            - ``types`` is a list containing the product of edge weights for
              each feedback loop. Interpretation:
              'non-essential' : np.nan, 'conditional' : 0, 
              'positive' : 1, 'negative' : -1
            - ``n_negative`` is a list containing the number of negative
              regulations in each feedback loop.
    
            If interaction weights are not defined, returns ``None``.
        """
        if self.weights is None:
            return None
    
        types: list[float] = []
        n_negative: list[int] = []
    
        for fbl in fbls:
            all_weights = []
    
            for u, v in zip(fbl, fbl[1:] + [fbl[0]]):
                # find index of regulator u in I[v]
                for idx, reg in enumerate(self.I[v]):
                    if reg == u:
                        w = self.weights[v][idx]
                        break
                else:
                    raise ValueError(f"No edge {u} -> {v} found in wiring diagram")
    
                all_weights.append(w)
    
            types.append(float(np.prod(all_weights)))
            n_negative.append(int(sum(w < 0 for w in all_weights)))
    
        return types, n_negative
