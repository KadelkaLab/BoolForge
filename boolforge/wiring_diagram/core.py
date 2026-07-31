#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections.abc import Sequence
import numpy as np

from .interoperability import WiringDiagramInteroperabilityMixin
from .modularity import WiringDiagramModularityMixin
from .motifs import WiringDiagramMotifMixin
from .plotting import WiringDiagramPlottingMixin

class WiringDiagram(
        WiringDiagramInteroperabilityMixin,
        WiringDiagramModularityMixin,
        WiringDiagramMotifMixin,
        WiringDiagramPlottingMixin,
        ):
    """
    Directed wiring diagram for a dynamical system.

    A wiring diagram specifies the regulatory topology of a dynamical system by
    listing, for each node, the indices of its regulators (incoming edges).
    It does not encode dynamical update functions or dynamical rules.

    Nodes with zero indegree are source nodes. Whether a source node represents
    a constant, an identity node, or a dynamic variable is determined only after
    dynamical update functions are assigned.

    Parameters
    ----------
    I : sequence of sequences of int
        Adjacency-list representation of the wiring diagram.
        Entry ``I[i]`` contains the indices of nodes regulating node ``i``.
        Regulator lists may have variable length.
    variables : list[str] or np.ndarray[str], optional
        Names of the variables corresponding to each node.
        Must have length ``N`` if provided. If None, default names
        ``['x0', 'x1', ..., 'x{N-1}']`` are assigned.
    weights : sequence of sequences of float, optional
        Interaction weights corresponding to ``I``.
        Entry ``weights[i][j]`` gives the weight of the interaction
        from regulator ``I[i][j]`` to node ``i``. Missing interactions
        may be encoded as ``np.nan``.

    Attributes
    ----------
    I : list[np.ndarray]
        Regulator index arrays, one per node.
    variables : np.ndarray[str]
        Names of variables corresponding to each node.
    N : int
        Total number of nodes, including source nodes.
    indegrees : np.ndarray[int]
        Indegree of each node.
    outdegrees : np.ndarray[int]
        Outdegree of each node.
    weights : list[np.ndarray] or None
        Interaction weights associated with ``I`` if provided.

    Notes
    -----
    - Node indices are zero-based.
    - Source nodes are identified solely by indegree == 0.
    - The wiring diagram encodes topology only and does not define dynamics.

    Examples
    --------
    Nodes with zero indegree are interpreted as source nodes.
    
    >>> from boolforge import WiringDiagram
    >>> I = [
            [],        # node 0 has no regulators -> source node
            [0],       # node 1 is regulated by node 0
            [0, 1],    # node 2 is regulated by nodes 0 and 1
        ]
    >>> W = WiringDiagram(I)
    
    >>> W.N
    3
    
    >>> W.get_source_nodes(as_dict=True)
    {0: True, 1: False, 2: False}
    
    >>> W.get_source_nodes(as_dict=False)
    array([0])
    
    See Also
    --------
    boolforge.BooleanNetwork
    """
    
    def __init__(
        self,
        I : Sequence[Sequence[int]],
        variables : list[str] | np.ndarray | None = None,
        weights : Sequence[Sequence[float]] | None = None,
    ):
        """
        Initialize a wiring diagram.
    
        Parameters
        ----------
        I : sequence of sequences of int
            Adjacency list representation of the wiring diagram.
            Entry ``I[i]`` contains the indices of nodes regulating node ``i``.
            Regulator lists may have variable length.
        variables : list[str] or np.ndarray[str], optional
            Names of the variables corresponding to each node.
            Must have length ``N`` if provided. If None, default names
            ``['x0', 'x1', ..., 'x{N-1}']`` are assigned.
        weights : sequence of sequences of float, optional
            Optional interaction weights corresponding to ``I``.
            Each entry ``weights[i][j]`` gives the weight of the interaction
            from regulator ``I[i][j]`` to node ``i``.
            Missing or undefined interactions may be encoded as ``np.nan``.
    
        Raises
        ------
        TypeError
            If ``I`` is not a sequence of sequences of integers.
            If ``weights`` is provided and is not a sequence of sequences of numbers.
            If any entry ``weights[i]`` is not a sequence of numbers.
        ValueError
            If ``variables`` is provided and does not have length ``N``.
            If ``weights`` is provided and does not have length ``N``.
            If any entry ``weights[i]`` does not have the same length as ``I[i]``.

    
        Notes
        -----
        Source nodes are identified automatically as nodes with zero indegree.
        """
        if not isinstance(I, Sequence) or isinstance(I, (str, bytes)):
            raise TypeError("I must be a sequence of sequences of int")
            
        if variables is not None and len(I) != len(variables):
            raise ValueError("len(I) == len(variables) required if variable names are provided")

        # ---- Properties (empty initially) -----------------------------------
        self._properties_exact = {}
        self._properties_estimated = {}
        
        self.I = [np.array(regulators,dtype=int) for regulators in I]
        self.N = len(I)
        self.indegrees = np.array([len(regulators) for regulators in self.I], dtype=int)
        
        if variables is None:
            variables = ['x'+str(i) for i in range(self.N)]
        
        self.variables = np.array(variables, dtype=str)
        
        self.outdegrees = self.get_outdegrees()
        
        if weights is not None:
            if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)):
                raise TypeError("weights must be None or a sequence of sequences of numbers")

            if len(weights) != self.N:
                raise ValueError("weights must have the same length as I")
        
            self.weights = []
            for i, (regs, row) in enumerate(zip(self.I, weights)):
                if not isinstance(row, Sequence):
                    raise TypeError(f"weights[{i}] must be a sequence of floats")
                if len(row) != len(regs):
                    raise ValueError(f"weights[{i}] must have the same length as I[{i}]")
                self.weights.append(np.array(row, dtype=float))
        else:
            self.weights = None

        
    def _make_property_key(self, name, context=None):
        if context is None:
            return name
        context = str(context).lower()
        return (name, context)

    def _set_property(self, name, value, context=None, exact=True):
        key = self._make_property_key(name, context)
    
        if exact:
            self._properties_exact[key] = value
            self._properties_estimated.pop(key, None)
        elif key not in self._properties_exact:
                self._properties_estimated[key] = value
    
    def _get_property(self, name, context=None):
        key = self._make_property_key(name, context)
    
        if key in self._properties_exact:
            return self._properties_exact[key], "exact"
    
        if key in self._properties_estimated:
            return self._properties_estimated[key], "estimated"

        # detect missing context argument
        if context is None:
            for dictionary in (self._properties_exact, self._properties_estimated):
                for k in dictionary:
                    if isinstance(k, tuple) and k[0] == name:
                        raise ValueError(
                            f"Property '{name}' depends on the context. Specify context."
                        )
            
        return None, None
                
    def __str__(self):
        return (
            f"WiringDiagram(N={self.N}, "
            f"indegrees={self.indegrees.tolist()})"
        )

    def __getitem__(self, index):
        return self.I[index]

    def __repr__(self):
        return f"{type(self).__name__}(N={self.N})"

    def __len__(self):
        return self.N

    def get_outdegrees(self) -> np.ndarray:
        """
        Compute the outdegree of each node.
    
        The outdegree of a node is the number of nodes it regulates, i.e.,
        the number of outgoing edges from that node in the wiring diagram.
    
        Returns
        -------
        np.ndarray
            One-dimensional array of length ``N`` containing the outdegree of
            each node.
        """
        outdegrees = np.zeros(self.N, dtype=int)
        for regulators in self.I:
            for regulator in regulators:
                outdegrees[regulator] += 1
        return outdegrees

    def get_source_nodes(
        self, 
        as_dict: bool = True
    ) -> dict[int, bool] | np.ndarray:
        """
        Identify source nodes in the wiring diagram.
    
        A source node is a node with zero indegree. Source nodes represent
        inputs to the wiring diagram; whether they act as constants or
        identity nodes in dynamical systems depends on the associated
        dynamical update functions and is not determined at the wiring-diagram
        level.
    
        Parameters
        ----------
        as_dict : bool, optional
            If True (default), return a dictionary mapping node indices to
            Boolean values indicating whether each node is a source node.
            If False, return an array of indices corresponding to source nodes.
    
        Returns
        -------
        dict[int, bool] or np.ndarray
            If ``as_dict`` is True, a dictionary where keys are node indices and
            values indicate whether the node is a source node.
            If ``as_dict`` is False, a one-dimensional array containing the
            indices of source nodes.
        """
        is_source = self.indegrees == 0
        if as_dict:
            return dict(enumerate(is_source.tolist()))
        return np.where(is_source)[0]
