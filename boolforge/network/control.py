#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 01:03:29 2026

@author: ckadelka
"""

from collections.abc import Sequence
from copy import deepcopy
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import BooleanNetwork

class BooleanNetworkControlMixin:
    def get_network_with_fixed_identity_nodes(
        self,
        values_identity_nodes: Sequence[int],
        keep_controlled_nodes: bool = False,
        simplify_recursively: bool = False,
    ) -> "BooleanNetwork":
        """
        Construct a Boolean network with identity nodes fixed to given values.
    
        Identity nodes are nodes with a single self-regulatory edge and identity
        update rule ``f(x) = x``. This method fixes the values of such nodes and
        returns a new BooleanNetwork with the corresponding constants removed.
    
        Parameters
        ----------
        values_identity_nodes : sequence of int
            Values to fix for each identity node, in the order returned by
            ``get_identity_nodes(as_dict=False)``. Each value must be either 0 or 1.
        keep_controlled_nodes : bool, optional
            If True, controlled nodes are retained in the network as identity
            nodes with self-loops. If False (default), controlled nodes are
            eliminated as constants.
        simplify_recursively : bool, optional
            If True, recursively propagate fixed values through the network and
            eliminate any nodes whose update functions become constant as a result.
            This logical reduction is repeated until no further simplifications are
            possible. If False (default), only the explicitly controlled nodes are
            fixed and no additional recursive simplification is performed. 
            
        Returns
        -------
        BooleanNetwork
            A new BooleanNetwork with the specified identity nodes fixed.
        """
        indices_identity_nodes = self.get_identity_nodes(as_dict=False)
    
        if len(values_identity_nodes) != len(indices_identity_nodes):
            raise ValueError(
                f"The number of values provided ({len(values_identity_nodes)}) must "
                f"match the number of identity nodes ({len(indices_identity_nodes)})."
            )
    
        for v in values_identity_nodes:
            if v not in (0, 1):
                raise ValueError("Identity node values must be 0 or 1.")
    
        return self.get_network_with_node_controls(
            indices_controlled_nodes=indices_identity_nodes,
            values_controlled_nodes=values_identity_nodes,
            keep_controlled_nodes=keep_controlled_nodes,
            simplify_recursively=simplify_recursively,
        )


    def get_network_with_node_controls(
        self,
        indices_controlled_nodes: Sequence[int],
        values_controlled_nodes: Sequence[int],
        keep_controlled_nodes: bool = False,
        simplify_recursively: bool = False,
    ) -> "BooleanNetwork":
        """
        Construct a Boolean network with specified nodes fixed to given values.
    
        This method applies node-level interventions by fixing selected nodes to
        constant Boolean values. Controlled nodes may either be removed from the
        dynamic network as constants or retained as identity-clamped nodes.
    
        Parameters
        ----------
        indices_controlled_nodes : sequence of int
            Indices of nodes to be fixed.
        values_controlled_nodes : sequence of int
            Values to fix for each specified node, in the same order as
            ``indices_controlled_nodes``. Each value must be either 0 or 1.
        keep_controlled_nodes : bool, optional
            If True, controlled nodes are retained in the network as identity
            nodes with self-loops. If False (default), controlled nodes are
            eliminated as constants.
        simplify_recursively : bool, optional
            If True, recursively propagate fixed values through the network and
            eliminate any nodes whose update functions become constant as a result.
            This logical reduction is repeated until no further simplifications are
            possible. If False (default), only the explicitly controlled nodes are
            fixed and no additional recursive simplification is performed. 
            
        Returns
        -------
        BooleanNetwork
            A new BooleanNetwork with the specified node controls applied.
        """
        if len(indices_controlled_nodes) != len(values_controlled_nodes):
            raise ValueError(
                f"The number of controlled nodes ({len(indices_controlled_nodes)}) "
                f"must match the number of values provided ({len(values_controlled_nodes)})."
            )
    
        for node in indices_controlled_nodes:
            if not isinstance(node, (int, np.integer)) or node < 0 or node >= self.N:
                raise ValueError(f"Invalid node index: {node, not isinstance(node, int), node < 0 , node >= self.N}")
    
        for v in values_controlled_nodes:
            if v not in (0, 1):
                raise ValueError("Controlled node values must be 0 or 1.")
    
        if simplify_recursively and keep_controlled_nodes:
            raise ValueError(
                "Cannot simplify recursively when keep_controlled_nodes=True."
            )
    
        F = deepcopy(self.F)
        I = deepcopy(self.I)
        controlled_variables = [str(self.variables[int(i)]) for i in indices_controlled_nodes]
    
        for node, value in zip(indices_controlled_nodes, values_controlled_nodes):
            if keep_controlled_nodes:
                # Identity-clamped node
                F[node].f = np.array([value, value], dtype=int)
                F[node].n = 1
                I[node] = np.array([node], dtype=int)
            else:
                # Structural constant (to be removed)
                F[node].f = np.array([value], dtype=int)
                F[node].n = 0
                I[node] = np.array([], dtype=int)
    
        bn2 = self.__class__(F, I, self.variables) #__init__ removes fixated control nodes
    
        # Preserve previously removed constants if controlled nodes are eliminated
        if simplify_recursively:
            dummy = bn2.propagate_constants()
            bn3 = dummy['ReducedNetwork']
            fixation_layers = dummy['FixationLayers']
            bn3.constants.update(self.constants)
            bn3.constants.update(bn2.constants)
            bn3.fixation_layers = [controlled_variables] + fixation_layers
            return bn3
        else:
            if not keep_controlled_nodes:
                bn2.constants.update(self.constants)
                bn2.fixation_layers = [controlled_variables]
            return bn2

    def get_network_with_edge_controls(
        self,
        control_targets: Sequence[int],
        control_sources: Sequence[int],
        values_edge_controls: Sequence[int] | None = None,
        keep_fully_controlled_nodes: bool = True
    ) -> "BooleanNetwork":
        """
        Construct a Boolean network with specified regulatory edges controlled.
    
        This method fixes the influence of selected source nodes on selected target
        nodes by restricting the target's Boolean update function to entries where
        the source assumes a specified value, and then removing the corresponding
        regulatory edge.
    
        Parameters
        ----------
        control_targets : sequence of int
            Indices of target nodes.
        control_sources : sequence of int
            Indices of source nodes whose influence on the corresponding targets
            is to be controlled.
        values_edge_controls : sequence of int, optional
            Fixed values (0 or 1) imposed on each controlled edge. If None, all
            controlled edges are fixed to 0.
        keep_fully_controlled_nodes : bool, optional
            If True (default), nodes without any remaining regulation are retained
            in the network as identity nodes with self-loops. 
            If False, fully controlled nodes are eliminated as constants.
            
        Returns
        -------
        BooleanNetwork
            A new BooleanNetwork with the specified edge controls applied.
    
        Raises
        ------
        ValueError
            If input lengths do not match, indices are invalid, or edge values are
            not in {0, 1}.
        """
        if len(control_targets) != len(control_sources):
            raise ValueError("control_targets and control_sources must have equal length.")
    
        if values_edge_controls is None:
            values_edge_controls = [0] * len(control_targets)
    
        if len(values_edge_controls) != len(control_targets):
            raise ValueError(
                "values_edge_controls must have the same length as control_targets."
            )
    
        F_new = deepcopy(self.F)
        I_new = deepcopy(self.I)
    
        for target, source, fixed_value in zip(
            control_targets, control_sources, values_edge_controls
        ):
            if fixed_value not in (0, 1):
                raise ValueError("Edge control values must be 0 or 1.")
    
            if not (0 <= target < self.N):
                raise ValueError(f"Invalid target index: {target}")
    
            if not (0 <= source < self.N):
                raise ValueError(f"Invalid source index: {source}")
    
            if source not in I_new[target]:
                raise ValueError(
                    f"Source node {source} is not a regulator of target node {target}."
                )
    
            idx_reg = list(I_new[target]).index(source)
            n_inputs = F_new[target].n
    
            truth_indices = np.arange(2**n_inputs, dtype=np.uint32)
            mask = ((truth_indices >> (n_inputs - 1 - idx_reg)) & 1) == fixed_value
    
            # Restrict truth table
            F_new[target].f = F_new[target].f[mask]
            F_new[target].n -= 1
    
            # Remove regulator
            I_new[target] = np.delete(I_new[target], idx_reg)
            
            # ---- NEW LOGIC: fully controlled node -----------------------------
            if F_new[target].n == 0:
                if keep_fully_controlled_nodes:
                    value = int(F_new[target].f[0])
                    # Identity-clamped node
                    F_new[target].f = np.array([value, value], dtype=int)
                    F_new[target].n = 1
                    I_new[target] = np.array([target], dtype=int)

        return self.__class__(F_new, I_new, self.variables)
