#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:52:32 2026

@author: ckadelka
"""


import warnings

from collections.abc import Sequence
from collections import deque
from copy import deepcopy
import numpy as np

from ..utils import get_left_side_of_truth_table
from ..boolean_function import BooleanFunction
from ..wiring_diagram import WiringDiagram
    
from .interoperability import BooleanNetworkInteroperabilityMixin
from .control import BooleanNetworkControlMixin
from .dynamics_sync import BooleanNetworkDynamicsSyncMixin
from .dynamics_async import BooleanNetworkDynamicsAsyncMixin
from .robustness_sync import BooleanNetworkRobustnessSyncMixin
from .robustness_sync import get_entropy_of_basin_size_distribution
from .robustness_async import BooleanNetworkRobustnessAsyncMixin
from .modularity import BooleanNetworkModularityMixin


dict_weights = {'non-essential' : np.nan, 'conditional' : 0, 'positive' : 1, 'negative' : -1}

class BooleanNetwork(
        WiringDiagram,
        BooleanNetworkInteroperabilityMixin,
        BooleanNetworkControlMixin,
        BooleanNetworkDynamicsSyncMixin,
        BooleanNetworkDynamicsAsyncMixin,
        BooleanNetworkRobustnessSyncMixin,
        BooleanNetworkRobustnessAsyncMixin,
        BooleanNetworkModularityMixin,
        ):
    """
    Representation of a Boolean network.

    A Boolean network consists of a wiring diagram specifying regulatory
    interactions between nodes and a collection of Boolean update functions
    defining the dynamics at each node.
    
    In a BooleanNetwork, constant nodes are removed during initialization, 
    so all nodes represent dynamic variables.

    Parameters
    ----------
    F : sequence
        Sequence of Boolean update functions or truth tables. Each entry may
        be a ``BooleanFunction`` instance, a truth table, or a Boolean
        expression. The length of ``F`` must match the number of nodes in the
        wiring diagram.
    I : sequence of sequences of int or WiringDiagram
        Wiring diagram specifying the regulators of each node, or an existing
        ``WiringDiagram`` instance.
    variables : sequence of str, optional
        Names of the variables corresponding to each node. Ignored if ``I`` is
        provided as a ``WiringDiagram``.
    simplify_functions : bool, optional
        If True, simplify Boolean update functions after initialization.
        Default is False.

    Attributes
    ----------
    F : list[BooleanFunction]
        Boolean update functions for each node.
    I : list[np.ndarray[int]]
        Wiring diagram specifying the regulators of each node.
    variables : np.ndarray[str]
        Names of the variables corresponding to each node.
    N : int
        Number of dynamic (non-constant) nodes in the network.
    indegrees : np.ndarray[int]
        Indegree of each node.
    outdegrees : np.ndarray[int]
        Outdegree of each node.
    constants : dict[str, dict[str, int | list[str]]]
        Mapping of node indices to constant values.
    weights : list[np.ndarray[float]] or None
        Interaction weights associated with the wiring diagram.
    STG : dict or None
        State transition graph, initialized to None and computed on demand.
    fixation_layers : list of list of str or None
        Sequential record of structural node fixations produced by recursive
        constant propagation. Each inner list contains the variable names that
        became fixed at the same propagation step. The first layer corresponds
        to externally controlled nodes (if any), and subsequent layers represent
        nodes that became constant due to earlier fixations. If no recursive
        simplification was performed, this attribute is None.
    """

    def __init__(
        self,
        F: Sequence[BooleanFunction | list[int] | np.ndarray],
        I: Sequence[Sequence[int]] | WiringDiagram,
        variables: Sequence[str] | None = None,
        simplify_functions: bool = False,
    ):
        """
        Initialize a Boolean network.
    
        A Boolean network is defined by a wiring diagram specifying regulatory
        interactions between nodes and a collection of Boolean update functions
        defining the dynamics at each node. Constant nodes (nodes with no
        regulators) are automatically eliminated during initialization and
        stored in the ``constants`` attribute.
    
        Parameters
        ----------
        F : sequence of BooleanFunction or array-like of int
            Boolean update functions for each node. Each entry must be either a
            ``BooleanFunction`` instance or a truth table encoding the function
            outputs. The length of ``F`` must match the number of nodes in the
            wiring diagram, and each function must have an arity consistent with
            the indegree of the corresponding node.
        I : sequence of sequences of int or WiringDiagram
            Wiring diagram specifying the regulators of each node, or an existing
            ``WiringDiagram`` instance. Regulator indices are assumed to be
            zero-based.
        variables : sequence of str, optional
            Names of the variables corresponding to each node. Ignored if ``I`` is
            provided as a ``WiringDiagram``.
        simplify_functions : bool, optional
            If True, Boolean update functions are simplified after initialization.
            Default is False.
    
        Raises
        ------
        TypeError
            If ``F`` is not a sequence of ``BooleanFunction`` objects or truth
            tables, or if ``I`` is not a valid wiring diagram specification.
        ValueError
            If the length of ``F`` does not match the number of nodes in the wiring
            diagram, or if a Boolean function has an arity inconsistent with the
            wiring diagram.
    
        Notes
        -----
        - Constant nodes are removed from the dynamic network during
          initialization and recorded in the ``constants`` attribute.
        - After initialization, the attribute ``N`` refers to the number of
          remaining dynamic nodes.
        - The state transition graph (``STG``) is initialized to ``None`` and
          computed on demand.
        """

        # ---- Validate inputs -------------------------------------------------
        if isinstance(F, (str, bytes)) or not isinstance(F, Sequence):
            raise TypeError(
                "F must be a sequence of BooleanFunction objects or truth tables"
            )
    
        if isinstance(I, (str, bytes)) or not isinstance(I, (Sequence, WiringDiagram)):
            raise TypeError(
                "I must be a sequence of sequences of int or a WiringDiagram instance"
            )
    
        # ---- Initialize wiring diagram --------------------------------------
        if isinstance(I, WiringDiagram):
            if variables is not None:
                warnings.warn(
                    "Provided variables ignored; using variables from WiringDiagram.",
                    UserWarning,
                )
            super().__init__(I.I, I.variables)
        else:
            super().__init__(I, variables)
    
        if len(F) != self.N:
            raise ValueError("len(F) must match the number of nodes in the wiring diagram")
    
        # ---- Initialize Boolean functions -----------------------------------
        self.F = []
    
        for i, f in enumerate(F):
            if isinstance(f, (list, np.ndarray)):
                bf = BooleanFunction(f, name=self.variables[i], variables=self.variables[self.I[i]])
            elif isinstance(f, BooleanFunction):
                bf = f
                bf.name = self.variables[i]
                bf.variables = self.variables[self.I[i]]
            else:
                raise TypeError(
                    f"Invalid entry in F at index {i}: expected BooleanFunction, "
                    f"truth table, got {type(f)}"
                )
    
            if bf.n != self.indegrees[i]:
                raise ValueError(
                    f"Index {i}: function has {bf.n} inputs but wiring diagram "
                    f"has indegree {self.indegrees[i]}"
                )
    
            self.F.append(bf)
    
        # ---- Constant bookkeeping -------------------------------------------
        # Always initialize (may already exist if called from get_network_with_fixed_source_nodes, etc)
        self.constants = {}
    
        # IMPORTANT: remove constants based on topology, not on dict contents
        if np.any(self.indegrees == 0):
            self.remove_constants()
    
        # ---- State transition graph -----------------------------------------
        self.STG = None
        
        # ---- Fixation layers ------------------------------------------------
        self.fixation_layers = None
    
        # ---- Optional simplification ----------------------------------------
        if simplify_functions:
            self.simplify_functions()
        

                
    def remove_constants(self) -> None:
        """
        Remove structurally constant nodes from the Boolean network.
    
        A node is considered constant if it has no regulators (indegree zero).
        Such nodes are eliminated from the dynamic network by propagating their
        fixed Boolean values to downstream nodes. Eliminated constants and their
        effects are recorded in the ``constants`` attribute.
    
        Notes
        -----
        - The Boolean value of a constant node is taken from its Boolean function.
        - After removal, ``self.N`` refers to the number of remaining dynamic nodes.
        - Nodes that lose all regulators as a result of constant removal are
          assigned a non-essential self-loop to preserve network structure.
        """
        # Identify constant nodes from topology
        # In this model, source nodes (indegree 0) are exactly the semantic constants
        # at initialization time.
        indices_constants = self.get_source_nodes(as_dict=False)
        if len(indices_constants) == 0:
            return
    
        dict_constants = self.get_source_nodes(as_dict=True)
        values_constants = [int(self.F[c][0]) for c in indices_constants]
    
        # Propagate constant values downstream
        for id_constant, value in zip(indices_constants, values_constants):    
            for i in range(self.N):
                if dict_constants[i]:
                    continue
    
                try:
                    index = list(self.I[i]).index(id_constant)
                except ValueError:
                    continue
    
                truth_table = get_left_side_of_truth_table(self.indegrees[i])
                indices_to_keep = np.where(truth_table[:, index] == value)[0]
    
                self.F[i].f = self.F[i].f[indices_to_keep]
    
                if self.weights is not None:
                    self.weights[i] = self.weights[i][self.I[i] != id_constant]
    
                self.I[i] = self.I[i][self.I[i] != id_constant]
                self.indegrees[i] -= 1
                self.F[i].n -= 1
    
    
            self.constants[str(self.variables[id_constant])] = value
    
        # Ensure no remaining node loses all regulators
        for i in range(self.N):
            if dict_constants[i]:
                continue
    
            if self.indegrees[i] == 0:
                self.indegrees[i] = 1
                self.F[i].n = 1
                self.F[i].f = np.array([self.F[i][0], self.F[i][0]], dtype=int)
                self.I[i] = np.array([i], dtype=int)
    
                if self.weights is not None:
                    self.weights[i] = np.array([np.nan], dtype=float)
    
        # Remove constant nodes structurally (using original mask)
        self.F = [self.F[i] for i in range(self.N) if not dict_constants[i]]
    
        adjustment_for_I = np.cumsum([dict_constants[i] for i in range(self.N)])
        self.I = [
            self.I[i] - adjustment_for_I[self.I[i]]
            for i in range(self.N)
            if not dict_constants[i]
        ]
    
        if self.weights is not None:
            self.weights = [self.weights[i] for i in range(self.N) if not dict_constants[i]]
    
        self.variables = np.array(
            [self.variables[i] for i in range(self.N) if not dict_constants[i]],
            dtype=str,
        )
    
        self.indegrees = np.array(
            [self.indegrees[i] for i in range(self.N) if not dict_constants[i]],
            dtype=int,
        )
    
        # Update network size and recompute outdegrees
        self.N -= len(indices_constants)
        self.outdegrees = self.get_outdegrees()


    def __len__(self):
        return self.N
    
    
    def __str__(self):
        return (
            f"BooleanNetwork(N={self.N}, "
            f"indegrees={self.indegrees.tolist()})"
        )
    
    def __getitem__(self, index):
        return self.F[index]
    
    
    def __repr__(self):
        return f"{type(self).__name__}(N={self.N}, average degree={np.round(self.indegrees.mean(),3)})"
    
    
    def clear_cache(self):
        """
        Remove all cached exact and approximate network properties.
    
        This method should be called after modifying network functions,
        regulators, or update rules directly.
        """
        self._properties_estimated.clear()
        self._properties_exact.clear()
    

    def summary(self, compute_all: bool = False, *, as_dict: bool = False):
        """
        Return a concise summary of the Boolean network.
    
        The summary includes basic structural and statistical properties of the
        Boolean network and, optionally, additional properties that may require
        nontrivial computation.
    
        Parameters
        ----------
        compute_all : bool, optional
            If ``True``, additional properties are computed and included in the
            summary. These computations may be expensive. If ``False`` (default),
            only already available properties are included.
        as_dict : bool, optional
            If ``True``, return the summary as a dictionary. If ``False`` (default),
            return a formatted string.
    
        Returns
        -------
        str or dict
            Summary of the Boolean network, either as a formatted string or as
            a dictionary depending on the value of ``as_dict``.
        """
        indices_identity_nodes = self.get_identity_nodes(True)
        indices_identity_nodes = np.array(list(indices_identity_nodes.values()))
        
        N_identity_nodes = int(indices_identity_nodes.sum())
        N_regulated_nodes = self.N - N_identity_nodes
        N_constants = len(self.constants)
        
        regulated_nodes = self.variables[~indices_identity_nodes].tolist()
        identity_nodes = self.variables[indices_identity_nodes].tolist()
        
        
        core_summary = {"Number of nodes": self.N,
                        "Number of regulated nodes": N_regulated_nodes}
        if N_identity_nodes>0:
            core_summary["Number of identity nodes"] = N_identity_nodes
        if N_constants>0:
            core_summary["Number of constants (removed)"] =  N_constants

        core_summary['Average degree'] = np.mean(self.indegrees)
        core_summary['Largest in-degree'] = int(np.max(self.indegrees))
        core_summary['Largest out-degree'] = int(np.max(self.get_outdegrees())) 

        core_summary["Regulated nodes"] = regulated_nodes
        if N_identity_nodes>0:
            core_summary['Identity nodes (inputs)'] = identity_nodes
        if N_constants>0:
            core_summary['Constants'] = self.constants
        
        special_formatting = {
            "Average degree" : ".3f",
            "Largest basin size" : ".3f",
            "Basin size entropy" : ".3f",
            "Coherence" : ".3f",
            "Fragility" : ".3f",
            "Derrida value" : ".3f"
        }
        
        summary = core_summary.copy()
    
        if compute_all:
            if self.N <= 15:
                additional_info = self.get_attractors_and_robustness_synchronous_exact()
                summary["Number of attractors"] = additional_info["NumberOfAttractors"]
                derrida = self.get_derrida_value(exact=True)
            else:
                additional_info = self.get_attractors_and_robustness_synchronous()
                summary["Minimal number of attractors"] = additional_info["NumberOfAttractors"]
                derrida = self.get_derrida_value()
            
            summary['Largest basin size'] = max(additional_info['BasinSizes'])
            entropy = get_entropy_of_basin_size_distribution(additional_info['BasinSizes'])
            summary['Basin size entropy'] = entropy
            summary['Derrida value'] = derrida
            summary['Coherence'] = additional_info['Coherence']
            summary['Fragility'] = additional_info['Fragility']
    
        if as_dict:
            return summary
    
        title = "BooleanNetwork"
            
        lines = [title, "-" * len(title)]
        
        for key, value in summary.items():
            if key not in special_formatting:
                lines.append(f"{key+':':30}{value}")
            else:
                lines.append(f"{key+':':30}{value:{special_formatting[key]}}")
        
        return "\n".join(lines)
    

    def get_types_of_regulation(self) -> list[np.ndarray]:
        """
        Compute and return regulation types (weights) for all nodes in the network.
    
        For each Boolean function, the type of each input regulation is determined
        via ``BooleanFunction.get_type_of_inputs`` and mapped to numerical weights
        using ``dict_weights``. The resulting weights are stored in the
        ``self.weights`` attribute and also returned.
    
        Returns
        -------
        list of np.ndarray
            Regulation weights for each node, aligned with the wiring diagram.
    
        Notes
        -----
        - This method recomputes ``self.weights`` from scratch.
        - Calling this method overwrites any existing values in ``self.weights``.
        """
        self.weights = [
            np.array([dict_weights[el] for el in bf.get_type_of_inputs()], dtype=float)
            for bf in self.F
        ]
        return self.weights


    ## Transform Boolean networks
    def simplify_functions(self) -> None:
        """
        Remove all non-essential regulators from the Boolean network.
    
        For each node, non-essential regulators (identified via ``np.nan`` entries
        in ``self.weights``) are removed from the wiring diagram and the associated
        Boolean function is restricted to its essential inputs. Nodes that would
        otherwise lose all regulators are assigned an identity self-loop to preserve
        network structure.
    
        Notes
        -----
        - This method modifies the network in place.
        - Regulation types (``self.weights``) are recomputed if necessary.
        - Identity self-loops introduced here are structural artifacts and do not
          represent genuine regulatory interactions.
        """
        # Ensure regulation types / weights are available
        self.get_types_of_regulation()
    
        for i in range(self.N):
            regulator_is_non_essential = np.isnan(self.weights[i])
    
            # All regulators are essential
            if not np.any(regulator_is_non_essential):
                continue
    
            non_essential_variables = np.where(regulator_is_non_essential)[0]
            essential_variables = np.where(~regulator_is_non_essential)[0]
    
            # Update outdegrees (each regulator appears at most once in I[i])
            self.outdegrees[non_essential_variables] -= 1
    
            # No essential regulators: introduce identity self-loop
            if len(essential_variables) == 0:
                self.indegrees[i] = 1
                self.F[i].f = np.array([self.F[i][0], self.F[i][0]], dtype=int)
                self.F[i].n = 1
                self.F[i].variables = np.array([self.variables[i]], dtype=str)
                self.I[i] = np.array([i], dtype=int)
                self.weights[i] = np.array([np.nan], dtype=float)
                self.outdegrees[i] += 1  # keep sum(outdegrees) == sum(indegrees)
                continue
    
            # Restrict truth table to essential inputs
            left_side = get_left_side_of_truth_table(self.indegrees[i])
            mask = np.sum(left_side[:, non_essential_variables], axis=1) == 0
    
            self.F[i].f = self.F[i][mask]
            self.F[i].n = len(essential_variables)
            self.F[i].variables = self.F[i].variables[~regulator_is_non_essential]
            self.I[i] = self.I[i][essential_variables]
            self.weights[i] = self.weights[i][essential_variables]
            self.indegrees[i] = len(essential_variables)


    def get_identity_nodes(
        self, 
        as_dict: bool = False
    ) -> dict[int, bool] | np.ndarray:
        """
        Identify identity nodes in the Boolean network.
    
        An identity node is a node with a single self-regulatory edge whose
        Boolean update function is the identity function ``f(x) = x``. Such
        nodes retain their state over time unless externally modified.
    
        Parameters
        ----------
        as_dict : bool, optional
            If True, return a dictionary mapping node indices to booleans.
            If False (default), return an array of indices of identity nodes.
    
        Returns
        -------
        dict[int, bool] or np.ndarray
            If ``as_dict`` is True, a dictionary indicating which nodes are
            identity nodes.
            If ``as_dict`` is False, an array of indices of identity nodes.
        """
        is_identity = np.array(
            [
                self.indegrees[i] == 1
                and self.I[i][0] == i
                and self.F[i][0] == 0
                and self.F[i][1] == 1
                for i in range(self.N)
            ],
            dtype=bool,
        )
    
        if as_dict:
            return dict(enumerate(is_identity.tolist()))
        return np.where(is_identity)[0]
    
    
    def propagate_constants(self) -> "BooleanNetwork":
        """
        Recursively propagate constants through the network.
    
        Any node whose update function becomes constant is converted
        into a structural constant. Removal of such nodes and updating
        of self.constants is handled by __init__.
        """
    
        F = deepcopy(self.F)
        I = deepcopy(self.I)
    
        n = len(F)
    
        # Build reverse dependency graph
        dependents = {i: [] for i in range(n)}
        for node in range(n):
            for inp in I[node]:
                dependents[inp].append(node)
    
        fixed = {}
        queue = deque()
        indices_fixation_layers = []  # <-- new
    
        # ----------------------------------------------------------
        # Initial scan (Layer 0)
        # ----------------------------------------------------------
        initial_layer = []
    
        for node in range(n):
            if F[node].is_constant():
                constant_value = F[node][0]
                fixed[node] = constant_value
                queue.append(node)
                initial_layer.append(node)
    
                F[node].f = np.array([constant_value], dtype=int)
                F[node].n = 0
                I[node] = np.array([], dtype=int)
    
        if initial_layer:
            indices_fixation_layers.append(initial_layer)
    
        # ----------------------------------------------------------
        # Propagation
        # ----------------------------------------------------------
        while queue:
    
            current_layer_size = len(queue)
            next_layer = []
    
            for _ in range(current_layer_size):
    
                fixed_node = queue.popleft()
                value = fixed[fixed_node]
    
                for node in dependents[fixed_node]:
    
                    if node in fixed:
                        continue
    
                    inputs = I[node]
                    positions = np.where(inputs == fixed_node)[0]
                    if len(positions) == 0:
                        continue
    
                    pos = positions[0]
                    k = len(inputs)
                    old_table = F[node].f
                    new_table = []
    
                    for r in range(len(old_table)):
                        bit = (r >> (k - pos - 1)) & 1
                        if bit == value:
                            new_table.append(old_table[r])
    
                    new_table = np.array(new_table, dtype=int)
    
                    F[node].f = new_table
                    F[node].n = k - 1
                    I[node] = np.delete(inputs, pos)
    
                    if len(new_table) == 1 or np.all(new_table == new_table[0]):
                        const_value = int(new_table[0])
    
                        F[node].f = np.array([const_value], dtype=int)
                        F[node].n = 0
                        I[node] = np.array([], dtype=int)
    
                        fixed[node] = const_value
                        queue.append(node)
                        next_layer.append(node)
    
            if next_layer:
                indices_fixation_layers.append(next_layer)
        
        fixation_layers = [
            [str(self.variables[i]) for i in layer]
            for layer in indices_fixation_layers
        ]
        
        # Reinitialize — constructor removes structural constants
        reduced_bn = self.__class__(F, I, self.variables)
        return {'ReducedNetwork' : reduced_bn,
                'FixationLayers' : fixation_layers}