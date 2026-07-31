#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:52:32 2026

@author: ckadelka
"""


from collections.abc import Sequence
import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import gmres
from scipy.sparse.csgraph import connected_components

from .. import utils

from ..backend._numba import _numba_required, __LOADED_NUMBA__
if __LOADED_NUMBA__:
    from ..backend.dynamics_async import _build_async_transition_coo



class BooleanNetworkDynamicsAsyncMixin:
    def get_asynchronous_transition_matrix(self) -> csr_matrix:
        """
        Construct and return the exact asynchronous state transition graph.
        
        The asynchronous state transition graph (STG) is represented as a
        row-stochastic sparse matrix whose rows correspond to network states
        and whose nonzero entries encode one-step asynchronous transitions.
        
        The matrix is cached after the first computation.
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse transition matrix of shape ``(2**N, 2**N)``.
        """
        if ('STG', 'asynchronous') in self._properties_exact:
            return self._properties_exact[('STG', 'asynchronous')]
        if not __LOADED_NUMBA__:
            _numba_required("Asynchronous exact dynamics computation")
        
        F_list = [np.asarray(f.f, dtype=np.uint8) for f in self.F]
        I_list = [np.asarray(regs, dtype=np.int32) for regs in self.I]

        # fast direct CSR construction
        rows, cols, data = _build_async_transition_coo(F_list,I_list,self.N)
        STG = csr_matrix(
            (data, (rows, cols)),
            shape=((1 << self.N), (1 << self.N)),
            dtype=np.float32
        )

        self._set_property('STG', STG, context='asynchronous', exact=True)
        
        return STG
                
    
    def get_terminal_sccs_asynchronous_exact(self) -> list[list[int]]:
        """
        Compute the terminal strongly connected components of the asynchronous STG.
        
        A terminal SCC is a strongly connected component with no outgoing
        transitions to states outside the component. Terminal SCCs correspond
        to asynchronous attractors, including both steady states and cyclic
        attractors.
        
        Results are cached after the first computation.
        
        Returns
        -------
        list of list of int
            Terminal SCCs represented as lists of decimal-encoded states.
        """
        if ('terminal_sccs', 'asynchronous') in self._properties_exact:
            return self._properties_exact[('terminal_sccs', 'asynchronous')]
        
        STG = self.get_asynchronous_transition_matrix()        
        n_components, labels = connected_components(STG, 
                                                    directed=True, 
                                                    connection='strong'
                                                    )
        terminal_sccs = []
        for c in range(n_components):
            states = np.where(labels == c)[0]
            for s in states:
                start, end = STG.indptr[s], STG.indptr[s + 1]
                if np.any(labels[STG.indices[start:end]] != c):
                    break
            else:
                terminal_sccs.append([int(s) for s in states])
        
        self._set_property('terminal_sccs', terminal_sccs,
                           context='asynchronous', exact=True)
        self._set_property('number_of_terminal_sccs', len(terminal_sccs),
                           context='asynchronous', exact=True)
        return terminal_sccs
    
    def get_minimal_trap_spaces_asynchronous_exact(self) -> np.ndarray:
        """
        Compute the minimal trap space associated with each terminal SCC.
        
        For each terminal SCC, nodes that take the same value in every state
        are marked by their fixed value (0 or 1), whereas nodes that vary
        across the SCC are marked as -1.
        
        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_terminal_sccs, N)`` whose rows represent
            minimal trap spaces.
        """
        terminal_sccs = self.get_terminal_sccs_asynchronous_exact()
        return np.array(
            [utils.get_minimal_trap_space(states, self.N) 
             for states in terminal_sccs]
        )
    
    def get_number_frozen_nodes_asynchronous_exact(self) -> int:
        """
        Compute the number of frozen nodes in the asynchronous dynamics.
    
        A node is considered frozen if it takes the same value in every
        attractor state. For asynchronous dynamics, attractor states are
        defined as the states belonging to terminal strongly connected
        components (terminal SCCs) of the asynchronous state transition
        graph.
    
        Returns
        -------
        int
            Number of nodes whose value is identical across all attractor
            states.
        """
        terminal_sccs = self.get_terminal_sccs_asynchronous_exact()
        return self.N - utils.get_number_of_varying_nodes(
            utils.flatten(terminal_sccs)
        )
        

    def get_absorption_probabilities_exact(self) -> np.ndarray:
        """
        Compute exact absorption probabilities for the asynchronous dynamics.
        
        For every network state and every terminal SCC, this method computes
        the probability that an asynchronous trajectory starting from that
        state is eventually absorbed into the corresponding terminal SCC.
        
        Probabilities are obtained by solving the standard absorbing Markov
        chain equations and are cached after the first computation.
        
        Returns
        -------
        numpy.ndarray
            Array of shape ``(2**N, n_terminal_sccs)`` where entry
            ``[x, a]`` is the probability that state ``x`` eventually
            reaches terminal SCC ``a``.
        """
        if ('absorption_probabilities', 'asynchronous') in self._properties_exact:
            return self._properties_exact[('absorption_probabilities', 'asynchronous')]
        
        STG = self.get_asynchronous_transition_matrix()
        terminal_sccs = self.get_terminal_sccs_asynchronous_exact()
        
        transient_states = np.setdiff1d(np.arange(1 << self.N),
                                        np.concatenate(terminal_sccs))
        
        n_terminal_sccs = len(terminal_sccs)
        n_transients = len(transient_states)
        absorption_probs = np.zeros(((1 << self.N), n_terminal_sccs),dtype=np.float32)

        if n_transients > 0:
            transient_mask = np.zeros(1 << self.N, dtype=bool)
            transient_mask[transient_states] = True
    
            transient_index = -np.ones(1 << self.N, dtype=np.int32)
            transient_index[transient_states] = np.arange(n_transients)
    
            # build Q and R without giant reorder/slicing
            rows_Q = []
            cols_Q = []
            vals_Q = []
            R = np.zeros((n_transients, n_terminal_sccs), dtype=np.float32)
    
            terminal_scc_lookup = {}
            for a, states in enumerate(terminal_sccs):
                for s in states:
                    terminal_scc_lookup[s] = a
    
            for s in transient_states:
                s_local = transient_index[s]
                start = STG.indptr[s]
                end = STG.indptr[s + 1]
                succs = STG.indices[start:end]
                probs = STG.data[start:end]
    
                for y, p in zip(succs, probs):
                    if transient_mask[y]:
                        rows_Q.append(s_local)
                        cols_Q.append(transient_index[y])
                        vals_Q.append(p)
                    else:
                        a = terminal_scc_lookup[y]
                        R[s_local, a] += p
    
            Q = csr_matrix(
                (vals_Q, (rows_Q, cols_Q)),
                shape=(n_transients,n_transients),
                dtype=np.float32
            )
            A = identity(n_transients, dtype=np.float32, format='csr') - Q

            #compute absorption probabilities
            for a in range(n_terminal_sccs):
                b = R[:, a]
                x,_ = gmres(A,b,atol=1e-10)
                x = np.clip(x.astype(np.float32), 0.0, 1.0)
                absorption_probs[transient_states,a] = x
            
            #correct for potential tiny numerical errors
            row_sums = absorption_probs[transient_states].sum(axis=1, keepdims=True)
            absorption_probs[transient_states] /= row_sums
        
        for a, states in enumerate(terminal_sccs):
            absorption_probs[states, a] = 1.0   
            
        self._set_property('absorption_probabilities', absorption_probs,
                           context='asynchronous', exact=True)
        
        return absorption_probs
    
    
    def get_steady_states_asynchronous(
        self,
        n_simulations: int = 500,
        initial_states: Sequence[int] | None = None,
        search_depth: int = 50,
        debug: bool = False,
        *,
        rng=None,
    ) -> dict:
        """
        Approximate steady states of a Boolean network under asynchronous updates.
    
        This method performs a Monte Carlo–style exploration of the asynchronous
        state space by simulating asynchronous updates from a collection of initial
        states. Each simulation proceeds until a steady state is reached or until
        a maximum search depth is exceeded.
    
        Unlike ``get_steady_states_asynchronous_exact``, this method does *not*
        exhaustively explore the full state space and does not guarantee that all
        steady states will be found. It is intended for large networks where exact
        enumeration is infeasible.
    
        Parameters
        ----------
        n_simulations : int, optional
            Number of asynchronous simulations to perform (default is 500).
        initial_states : sequence of int or None, optional
            Initial states to use for the simulations, given as decimal
            representations of network states. If None (default), ``n_simulations``
            random initial states are generated.
        search_depth : int, optional
            Maximum number of asynchronous update steps per simulation before
            giving up on convergence (default is 50).
        debug : bool, optional
            If True, print detailed debugging information during simulation.
        rng : optional
            Random number generator or seed, passed to ``utils._coerce_rng``.
    
        Returns
        -------
        dict
            Dictionary with the following entries:
    
            - SteadyStates : list of int  
              Decimal representations of steady states encountered.
            - NumberOfSteadyStatesLowerBound : int  
              Number of unique steady states found.
            - BasinSizesApproximation : list of int  
              Proportion of simulations that converged to each steady state.
            - STGAsynchronous : dict  
              Partial cache of asynchronous transitions encountered during
              simulation. Keys are ``(state, node_index)`` and values are
              successor states (all in decimal form).
            - InitialSamplePoints : list of int  
              Decimal initial states used in the simulations (either provided
              explicitly or generated randomly).
    
        Notes
        -----
        - This method detects only *steady states* (fixed points). If the
          asynchronous dynamics contain limit cycles, simulations may fail
          to converge within ``search_depth``.
        - The returned asynchronous transition graph is generally incomplete
          and should be interpreted as a cache of explored transitions rather
          than the full STG.
        - There is no guarantee that all steady states will be identified.
        """
        rng = utils._coerce_rng(rng)
    
        sampled_states: list[int] = []
        STG_asynchronous: dict[tuple[int, int], int] = {}
    
        steady_states: list[int] = []
        basin_sizes: list[int] = []
        steady_state_dict: dict[int, int] = {}
    
        for iteration in range(n_simulations):
            # Initialize state
            if initial_states is None:
                x = rng.integers(2, size=self.N)
                xdec = utils.bin2dec(x)
                sampled_states.append(xdec)
            else:
                xdec = initial_states[iteration]
                x = utils.dec2bin(xdec, self.N)

            for step in range(search_depth):
                found_new_state = False
    
                # Check if state is already known to be steady
                if xdec in steady_state_dict:
                    basin_sizes[steady_state_dict[xdec]] += 1
                    break
    
                update_order = rng.permutation(self.N)
                for i in map(int, update_order):
                    try:
                        fxdec = STG_asynchronous[(xdec, i)]
                    except KeyError:
                        fx_i = self.update_single_node(i, x[self.I[i]])
                        if fx_i > x[i]:
                            fxdec = xdec + 2 ** (self.N - 1 - i)
                            x[i] = 1
                            found_new_state = True
                        elif fx_i < x[i]:
                            fxdec = xdec - 2 ** (self.N - 1 - i)
                            x[i] = 0
                            found_new_state = True
                        else:
                            fxdec = xdec
                        STG_asynchronous[(xdec, i)] = fxdec
    
                    if fxdec != xdec:
                        xdec = fxdec
                        found_new_state = True
                        break

                if not found_new_state:
                    # New steady state found
                    if xdec in steady_state_dict:
                        basin_sizes[steady_state_dict[xdec]] += 1
                    else:
                        steady_state_dict[xdec] = len(steady_states)
                        steady_states.append(xdec)
                        basin_sizes.append(1)
                    break

        if sum(basin_sizes) < n_simulations:
            print(
                f"Warning: only {sum(basin_sizes)} of the {n_simulations} simulations "
                "reached a steady state. Consider increasing search_depth. "
                "The network may also contain asynchronous limit cycles."
            )
        
        if sum(basin_sizes)>0:
            sum_basin_sizes  = sum(basin_sizes)
            basin_sizes = np.array([size/sum_basin_sizes for size in basin_sizes])
        
        return {
            "SteadyStates": steady_states,
            "NumberOfSteadyStatesLowerBound": len(steady_states),
            "BasinSizesApproximation": basin_sizes,
            "STGAsynchronous": STG_asynchronous,
            "InitialSamplePoints": (
                initial_states if initial_states is not None else sampled_states
            ),
        }
    
    def get_steady_states_asynchronous_given_one_initial_condition(
        self,
        initial_condition: int | Sequence[int] = 0,
        n_simulations: int = 500,
        stochastic_weights: Sequence[float] | None = None,
        search_depth: int = 50,
        debug: bool = False,
        *,
        rng=None,
    ) -> dict:
        """
        Approximate steady states reachable from a single initial condition under
        asynchronous updates.
    
        This method performs multiple asynchronous simulations starting from the
        same initial condition. In each simulation, nodes are updated one at a time
        according to either a uniform random order or node-specific stochastic
        update propensities. The simulation proceeds until a steady state is reached
        or a maximum number of update steps is exceeded.
    
        The method is sampling-based and does *not* guarantee that all reachable
        steady states are found. It is intended for exploratory analysis and for
        networks where exhaustive asynchronous analysis is infeasible.
    
        Parameters
        ----------
        initial_condition : int or sequence of int, optional
            Initial network state. If an integer is provided, it is interpreted as
            the decimal encoding of a Boolean state. If a sequence is provided, it
            must be a binary vector of length ``N``. Default is 0.
        n_simulations : int, optional
            Number of asynchronous simulation runs (default is 500).
        stochastic_weights : sequence of float or None, optional
            Relative update propensities for each node. If provided, must have
            length ``N`` and be strictly positive. The weights are normalized
            internally. If None (default), nodes are updated uniformly at random.
        search_depth : int, optional
            Maximum number of asynchronous update steps per simulation.
        debug : bool, optional
            If True, print detailed debugging information during simulation.
        rng : optional
            Random number generator or seed, passed to ``utils._coerce_rng``.
    
        Returns
        -------
        dict
            Dictionary with the following entries:
    
            - SteadyStates : list of int  
              Decimal representations of steady states reached.
            - NumberOfSteadyStatesLowerBound : int  
              Number of unique steady states found.
            - BasinSizesApproximation : list of int  
              Proportion of simulations converging to each steady state.
            - TransientTimes : list of list of int  
              For each steady state, a list of transient lengths (number of update
              steps before convergence).
            - STGAsynchronous : dict  
              Partial cache of asynchronous transitions encountered during
              simulation. Keys are ``(state, node_index)`` and values are successor
              states (all in decimal form).
            - UpdateQueues : list of list of int  
              For each simulation, the sequence of visited states (in decimal form).
    
        Notes
        -----
        - Only steady states (fixed points) are detected. If the asynchronous
          dynamics contain limit cycles, simulations may fail to converge within
          ``search_depth``.
        - The returned asynchronous transition graph is incomplete and represents
          only transitions encountered during sampling.
        - There is no guarantee that all steady states will be identified.
        """
        rng = utils._coerce_rng(rng)
    
        # --- Initialize initial condition ---
        if isinstance(initial_condition, int):
            x0 = utils.dec2bin(initial_condition, self.N)
            x0dec = initial_condition
        else:
            x0 = np.asarray(initial_condition, dtype=int)
            if x0.shape[0] != self.N:
                raise ValueError(
                    f"Initial condition must have length {self.N}, got {x0.shape[0]}."
                )
            x0dec = utils.bin2dec(x0)
    
        # --- Handle stochastic weights ---
        if stochastic_weights is not None:
            stochastic_weights = np.asarray(stochastic_weights, dtype=float)
            if stochastic_weights.shape[0] != self.N:
                raise ValueError("stochastic_weights must have length N.")
            if np.any(stochastic_weights <= 0):
                raise ValueError("stochastic_weights must be strictly positive.")
            stochastic_weights = stochastic_weights / stochastic_weights.sum()
    
        # --- Bookkeeping ---
        STG_async: dict[tuple[int, int], int] = {}
        steady_states: list[int] = []
        basin_sizes: list[int] = []
        transient_times: list[list[int]] = []
        steady_state_dict: dict[int, int] = {}
        queues: list[list[int]] = []
    
        # --- Simulations ---
        for iteration in range(n_simulations):
            x = x0.copy()
            xdec = x0dec
            queue = [xdec]
    
            for step in range(search_depth):
                found_new_state = False
    
                # If already known steady state, stop
                if xdec in steady_state_dict:
                    idx = steady_state_dict[xdec]
                    basin_sizes[idx] += 1
                    transient_times[idx].append(step)
                    queues.append(queue)
                    break
    
                # Choose update order
                if stochastic_weights is None:
                    update_order = rng.permutation(self.N)
                else:
                    update_order = rng.choice(
                        self.N, size=self.N, replace=False, p=stochastic_weights
                    )
    
                for i in map(int, update_order):
                    try:
                        fxdec = STG_async[(xdec, i)]
                    except KeyError:
                        fx_i = self.update_single_node(i, x[self.I[i]])
                        if fx_i > x[i]:
                            fxdec = xdec + 2 ** (self.N - 1 - i)
                            x[i] = 1
                        elif fx_i < x[i]:
                            fxdec = xdec - 2 ** (self.N - 1 - i)
                            x[i] = 0
                        else:
                            fxdec = xdec
                        STG_async[(xdec, i)] = fxdec
    
                    if fxdec != xdec:
                        xdec = fxdec
                        queue.append(xdec)
                        found_new_state = True
                        break
    
                if debug:
                    print(iteration, step, i, found_new_state, xdec, x)
    
                if not found_new_state:
                    # New steady state reached
                    if xdec in steady_state_dict:
                        idx = steady_state_dict[xdec]
                        basin_sizes[idx] += 1
                        transient_times[idx].append(step)
                    else:
                        steady_state_dict[xdec] = len(steady_states)
                        steady_states.append(xdec)
                        basin_sizes.append(1)
                        transient_times.append([step])
                    queues.append(queue)
                    break
    
            if debug:
                print()
    
        if sum(basin_sizes) < n_simulations:
            print(
                f"Warning: only {sum(basin_sizes)} of the {n_simulations} simulations "
                "reached a steady state. Consider increasing search_depth. "
                "The network may contain asynchronous limit cycles."
            )
            
        basin_sizes = np.array(basin_sizes)/n_simulations
    
        return {
            "SteadyStates": steady_states,
            "NumberOfSteadyStatesLowerBound": len(steady_states),
            "BasinSizesApproximation": basin_sizes,
            "TransientTimes": transient_times,
            "STGAsynchronous": STG_async,
            "UpdateQueues": queues,
        }