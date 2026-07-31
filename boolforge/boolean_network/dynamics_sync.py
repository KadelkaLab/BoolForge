#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:52:32 2026

@author: ckadelka
"""

from collections.abc import Sequence
from collections import deque
import numpy as np

from .. import utils

from ..backend._numba import __LOADED_NUMBA__
if __LOADED_NUMBA__:
    from ..backend._numba import List
    from ..backend.dynamics_sync import _update_network_synchronously_numba
    from ..backend.dynamics_sync import _compute_synchronous_stg_numba
    from ..backend.dynamics_sync import _compute_synchronous_stg_numba_low_memory
    from ..backend.dynamics_sync import _attractors_functional_graph
    from ..backend.dynamics_sync import _transient_lengths_functional_numba

class BooleanNetworkDynamicsSyncMixin:
    def update_single_node(
        self,
        index: int,
        states_regulators: Sequence[int],
    ) -> int:
        """
        Update the state of a single node.
    
        The new state is obtained by applying the Boolean update function to the
        states of its regulators.
    
        Parameters
        ----------
        index : int
            Index of the node to update.
        states_regulators : sequence of int
            Binary states of the node's regulators.
    
        Returns
        -------
        int
            Updated state of the node (0 or 1).
        """
        return self.F[index].f[utils.bin2dec(states_regulators)].item()


    def __call__(self, state):
        """
        Apply one synchronous update step to the Boolean network.
    
        The next state is obtained by evaluating each node's Boolean update
        function on the current values of its regulators.
    
        Parameters
        ----------
        state : sequence of int
            Current network state as a binary vector of length ``N``, ordered
            according to ``self.variables``.
    
        Returns
        -------
        np.ndarray
            The updated network state after one synchronous update.
        
        Notes
        -----
        This method is equivalent to calling ``update_network_synchronously``.
        """
        return self.update_network_synchronously(state)


    def update_network_synchronously(
        self,
        state: Sequence[int],
    ) -> np.ndarray:
        """
        Perform a synchronous update of the Boolean network.
    
        Parameters
        ----------
        state : sequence of int
            Binary state vector of length ``N``.
    
        Returns
        -------
        np.ndarray
            Updated state vector.
        """
        state = np.asarray(state, dtype=int)
    
        if state.shape[0] != self.N:
            raise ValueError(
                f"State vector must have length {self.N}, got {state.shape[0]}."
            )
    
        if not np.all((state == 0) | (state == 1)):
            raise ValueError("State vector must be binary (0 or 1).")
    
        return self._update_network_synchronously_unchecked(state)
    

    def _update_network_synchronously_unchecked(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """Internal fast path. Assumes validated binary state."""
        next_state = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            next_state[i] = self.F[i].f[utils.bin2dec(state[self.I[i]])]
        return next_state
    
    
    def update_network_SDDS(
        self,
        state: Sequence[int],
        P: np.ndarray,
        *,
        rng=None,
    ) -> np.ndarray:
        """
        Perform a stochastic discrete dynamical system (SDDS) update of the network.
    
        This update scheme follows the SDDS formalism: for each node, the
        deterministic Boolean update is first computed. If the update would
        increase the node's state, the change occurs with the node-specific
        activation probability. If the update would decrease the node's state,
        the change occurs with the node-specific degradation probability.
        Otherwise, the node's state remains unchanged.
    
        Parameters
        ----------
        state : sequence of int
            Current network state (binary vector of length ``N``).
        P : np.ndarray
            Array of shape ``(N, 2)``, where ``P[i, 0]`` is the activation
            probability and ``P[i, 1]`` is the degradation probability for node ``i``.
        rng : optional
            Random number generator or seed, passed to ``utils._coerce_rng``.
    
        Returns
        -------
        np.ndarray
            Updated network state after one stochastic SDDS update.
    
        Notes
        -----
        This implementation follows the SDDS framework introduced in:
    
        Murrugarra, D., Veliz-Cuba, A., Aguilar, B., Arat, S., & Laubenbacher, R.
        (2012). *Modeling stochasticity and variability in gene regulatory networks*.
        EURASIP Journal on Bioinformatics and Systems Biology, 2012(1), 5.
    
        The method assumes that ``state`` is a valid binary vector and that
        ``P`` has the correct shape; no additional validation is performed
        for performance reasons.
        """
        rng = utils._coerce_rng(rng)
        state = np.asarray(state, dtype=int)
    
        Fx = state.copy()
        for i in range(self.N):
            nextstep = self.update_single_node(
                index=i,
                states_regulators=state[self.I[i]],
            )
    
            if nextstep > state[i]:
                if rng.random() < P[i, 0]:
                    Fx[i] = nextstep
            elif nextstep < state[i]:
                if rng.random() < P[i, 1]:
                    Fx[i] = nextstep
    
        return Fx
    
    
    def get_attractors_synchronous(
        self,
        n_simulations: int = 500,
        initial_sample_points: Sequence[int | Sequence[int]] | None = None,
        n_steps_timeout: int = 1000,
        initial_sample_points_are_vectors: bool = False,
        use_numba: bool = True,
        *,
        rng=None,
    ) -> dict:
        """
        Approximate synchronous attractors of a Boolean network via sampling.
    
        This method estimates the synchronous attractors (fixed points and cycles)
        of a Boolean network by simulating synchronous updates from a collection
        of initial states. For each simulation, the network is updated until an
        attractor is reached or a maximum number of update steps is exceeded.
    
        The method is sampling-based and does *not* guarantee that all attractors
        are found. Basin sizes are lower-bound estimates based on the sampled
        initial conditions.
    
        If Numba is available and ``use_numba=True``, synchronous updates are
        accelerated using a compiled kernel.
    
        Parameters
        ----------
        n_simulations : int, optional
            Number of random initial conditions to sample (default is 500). 
            Ignored if ``initial_sample_points`` is provided.
        initial_sample_points : sequence of int or sequence of sequence of int, optional
            Initial states to use. If provided, its length determines the number
            of simulations. Interpretation depends on
            ``initial_sample_points_are_vectors``.
        n_steps_timeout : int, optional
            Maximum number of synchronous update steps per simulation before
            declaring a timeout (default is 1000).
        initial_sample_points_are_vectors : bool, optional
            If True, ``initial_sample_points`` are interpreted as binary vectors;
            otherwise (default) they are interpreted as decimal-encoded states.
        use_numba : bool, optional
            If True (default) and Numba is available, use a Numba-accelerated
            synchronous update kernel.
        rng : optional
            Random number generator or seed, passed to ``utils._coerce_rng``.
    
        Returns
        -------
        dict
            Dictionary with the following entries:
    
            - Attractors : list of list of int  
              Attractors found, each represented as a list of decimal states
              (cycles are given in cyclic order).
            - NumberOfAttractorsLowerBound : int
                Number of distinct attractors discovered (a lower bound on the true
                number of attractors).
            - BasinSizesApproximation : np.ndarray[float]
                Approximate basin size (fraction of sampled trajectories that end in
                each attractor). Sums to 1 in the absence of time-outs.
            - AttractorID : dict  
              Mapping from visited states (decimal) to attractor index.
            - InitialSamplePoints : list of int  
              Decimal initial states used for sampling.
            - STG : dict  
              Sampled synchronous state transition graph
              (state → successor state).
            - NumberOfTimeouts : int  
              Number of simulations that did not converge within
              ``n_steps_timeout``.
    
        Notes
        -----
        - This method is intended for large networks with long transient dynamics, 
          where exhaustive analysis is infeasible.
        - Basin sizes are *sampling-based estimates* and should not be interpreted
          as exact proportions of the state space.
        - There is no guarantee that all attractors are found. 
        """
        rng = utils._coerce_rng(rng)
    
        # --- Bookkeeping ---
        dictF: dict[int, int] = {}        # memorized synchronous transitions
        attractors: list[list[int]] = []  # attractor cycles
        basin_sizes: list[int] = []       # basin counts
        attr_dict: dict[int, int] = {}    # state -> attractor index
        STG: dict[int, int] = {}           # sampled synchronous STG
        n_timeout = 0
        sampled_points: list[int] = []
    
        initial_sample_points_empty = initial_sample_points is None
        if not initial_sample_points_empty:
            n_simulations = len(initial_sample_points)
    
        # --- Decide update backend ---
        use_numba = __LOADED_NUMBA__ and use_numba
    
        if use_numba:
            F_array_list = List([np.array(bf.f, dtype=np.uint8) for bf in self.F])
            I_array_list = List([np.array(regs, dtype=np.int64) for regs in self.I])
    
        # --- Main simulation loop ---
        for sim_idx in range(n_simulations):
            # Initialize state
            if initial_sample_points_empty:
                x = rng.integers(2, size=self.N, dtype=np.uint8)
                xdec = utils.bin2dec(x)
                sampled_points.append(xdec)
            else:
                if initial_sample_points_are_vectors:
                    x = np.asarray(initial_sample_points[sim_idx], dtype=np.uint8)
                    if x.shape[0] != self.N:
                        raise ValueError(
                            f"Initial state must have length {self.N}, got {x.shape[0]}."
                        )
                    xdec = utils.bin2dec(x)
                else:
                    xdec = int(initial_sample_points[sim_idx])
                    x = np.array(utils.dec2bin(xdec, self.N), dtype=np.uint8)
    
            visited = {xdec: 0}
            trajectory = [xdec]
            count = 0
    
            # --- Iterate until attractor or timeout ---
            while count < n_steps_timeout:
                if xdec in dictF:
                    fxdec = dictF[xdec]
                else:
                    if use_numba:
                        fx = _update_network_synchronously_numba(
                            x, F_array_list, I_array_list
                        )
                    else:
                        fx = self._update_network_synchronously_unchecked(x)
    
                    fxdec = utils.bin2dec(fx)
                    dictF[xdec] = fxdec
                    x = fx
    
                # record sampled STG edge (first visit only)
                if count == 0:
                    STG[xdec] = fxdec
    
                # already assigned to known attractor
                if fxdec in attr_dict:
                    idx_attr = attr_dict[fxdec]
                    basin_sizes[idx_attr] += 1
                    for s in trajectory:
                        attr_dict[s] = idx_attr
                    break
    
                # new attractor detected
                if fxdec in visited:
                    cycle_start = visited[fxdec]
                    attractor_states = trajectory[cycle_start:]
                    attractors.append(attractor_states)
                    basin_sizes.append(1)
                    idx_attr = len(attractors) - 1
                    for s in attractor_states:
                        attr_dict[s] = idx_attr
                    break
    
                # continue traversal
                visited[fxdec] = len(trajectory)
                trajectory.append(fxdec)
                xdec = fxdec
                count += 1
    
                if count == n_steps_timeout:
                    n_timeout += 1
                    break
        
        #normalize basin sizes to return proportions
        basin_sizes = np.array(basin_sizes) / max(1,n_simulations)
        
        self._set_property('attractors', attractors,
                           context='synchronous', exact=False)
        self._set_property('number_of_attractors', len(attractors),
                           context='synchronous', exact=False)
        self._set_property('basin_sizes', basin_sizes,
                           context='synchronous', exact=False)
        
        return {
            "Attractors": attractors,
            "NumberOfAttractorsLowerBound": len(attractors),
            "BasinSizesApproximation": basin_sizes,
            "AttractorID": attr_dict,
            "InitialSamplePoints": (
                sampled_points if initial_sample_points_empty else list(initial_sample_points)
            ),
            "STG": STG,
            "NumberOfTimeouts": n_timeout,
        }


    
    def compute_synchronous_state_transition_graph(
        self,
        use_numba: bool = True,
    ) -> None:
        """
        Compute the exact synchronous state transition graph (STG).
    
        The STG is stored in ``self.STG`` as a one-dimensional NumPy array of length
        ``2**N``, where ``self.STG[x]`` is the decimal representation of the successor
        state reached from state ``x`` under synchronous updating.
    
        This computation is exact and requires memory proportional to ``2**N``.
        It is therefore intended for small-to-moderate networks only.
    
        Parameters
        ----------
        use_numba : bool, optional
            If True (default) and Numba is available, use a compiled kernel to
            accelerate computation.
        """
        # Optional: avoid recomputation
        if self.STG is not None:
            return
    
        if __LOADED_NUMBA__ and use_numba:
            # Preprocess data into Numba-friendly types
            F_list = [np.array(bf.f, dtype=np.uint8) for bf in self.F]
            I_list = [np.array(regs, dtype=np.int64) for regs in self.I]
    
            if self.N <= 22:
                self.STG = _compute_synchronous_stg_numba(F_list, I_list, self.N)
            else:
                self.STG = _compute_synchronous_stg_numba_low_memory(
                    F_list, I_list, self.N
                )
            return
    
        # -------- Pure NumPy implementation --------
    
        # 1. Enumerate all states (binary)
        states = utils.get_left_side_of_truth_table(self.N)
    
        # 2. Allocate next-state matrix
        next_states = np.zeros_like(states, dtype=np.uint8)
    
        # Binary-to-decimal weights
        powers_of_two = (1 << np.arange(self.N))[::-1]
    
        # 3. Compute next state for each node
        for j, bf in enumerate(self.F):
            regulators = self.I[j]
    
            if len(regulators) == 0:
                # Constant node
                next_states[:, j] = bf.f[0]
                continue
    
            subspace = states[:, regulators]
            idx = np.dot(subspace, powers_of_two[-len(regulators):])
            next_states[:, j] = bf.f[idx]
    
        # 4. Convert next states to decimal
        self.STG = np.dot(next_states, powers_of_two).astype(np.int64)


    def get_attractors_synchronous_exact(
        self,
        use_numba: bool = True,
    ) -> dict:
        """
        Compute all attractors and their exact basin sizes under synchronous updating.
    
        This method computes the exact synchronous state transition graph (STG) and
        analyzes it as a functional graph on ``2**N`` states. All attractors (cycles),
        their basin sizes, and the attractor reached from each state are determined
        exactly.
    
        This computation requires memory and time proportional to ``2**N`` and is
        intended for small-to-moderate networks only.
    
        Parameters
        ----------
        use_numba : bool, optional
            If True (default) and Numba is available, use a compiled kernel for
            attractor detection.
    
        Returns
        -------
        dict
            Dictionary with keys:
    
            - Attractors : list[list[int]]
                Each attractor represented as a list of decimal states forming a cycle.
            - NumberOfAttractors : int
                Total number of attractors.
            - BasinSizes : np.ndarray[float]
                Fraction of all states belonging to each attractor basin.
            - AttractorID : np.ndarray[int]
                For each of the ``2**N`` states, the index of the attractor it reaches.
            - STG : np.ndarray[int]
                The synchronous state transition graph.
        """
        if self.STG is None:
            self.compute_synchronous_state_transition_graph(use_numba=use_numba)
            
        if ('attractors', 'synchronous') in self._properties_exact:
            return {
                "Attractors":
                    self._properties_exact[('attractors', 'synchronous')],
                "NumberOfAttractors":
                    self._properties_exact[('number_of_attractors', 'synchronous')],
                "BasinSizes":
                    self._properties_exact[('basin_sizes', 'synchronous')],
                "AttractorID":
                    self._properties_exact[('attractor_id', 'synchronous')],
                "STG": self.STG,
            }        
    
        attractors = []
    
        if __LOADED_NUMBA__ and use_numba:
            attractor_id, basin_sizes, cycle_rep, cycle_len, n_attr = (
                _attractors_functional_graph(self.STG)
            )
    
            for k in range(int(n_attr)):
                rep = int(cycle_rep[k])
                L = int(cycle_len[k])
                cyc = [rep]
                x = rep
                for _ in range(L - 1):
                    x = int(self.STG[x])
                    cyc.append(x)
                attractors.append(cyc)
    
        else:
            attractor_id = -np.ones(2**self.N, dtype=np.int32)
            basin_sizes = []
            n_attr = 0
    
            for xdec in range(2**self.N):
                if attractor_id[xdec] != -1:
                    continue
    
                cur = xdec
                queue = [cur]
    
                while True:
                    fxdec = int(self.STG[cur])
    
                    if attractor_id[fxdec] != -1:
                        idx_attr = attractor_id[fxdec]
                        basin_sizes[idx_attr] += len(queue)
                        for q in queue:
                            attractor_id[q] = idx_attr
                        break
    
                    if fxdec in queue:
                        idx = queue.index(fxdec)
                        cycle = queue[idx:]
                        attractors.append(cycle)
                        basin_sizes.append(len(queue))
                        for q in queue:
                            attractor_id[q] = n_attr
                        n_attr += 1
                        break
    
                    queue.append(fxdec)
                    cur = fxdec
    
        basin_sizes = np.array(basin_sizes, dtype=np.float64) / (2**self.N)

        self._set_property('attractors', attractors,
                           context='synchronous', exact=True)
        self._set_property('number_of_attractors', len(attractors),
                           context='synchronous', exact=True)
        self._set_property('basin_sizes', basin_sizes,
                           context='synchronous', exact=True)
        self._set_property('attractor_id', attractor_id,
                           context='synchronous', exact=True)    
        
        return {
            "Attractors": attractors,
            "NumberOfAttractors": len(attractors),
            "BasinSizes": basin_sizes,
            "AttractorID": attractor_id,
            "STG": self.STG,
        }
    
    
    def get_minimal_trap_spaces_synchronous_exact(
        self, 
        use_numba = True
    ) -> np.ndarray:
        """
        Compute the minimal trap space associated with each synchronous attractor.
        
        For each attractor, nodes that take the same value in every state
        are marked by their fixed value (0 or 1), whereas nodes that vary
        across the attractor are marked as -1.
        
        Parameters
        ----------
        use_numba : bool, optional
            If True (default) and Numba is available, use a compiled kernel for
            attractor detection.
        
        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_attractors, N)`` whose rows represent
            minimal trap spaces.
        """
        attr_info = self.get_attractors_synchronous_exact(use_numba = use_numba)
        attractors = attr_info['Attractors']
        return np.array(
            [utils.get_minimal_trap_space(states, self.N) 
             for states in attractors]
        )
    
    
    def get_transient_lengths_exact(
        self,
        use_numba : bool = True
    ) -> np.ndarray:
        """
        Compute the exact transient length of every state under synchronous updating.
        
        The transient length of a state is the number of update steps required
        to reach an attractor. States belonging to an attractor have transient
        length 0.
        
        This method first computes the exact synchronous attractors and then
        determines the shortest distance from every state to an attractor in
        the synchronous state transition graph (STG).
        
        Parameters
        ----------
        use_numba : bool, optional
            If True (default) and Numba is available, use a compiled kernel
            for transient-length computation.
        
        Returns
        -------
        numpy.ndarray
            Array of length ``2**N`` where entry ``i`` gives the transient
            length of state ``i``.
            
        Notes
        -----
        - This method requires the exact synchronous state transition graph
          and is therefore intended for small-to-moderate networks only.
        - Attractor states, including all states belonging to cyclic
          attractors, are assigned transient length 0.
        """
        attr_info = self.get_attractors_synchronous_exact(use_numba = use_numba)
        attractors = attr_info["Attractors"]   # list of cycles
        stg = self.STG                         # full mapping: successor(s)
        
        if __LOADED_NUMBA__ and use_numba:
            is_attr_mask = np.full(2**self.N, 0, dtype=np.uint8)
        
            for i, states in enumerate(attractors):
                states_arr = np.asarray(states, dtype=np.int64)
                is_attr_mask[states_arr] = 1
            return _transient_lengths_functional_numba(
                self.STG.astype(np.int64, copy=False),
                is_attr_mask
            )
        
        
        # Normalize STG to an integer array/list succ where succ[u] = v
        if isinstance(stg, np.ndarray):
            succ = stg.astype(int, copy=False)
            n = int(succ.shape[0])
        else:
            succ = list(stg)
            n = len(succ)
    
        # Build reverse adjacency list rev[v] = all u such that u -> v
        rev = [[] for _ in range(n)]
        for u in range(n):
            v = int(succ[u])
            if v < 0 or v >= n:
                raise ValueError(f"Invalid successor: {u} -> {v}")
            rev[v].append(u)
    
        # Initialize distances: all cycle states have transient length 0
        dist = np.full(n, -1, dtype=np.int64)
        bfs = deque()
    
        for cycle in attractors:
            for s in cycle:
                if dist[s] == -1:
                    dist[s] = 0
                    bfs.append(s)
    
        # Multi-source BFS outward from cycle states
        while bfs:
            v = bfs.popleft()
            for u in rev[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    bfs.append(u)
    
        # If STG is complete, every state must get a distance
        if any(d < 0 for d in dist):
            raise RuntimeError("Some states did not receive a transient length. Is STG complete?")
        
        return np.array(dist,dtype=int)
    
    

    