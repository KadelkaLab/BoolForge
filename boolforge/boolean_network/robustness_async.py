#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:54:16 2026

@author: ckadelka
"""

import numpy as np

from .. import utils

from ..backend._numba import __LOADED_NUMBA__, _numba_required
if __LOADED_NUMBA__:
    from ..backend.robustness_async import _compute_neighbor_attraction_probability


class BooleanNetworkRobustnessAsyncMixin():
    def get_terminal_sccs_and_robustness_asynchronous_exact(self, compute_absorption_times=False) -> dict:
        """
        Compute terminal SCCs (attractors) and exact robustness measures of an 
        asynchronously updated Boolean network.

        This method constructs the exact asynchronous state transition graph
        on ``2**N`` states and interprets the asynchronous dynamics as a finite 
        Markov chain. All attractors (terminal SCCs), basin sizes, and the 
        terminal SCCs reached  from each state are determined exactly. Based on
        this decomposition, exact coherence measures are computed for the full
        network, for each basin of attraction, and for each attractor.
        Entropy-based measures of basin structure and local divergence measures
        are also computed. Expected absorption times are computed only if
        requested via ``compute_absorption_times``, as they require solving an
        additional linear system.

        This computation requires memory and time proportional to ``2**N`` and
        is intended for small-to-moderate networks (e.g., ``N ≈ 18`` on typical 
        hardware).

        Parameters
        ----------
        compute_absorption_times : bool, optional
            If True, also compute exact expected absorption times (see
            ``ExpectedAbsorptionTimesAny`` and ``ExpectedAbsorptionTimesSpecific``
            below). This requires an additional linear solve on top of the rest
            of this method's computation and is skipped by default. If False,
            both fields are returned as ``None``. Default is False.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - TerminalSCCs : list[list[int]]
                Each terminal SCC represented as a recurrent communicating class.
            - NumberOfTerminalSCCs : int
                Total number of terminal SCCs.
            - LengthOfTerminalSCCs : np.ndarray of int
                Length of each terminal SCC.
            - TrapSpaceDimensions : np.ndarray of int
                Dimension of the minimal space containing each terminal SCC.
            - BasinSizes : np.ndarray of float
                Probability of reaching a terminal SCC from a random state.
            - AbsorptionProbabilities : np.ndarray of float
                For each of the ``2**N`` states, the probability that a specific
                terminal SCC is reached
            - Coherence : float
                Exact global network coherence.
            - BasinCoherences : np.ndarray of float
                Exact coherence of each basin of attraction.
            - TerminalSCCCoherencesUniform : np.ndarray of float
                Exact coherence of each terminal SCC (when weighting each 
                attractor state equally).
            - TerminalSCCCoherencesStationary : np.ndarray of float
                Exact coherence of each terminal SCC (when weighting each 
                attractor state based on the stationary distribution).
            - BasinEntropy : float
                Shannon entropy of the distribution of basin sizes across
                terminal SCCs.
            - StateEntropies : np.ndarray of float
                Array of shape ``(2**N,)``. Shannon entropy of each state's
                absorption-probability distribution over terminal SCCs.
            - MeanStateEntropyPerBasin : np.ndarray of float
                Mean state entropy among states with positive absorption
                probability into each terminal SCC.
            - ExpectedAbsorptionTimesAny : np.ndarray of float or None
                Array of shape ``(2**N,)``. Expected number of asynchronous
                update steps for each state to be absorbed into any terminal
                SCC. Terminal states have value 0. ``None`` unless
                ``compute_absorption_times`` is True.
            - ExpectedAbsorptionTimesSpecific : np.ndarray of float or None
                Array of shape ``(2**N, NumberOfTerminalSCCs)``. Expected number
                of steps for each state to be absorbed, conditioned on
                absorption into each specific terminal SCC. Entries are ``NaN``
                where absorption into that SCC from that state has zero
                probability. ``None`` unless ``compute_absorption_times`` is
                True.
            - NetworkDivergence : float
                Mean local divergence over all network states, where local
                divergence at a state is the mean Jensen-Shannon divergence
                between its absorption-probability distribution and those of
                its one-bit neighbors.
            - LocalDivergences : np.ndarray of float
                Array of shape ``(2**N,)``. Local divergence at each network
                state.
        """
        if not __LOADED_NUMBA__:
            _numba_required("Asynchronous exact robustness computation")
        
        terminal_sccs = self.get_terminal_sccs_asynchronous_exact()
        n_terminal_sccs = int(len(terminal_sccs))
        if compute_absorption_times:
            absorption_times_any, absorption_times_specific = self.get_expected_absorption_times_exact()
        else:
            absorption_times_any = None
            absorption_times_specific = None
        if n_terminal_sccs==1:
            return  {
                "TerminalSCCs": terminal_sccs,
                "NumberOfTerminalSCCs": n_terminal_sccs,
                "LengthOfTerminalSCCs": np.array(len(terminal_sccs[0])),
                "TrapSpaceDimensions": np.array(
                    utils.get_number_of_varying_nodes(terminal_sccs[0])
                ),
                "BasinSizes": np.array([1.],dtype=np.float32),
                "AbsorptionProbabilities": np.ones(
                    ((1 << self.N), 1), dtype=np.float32
                ),
                "Coherence": 1.,
                "BasinCoherences": np.ones(1),
                "TerminalSCCCoherencesUniform": np.ones(1),
                "TerminalSCCCoherencesStationary": np.ones(1),
                "BasinEntropy": 0.,
                "StateEntropies": np.zeros((1 << self.N),dtype=np.float32),
                "MeanStateEntropyPerBasin": np.zeros(1,dtype=np.float32),
                "ExpectedAbsorptionTimesAny": absorption_times_any,
                "ExpectedAbsorptionTimesSpecific": absorption_times_specific,
                "NetworkDivergence": 0.,
                "LocalDivergences": np.zeros((1 << self.N),dtype=np.float32)
            }
            
        absorption_probs = self.get_absorption_probabilities_exact()
        
        relative_basin_sizes = self.get_basin_sizes_asynchronous_exact()
        basin_sizes = relative_basin_sizes * float(1<<self.N)
        length_terminal_sccs = np.array(list(map(len,terminal_sccs)))
        dim_trap_spaces = np.array(list(map(utils.get_number_of_varying_nodes,
                                            terminal_sccs)))
        neighbor_attraction_probability = _compute_neighbor_attraction_probability(
            self.N,
            absorption_probs
        )
        basin_coherences = (
            absorption_probs * neighbor_attraction_probability
        ).sum(axis=0) / basin_sizes
        coherence = np.dot(basin_coherences, relative_basin_sizes)
        terminal_scc_coherences_uniform = np.array([
            neighbor_attraction_probability[a, i].mean()
            for i, a in enumerate(terminal_sccs)
        ])
        
        terminal_scc_coherences_stationary = np.zeros(len(terminal_sccs))
        for i,(length,terminal_scc) in enumerate(zip(length_terminal_sccs,terminal_sccs)):
            if length==1:
                terminal_scc_coherences_stationary[i] = \
                neighbor_attraction_probability[terminal_scc, i].mean()
                
            else:
                STG = self.get_asynchronous_transition_matrix()
                k = len(terminal_scc)
                
                P_A = STG[terminal_scc][:,terminal_scc].toarray()
                A = P_A.T - np.eye(k)
                A[-1,:] = 1.0
                b = np.zeros(k)
                b[-1] = 1.0
                
                psi = np.linalg.solve(A,b)
                terminal_scc_coherences_stationary[i] = np.dot(
                    psi,
                    neighbor_attraction_probability[terminal_scc, i]
                )
        entropies_dict = self.compute_entropy()
        network_divergence, local_divergences = self.get_divergence()
        return  {
            "TerminalSCCs": terminal_sccs,
            "NumberOfTerminalSCCs": n_terminal_sccs,
            "LengthOfTerminalSCCs": length_terminal_sccs,
            "TrapSpaceDimensions": dim_trap_spaces,
            "BasinSizes": relative_basin_sizes,
            "AbsorptionProbabilities": absorption_probs,
            "Coherence": coherence.item(),
            "BasinCoherences": basin_coherences,
            "TerminalSCCCoherencesUniform": terminal_scc_coherences_uniform,
            "TerminalSCCCoherencesStationary": terminal_scc_coherences_stationary,
            "BasinEntropy": entropies_dict["basin_entropy"],
            "StateEntropies": entropies_dict["state_entropies"],
            "MeanStateEntropyPerBasin":entropies_dict["basin_mean_state_entropies"],
            "ExpectedAbsorptionTimesAny": absorption_times_any,
            "ExpectedAbsorptionTimesSpecific": absorption_times_specific,
            "NetworkDivergence": network_divergence,
            "LocalDivergences": local_divergences
        }
