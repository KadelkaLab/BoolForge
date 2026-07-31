#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:54:16 2026

@author: ckadelka
"""

import math
from collections import defaultdict
from collections.abc import Sequence
import numpy as np

from .. import utils

from ..backend._numba import __LOADED_NUMBA__
if __LOADED_NUMBA__:
    from ..backend._numba import List
    from ..backend.dynamics_sync import _transient_lengths_functional_numba
    from ..backend.robustness_sync import _robustness_edge_traversal_numba_stratified
    from ..backend.robustness_sync import _robustness_edge_traversal_numba
    from ..backend.robustness_sync import _derrida_simulation

def get_entropy_of_basin_size_distribution(
    basin_sizes: Sequence[float]
) -> float:
    """
    Compute the Shannon entropy of a basin size distribution.

    The basin sizes are first normalized to form a probability distribution.
    The Shannon entropy is then computed as

    ``H = -sum(p_i * log(p_i))``,

    where ``p_i`` is the proportion of states in basin ``i``.

    Parameters
    ----------
    basin_sizes : Sequence[float]
        Sizes of the basins of attraction (raw counts or normalized weights),
        where each entry gives the number or proportion of initial conditions
        that converge to a given attractor.

    Returns
    -------
    float
        Shannon entropy of the basin size distribution.
    """
    return utils.get_shannon_entropy(basin_sizes)

class BooleanNetworkRobustnessSyncMixin:
    def get_attractors_and_robustness_synchronous_exact(
        self, 
        use_numba: bool = True,
        get_stratified_coherences : bool = False
    ) -> dict:
        """
        Compute attractors and exact robustness measures of a synchronously
        updated Boolean network.

        This method constructs the exact synchronous state transition graph
        (STG) on ``2**N`` states and analyzes it as a functional graph. All
        attractors (cycles), basin sizes, and the attractor reached from each
        state are determined exactly. Based on this decomposition, exact
        coherence and fragility measures are computed for the full network,
        for each basin of attraction, and for each attractor.

        Optionally, coherence can be stratified by the transient length
        (distance from the attractor) of each state, allowing robustness to be
        analyzed as a function of how far states lie from their eventual
        attractor.

        This computation requires memory and time proportional to ``2**N`` and
        is intended for small-to-moderate networks. When Numba is enabled,
        exact and stratified robustness measures remain feasible up to
        moderate values of ``N`` (e.g., ``N ≈ 20`` on typical hardware).

        Parameters
        ----------
        use_numba : bool, optional
            If True (default) and Numba is available, compiled kernels are used
            for robustness and transient-length computations, resulting in
            substantial speedups.
        get_stratified_coherences : bool, optional
            If True, coherence is additionally computed as a function of the
            transient length (distance to the attractor) of each state.
            When Numba is enabled, this option incurs only modest additional
            computational cost. Default is False.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - Attractors : list[list[int]]
                Each attractor represented as a list of decimal states forming
                a cycle.
            - NumberOfAttractors : int
                Total number of attractors.
            - BasinSizes : np.ndarray of float
                Fraction of all states belonging to each attractor basin.
            - AttractorID : np.ndarray of int
                For each of the ``2**N`` states, the index of the attractor it
                eventually reaches.
            - Coherence : float
                Exact global network coherence.
            - Fragility : float
                Exact global network fragility.
            - BasinCoherences : np.ndarray of float
                Exact coherence of each basin of attraction.
            - BasinFragilities : np.ndarray of float
                Exact fragility of each basin of attraction.
            - AttractorCoherences : np.ndarray of float
                Exact coherence of each attractor.
            - AttractorFragilities : np.ndarray of float
                Exact fragility of each attractor.

            If ``get_stratified_coherences`` is True, the dictionary additionally
            contains:

            - StratifiedCoherences : np.ndarray of float
                Coherence values stratified by attractor and transient length.
            - DistanceFromAttractorCount : np.ndarray of int
                Number of state–hypercube-edge incidences contributing to each
                stratified coherence entry.
            - DistanceFromAttractor : np.ndarray of int
                Transient length (distance to attractor) for each state.
        """
    
        # ------------------------------------------------------------------
        # 0) Attractors and basins
        # ------------------------------------------------------------------
        result = self.get_attractors_synchronous_exact(use_numba=use_numba)
    
        attractors = result["Attractors"]
        n_attractors = int(result["NumberOfAttractors"])
    
        basin_sizes = np.asarray(result["BasinSizes"], dtype=np.float64)
        attractor_id = np.asarray(result["AttractorID"], dtype=np.int64)
    
        n_states = 1 << self.N
    
        # ------------------------------------------------------------------
        # Single-attractor shortcut
        # ------------------------------------------------------------------
        if n_attractors == 1 and not get_stratified_coherences:
            basin_coherences = np.ones(1, dtype=np.float64)
            basin_fragilities = np.zeros(1, dtype=np.float64)
            attractor_coherences = np.ones(1, dtype=np.float64)
            attractor_fragilities = np.zeros(1, dtype=np.float64)

            for name, value in zip(
                ['coherence','fragility','basin_coherences','basin_fragilities',
                 'attractor_coherences','attractor_fragilities'],
                [1.0,0.0,basin_coherences,basin_fragilities,
                 attractor_coherences,attractor_fragilities]):
                self._set_property(name, value, context='synchronous', exact=True)

            return {
                "Attractors": attractors,
                "NumberOfAttractors": 1,
                "BasinSizes": basin_sizes,
                "AttractorID": attractor_id,
                "Coherence": 1.0,
                "Fragility": 0.0,
                "BasinCoherences": basin_coherences,
                "BasinFragilities": basin_fragilities,
                "AttractorCoherences": attractor_coherences,
                "AttractorFragilities": attractor_fragilities,
            }
    
        # ------------------------------------------------------------------
        # 1) Attractor membership and lengths
        # ------------------------------------------------------------------
        is_attr_mask = np.zeros(n_states, dtype=np.uint8)
        len_attractors = np.empty(n_attractors, dtype=np.int64)
    
        for i, states in enumerate(attractors):
            states_arr = np.asarray(states, dtype=np.int64)
            len_attractors[i] = states_arr.size
            is_attr_mask[states_arr] = 1
    
        # ------------------------------------------------------------------
        # 2) Mean binary vector per attractor
        # ------------------------------------------------------------------
        mean_states_attractors = np.empty((n_attractors, self.N), dtype=np.float64)
    
        for i, states in enumerate(attractors):
            if len(states) == 1:
                mean_states_attractors[i] = np.asarray(
                    utils.dec2bin(states[0], self.N), dtype=np.float64
                )
            else:
                arr = np.asarray(
                    [utils.dec2bin(s, self.N) for s in states], dtype=np.float64
                )
                mean_states_attractors[i] = arr.mean(axis=0)
    
        # ------------------------------------------------------------------
        # 3) Distance matrix between attractors
        # ------------------------------------------------------------------
        diff = mean_states_attractors[:, None, :] - mean_states_attractors[None, :, :]
        distance_between_attractors = np.sum(np.abs(diff), axis=2)
        distance_between_attractors = np.asarray(
            distance_between_attractors / float(self.N), dtype=np.float64
        )
    
        # ------------------------------------------------------------------
        # 4) Hypercube edge traversal
        # ------------------------------------------------------------------
        if __LOADED_NUMBA__ and use_numba:
            if get_stratified_coherences:
                distances_from_attractor = _transient_lengths_functional_numba(
                    self.STG.astype(np.int64, copy=False),
                    is_attr_mask
                )
                max_distance_from_attractor = int(distances_from_attractor.max())
        
                (
                    basin_coherences,
                    basin_fragilities,
                    attractor_coherences,
                    attractor_fragilities,
                    stratified_coherences,
                    n_states_with_specific_distance_from_attractor,
                ) = _robustness_edge_traversal_numba_stratified(
                    int(self.N),
                    attractor_id,
                    is_attr_mask,
                    distance_between_attractors,
                    distances_from_attractor,
                    max_distance_from_attractor,
                )
                    
                stratified_coherences = np.asarray(stratified_coherences, dtype=np.float64)
                n_states_with_specific_distance_from_attractor = np.asarray(n_states_with_specific_distance_from_attractor, dtype=int)
            else:
                (
                    basin_coherences,
                    basin_fragilities,
                    attractor_coherences,
                    attractor_fragilities,
                ) = _robustness_edge_traversal_numba(
                    int(self.N),
                    attractor_id,
                    is_attr_mask,
                    distance_between_attractors,
                )
    
            basin_coherences = np.asarray(basin_coherences, dtype=np.float64)
            basin_fragilities = np.asarray(basin_fragilities, dtype=np.float64)
            attractor_coherences = np.asarray(attractor_coherences, dtype=np.float64)
            attractor_fragilities = np.asarray(attractor_fragilities, dtype=np.float64)
    
        else:
            basin_coherences = np.zeros(n_attractors, dtype=np.float64)
            basin_fragilities = np.zeros(n_attractors, dtype=np.float64)
            attractor_coherences = np.zeros(n_attractors, dtype=np.float64)
            attractor_fragilities = np.zeros(n_attractors, dtype=np.float64)
            
            if get_stratified_coherences:
                distances_from_attractor = self.get_transient_lengths_exact(result)
                max_distance_from_attractor = max(distances_from_attractor)
                stratified_coherences = np.zeros((n_attractors,max_distance_from_attractor+1), dtype=np.float64)
                n_states_with_specific_distance_from_attractor = np.zeros((n_attractors,max_distance_from_attractor+1), dtype=int)
                
            for xdec in range(n_states):
                for bitpos in range(self.N):
                    if (xdec >> bitpos) & 1:
                        continue
    
                    ydec = xdec | (1 << bitpos)
    
                    idx_x = attractor_id[xdec]
                    idx_y = attractor_id[ydec]
                    
                    if get_stratified_coherences:
                        n_states_with_specific_distance_from_attractor[idx_x,distances_from_attractor[xdec]] += 1
                        n_states_with_specific_distance_from_attractor[idx_y,distances_from_attractor[ydec]] += 1
                        
                    if idx_x == idx_y:
                        basin_coherences[idx_x] += 2.0
                        if is_attr_mask[xdec]:
                            attractor_coherences[idx_x] += 1.0
                        if is_attr_mask[ydec]:
                            attractor_coherences[idx_y] += 1.0
                        if get_stratified_coherences:
                            stratified_coherences[idx_x,distances_from_attractor[xdec]] += 1.0
                            stratified_coherences[idx_y,distances_from_attractor[ydec]] += 1.0
                    else:
                        dxy = float(distance_between_attractors[idx_x, idx_y])
                        basin_fragilities[idx_x] += dxy
                        basin_fragilities[idx_y] += dxy
                        if is_attr_mask[xdec]:
                            attractor_fragilities[idx_x] += dxy
                        if is_attr_mask[ydec]:
                            attractor_fragilities[idx_y] += dxy
    
        # ------------------------------------------------------------------
        # 5) Normalization
        # ------------------------------------------------------------------
        basin_counts = basin_sizes * float(n_states)
    
        if get_stratified_coherences:
            n_states_with_specific_distance_from_attractor //= self.N
    
        for i in range(n_attractors):
            if basin_counts[i] > 0.0:
                basin_coherences[i] /= basin_counts[i] * self.N
                basin_fragilities[i] /= basin_counts[i] * self.N
    
            if len_attractors[i] > 0:
                attractor_coherences[i] /= len_attractors[i] * self.N
                attractor_fragilities[i] /= len_attractors[i] * self.N
                
            if get_stratified_coherences:
                for d in range(max_distance_from_attractor+1):
                    if n_states_with_specific_distance_from_attractor[i,d] > 0.0:
                        stratified_coherences[i,d] /= n_states_with_specific_distance_from_attractor[i,d] * self.N
                    else:
                        stratified_coherences[i,d] = np.nan
                        
        coherence = float(np.dot(basin_sizes, basin_coherences))
        fragility = float(np.dot(basin_sizes, basin_fragilities))
    
        # ------------------------------------------------------------------
        # Final return
        # ------------------------------------------------------------------
        for name, value in zip(
            ['coherence','fragility','basin_coherences','basin_fragilities',
             'attractor_coherences','attractor_fragilities'],
            [coherence,fragility,basin_coherences,basin_fragilities,
             attractor_coherences,attractor_fragilities]):
            self._set_property(name, value, context='synchronous', exact=True)

        return_dict =  {
            "Attractors": attractors,
            "NumberOfAttractors": int(n_attractors),
            "BasinSizes": basin_sizes,
            "AttractorID": attractor_id,
            "Coherence": coherence,
            "Fragility": fragility,
            "BasinCoherences": basin_coherences,
            "BasinFragilities": basin_fragilities,
            "AttractorCoherences": attractor_coherences,
            "AttractorFragilities": attractor_fragilities,
        }
        if get_stratified_coherences:
            return_dict['StratifiedCoherences'] = stratified_coherences
            return_dict['DistanceFromAttractorsCount'] = n_states_with_specific_distance_from_attractor
            return_dict['DistanceFromAttractors'] = distances_from_attractor
        return return_dict


    def get_attractors_and_robustness_synchronous(
        self,
        n_simulations: int = 500,
        return_attractor_coherence: bool = True,
        *,
        rng=None,
    ) -> dict:
        """
        Approximate attractors and robustness measures under synchronous updating.
    
        This method samples the attractor landscape by simulating the network from
        multiple random initial conditions (ICs) and their single-bit perturbations.
        It returns Monte-Carlo approximations of global coherence, fragility, and a
        final Hamming-distance-based measure, along with per-basin approximations.
        Optionally, it additionally estimates attractor-level coherence and fragility
        by perturbing attractor states found during sampling.
    
        Notes
        -----
        - The attractor set returned is a *lower bound* on the true number of
          attractors, because only the sampled portion of state space is explored.
        - For ``N >= 64``, decimal encoding of states may exceed ``np.int64`` and
          this method uses bitstrings (type ``str``) as state identifiers.
    
        Parameters
        ----------
        n_simulations : int, optional
            Number of random initial conditions to sample (default is 500). For each
            IC, the method also simulates one randomly chosen single-bit perturbation.
        return_attractor_coherence : bool, optional
            If True (default), also compute attractor-level coherence and fragility
            by perturbing attractor states found during sampling.
        rng : None or numpy.random.Generator, optional
            Random number generator or seed-like object. Passed to
            ``utils._coerce_rng``.
    
        Returns
        -------
        dict
            Dictionary with keys:
    
            - Attractors : list[list[int]] or list[list[str]]
                List of discovered attractors, each represented as a list of states
                forming a cycle. States are decimals (``int``) for ``N < 64`` and
                bitstrings (``str``) for ``N >= 64``.
            - NumberOfAttractorsLowerBound : int
                Number of distinct attractors discovered (a lower bound on the true
                number of attractors).
            - BasinSizesApproximation : np.ndarray[float]
                Approximate basin size (fraction of sampled trajectories that end in
                each attractor). Sums to ~1 over discovered attractors.
            - CoherenceApproximation : float
                Approximate global coherence: probability that a random IC and its
                single-bit perturbation reach the same attractor.
            - FragilityApproximation : float
                Approximate global fragility: expected normalized difference between
                reached attractors when the IC and perturbation reach different
                attractors. Normalized by ``N``.
            - FinalHammingDistanceApproximation : float
                Approximate final Hamming distance between the two periodic
                trajectories when comparing the IC and its perturbation. This is a
                *distance* in [0, 1], where 0 means identical and 1 means completely
                different.
            - BasinCoherencesApproximation : np.ndarray[float]
                Approximate coherence per basin (same definition as coherence but
                conditioned on having reached that basin).
            - BasinFragilitiesApproximation : np.ndarray[float]
                Approximate fragility per basin (same definition as fragility but
                conditioned on having reached that basin).
            - AttractorCoherences : np.ndarray[float], optional
                If ``return_attractor_coherence`` is True: attractor-level
                coherence (probability that a single-bit perturbation of an attractor
                state returns to the same attractor). For all discovered attractors,
                these values are exact!
            - AttractorFragilities : np.ndarray[float], optional
                If ``return_attractor_coherence`` is True:  attractor-level
                fragility based on differences between the original attractor and the
                attractor reached after perturbation. For all discovered attractors,
                these values are exact!
    
        References
        ----------
        Park, K. H., Costa, F. X., Rocha, L. M., Albert, R., & Rozum, J. C. (2023).
        Models of cell processes are far from the edge of chaos. PRX Life, 1(2), 023009.
    
        Bavisetty, V. S. N., Wheeler, M., & Kadelka, C. (2025).
        Attractors are less stable than their basins: Canalization creates a coherence
        gap in gene regulatory networks. bioRxiv 2025-11.
        """
        rng = utils._coerce_rng(rng)
    
        def lcm(a: int, b: int) -> int:
            return abs(a * b) // math.gcd(a, b)
    
        # ------------------------------------------------------------------
        # Initialization
        # ------------------------------------------------------------------
        dictF = {}
        attractors = []
        ICs_per_attractor_state = []
        basin_sizes = []
        attractor_dict = {}
        attractor_state_dict = []
        distance_from_attractor_state_dict = []
        counter_phase_shifts = []
    
        powers_of_2s = [
            np.asarray([2**i for i in range(NN)][::-1], dtype=np.int64)
            for NN in range(max(self.indegrees) + 1)
        ]
    
        if self.N < 64:
            powers_of_2 = np.asarray([2**i for i in range(self.N)][::-1], dtype=np.int64)
    
        robustness_approximation = 0
        fragility_sum = 0.0
        basin_robustness = defaultdict(float)
        basin_fragility = defaultdict(float)
        final_hamming_distance_approximation = 0.0
    
        mean_states_attractors = []
        states_attractors = []
    
        # ------------------------------------------------------------------
        # Sampling phase
        # ------------------------------------------------------------------
        for _ in range(n_simulations):
            index_attractors = []
            index_within_attr = []
            dist_from_attr = []
    
            for j in range(2):
                if j == 0:
                    x = rng.integers(2, size=self.N, dtype=np.uint8)
                    if self.N < 64:
                        xdec = int(np.dot(x, powers_of_2))
                    else:
                        xdec = "".join(str(int(b)) for b in x)
                    x_old = x.copy()
                else:
                    x = x_old.copy()
                    bit = int(rng.integers(self.N))
                    x[bit] ^= 1
                    if self.N < 64:
                        xdec = int(np.dot(x, powers_of_2))
                    else:
                        xdec = "".join(str(int(b)) for b in x)
    
                queue = [xdec]
    
                try:
                    idx_attr = attractor_dict[xdec]
                except KeyError:
                    while True:
                        try:
                            fxdec = dictF[xdec]
                        except KeyError:
                            fx = np.empty(self.N, dtype=np.uint8)
                            for jj in range(self.N):
                                if self.indegrees[jj] > 0:
                                    fx[jj] = self.F[jj].f[
                                        int(
                                            np.dot(
                                                x[self.I[jj]],
                                                powers_of_2s[self.indegrees[jj]],
                                            )
                                        )
                                    ]
                                else:
                                    fx[jj] = self.F[jj].f[0]
    
                            if self.N < 64:
                                fxdec = int(np.dot(fx, powers_of_2))
                            else:
                                fxdec = "".join(str(int(b)) for b in fx)
    
                            dictF[xdec] = fxdec
    
                        try:
                            idx_attr = attractor_dict[fxdec]
                            idx_state = attractor_state_dict[idx_attr][fxdec]
                            dist_state = distance_from_attractor_state_dict[idx_attr][fxdec]
    
                            attractor_dict.update({q: idx_attr for q in queue})
                            attractor_state_dict[idx_attr].update(
                                {q: idx_state for q in queue}
                            )
                            distance_from_attractor_state_dict[idx_attr].update(
                                {
                                    q: d
                                    for q, d in zip(
                                        queue,
                                        range(len(queue) + dist_state, dist_state, -1),
                                    )
                                }
                            )
                            break
    
                        except KeyError:
                            if fxdec in queue:
                                idx = queue.index(fxdec)
                                idx_attr = len(attractors)
    
                                attractors.append(queue[idx:])
                                basin_sizes.append(1)
                                ICs_per_attractor_state.append(
                                    [0] * len(attractors[-1])
                                )
                                counter_phase_shifts.append(
                                    [0] * len(attractors[-1])
                                )
    
                                attractor_dict.update({q: idx_attr for q in queue})
                                attractor_state_dict.append(
                                    {
                                        q: (0 if q in queue[:idx] else queue[idx:].index(q))
                                        for q in queue
                                    }
                                )
                                distance_from_attractor_state_dict.append(
                                    {
                                        q: (idx - queue.index(q))
                                        if q in queue[:idx]
                                        else 0
                                        for q in queue
                                    }
                                )
    
                                if len(attractors[-1]) == 1:
                                    fp = (
                                        np.asarray(
                                            utils.dec2bin(queue[idx], self.N),
                                            dtype=np.float64,
                                        )
                                        if self.N < 64
                                        else np.asarray(list(queue[idx]), dtype=np.float64)
                                    )
                                    states_attractors.append(fp.reshape(1, self.N))
                                    mean_states_attractors.append(fp)
                                else:
                                    lc = (
                                        np.asarray(
                                            [
                                                utils.dec2bin(s, self.N)
                                                for s in queue[idx:]
                                            ],
                                            dtype=np.float64,
                                        )
                                        if self.N < 64
                                        else np.asarray(
                                            [list(s) for s in queue[idx:]],
                                            dtype=np.float64,
                                        )
                                    )
                                    states_attractors.append(lc)
                                    mean_states_attractors.append(lc.mean(axis=0))
                                break
                            else:
                                x = fx.copy()
                                queue.append(fxdec)
                                xdec = fxdec
    
                index_attractors.append(idx_attr)
                index_within_attr.append(attractor_state_dict[idx_attr][xdec])
                dist_from_attr.append(
                    distance_from_attractor_state_dict[idx_attr][xdec]
                )
    
                basin_sizes[idx_attr] += 1
                ICs_per_attractor_state[idx_attr][
                    attractor_state_dict[idx_attr][xdec]
                ] += 1
    
            if index_attractors[0] == index_attractors[1]:
                robustness_approximation += 1
                basin_robustness[index_attractors[0]] += 1
                ps = max(index_within_attr) - min(index_within_attr)
                counter_phase_shifts[index_attractors[0]][ps] += 1
            else:
                d = np.sum(
                    np.abs(
                        mean_states_attractors[index_attractors[0]]
                        - mean_states_attractors[index_attractors[1]]
                    )
                )
                fragility_sum += d
                basin_fragility[index_attractors[0]] += d
    
                L = lcm(
                    len(attractors[index_attractors[0]]),
                    len(attractors[index_attractors[1]]),
                )
    
                s0 = states_attractors[index_attractors[0]]
                s1 = states_attractors[index_attractors[1]]
    
                p0 = np.tile(s0, (L // len(s0) + 1, 1))[
                    index_within_attr[0] : index_within_attr[0] + L
                ]
                p1 = np.tile(s1, (L // len(s1) + 1, 1))[
                    index_within_attr[1] : index_within_attr[1] + L
                ]
    
                final_hamming_distance_approximation += np.mean(p0 == p1)
    
        # ------------------------------------------------------------------
        # Aggregation
        # ------------------------------------------------------------------
        lower_bound_number_of_attractors = len(attractors)
    
        approximate_basin_sizes = (
            np.asarray(basin_sizes, dtype=np.float64)
            / (2.0 * float(n_simulations))
        )
    
        approximate_coherence = robustness_approximation / float(n_simulations)
        approximate_fragility = fragility_sum / float(n_simulations) / float(self.N)
    
        approximate_basin_coherence = np.asarray(
            [
                2.0 * basin_robustness[i] / basin_sizes[i]
                for i in range(lower_bound_number_of_attractors)
            ],
            dtype=np.float64,
        )
    
        approximate_basin_fragility = np.asarray(
            [
                2.0 * basin_fragility[i] / basin_sizes[i] / float(self.N)
                for i in range(lower_bound_number_of_attractors)
            ],
            dtype=np.float64,
        )
    
        final_hamming_distance_approximation /= float(n_simulations)
    
        results = [
            attractors,
            lower_bound_number_of_attractors,
            approximate_basin_sizes,
            approximate_coherence,
            approximate_fragility,
            final_hamming_distance_approximation,
            approximate_basin_coherence,
            approximate_basin_fragility,
        ]
        
        self._set_property('coherence', approximate_coherence,
                           context='synchronous', exact=False)
        self._set_property('fragility', approximate_fragility,
                           context='synchronous', exact=False)
        self._set_property('basin_coherences', approximate_basin_coherence,
                           context='synchronous', exact=False)
        self._set_property('basin_fragilities', approximate_basin_fragility,
                           context='synchronous', exact=False)
    
        return_dict = dict(
            zip(
                [
                    "Attractors",
                    "NumberOfAttractorsLowerBound",
                    "BasinSizesApproximation",
                    "CoherenceApproximation",
                    "FragilityApproximation",
                    "FinalHammingDistanceApproximation",
                    "BasinCoherencesApproximation",
                    "BasinFragilitiesApproximation",
                ],
                results,
            )
        )
    
        if not return_attractor_coherence:
            return return_dict
    
        # ------------------------------------------------------------------
        # Attractor-level coherence / fragility
        # ------------------------------------------------------------------        
        attractor_coherences = np.zeros(lower_bound_number_of_attractors, dtype=np.float64)
        attractor_fragilities = np.zeros(lower_bound_number_of_attractors, dtype=np.float64)
    
        attractors_original = attractors[:]
    
        for idx0, attractor in enumerate(attractors_original):
            for state in attractor:
                for i in range(self.N):
                    x = (
                        np.asarray(utils.dec2bin(state, self.N), dtype=np.uint8)
                        if self.N < 64
                        else np.asarray(list(state), dtype=np.uint8)
                    )
                    x[i] ^= 1
    
                    if self.N < 64:
                        xdec = int(np.dot(x, powers_of_2))
                    else:
                        xdec = "".join(str(int(b)) for b in x)
    
                    try:
                        idx1 = attractor_dict[xdec]
                    except KeyError:
                        # --- safe forward-walk without touching basin counts
                        queue = [xdec]
                        x_local = x.copy()
                        while True:
                            try:
                                fxdec = dictF[xdec]
                            except KeyError:
                                fx = np.empty(self.N, dtype=np.uint8)
                                for jj in range(self.N):
                                    if self.indegrees[jj] > 0:
                                        fx[jj] = self.F[jj].f[
                                            int(
                                                np.dot(
                                                    x_local[self.I[jj]],
                                                    powers_of_2s[self.indegrees[jj]],
                                                )
                                            )
                                        ]
                                    else:
                                        fx[jj] = self.F[jj].f[0]
    
                                if self.N < 64:
                                    fxdec = int(np.dot(fx, powers_of_2))
                                else:
                                    fxdec = "".join(str(int(b)) for b in fx)
    
                                dictF[xdec] = fxdec
    
                            if fxdec in attractor_dict:
                                idx1 = attractor_dict[fxdec]
                                break
    
                            if fxdec in queue:
                                idx = queue.index(fxdec)
                                idx1 = len(attractors)
                                attractors.append(queue[queue.index(fxdec):])
                                attractor_dict.update(
                                    {q: idx1 for q in queue}
                                )
                                
                                if len(attractors[-1]) == 1:
                                    fp = (
                                        np.asarray(
                                            utils.dec2bin(queue[idx], self.N),
                                            dtype=np.float64,
                                        )
                                        if self.N < 64
                                        else np.asarray(list(queue[idx]), dtype=np.float64)
                                    )
                                    states_attractors.append(fp.reshape(1, self.N))
                                    mean_states_attractors.append(fp)
                                else:
                                    lc = (
                                        np.asarray(
                                            [
                                                utils.dec2bin(s, self.N)
                                                for s in queue[idx:]
                                            ],
                                            dtype=np.float64,
                                        )
                                        if self.N < 64
                                        else np.asarray(
                                            [list(s) for s in queue[idx:]],
                                            dtype=np.float64,
                                        )
                                    )
                                    states_attractors.append(lc)
                                    mean_states_attractors.append(lc.mean(axis=0))
                                break
    
                            queue.append(fxdec)
                            xdec = fxdec
                            x_local = fx.copy()
    
                    if idx0 == idx1:
                        attractor_coherences[idx0] += 1.0
                    else:
                        attractor_fragilities[idx0] += np.sum(
                            np.abs(
                                mean_states_attractors[idx0]
                                - mean_states_attractors[idx1]
                            )
                        )
    
        attractor_coherences /= (
            float(self.N)
            * np.asarray(list(map(len, attractors_original)), dtype=np.float64)
        )
    
        attractor_fragilities /= (
            float(self.N) ** 2
            * np.asarray(list(map(len, attractors_original)), dtype=np.float64)
        )
    
        results[0] = attractors_original

        self._set_property('attractor_coherences', attractor_coherences,
                           context='synchronous', exact=False)
        self._set_property('attractor_fragilities', attractor_fragilities,
                           context='synchronous', exact=False)
    
        return_dict["AttractorCoherences"] = attractor_coherences
        return_dict["AttractorFragilities"] = attractor_fragilities
        return return_dict
    
    
    def get_derrida_value(
        self,
        n_simulations: int = 1000,
        exact: bool | None = None,
        use_numba: bool = True,
        *,
        rng=None,
    ) -> float:
        """
        Compute the Derrida value of a Boolean network.
    
        The Derrida value measures the average Hamming distance between the
        one-step synchronous updates of two states that differ by a single-bit
        perturbation. It quantifies the short-term sensitivity of the network
        dynamics to small perturbations.
    
        If ``exact`` is True, the Derrida value is computed exactly as the mean
        (unnormalized) average sensitivity of the Boolean update functions.
        Otherwise, it is approximated via Monte Carlo simulation.
    
        Parameters
        ----------
        n_simulations : int, optional
            Number of Monte Carlo simulations to perform (default is 1000).
            Ignored if ``exact`` is True.
        exact : bool, optional
            If True, compute the exact Derrida value. 
            If False,approximate using Monte Carlo simulation.
            If None (default), compute exactly for small networks with N <= 15,
            and approximate for larger networks.
        use_numba : bool, optional
            If True (default) and Numba is available, use a compiled kernel for
            Monte Carlo simulation.
        rng : None or np.random.Generator, optional
            Random number generator, passed through ``utils._coerce_rng``.
    
        Returns
        -------
        float
            The Derrida value, defined as the average Hamming distance after
            one synchronous update following a single-bit perturbation.
    
        References
        ----------
        Derrida, B., & Pomeau, Y. (1986).
        Random networks of automata: a simple annealed approximation.
        *Europhysics Letters*, 1(2), 45.
        """
        if exact is None:
            exact = True if self.N <= 15 else False

        if exact:
            # ------------------------------------------------------------------
            # Exact computation
            # ------------------------------------------------------------------
            derrida_value = np.mean(
                [
                    bf.get_average_sensitivity(
                        exact=True, normalized=False
                    )
                    for bf in self.F
                ]
            )
        else:
            # ------------------------------------------------------------------
            # Monte Carlo approximation
            # ------------------------------------------------------------------
            rng = utils._coerce_rng(rng)
        
            if __LOADED_NUMBA__ and use_numba:
                # Prepare Numba-friendly inputs
                F_array_list = List(
                    [np.asarray(bf.f, dtype=np.uint8) for bf in self.F]
                )
                I_array_list = List(
                    [np.asarray(regs, dtype=np.int64) for regs in self.I]
                )
        
                seed = int(rng.integers(0, 2**31 - 1))
        
                derrida_value = float(
                    _derrida_simulation(
                        F_array_list,
                        I_array_list,
                        int(self.N),
                        int(n_simulations),
                        seed,
                    )
                )
                
            else:
                # ------------------------------------------------------------------
                # Pure Python fallback
                # ------------------------------------------------------------------
                total_dist: float = 0.0
            
                for _ in range(int(n_simulations)):
                    x = rng.integers(0, 2, size=self.N, dtype=np.uint8)
                    y = x.copy()
            
                    idx = int(rng.integers(0, self.N))
                    y[idx] ^= np.uint8(1)
            
                    fx = np.asarray(
                        self._update_network_synchronously_unchecked(x),
                        dtype=np.uint8,
                    )
                    fy = np.asarray(
                        self._update_network_synchronously_unchecked(y),
                        dtype=np.uint8,
                    )
            
                    total_dist += float(np.sum(fx != fy))
            
                derrida_value = float(total_dist / float(n_simulations))
            
        self._set_property('derrida_value', derrida_value,
                           exact=exact)
        return derrida_value