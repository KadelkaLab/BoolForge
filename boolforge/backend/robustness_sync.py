#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

# load optional but desirable package
from ._numba import njit, __LOADED_NUMBA__
from .dynamics_sync import _update_network_synchronously_numba

if __LOADED_NUMBA__:
    @njit(fastmath=True)  # safe: operations are integer-only
    def _hamming_distance(a, b):
        """
        Compute the Hamming distance between two binary vectors.
    
        Parameters
        ----------
        a : np.ndarray
            One-dimensional array of dtype ``uint8``.
        b : np.ndarray
            One-dimensional array of dtype ``uint8`` with the same shape as ``a``.
    
        Returns
        -------
        int
            Number of positions at which ``a`` and ``b`` differ.
        """
        dist = 0
        for i in range(a.size):
            dist += a[i] != b[i]
        return dist
    
    
    @njit(fastmath=True)  # safe: operations are integer-only
    def _derrida_simulation(
        F_array_list,
        I_array_list,
        N,
        n_simulations,
        seed
    ):
        """
        Perform a Monte Carlo simulation to estimate the Derrida value.
    
        This function estimates the Derrida value by repeatedly sampling a random
        initial state, flipping a single randomly chosen bit, synchronously
        updating both states, and computing the Hamming distance between the
        resulting successor states.
    
        Parameters
        ----------
        F_array_list : list[np.ndarray]
            List of Boolean update tables for each node.
        I_array_list : list[np.ndarray]
            List of regulator index arrays for each node.
        N : int
            Number of variables (nodes) in the network.
        n_simulations : int
            Number of Monte Carlo simulations to perform.
        seed : int
            Seed for the Numba-compatible random number generator.
    
        Returns
        -------
        float
            Estimated Derrida value, i.e., the expected Hamming distance after one
            synchronous update following a random single-bit perturbation.
        """

        # Numba RNG: seed once
        np.random.seed(seed)
        total_dist = 0.0
    
        X = np.empty(N, dtype=np.uint8)
        Y = np.empty(N, dtype=np.uint8)
    
        for _ in range(n_simulations):
            # Random initial state
            for i in range(N):
                X[i] = np.random.randint(0, 2)
            Y[:] = X
    
            # Flip one random bit
            idx = np.random.randint(0, N)
            Y[idx] = 1 - Y[idx]
    
            # Synchronous updates
            FX = _update_network_synchronously_numba(X, F_array_list, I_array_list)
            FY = _update_network_synchronously_numba(Y, F_array_list, I_array_list)
    
            total_dist += _hamming_distance(FX, FY)
    
        return total_dist / n_simulations


    @njit(cache=True)
    def _robustness_edge_traversal_numba(
        N,
        attractor_idx,
        is_attr_mask,
        dist_attr
    ):
        """
        Traverse hypercube edges to compute basin and attractor robustness measures.
    
        This function iterates over all undirected edges of the Boolean hypercube
        exactly once and accumulates coherence and fragility contributions for
        basins of attraction and for attractor states.
    
        Parameters
        ----------
        N : int
            Number of variables (dimension of the Boolean hypercube).
        attractor_idx : np.ndarray
            Integer array of shape ``(2**N,)`` mapping each state to its attractor
            index in ``[0, n_attr - 1]``.
        is_attr_mask : np.ndarray
            Boolean or uint8 array of shape ``(2**N,)`` indicating whether a state
            lies on an attractor.
        dist_attr : np.ndarray
            Two-dimensional array of shape ``(n_attr, n_attr)`` giving pairwise
            distances between attractors.
    
        Returns
        -------
        basin_coh : np.ndarray
            Array of length ``n_attr`` containing basin coherence values.
        basin_frag : np.ndarray
            Array of length ``n_attr`` containing basin fragility values.
        attr_coh : np.ndarray
            Array of length ``n_attr`` containing attractor coherence values.
        attr_frag : np.ndarray
            Array of length ``n_attr`` containing attractor fragility values.
        """

        n_states = attractor_idx.shape[0]
        n_attr = dist_attr.shape[0]
    
        basin_coh = np.zeros(n_attr, dtype=np.float64)
        basin_frag = np.zeros(n_attr, dtype=np.float64)
        attr_coh = np.zeros(n_attr, dtype=np.float64)
        attr_frag = np.zeros(n_attr, dtype=np.float64)
    
        # Iterate each undirected hypercube edge exactly once:
        # For x, flip only bits that are 0 -> y = x | (1<<bit), which guarantees y > x.
        for xdec in range(n_states):
            idx_x = attractor_idx[xdec]
            # (Should never be -1 if attractor_idx is filled for all states)
            for bit in range(N):
                if (xdec >> bit) & 1:
                    continue
                ydec = xdec | (1 << bit)
                idx_y = attractor_idx[ydec]
    
                if idx_x == idx_y:
                    # same basin: count both directions (like your +2)
                    basin_coh[idx_x] += 2.0
                    if is_attr_mask[xdec]:
                        attr_coh[idx_x] += 1.0
                    if is_attr_mask[ydec]:
                        attr_coh[idx_y] += 1.0
                else:
                    dxy = dist_attr[idx_x, idx_y]
                    basin_frag[idx_x] += dxy
                    basin_frag[idx_y] += dxy
                    if is_attr_mask[xdec]:
                        attr_frag[idx_x] += dxy
                    if is_attr_mask[ydec]:
                        attr_frag[idx_y] += dxy
    
        return basin_coh, basin_frag, attr_coh, attr_frag


    @njit(cache=True)
    def _robustness_edge_traversal_numba_stratified(
        N,
        attractor_idx,
        is_attr_mask,
        dist_attr,
        dist_state,
        max_dist
    ):
        n_states = attractor_idx.shape[0]
        n_attr = dist_attr.shape[0]
    
        basin_coh = np.zeros(n_attr, dtype=np.float64)
        basin_frag = np.zeros(n_attr, dtype=np.float64)
        attr_coh = np.zeros(n_attr, dtype=np.float64)
        attr_frag = np.zeros(n_attr, dtype=np.float64)
    
        strat_coh = np.zeros((n_attr, max_dist + 1), dtype=np.float64)
        strat_cnt = np.zeros((n_attr, max_dist + 1), dtype=np.int64)
    
        for xdec in range(n_states):
            dx = dist_state[xdec]
            idx_x = attractor_idx[xdec]
    
            for bit in range(N):
                if (xdec >> bit) & 1:
                    continue
    
                ydec = xdec | (1 << bit)
                dy = dist_state[ydec]
                idx_y = attractor_idx[ydec]
    
                strat_cnt[idx_x, dx] += 1
                strat_cnt[idx_y, dy] += 1
    
                if idx_x == idx_y:
                    basin_coh[idx_x] += 2.0
    
                    if is_attr_mask[xdec]:
                        attr_coh[idx_x] += 1.0
                    if is_attr_mask[ydec]:
                        attr_coh[idx_y] += 1.0
    
                    strat_coh[idx_x, dx] += 1.0
                    strat_coh[idx_y, dy] += 1.0
                else:
                    dxy = dist_attr[idx_x, idx_y]
                    basin_frag[idx_x] += dxy
                    basin_frag[idx_y] += dxy
    
                    if is_attr_mask[xdec]:
                        attr_frag[idx_x] += dxy
                    if is_attr_mask[ydec]:
                        attr_frag[idx_y] += dxy
    
        return (
            basin_coh,
            basin_frag,
            attr_coh,
            attr_frag,
            strat_coh,
            strat_cnt,
        )