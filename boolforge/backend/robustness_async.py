#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ._numba import njit, __LOADED_NUMBA__

if __LOADED_NUMBA__:
    @njit(parallel=True)
    def _compute_local_coherence_async_numba(
        N,
        absorption_probs
    ):
        n_states = absorption_probs.shape[0]
        n_attr = absorption_probs.shape[1]
    
        local_coherence = np.zeros(n_states, dtype=np.float32)
        for x in range(n_states):
            total = 0.0
            for bit in range(N):
                y = x ^ (1 << bit)
                for a in range(n_attr):
                    total += absorption_probs[x, a] * absorption_probs[y, a]
            local_coherence[x] = total / N
        return local_coherence
    
    @njit
    def _compute_neighbor_attraction_probability(
            N, 
            absorption_probs
    ):
        n_states, n_attr = absorption_probs.shape
        out = np.zeros((n_states, n_attr))
        for x in range(n_states):
            for bit in range(N):
                y = x ^ (1 << bit)
                out[x] += absorption_probs[y]
        out /= N
        return out
    
