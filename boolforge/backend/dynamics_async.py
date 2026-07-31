#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ._numba import njit, __LOADED_NUMBA__

if __LOADED_NUMBA__:
    @njit
    def _build_async_transition_coo(
        F_array_list,
        I_array_list,
        N
    ):
        nstates = 1 << N
        max_edges = nstates * N
    
        rows = np.empty(max_edges, dtype=np.int32)
        cols = np.empty(max_edges, dtype=np.int32)
        data = np.empty(max_edges, dtype=np.float32)
    
        edge_count = 0
        powers = np.empty(N, dtype=np.int32)
        for j in range(N):
            powers[j] = 1 << (N - 1 - j)
    
        for s in range(nstates):
            unstable_count = 0
            # count unstable nodes
            for j in range(N):
                regs = I_array_list[j]
                idx = 0
                for k in range(len(regs)):
                    bit = (s >> (N - 1 - regs[k])) & 1
                    idx = (idx << 1) | bit
                new_val = F_array_list[j][idx]
                current = (s >> (N - 1 - j)) & 1
                if new_val != current:
                    unstable_count += 1
    
            # fixed point self-loop
            if unstable_count == 0:
                rows[edge_count] = s
                cols[edge_count] = s
                data[edge_count] = 1.0
                edge_count += 1
                continue
    
            p = 1.0 / unstable_count
    
            # emit transitions
            for j in range(N):
                regs = I_array_list[j]
                idx = 0
                for k in range(len(regs)):
                    bit = (s >> (N - 1 - regs[k])) & 1
                    idx = (idx << 1) | bit
                new_val = F_array_list[j][idx]
                current = (s >> (N - 1 - j)) & 1
                if new_val != current:
                    y = s ^ powers[j]
                    rows[edge_count] = s
                    cols[edge_count] = y
                    data[edge_count] = p
                    edge_count += 1
        return (
            rows[:edge_count],
            cols[:edge_count],
            data[:edge_count]
        )


def get_dimension_trap_space(terminal_scc):
    ref = terminal_scc[0]
    varying = 0
    for s in terminal_scc[1:]:
        varying |= (ref ^ s)
    return varying.bit_count()    
