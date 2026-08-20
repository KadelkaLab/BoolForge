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

    @njit
    def _single_bit_position(lowbit):
        pos = 0
        while lowbit > 1:
            lowbit >>= 1
            pos += 1
        return pos

    @njit
    def _build_sdds_transition_csr(
        STG_synchronous_exact,
        p_degradation,
        p_activation,
        N
    ):
        """
        Build the exact SDDS state transition matrix in CSR representation.

        Parameters
        ----------
        STG_synchronous_exact : np.ndarray
            1D integer array of length 2**N. Entry x gives the synchronous
            Boolean successor F(x), encoded as a decimal integer.

        p_degradation : np.ndarray
            Length-N array. p_degradation[i] is the probability that node i
            changes 1 -> 0 when its Boolean update calls for degradation.

        p_activation : np.ndarray
            Length-N array. p_activation[i] is the probability that node i
            changes 0 -> 1 when its Boolean update calls for activation.

        N : int
            Number of Boolean variables.

        Returns
        -------
        indptr : np.ndarray
            CSR row pointer array of length 2**N + 1.

        indices : np.ndarray
            CSR column indices.

        data : np.ndarray
            CSR transition probabilities.

        Notes
        -----
        Assumes 0 < p_activation[i] < 1 and
                0 < p_degradation[i] < 1
        for all i. Under this assumption, if m nodes differ between x and
        F(x), exactly 2**m transitions have nonzero probability.
        """

        nstates = STG_synchronous_exact.shape[0]

        # ==============================================================
        # PASS 1
        # Count the number of nonzero transitions in every row and build
        # the CSR row pointer directly.
        # ==============================================================

        indptr = np.empty(nstates + 1, dtype=np.int64)
        indptr[0] = 0

        for x in range(nstates):

            fx = STG_synchronous_exact[x]
            diff = x ^ fx

            # Popcount using Kernighan's method.
            m = 0
            while diff != 0:
                diff &= diff - 1
                m += 1

            indptr[x + 1] = indptr[x] + (1 << m)

        n_non_zero = indptr[nstates]

        # ==============================================================
        # Allocate exact-size CSR arrays.
        # ==============================================================

        indices = np.empty(n_non_zero, dtype=np.int64)
        data = np.empty(n_non_zero, dtype=np.float64)

        # Maximum possible number of simultaneously changing nodes is N.
        # Store their bit positions and corresponding change probabilities.
        diff_bits = np.empty(N, dtype=np.int64)
        q = np.empty(N, dtype=np.float64)

        # ==============================================================
        # PASS 2
        #
        # For each state:
        #   1. extract the differing bits;
        #   2. determine activation/degradation probabilities;
        #   3. progressively expand the row.
        #
        # If the first k nodes have been processed, there are 2**k
        # entries already constructed. Processing the next node doubles
        # the row:
        #
        #   old state       -> stays       probability * (1-q)
        #   old state XOR b -> changes     probability * q
        #
        # Thus we never recompute a product over all m nodes for every
        # subset.
        # ==============================================================

        for x in range(nstates):

            fx = STG_synchronous_exact[x]
            diff = x ^ fx

            # ----------------------------------------------------------
            # Extract set bits of diff.
            #
            # bit position b corresponds to node i = N - 1 - b.
            # ----------------------------------------------------------

            m = 0
            tmp = diff

            while tmp != 0:

                # Isolate lowest set bit.
                lowbit = tmp & -tmp

                # Position of that bit (0 = least significant bit).
                bit_position = _single_bit_position(lowbit)
                node = N - 1 - bit_position

                diff_bits[m] = bit_position

                # Since this bit differs between x and F(x), knowing x's
                # current value tells us whether this is activation or degradation.
                if (x >> bit_position) & 1: #degradation
                    q[m] = p_degradation[node]
                else: #activation
                    q[m] = p_activation[node]

                m += 1

                # Remove lowest set bit.
                tmp ^= lowbit

            # ----------------------------------------------------------
            # Build the row incrementally.
            # ----------------------------------------------------------

            row_start = indptr[x]

            # Initially there is one possible outcome:
            #
            #   x with probability 1.
            #
            indices[row_start] = x
            data[row_start] = 1.0

            n_current = 1

            for k in range(m):

                bit_position = diff_bits[k]
                qk = q[k]
                bit_mask = 1 << bit_position

                old_start = row_start

                # The existing outcomes become the "node does not change"
                # outcomes, while duplicated entries become the
                # "node changes" outcomes.
                for j in range(n_current):

                    old_pos = old_start + j
                    new_pos = old_start + n_current + j

                    old_state = indices[old_pos]
                    old_prob = data[old_pos]

                    # Node does not change.
                    data[old_pos] = old_prob * (1.0 - qk)

                    # Node changes.
                    indices[new_pos] = old_state ^ bit_mask
                    data[new_pos] = old_prob * qk

                n_current *= 2

        return indptr, indices, data


@njit
def _build_sdds_transition_csr_with_endpoint_probabilities(
    STG_synchronous_exact,
    p_degradation,
    p_activation,
    N
):

    """
    Build the exact SDDS state transition matrix in CSR representation.

    For each state x, let F(x) be its synchronous Boolean successor.
    Nodes for which F_i(x) != x_i may change according to their
    activation/degradation probabilities:

        q = 0       node cannot change
        q = 1       node must change
        0 < q < 1   node changes stochastically

    Only nodes with 0 < q < 1 introduce branching. If m such nodes
    exist, the row has 2**m nonzero transitions.

    Parameters
    ----------
    STG_synchronous_exact : np.ndarray
        1D integer array of length 2**N. Entry x gives the synchronous
        Boolean successor F(x), encoded as a decimal integer.

    p_degradation : np.ndarray
        Length-N array. p_degradation[i] is the probability that node i
        changes 1 -> 0 when its Boolean update calls for degradation.

    p_activation : np.ndarray
        Length-N array. p_activation[i] is the probability that node i
        changes 0 -> 1 when its Boolean update calls for activation.

    N : int
        Number of Boolean variables.

    Returns
    -------
    indptr : np.ndarray
        CSR row pointer array of length 2**N + 1.

    indices : np.ndarray
        CSR column indices.

    data : np.ndarray
        CSR transition probabilities.

    """
    nstates = STG_synchronous_exact.shape[0]

    # ==============================================================
    # PASS 1
    # Count the number of nonzero transitions in each row.
    # Only genuinely stochastic nodes contribute a factor of 2.
    # ==============================================================

    indptr = np.empty(nstates + 1, dtype=np.int64)
    indptr[0] = 0

    for x in range(nstates):

        fx = STG_synchronous_exact[x]
        diff = x ^ fx

        m = 0
        tmp = diff

        while tmp != 0:

            lowbit = tmp & -tmp
            bit_position = _single_bit_position(lowbit)
            node = N - 1 - bit_position

            # Current node value determines activation vs degradation.
            if (x >> bit_position) & 1:
                q_i = p_degradation[node]
            else:
                q_i = p_activation[node]

            # Only an interior probability creates two possible outcomes.
            if 0.0 < q_i < 1.0:
                m += 1

            tmp ^= lowbit

        indptr[x + 1] = indptr[x] + (1 << m)

    n_non_zero = indptr[nstates]

    # ==============================================================
    # Allocate exact-size CSR arrays.
    # ==============================================================

    indices = np.empty(n_non_zero, dtype=np.int64)
    data = np.empty(n_non_zero, dtype=np.float64)

    # At most N genuinely stochastic nodes.
    diff_bits = np.empty(N, dtype=np.int64)
    q = np.empty(N, dtype=np.float64)

    # ==============================================================
    # PASS 2
    # Build each row.
    #
    # Deterministic changes (q=1) are applied immediately to
    # base_successor. Deterministic non-changes (q=0) do nothing.
    # Only 0 < q < 1 nodes are stored in diff_bits/q and enumerated.
    # ==============================================================

    for x in range(nstates):

        fx = STG_synchronous_exact[x]
        diff = x ^ fx

        m = 0
        tmp = diff

        # Base successor after all deterministic changes.
        base_successor = x

        while tmp != 0:

            lowbit = tmp & -tmp
            bit_position = _single_bit_position(lowbit)
            node = N - 1 - bit_position

            if (x >> bit_position) & 1:
                # 1 -> 0: degradation
                q_i = p_degradation[node]
            else:
                # 0 -> 1: activation
                q_i = p_activation[node]

            if q_i == 1.0:
                # This node must change.
                base_successor ^= lowbit

            elif q_i > 0.0:
                # 0 < q_i < 1: this node creates branching.
                diff_bits[m] = bit_position
                q[m] = q_i
                m += 1

            # q_i == 0.0:
            # The node cannot change, so base_successor is unchanged.

            tmp ^= lowbit

        # ----------------------------------------------------------
        # Enumerate only genuinely stochastic branches.
        # ----------------------------------------------------------

        row_start = indptr[x]
        n_successors = 1 << m

        for subset in range(n_successors):

            successor = base_successor
            probability = 1.0

            for k in range(m):

                bit_position = diff_bits[k]
                qk = q[k]

                if (subset >> k) & 1:
                    # Node changes.
                    successor ^= (1 << bit_position)
                    probability *= qk
                else:
                    # Node does not change.
                    probability *= (1.0 - qk)

            pos = row_start + subset

            indices[pos] = successor
            data[pos] = probability

    return indptr, indices, data

def get_dimension_trap_space(terminal_scc):
    ref = terminal_scc[0]
    varying = 0
    for s in terminal_scc[1:]:
        varying |= (ref ^ s)
    return varying.bit_count()    

# def _get_sdds_context(self, p_degradation, p_activation):
#     p_degradation = np.asarray(p_degradation, dtype=np.float64)
#     p_activation = np.asarray(p_activation, dtype=np.float64)

#     return (
#         f"sdds_"
#         f"degradation={tuple(p_degradation.tolist())}_"
#         f"activation={tuple(p_activation.tolist())}"
#     )