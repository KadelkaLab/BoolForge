#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:52:32 2026

@author: ckadelka
"""


from collections.abc import Sequence
import numpy as np
from scipy.stats import entropy
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import gmres
from scipy.sparse.csgraph import connected_components

from .. import utils

from ..backend._numba import _numba_required, __LOADED_NUMBA__
if __LOADED_NUMBA__:
    from ..backend.dynamics_async import _build_async_transition_coo, _build_sdds_transition_csr, _build_sdds_transition_csr_with_endpoint_probabilities



class BooleanNetworkDynamicsAsyncMixin:
    def _get_stochastic_context(
        self,
        update_scheme: str = "asynchronous",
        *,
        p_degradation: Sequence[float] | None = None,
        p_activation: Sequence[float] | None = None,
    ):
        update_scheme = update_scheme.lower()

        if update_scheme == "asynchronous":
            return "asynchronous"

        elif update_scheme == "sdds":
            return (
                f"sdds_"
                f"degradation={tuple(p_degradation.tolist())}_"
                f"activation={tuple(p_activation.tolist())}"
            )

        else:
            raise ValueError(
                f"Unknown stochastic update scheme: {update_scheme!r}."
            )

    def _validate_stochastic_update_scheme(
        self,
        update_scheme: str = "asynchronous",
        *,
        p_degradation: Sequence[float] | None = None,
        p_activation: Sequence[float] | None = None,
    ) -> tuple[str, np.ndarray | None, np.ndarray | None]:
        update_scheme = update_scheme.lower()

        if update_scheme not in {"asynchronous", "sdds"}:
            raise ValueError(
                f"Unknown stochastic update scheme: {update_scheme!r}. "
                "Expected 'asynchronous' or 'sdds'."
            )

        if update_scheme == "asynchronous":
            if p_degradation is not None or p_activation is not None:
                raise ValueError(
                    "p_degradation and p_activation are only valid "
                    "for the SDDS update scheme."
                )

            return update_scheme, None, None
        elif update_scheme == "sdds":
            if p_degradation is None or p_activation is None:
                raise ValueError(
                    "p_degradation and p_activation must be provided "
                    "for the SDDS update scheme."
                )

            p_degradation = np.asarray(p_degradation, dtype=np.float64)
            p_activation = np.asarray(p_activation, dtype=np.float64)

            if p_degradation.shape != (self.N,):
                raise ValueError(
                    f"p_degradation must have shape ({self.N},), "
                    f"got {p_degradation.shape}."
                )

            if p_activation.shape != (self.N,):
                raise ValueError(
                    f"p_activation must have shape ({self.N},), "
                    f"got {p_activation.shape}."
                )

            if np.any((p_degradation < 0.0) | (p_degradation > 1.0)):
                raise ValueError(
                    "p_degradation must contain values in [0, 1]."
                )

            if np.any((p_activation < 0.0) | (p_activation > 1.0)):
                raise ValueError(
                    "p_activation must contain values in [0, 1]."
                )

            return update_scheme, p_degradation, p_activation
        else:
            raise ValueError(
                f"Unknown stochastic update scheme: {update_scheme!r}."
            )

    def get_stochastic_transition_matrix(
        self,
        update_scheme: str = "asynchronous",
        *,
        p_degradation: Sequence[float] | None = None,
        p_activation: Sequence[float] | None = None
    ) -> csr_matrix:
        """
        Construct and return the exact stochastic state transition graph.

        The stochastic state transition graph (STG) is represented as a
        row-stochastic sparse matrix whose rows correspond to network states
        and whose nonzero entries encode one-step stochastic transitions.

        The matrix is cached after the first computation.

        Parameters
        ----------  
            update_scheme : {"asynchronous", "sdds"}, optional
                Stochastic update scheme. Default is "asynchronous".
            p_degradation : Sequence[float] or None, optional
                Node-specific degradation probabilities. Required for SDDS.
            p_activation : Sequence[float] or None, optional
                Node-specific activation probabilities. Required for SDDS.
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse transition matrix of shape ``(2**N, 2**N)``.

        Notes
        -----
        The transition matrix depends on both the Boolean network and, 
        for SDDS, the supplied activation/degradation probabilities, so the
        transition matrix is cached separately for each propensity parameterization.
        """
        (
            update_scheme,
            p_degradation,
            p_activation,
        ) = self._validate_stochastic_update_scheme(
            update_scheme=update_scheme,
            p_degradation=p_degradation,
            p_activation=p_activation,
        )

        if update_scheme == "asynchronous":
            return self.get_asynchronous_transition_matrix()
        elif update_scheme == "sdds":
            return self.get_sdds_transition_matrix(
                p_degradation=p_degradation,
                p_activation=p_activation,
            )
        else: # should be impossible because of validation beforehand
            raise RuntimeError(
                f"Unhandled stochastic update scheme: {update_scheme!r}."
            )


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

        self._set_property('STG', STG, 
                           context='asynchronous', exact=True)
        
        return STG


    def get_sdds_transition_matrix(
        self,
        p_degradation : Sequence[float],
        p_activation : Sequence[float],
    ) -> csr_matrix:
        """
        Construct and return the exact SDDS state transition graph.

        The SDDS state transition graph (STG) is represented as a
        row-stochastic sparse matrix whose rows correspond to network states
        and whose nonzero entries encode one-step SDDS transition
        probabilities.

        Parameters
        ----------
        p_degradation : Sequence[float]
            Node-specific probabilities of degradation (1 -> 0), of length N.

        p_activation : Sequence[float]
            Node-specific probabilities of activation (0 -> 1), of length N.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse transition matrix of shape ``(2**N, 2**N)``.

        Notes
        -----
        The transition matrix depends on both the Boolean network and the
        supplied activation/degradation probabilities, so the SDDS transition
        matrix is cached separately for each propensity parameterization.

        References
        ----------
        [1] Murrugarra, D., Veliz-Cuba, A., Aguilar, B., Laubenbacher, R. (2012).
            Modeling stochasticity and variability in gene regulatory networks.
            EURASIP Journal on Bioinformatics and Systems Biology, 2012(1), 8.
        """
        (
            update_scheme,
            p_degradation,
            p_activation,
        ) = self._validate_stochastic_update_scheme(
            update_scheme='sdds',
            p_degradation=p_degradation,
            p_activation=p_activation,
        )

        context = self._get_stochastic_context(
            update_scheme='sdds',
            p_degradation=p_degradation,
            p_activation=p_activation,
        )

        STG, status = self._get_property("STG", context=context)
        if status == "exact":
            return STG

        if not __LOADED_NUMBA__:
            _numba_required("SDDS exact dynamics computation")

        # The synchronous STG provides F(x) for every state x. The SDDS
        # transition kernel then converts each deterministic transition
        # x -> F(x) into its probabilistic SDDS successors.
        if self.STG is None:
            self.compute_synchronous_state_transition_graph()

        has_endpoint_probabilities = (
            np.any(p_activation == 0.0)
            or np.any(p_activation == 1.0)
            or np.any(p_degradation == 0.0)
            or np.any(p_degradation == 1.0)
        )

        if has_endpoint_probabilities:
            _build_sdds_transition_csr_method = _build_sdds_transition_csr_with_endpoint_probabilities
        else:
            _build_sdds_transition_csr_method = _build_sdds_transition_csr

        indptr, indices, data = _build_sdds_transition_csr_method(
            STG_synchronous_exact,
            p_degradation=p_degradation,
            p_activation=p_activation,
            N=self.N,
            )

        STG = csr_matrix(
            (data, indices, indptr),
            shape=(1 << self.N, 1 << self.N),
            dtype=np.float64,
        )

        self._set_property("STG", STG, 
                           context=context, exact=True)

        return STG

    def _get_terminal_sccs_from_stg(self, STG):
        """
        Compute the terminal strongly connected components of a given STG.

        A terminal SCC is a strongly connected component with no outgoing
        transitions to states outside the component. Terminal SCCs correspond
        to attractors, including both steady states and cyclic attractors.

        Parameters
        ----------
        STG : scipy.sparse.csr_matrix
            Sparse transition matrix of shape ``(2**N, 2**N)``.

        Returns
        -------
        list of list of int
            Terminal SCCs represented as lists of decimal-encoded states.
        """
        n_components, labels = connected_components(STG, directed=True, connection='strong')
        terminal_sccs = []
        for c in range(n_components):
            states = np.where(labels == c)[0]
            for s in states:
                start, end = STG.indptr[s], STG.indptr[s + 1]
                if np.any(labels[STG.indices[start:end]] != c):
                    break
            else:
                terminal_sccs.append([int(s) for s in states])
        
        return terminal_sccs

    def get_terminal_sccs_stochastic_exact(
        self,
        update_scheme: str = "asynchronous",
        *,
        p_degradation: Sequence[float] | None = None,
        p_activation: Sequence[float] | None = None,
    ) -> list[list[int]]:
        """
        Compute the terminal SCCs of a stochastic Boolean-network STG.

        A terminal SCC is a strongly connected component with no outgoing
        transitions to states outside the component. Terminal SCCs correspond
        to attractors under stochastic update, including both steady states and cyclic
        attractors.

        Results are cached after the first computation.

        Parameters
        ----------
        update_scheme : {"asynchronous", "sdds"}, optional
            Stochastic update scheme. Default is "asynchronous".
        p_degradation : Sequence[float] or None, optional
            Node-specific degradation probabilities. Required for SDDS. 
        p_activation : Sequence[float] or None, optional
            Node-specific activation probabilities. Required for SDDS.

        Returns
        -------
        list of list of int
            Terminal SCCs represented as lists of decimal-encoded states.
        """

        (
            update_scheme,
            p_degradation,
            p_activation,
        ) = self._validate_stochastic_update_scheme(
            update_scheme=update_scheme,
            p_degradation=p_degradation,
            p_activation=p_activation,
        )

        context = self._get_stochastic_context(
            update_scheme,
            p_degradation=p_degradation,
            p_activation=p_activation,
        )

        terminal_sccs, status = self._get_property(
            "terminal_sccs",
            context=context,
        )

        if status == "exact":
            return terminal_sccs

        if update_scheme == "asynchronous":
            STG = self.get_asynchronous_transition_matrix()
        elif update_scheme == "sdds":
            STG = self.get_sdds_transition_matrix(
                p_degradation=p_degradation,
                p_activation=p_activation,
            )

        terminal_sccs = self._get_terminal_sccs_from_stg(STG)

        self._set_property(
            "terminal_sccs", terminal_sccs,
            context=context, exact=True,
        )
        self._set_property(
            "number_of_terminal_sccs", len(terminal_sccs),
            context=context, exact=True,
        )

        return terminal_sccs
    
    def get_terminal_sccs_asynchronous_exact(self) -> list[list[int]]:
        """
        Compute the terminal SCCs under general asynchronous updating.

        Notes
        -----
        This method is retained for backward compatibility. For new code, use
        ``get_terminal_sccs_stochastic_exact(update_scheme="asynchronous")``
        instead.
        """
        return self.get_terminal_sccs_stochastic_exact(
                update_scheme="asynchronous"
            )

    def get_minimal_trap_spaces_stochastic_exact(
        self,
        update_scheme: str = "asynchronous",
        *,
        p_degradation: Sequence[float] | None = None,
        p_activation: Sequence[float] | None = None,
    ) -> np.ndarray:
        """
        Compute the minimal trap space associated with each terminal SCC
        under a stochastic update scheme.

        For each terminal SCC, nodes that take the same value in every state
        are marked by their fixed value (0 or 1), whereas nodes that vary
        across the SCC are marked as -1.

        Parameters
        ----------
        update_scheme : {"asynchronous", "sdds"}, optional
            Stochastic update scheme. Default is "asynchronous".
        p_degradation : Sequence[float] or None, optional
            Node-specific degradation probabilities. Required for SDDS.
        p_activation : Sequence[float] or None, optional
            Node-specific activation probabilities. Required for SDDS.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_terminal_sccs, N)`` whose rows represent
            minimal trap spaces.
        """
        terminal_sccs = self.get_terminal_sccs_stochastic_exact(
            update_scheme=update_scheme,
            p_degradation=p_degradation,
            p_activation=p_activation,
        )

        return np.array(
            [
                utils.get_minimal_trap_space(states, self.N)
                for states in terminal_sccs
            ]
        )
    
    def get_minimal_trap_spaces_asynchronous_exact(self) -> np.ndarray:
        """
        Compute the minimal trap space associated with each terminal SCC
        under general asynchronous updating.

        Notes
        -----
        This method is retained for backward compatibility. For new code, use
        ``get_minimal_trap_spaces_stochastic_exact(update_scheme="asynchronous")``
        instead.
        """
        return self.get_minimal_trap_spaces_stochastic_exact(
            update_scheme="asynchronous"
        )

    def get_number_frozen_nodes_stochastic_exact(self,
        update_scheme: str = "asynchronous",
        *,
        p_degradation: Sequence[float] | None = None,
        p_activation: Sequence[float] | None = None,
    ) -> int:
        """
        Compute the number of frozen nodes in the stochastic dynamics.

        A node is considered frozen if it takes the same value in every
        attractor state. For stochastic dynamics, attractor states are
        defined as the states belonging to terminal strongly connected
        components (terminal SCCs) of the stochastic state transition
        graph.

        Parameters
        ----------
        update_scheme : {"asynchronous", "sdds"}, optional
            Stochastic update scheme. Default is "asynchronous".
        p_degradation : Sequence[float] or None, optional
            Node-specific degradation probabilities. Required for SDDS. 
        p_activation : Sequence[float] or None, optional
            Node-specific activation probabilities. Required for SDDS.


        Returns
        -------
        int
            Number of nodes whose value is identical across all attractor
            states.
        """
        terminal_sccs = self.get_terminal_sccs_stochastic_exact(
            update_scheme=update_scheme,
            p_degradation=p_degradation,
            p_activation=p_activation,
        )
        return self.N - utils.get_number_of_varying_nodes(
            utils.flatten(terminal_sccs)
        )

    
    def get_number_frozen_nodes_asynchronous_exact(self) -> int:
        """
        Compute the number of frozen nodes in the asynchronous dynamics.
    
        Notes:
        -----
        This method is retained for backward compatibility. For new code, use
        ``get_number_frozen_nodes_stochastic_exact(update_scheme="asynchronous")``
        instead.
        """
        return self.get_number_frozen_nodes_stochastic_exact(
            update_scheme="asynchronous"
        )
        

    def _build_absorption_system(self) -> tuple[csr_matrix, np.ndarray]:
        """
        Build the linear system (A, R) used to solve for absorption
        probabilities of an absorbing Markov chain.

        Restricts the asynchronous transition matrix to transient states,
        producing ``A = I - Q`` (where ``Q`` is the transient-to-transient
        sub-matrix) and ``R`` (one-step transition probabilities from
        transient states directly into each terminal SCC). Solving
        ``A x = R`` for `x` gives absorption probabilities; solving
        ``A x = (A^-1 R)`` gives expected absorption times.

        Returns
        -------
        fundamental_matrix : scipy.sparse.csr_matrix
            The matrix ``I - Q`` of shape ``(n_transients, n_transients)``,
            where ``Q`` is the transition matrix restricted to transient
            states.
        transient_to_absorbing_matrix : numpy.ndarray
            Matrix of shape ``(n_transients, n_terminal_sccs)`` giving the
            one-step transition probabilities from transient states directly
            into each terminal SCC.
        """
        STG = self.get_asynchronous_transition_matrix()
        terminal_sccs = self.get_terminal_sccs_asynchronous_exact()
        n_terminal_sccs = len(terminal_sccs)
        transient_states = np.setdiff1d(np.arange(1 << self.N),
                                        np.concatenate(terminal_sccs))
        n_transients = len(transient_states)

        transient_mask = np.zeros(1 << self.N, dtype=bool)
        transient_mask[transient_states] = True
        transient_index = -np.ones(1 << self.N, dtype=np.int32)
        transient_index[transient_states] = np.arange(n_transients)

        # build Q and R without giant reorder/slicing
        rows_Q = []
        cols_Q = []
        vals_Q = []
        transient_to_absorbing_matrix = np.zeros((n_transients, n_terminal_sccs), dtype=np.float32)

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
                    transient_to_absorbing_matrix[s_local, a] += p

        Q = csr_matrix(
            (vals_Q, (rows_Q, cols_Q)),
            shape=(n_transients, n_transients),
            dtype=np.float32
        )
        uninverted_fundamental_matrix = identity(n_transients, dtype=np.float32, format='csr') - Q

        return uninverted_fundamental_matrix, transient_to_absorbing_matrix


    @staticmethod
    def _gmres(A: csr_matrix, B: np.ndarray, probability_cutoff: bool) -> np.ndarray:
        """
        Solve ``A x = b`` column-by-column for each column ``b`` of `B` using
        GMRES.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            Square coefficient matrix of shape ``(n, n)``.
        B : numpy.ndarray
            Right-hand side matrix of shape ``(n, m)``. Each column is
            solved for independently.
        probability_cutoff : bool
            If True, clip each solution to ``[0, 1]`` and renormalize each
            row of the result to sum to 1 (for absorption probabilities).
            If False, only clip below at 0 with no upper bound and no
            renormalization (for expected absorption times, which are
            non-negative but otherwise unbounded).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n, m)`` where column ``a`` is the solution to
            ``A x = B[:, a]``.
        """
        dims = (np.shape(A)[1], np.shape(B)[1])    
        out_matrix = np.zeros(dims, dtype=np.float64)
        if probability_cutoff:
            cutoff = 1.0
        else:
            cutoff = None
        for a in range(dims[1]):
            b = B[:, a]
            x,_ = gmres(A,b,atol=1e-10)
            x = np.clip(x.astype(np.float64), 0.0, cutoff)
            out_matrix[:,a] = x
        if probability_cutoff:
            row_sums = out_matrix.sum(axis=1, keepdims=True)
            out_matrix /= row_sums
        return out_matrix


    def get_absorption_probabilities_stochastic_exact(
            self,
            update_scheme: str = "asynchronous",
            *,
            p_degradation: Sequence[float] | None = None,
            p_activation: Sequence[float] | None = None,
        ) -> np.ndarray:
        """
        Compute exact absorption probabilities under a stochastic update scheme.

        For every network state and every terminal SCC, this method computes
        the probability that a stochastic trajectory starting from that
        state is eventually absorbed into the corresponding terminal SCC.
        
        Probabilities are obtained by solving the standard absorbing Markov
        chain equations and are cached after the first computation.
        
        Parameters
        ----------
        update_scheme : {"asynchronous", "sdds"}, optional
            Stochastic update scheme. Default is "asynchronous".
        p_degradation : Sequence[float] or None, optional
            Node-specific degradation probabilities. Required for SDDS. 
        p_activation : Sequence[float] or None, optional
            Node-specific activation probabilities. Required for SDDS.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(2**N, n_terminal_sccs)`` where entry
            ``[x, a]`` is the probability that state ``x`` eventually
            reaches terminal SCC ``a``.
        """

        (
            update_scheme,
            p_degradation,
            p_activation,
        ) = self._validate_stochastic_update_scheme(
            update_scheme=update_scheme,
            p_degradation=p_degradation,
            p_activation=p_activation,
        )

        context = self._get_stochastic_context(
            update_scheme,
            p_degradation=p_degradation,
            p_activation=p_activation,
        )

        absorption_probabilities, status = self._get_property(
            "absorption_probabilities",
            context=context,
        )

        if status == "exact":
            return absorption_probabilities

        STG = self.get_stochastic_transition_matrix(
            update_scheme=update_scheme,
            p_degradation=p_degradation,
            p_activation=p_activation
        )
        terminal_sccs = self.get_terminal_sccs_stochastic_exact(
            update_scheme=update_scheme,
            p_degradation=p_degradation,
            p_activation=p_activation
        )
        
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
                           context=context, exact=True)
        
        return absorption_probs


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
        
        terminal_sccs = self.get_terminal_sccs_asynchronous_exact()
        n_terminal_sccs = len(terminal_sccs)
        transient_states = np.setdiff1d(np.arange(1 << self.N),
                                        np.concatenate(terminal_sccs))
        n_transients = len(transient_states)
        absorption_probs = np.zeros(((1 << self.N), n_terminal_sccs), dtype=np.float32)
        
        if n_transients > 0:
            A, R = self._build_absorption_system()
            absorption_probs[transient_states] = self._gmres(A, R, True)
        
        for a, states in enumerate(terminal_sccs):
            absorption_probs[states, a] = 1.0   
            
        self._set_property('absorption_probabilities', absorption_probs,
                        context='asynchronous', exact=True)
        
        return absorption_probs


    def get_expected_absorption_times_exact(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute exact expected absorption times for the asynchronous dynamics.

        For every network state, this method computes the expected number of
        asynchronous update steps until the trajectory is absorbed into some
        terminal SCC, as well as the expected number of steps conditioned on
        absorption into each specific terminal SCC.

        Both quantities are obtained from the fundamental-matrix identity for
        absorbing Markov chains (via ``A x = N R`` where ``N = A^-1``, using
        the previously computed absorption probabilities ``N R`` as the new
        right-hand side) and are cached after the first computation.

        Returns
        -------
        mean_absorption_times_to_any_scc : numpy.ndarray
            Array of shape ``(2**N,)`` where entry ``[x]`` is the expected
            number of steps for state ``x`` to be absorbed into any terminal
            SCC. Terminal states have value 0.
        mean_absorption_times_to_specific_sccs : numpy.ndarray
            Array of shape ``(2**N, n_terminal_sccs)`` where entry
            ``[x, a]`` is the expected number of steps for state ``x`` to be
            absorbed, conditioned on absorption occurring into terminal SCC
            ``a``. Entries are ``NaN`` where absorption into SCC ``a`` from
            state ``x`` has zero probability. State ``x``'s own terminal SCC
            entry is 0.
        """
        if ('mean_absorption_times_to_any_scc', 'asynchronous') in self._properties_exact:
            return self._properties_exact[('mean_absorption_times_to_any_scc', 'asynchronous')], self._properties_exact[('mean_absorption_times_to_specific_sccs', 'asynchronous')]

        absorption_probs = self.get_absorption_probabilities_exact()
        terminal_sccs = self.get_terminal_sccs_asynchronous_exact()
        transient_states = np.setdiff1d(np.arange(1 << self.N),
                                        np.concatenate(terminal_sccs))
        relavent_probs = absorption_probs[transient_states, :]
        mean_absorption_times_to_any_scc = np.zeros(1 << self.N, dtype=np.float32)
        mean_absorption_times_to_specific_sccs = np.full(np.shape(absorption_probs), np.nan, dtype=np.float32)
        if len(transient_states)>0:
            A, _ = self._build_absorption_system()
            cap_N_squared_R = self._gmres(A, relavent_probs, False)
            mean_absorption_times_to_any_scc[transient_states] = cap_N_squared_R.sum(axis=1)
            mean_absorption_times_to_specific_sccs[transient_states] = np.divide(
                cap_N_squared_R, relavent_probs,
                out=np.full_like(cap_N_squared_R, np.nan),
                where=relavent_probs>0)
        for a, states in enumerate(terminal_sccs):
            mean_absorption_times_to_any_scc[states] = 0.0
            mean_absorption_times_to_specific_sccs[states, a] = 0.0
        self._set_property('mean_absorption_times_to_any_scc', mean_absorption_times_to_any_scc,
                            context='asynchronous', exact=True)
        self._set_property('mean_absorption_times_to_specific_sccs', mean_absorption_times_to_specific_sccs,
                        context='asynchronous', exact=True)
        return mean_absorption_times_to_any_scc, mean_absorption_times_to_specific_sccs


    def get_basin_sizes_asynchronous_exact(self, relative=True) -> np.ndarray:
        """
        Compute the exact basin sizes of the asynchronous terminal SCCs.

        Basin size is defined as the mean absorption probability over all
        network states. Thus, the relative basin sizes sum to one.

        Parameters
        ----------
        relative : bool, optional
            If True, return basin sizes as proportions of the state space.
            If False, return basin sizes as numbers of states. Default is True.

        Returns
        -------
        numpy.ndarray
            One basin size for each terminal SCC, in the same order as returned
            by ``get_terminal_sccs_asynchronous_exact``.
        """
        if ('BasinSizes', 'asynchronous') in self._properties_exact:
            basin_sizes = self._properties_exact[('BasinSizes', 'asynchronous')]
        else:
            absorption_probs = self.get_absorption_probabilities_exact()
            basin_sizes = np.sum(absorption_probs, axis=0) / (2 ** self.N)
            self._set_property('BasinSizes', basin_sizes,
                               context='asynchronous', exact=True)
        if relative:
            return basin_sizes
        else:
            return basin_sizes * (2**self.N)


    def compute_entropy(self) -> dict:
        """
        Compute entropy-based measures of asynchronous attractor structure.

        For each network state, computes the Shannon entropy of its absorption
        probability distribution over terminal SCCs. Also computes the mean
        state entropy, the entropy of the basin-size distribution, and the mean
        state entropy within each basin.

        Returns
        -------
        dict
            Dictionary containing:

            - ``state_entropies`` : numpy.ndarray
                Shannon entropy of the absorption probabilities for each state.
            - ``basin_entropy`` : float
                Shannon entropy of the distribution of basin sizes.
            - ``mean_state_entropy`` : float
                Mean state entropy over all network states.
            - ``basin_mean_state_entropies`` : numpy.ndarray
                Mean state entropy among states having positive absorption
                probability into each terminal SCC.
        """
        if ('mean_state_entropy', 'asynchronous') in self._properties_exact:
            return {'state_entropies' : self._properties_exact[('state_entropies', 'asynchronous')],
                    'basin_entropy' : self._properties_exact[('basin_entropy', 'asynchronous')],
                    'mean_state_entropy' : self._properties_exact[('mean_state_entropy', 'asynchronous')],
                    'basin_mean_state_entropies' : self._properties_exact[('basin_mean_state_entropies', 'asynchronous')]}
        absorption_probabilities = self.get_absorption_probabilities_exact()

        xlnx_mat = np.multiply(np.log(absorption_probabilities, 
                                    out=np.zeros_like(absorption_probabilities), 
                                    where=absorption_probabilities>0),
                            absorption_probabilities)
        state_entropies = -np.sum(xlnx_mat, 1)
        mean_state_entropy = np.mean(state_entropies)

        basin_sizes = self.get_basin_sizes_asynchronous_exact()
        basin_entropy = entropy(basin_sizes)

        basin_mean_state_entropies = np.divide(
            np.nanmean(
                np.multiply(absorption_probabilities,
                            np.where(absorption_probabilities>0, 
                                     state_entropies[:, None], 
                                     np.nan)), 
                       axis=0),
                                               basin_sizes)
        
        self._set_property('state_entropies', state_entropies,
                        context='asynchronous', exact=True)
        self._set_property('basin_entropy', basin_entropy,
                            context='asynchronous', exact=True)
        self._set_property('mean_state_entropy', mean_state_entropy,
                            context='asynchronous', exact=True)
        self._set_property('basin_mean_state_entropies', basin_mean_state_entropies,
                            context='asynchronous', exact=True)

        return {'state_entropies':state_entropies, 'basin_entropy':basin_entropy, 
                'mean_state_entropy':mean_state_entropy, 'basin_mean_state_entropies':basin_mean_state_entropies}


    def _compute_local_divergence_async(self):
        """
        Compute the local divergence at every network state.

        For each state, local divergence is the mean Jensen-Shannon divergence
        between its absorption-probability distribution and those of its
        one-bit neighbors.

        Returns
        -------
        numpy.ndarray
            Array of length ``2**N`` containing the local divergence at each
            network state.
        """
        absorption_probabilities = self.get_absorption_probabilities_exact()
        entropies = self.compute_entropy()
        state_entropies = entropies['state_entropies']
        n_states = absorption_probabilities.shape[0]    
        local_divergence = np.zeros(n_states, dtype=np.float32)
        for x in range(n_states):
            total = 0.0
            for bit in range(self.N):
                y = x ^ (1 << bit)
                total += entropy((absorption_probabilities[x]+absorption_probabilities[y])/2) - (state_entropies[x]+state_entropies[y])/2
            local_divergence[x] = total / self.N
        return local_divergence


    def get_divergence(self):
        """
        Compute the mean and state-resolved divergence of the asynchronous dynamics.

        Returns
        -------
        network_divergence : float
            Mean local divergence over all network states.
        local_divergence : numpy.ndarray
            Local divergence at each network state.
        """
        local_divergence = self._compute_local_divergence_async()
        network_divergence = np.mean(local_divergence)
        return network_divergence, local_divergence


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