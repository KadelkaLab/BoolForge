#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from itertools import combinations
from collections.abc import Sequence
from .. import utils
from .. import utils_multistate

class MultistateFunctionCanalizationMixin:
    def get_layer_structure_from_canalized_outputs(
            outputs : Sequence[int]
        ) -> list:
        """
        Compute the canalizing layer structure from canalized outputs.

        Consecutive identical canalized output values are grouped into the same
        canalizing layer. The size of each layer corresponds to the number of
        variables in that layer.

        Parameters
        ----------
        outputs : Sequence[int]
            Sequence of canalized output values in the order in which canalizing
            variables are identified.

        Returns
        -------
        list[int]
            List specifying the number of variables in each canalizing layer.
        """
        
        return

    def _permissible_sets(r_i: int, segments_only: bool) -> list:
            result = []
            all_values = list(range(r_i))
            if segments_only:
                for j in range(1, r_i):
                    result.append(set(range(0, j)))
                for j in range(r_i - 1, 0, -1):
                    result.append(set(range(j, r_i)))
            else:
                for size in range(1, r_i):
                    for S in combinations(all_values, size):
                        result.append(set(S))
            return result

    def _row_mask(r_inputs, i, S) -> np.ndarray:
        """
        Boolean mask over all truth table rows, True where variable i takes a
        value in S.

        Parameters
        ----------
        r_inputs : np.ndarray
            Radices of the variables of the function being masked. Passed
            explicitly rather than read from self, because recursive routines
            operate on reduced subfunctions with fewer variables.
        i : int
            Index of the variable to test.
        S : set | frozenset
            Values of variable i that trigger the canalizing condition.

        Returns
        -------
        np.ndarray
            Boolean array of length prod(r_inputs).
        """
        total_rows = int(np.prod(r_inputs))
        return np.array([
            utils_multistate.dec2mix(row, r_inputs)[i] in S
            for row in range(total_rows)
        ])

    def _depends_on(f, r_inputs, i) -> bool:
        """
        Whether f actually depends on variable i.

        Compares the slices of f obtained by fixing variable i to each of its
        values. If all slices are identical, f does not depend on variable i.
        """
        base = None
        for a in range(r_inputs[i]): #loop over all possible values of variable i
            sl = f[_row_mask(r_inputs, i, {a})] #extract the values of f where x_i is equal to a
            if base is None: #if first slice, store it as the base for comparison
                base = sl
            elif not np.array_equal(base, sl): #if the current slice is not equal to the base sl, f depends on x_i
                return True
        return False


    class MultistateFunctionCanalizationMixin:

        @property
        def canalizing(self) -> bool:
            """Check whether the multistate function is canalizing."""
            # return self.is_canalizing()
            return

        @property
        def nested_canalizing(self) -> bool:
            """Check whether the multistate function is nested canalizing."""
            # return self.is_ncf()
            return

        @property
        def canalizing_depth(self) -> int:
            """Determine the canalizing depth of the multistate function."""
            # return self.get_canalizing_depth()
            return

        @property
        def layer_structure(self):
            # if "LayerStructure" not in self.properties:
            #     call self.get_layer_structure() to compute and cache it
            # return self.properties["LayerStructure"]
            return

        
        def is_canalizing(self) -> bool:
            """
            Determine whether the multistate function is canalizing.

            A multistate function is canalizing if there exists at least one
            variable x_i and a nonempty proper subset S of its state space
            {0, ..., r_inputs[i] - 1} such that fixing x_i to any value in S
            forces the output to a constant b, regardless of all other inputs.

            Parameters
            ----------
            segments_only : bool, optional
                If ``False`` (default), S can be any nonempty proper subset of
                {0, ..., r_inputs[i] - 1}. This corresponds to the
                Murrugarra--Dimitrova definition.
                If ``True``, S is restricted to segments of the form {0, ..., i}
                or {i, ..., r_inputs[i] - 1}. This corresponds to the
                Remy--Ruet definition.

            Returns
            -------
            bool
                ``True`` if the multistate function is canalizing, ``False`` otherwise.

            References
            ----------
            Murrugarra, D., & Dimitrova, E. S. (2013).
                Quantifying the Connectivity of a Network and Its Application
                to the Analysis of the Phenomenon of Canalization.

            Remy, E., & Ruet, P. (2008).
                From Minimal Signed Circuits to the Dynamics of Boolean Regulatory
                Networks.
            """
            # Unlike the Boolean case we cannot use bitmasking because variables
            # have different radices. Instead, for each row index we use
            # dec2mix(row, self.r_inputs) to decode the row into its input vector
            # and read off the value of each variable.
            #
            # for each variable i in range(self.n):
            #     let r_i = self.r_inputs[i]
            #
            #     if segments_only is False:
            #         candidate_sets = all nonempty proper subsets of {0, ..., r_i - 1}
            #                          ordered by size (size-1 first, then size-2, etc.)
            #     if segments_only is True:
            #         candidate_sets = all segments of {0, ..., r_i - 1}
            #                          i.e. sets of the form {0,..,j} or {j,..,r_i-1}
            #
            #     for each candidate set S in candidate_sets:
            #
            #         build a boolean mask over all len(self.f) rows:
            #             mask[row] = True if dec2mix(row, self.r_inputs)[i] ∈ S
            #
            #         f_S     = self.f[mask]
            #         f_not_S = self.f[~mask]
            #
            #         if f_S is all the same value b:
            #             if f_not_S is NOT all equal to b:
            #                 return True
            #
            # return False
            
            if np.all(self.f == self.f[0]):
                return False

            for i in range(self.n):
                for a in range(int(self.r_inputs[i])):
                    mask    = _row_mask(self.r_inputs, i, {a})
                    f_S     = self.f[mask]
                    f_not_S = self.f[~mask]

                    if not np.all(f_S == f_S[0]):
                        continue
                    if not np.all(f_not_S == f_S[0]):
                        return True

            return False
            
            

        def is_k_canalizing(self, k: int, segments_only: bool = False) -> bool:
            """
            Determine whether the multistate function is k-canalizing.

            A multistate function is k-canalizing if it has a sequence of at
            least k canalizing variables. After fixing the first canalizing
            variable to its canalizing input set S, the resulting subfunction
            must itself be (k-1)-canalizing, recursively.

            Parameters
            ----------
            k : int
                Desired canalizing depth, with ``0 <= k <= n``. Every multistate
                function is trivially 0-canalizing.
            segments_only : bool, optional
                If ``False`` (default), S can be any nonempty proper subset.
                Corresponds to the Murrugarra--Dimitrova definition.
                If ``True``, S is restricted to segments. Corresponds to the
                Remy--Ruet definition.

            Returns
            -------
            bool
                ``True`` if the multistate function is k-canalizing, ``False`` otherwise.

            Notes
            -----
            This method has exponential time complexity in ``n`` and is intended for
            small multistate functions.

            References
            ----------
            Murrugarra, D., & Dimitrova, E. S. (2013).
                Quantifying the Connectivity of a Network and Its Application
                to the Analysis of the Phenomenon of Canalization.

            Remy, E., & Ruet, P. (2008).
                From Minimal Signed Circuits to the Dynamics of Boolean Regulatory
                Networks.
            """
            

            if k > self.n: #can't have more canalizing variables than total variables
                return False
            if k == 0: #trivially true for any function
                return True
            if np.all(self.f == self.f[0]): #constant functions are not considered canalizing
                return False

            for i in range(self.n):
                r_i = self.r_inputs[i]

                for S in _permissible_sets(r_i, segments_only): #loop over all nonempty propoer subsets of {0,...,r_i-1} for each of the i variables

                    mask    = _row_mask(self.r_inputs, i, S) # Returns a True value for all the rows where x_i is inside S
                    f_S     = self.f[mask] # Extracting the values of f where x_i is inside S
                    f_not_S = self.f[~mask] #Extracting the values of f where X_i is outside S

                    if not np.all(f_S == f_S[0]): # If f_S is not all the same value, then x_i is not canalizing for this S
                        continue
                    b = f_S[0]
                    if np.all(f_not_S == b): # if f_not_S is all equal to b, then the complement g is constant and does not depend on x_i, so x_i is not canalizing for this S
                        continue

                    # the complement must not depend on x_i, so that g is a function
                    # on the remaining variables only
                    r_rem = r_i - len(S)
                    if r_rem > 1:
                        r_inputs_rem = self.r_inputs.copy()
                        r_inputs_rem[i] = r_rem
                        if _depends_on(f_not_S, r_inputs_rem, i): # checking if the complment (g) depends on x_i, if it does, then x_i is not canalizing for this S
                            continue

                    if k == 1: # 1-canalizing found
                        return True

                    # x_i can now be dropped: collapse its identical slices
                    r_inputs_new = np.delete(self.r_inputs, i) # describes the radices of f_new that is the subfunction after removing x_i
                    if r_rem > 1:
                        r_inputs_rem = self.r_inputs.copy() # describes the radices of f_not_S
                        r_inputs_rem[i] = r_rem # update the radix of x_i
                        keep = _row_mask(r_inputs_rem, i, {0}) # returns a mask of the rows where x_i is equal to 0, choice of 0 is arbitrary because all slices of f_not_S are identical since f_not_S does not depend on x_i
                        f_new = f_not_S[keep] # Extracting the values of f_not_S where x_i = 0, this is the subfunction after removing x_i = canalizing variable
                    else:
                        f_new = f_not_S

                    sub = self.__class__(f_new.tolist(), self.r, r_inputs_new.tolist()) # create a new MultistateFunction object for the subfunction after removing x_i
                    if sub.is_k_canalizing(k - 1, segments_only=segments_only):
                        return True

            return False

            
            

        def is_ncf(self, segments_only: bool = False) -> bool:
            """
            Determine whether the multistate function is nested canalizing (NCF).

            A multistate function is nested canalizing if it is n-canalizing,
            i.e. every variable appears in the canalizing nesting.

            Parameters
            ----------
            segments_only : bool, optional
                If ``False`` (default), canalizing input sets S can be any nonempty
                proper subset. Corresponds to the Murrugarra--Dimitrova NCF definition.
                If ``True``, S is restricted to segments. Corresponds to the
                Remy--Ruet NC definition.

            Returns
            -------
            bool
                ``True`` if the multistate function is nested canalizing.

            References
            ----------
            Murrugarra, D., & Dimitrova, E. S. (2013).
                Quantifying the Connectivity of a Network and Its Application
                to the Analysis of the Phenomenon of Canalization.

            Remy, E., & Ruet, P. (2008).
                From Minimal Signed Circuits to the Dynamics of Boolean Regulatory
                Networks.
            """
            
            return self.is_k_canalizing(self.n, segments_only=segments_only)
            

        def is_wnc(self) -> bool:
            """
            Determine whether the multistate function is weakly nested canalizing (WNC).

            The class of weakly nested canalizing functions is defined inductively
            on the size of the domain |Ω| = prod(r_inputs):
            - If |Ω| == 1, any function is WNC (base case).
            - If |Ω| > 1, f is WNC if it is weakly canalizing with respect to
                some variable x_i and value a (i.e. f(x) = b whenever x_i = a,
                with no condition on other rows), the canalizing input set must be
                a segment, and the restriction of f to x_i ≠ a is WNC on the
                strictly smaller domain.

            Unlike is_canalizing, constant functions ARE weakly canalizing and
            therefore WNC. No non-constancy condition is imposed on g.

            Returns
            -------
            bool
                ``True`` if the multistate function is weakly nested canalizing.

            References
            ----------
            Remy, E., & Ruet, P. (2008).
                From Minimal Signed Circuits to the Dynamics of Boolean Regulatory
                Networks.
            """
            # BASE CASE:
            # if np.prod(self.r_inputs) == 1:   (domain has exactly one point)
            #     return True
            #
            # INDUCTIVE STEP:
            # for each variable i in range(self.n):
            #     let r_i = self.r_inputs[i]
            #     if r_i < 2:
            #         continue                   (need at least 2 states to restrict)
            #
            #     for each segment S of {0, ..., r_i - 1}:
            #         (segments are sets of the form {0,..,j} or {j,..,r_i-1})
            #
            #         build a boolean mask over all len(self.f) rows:
            #             mask[row] = True if dec2mix(row, self.r_inputs)[i] ∈ S
            #
            #         f_S = self.f[mask]
            #
            #         if f_S is all the same value b:    (weakly canalizing — no
            #                                             condition on f_not_S)
            #             f_not_S  = self.f[~mask]
            #             r_inputs_new = self.r_inputs with entry i removed
            #                            BUT r_inputs_new[i] reduced by |S|
            #                            (domain of x_i shrinks to r_i - |S| states)
            #
            #             sub = self.__class__(f_not_S.tolist(), self.r, r_inputs_new.tolist())
            #             if sub.is_wnc():
            #                 return True         (found a valid inductive decomposition)
            #
            # return False
            pass

        def _get_layer_structure(
            self,
            can_inputs,
            can_outputs,
            can_order,
            variables,
            r_inputs_current,
            depth,
            number_layers
        ):
            """
            Internal recursive routine for computing the canalizing layer structure.

            This method identifies all canalizing variables at the current recursion
            level, removes them simultaneously, and recurses on the resulting
            subfunction.

            Parameters
            ----------
            can_inputs : list
                Accumulated canalizing input sets (one frozenset per variable).
            can_outputs : list
                Accumulated canalized output values.
            can_order : list
                Accumulated order of canalizing variables (original indices).
            variables : list[int]
                Original indices of variables remaining in the current subfunction.
            r_inputs_current : np.ndarray
                Radices of the variables remaining in the current subfunction.
                Passed explicitly because self.r_inputs always reflects the full
                original function, not the reduced subfunction at each recursion step.
            depth : int
                Current canalizing depth.
            number_layers : int
                Current number of identified canalizing layers.

            Returns
            -------
            tuple
                A tuple containing the updated canalizing depth, number of layers,
                canalizing inputs, canalized outputs, core multistate function, and
                canalizing variable order.

            Notes
            -----
            r_inputs_current must be passed explicitly at each recursion step
            because self.r_inputs always reflects the full original function.
            In the Boolean version this was unnecessary since all variables
            always have radix 2 and the subfunction size is inferred from len(f).
            Here different variables can have different radices, so we must
            track which radices remain after each peeling step.
            """
            # BASE CASE:
            # if np.all(self.f == self.f[0]):
            #     return (depth, number_layers, can_inputs, can_outputs,
            #             self, can_order)
            #
            # if variables is empty:
            #     variables = list(range(len(r_inputs_current)))
            #
            # FIND ALL CANALIZING VARIABLES AT THIS LEVEL (one layer):
            # new_canalizing_vars = []   ← original indices of canalizing variables
            # new_can_inputs      = []   ← canalizing input sets (frozensets)
            # new_can_outputs     = []   ← canalized output values
            #
            # for each local index i in range(len(r_inputs_current)):
            #     let r_i = r_inputs_current[i]
            #
            #     for each nonempty proper subset S of {0, ..., r_i - 1}:
            #         (try size-1 subsets first, then size-2, etc.)
            #
            #         build a boolean mask over all len(self.f) rows:
            #             mask[row] = True if dec2mix(row, r_inputs_current)[i] ∈ S
            #
            #         f_S     = self.f[mask]
            #         f_not_S = self.f[~mask]
            #
            #         if f_S is all the same value b
            #         AND f_not_S is NOT all equal to b:
            #             new_canalizing_vars.append(variables[i])
            #             new_can_inputs.append(frozenset(S))
            #             new_can_outputs.append(b)
            #             break out of subset search for this variable
            #
            # NO CANALIZING VARIABLES FOUND:
            # if new_canalizing_vars is empty:
            #     return (depth, number_layers, can_inputs, can_outputs,
            #             self, can_order)
            #
            # BUILD THE RESTRICTED SUBFUNCTION:
            # mask_keep = np.ones(len(self.f), dtype=bool)
            # for each (local index i, S) in this layer:
            #     set mask_keep to False for all rows where
            #         dec2mix(row, r_inputs_current)[i] ∈ S
            # new_f = self.f[mask_keep]
            #
            # REMOVE CANALIZING VARIABLES FROM r_inputs_current:
            # remaining_local    = local indices NOT in new_canalizing_vars
            # r_inputs_new       = r_inputs_current[remaining_local]
            # remaining_original = [variables[j] for j in remaining_local]
            #
            # RECURSE:
            # new_mf = self.__class__(new_f.tolist(), self.r, r_inputs_new.tolist())
            # return new_mf._get_layer_structure(
            #     can_inputs  + new_can_inputs,
            #     can_outputs + new_can_outputs,
            #     can_order   + new_canalizing_vars,
            #     remaining_original,
            #     r_inputs_new,
            #     depth        + len(new_canalizing_vars),
            #     number_layers + 1
            # )
            pass

        def get_layer_structure(self) -> dict:
            """
            Determine the canalizing layer structure of a multistate function.

            This method decomposes a multistate function into its canalizing layers
            by recursively identifying and removing canalizing variables. All
            variables that canalize the function at the same recursion step form
            one canalizing layer and are removed simultaneously.

            The decomposition yields the canalizing depth, the number of canalizing
            layers, the canalizing inputs and outputs, the order of canalizing
            variables, and the remaining non-canalizing core function.

            Returns
            -------
            dict
                Dictionary containing the canalizing layer structure with the
                following entries:

                - ``CanalizingDepth`` : int
                Total number of canalizing variables.

                - ``NumberOfLayers`` : int
                Number of distinct canalizing layers.

                - ``CanalizingInputs`` : list[frozenset]
                Canalizing input set for each canalizing variable.
                Each entry is a frozenset of values that trigger the output b.
                (In the Boolean case this was a single integer 0 or 1.)

                - ``CanalizedOutputs`` : list[int]
                Output value forced by each canalizing variable.

                - ``CoreFunction`` : MultistateFunction
                Core multistate function obtained after removing all canalizing
                variables.

                - ``OrderOfCanalizingVariables`` : list[int]
                Order in which canalizing variables are identified.

                - ``LayerStructure`` : list[int]
                Number of canalizing variables in each layer.

            Notes
            -----
            The result is cached in ``self.properties`` and recomputed only if the
            canalizing structure has not been computed previously.

            This method has exponential time complexity in ``n`` and is intended for
            smaller multistate functions.

            References
            ----------
            Murrugarra, D., & Dimitrova, E. S. (2013).
                Quantifying the Connectivity of a Network and Its Application
                to the Analysis of the Phenomenon of Canalization.
            """
            # if "CanalizingDepth" not in self.properties:
            #     dummy = dict(zip(
            #         ["CanalizingDepth", "NumberOfLayers", "CanalizingInputs",
            #          "CanalizedOutputs", "CoreFunction", "OrderOfCanalizingVariables"],
            #         self._get_layer_structure(
            #             can_inputs       = [],
            #             can_outputs      = [],
            #             can_order        = [],
            #             variables        = [],
            #             r_inputs_current = self.r_inputs.copy(),
            #             depth            = 0,
            #             number_layers    = 0
            #         )
            #     ))
            #     dummy.update({
            #         "LayerStructure": get_layer_structure_from_canalized_outputs(
            #             dummy["CanalizedOutputs"]
            #         )
            #     })
            #     self.properties.update(dummy)
            #     return dummy
            # else:
            #     return {key: self.properties[key] for key in [
            #         "CanalizingDepth", "NumberOfLayers", "CanalizingInputs",
            #         "CanalizedOutputs", "CoreFunction",
            #         "OrderOfCanalizingVariables", "LayerStructure"
            #     ]}
            pass

        def get_canalizing_depth(self) -> int:
            """
            Return the canalizing depth of the multistate function.

            The canalizing depth is the total number of canalizing variables
            identified in the canalizing layer decomposition.

            Returns
            -------
            int
                Canalizing depth of the multistate function.
            """
            # if "CanalizingDepth" not in self.properties:
            #     self.get_layer_structure()
            # return self.properties["CanalizingDepth"]
            pass

        