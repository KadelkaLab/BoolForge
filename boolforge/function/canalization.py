#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections.abc import Sequence

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
    canalizing_depth = len(outputs)
    if canalizing_depth == 0:
        return []
    size_of_layer = 1
    layer_structure = []
    for i in range(1, canalizing_depth):
        if outputs[i] == outputs[i - 1]:
            size_of_layer += 1
        else:
            layer_structure.append(size_of_layer)
            size_of_layer = 1
    layer_structure.append(size_of_layer)
    return layer_structure

def _is_k_canalizing_recursive(
        f: np.ndarray,
        r_inputs: np.ndarray,
        k: int,
        segments_only: bool,
) -> bool:
    """
    Recursively test strong k-canalization.

    Operates directly on truth tables without constructing Function
    objects. Binary inputs use a simplified recursion.
    """
    if k == 0:
        return True

    if np.all(f == f.flat[0]):
        return False

    table = f.reshape(tuple(r_inputs))

    for i, r_i in enumerate(r_inputs):
        slices = np.moveaxis(table, i, 0)

        if r_i == 2:
            # Binary inputs: weak and strong canalization coincide.
            f0, f1 = slices[0], slices[1]

            if np.all(f0 == f0.flat[0]):
                if k == 1:
                    return True

                r_new = np.delete(r_inputs, i)
                if _is_k_canalizing_recursive(
                        f1.ravel(), r_new, k - 1, segments_only
                ):
                    return True

            if np.all(f1 == f1.flat[0]):
                if k == 1:
                    return True

                r_new = np.delete(r_inputs, i)
                if _is_k_canalizing_recursive(
                        f0.ravel(), r_new, k - 1, segments_only
                ):
                    return True

        else:
            # Multistate inputs: find a strong canalizing decomposition.
            for a in range(r_i):
                g = slices[a]

                # States whose slices differ from candidate residual g.
                differs = np.any(
                    slices != g,
                    axis=tuple(range(1, slices.ndim))
                )

                if not np.any(differs):
                    continue

                # All differing slices must be constant with
                # the same canalized output.
                canalized = slices[differs]

                if not np.all(canalized == canalized.flat[0]):
                    continue

                if segments_only:
                    indices = np.flatnonzero(differs)

                    # Canalizing set must be a lower or upper segment.
                    if (indices[-1] != len(indices) - 1
                            and indices[0] != r_i - len(indices)):
                        continue

                if k == 1:
                    return True

                r_new = np.delete(r_inputs, i)

                if _is_k_canalizing_recursive(
                        g.ravel(), r_new, k - 1, segments_only
                ):
                    return True

    return False



def _get_layer_structure_recursive(
        f: np.ndarray,
        r_inputs: np.ndarray,
        variables: list[int],
        can_inputs: list,
        can_outputs: list[int],
        can_order: list[int],
        number_layers: int,
        segments_only: bool = False,
):
    """
    Recursively identify and remove all strongly canalizing
    variables at each layer.

    Operates directly on truth tables and input cardinalities.
    """
    if np.all(f == f[0]):
        return (
            number_layers, can_inputs, can_outputs,
            f, r_inputs, variables, can_order
        )

    table = f.reshape(tuple(r_inputs))
    candidates = []

    for i, r_i in enumerate(r_inputs):
        slices = np.moveaxis(table, i, 0)

        if r_i == 2:
            # Fast path for binary inputs.
            f0, f1 = slices[0], slices[1]

            if np.all(f0 == f0.flat[0]):
                candidates.append((i, {0}, int(f0.flat[0]), 1))
            elif np.all(f1 == f1.flat[0]):
                candidates.append((i, {1}, int(f1.flat[0]), 0))

        else:
            # Strong multistate canalization.
            for a in range(r_i):
                g = slices[a]

                differs = np.any(
                    slices != g,
                    axis=tuple(range(1, slices.ndim))
                )

                if not np.any(differs):
                    continue

                canalized = slices[differs]

                if not np.all(canalized == canalized.flat[0]):
                    continue

                S = set(np.flatnonzero(differs).tolist())

                if segments_only:
                    indices = sorted(S)
                    if (indices != list(range(len(S)))
                            and indices != list(range(r_i - len(S), r_i))):
                        continue

                candidates.append((i, S, int(canalized.flat[0]), a))
                break

    if not candidates:
        return (
            number_layers, can_inputs, can_outputs,
            f, r_inputs, variables, can_order
        )

    # All distinct strongly canalizing variables have the
    # same canalized output, so they form one layer.
    index = [slice(None)] * len(r_inputs)
    removed = set()

    for i, S, b, a in candidates:
        index[i] = a
        removed.add(i)

        can_order.append(variables[i])
        can_outputs.append(b)
        can_inputs.append(
            next(iter(S)) if r_inputs[i] == 2 else S
        )

    # Restrict all canalizing variables simultaneously.
    new_f = np.asarray(table[tuple(index)]).reshape(-1)

    new_r_inputs = np.array([
        r for i, r in enumerate(r_inputs) if i not in removed
    ], dtype=int)

    new_variables = [
        v for i, v in enumerate(variables) if i not in removed
    ]

    return _get_layer_structure_recursive(
        new_f,
        new_r_inputs,
        new_variables,
        can_inputs,
        can_outputs,
        can_order,
        number_layers + 1,
        segments_only,
    )

class FunctionCanalizationMixin:
    @property
    def canalizing(self) -> bool:
        """Whether the function is canalizing under the default definition."""
        return self.is_canalizing()

    @property
    def nested_canalizing(self) -> bool:
        """Whether the function is nested canalizing under the strong definition."""
        return self.is_k_canalizing(self.n)

    @property
    def canalizing_depth(self) -> int:
        """Return the canalizing depth of the function."""
        return self.get_canalizing_depth()

    @property
    def layer_structure(self) -> list[int]:
        """Return the canalizing layer structure of the function."""
        if "LayerStructure" not in self.properties:
            self.get_layer_structure()
        return self.properties["LayerStructure"]

    def is_canalizing(
            self,
            *,
            strong: bool = False,
            constant_functions_canalizing: bool = True,
    ) -> bool:
        """
        Determine whether the function is canalizing.

        A nonconstant function is weakly canalizing if there exists an
        input variable and a nonempty proper subset of its states such
        that restricting the variable to that subset forces a constant
        output.

        It is strongly canalizing if, additionally, the function outside
        that subset is independent of the canalizing input.

        Constant functions are considered canalizing by convention when
        ``constant_functions_canalizing=True``, provided the function
        has at least one input variable.

        Parameters
        ----------
        strong : bool, optional
            If True, require strong canalization. Default is False.
        constant_functions_canalizing : bool, optional
            Whether constant functions are considered canalizing.
            Default is True.

        Returns
        -------
        bool
            Whether the function is canalizing.
        """
        if self.n == 0:
            return False

        if self.constant:
            return constant_functions_canalizing

        table = self.f.reshape(tuple(self.r_inputs))

        for i in range(self.n):
            slices = np.moveaxis(table, i, 0)

            if not strong or self.r_inputs[i] == 2:
                # A constant slice suffices for weak canalization.
                # For binary inputs, weak and strong canalization coincide.
                for a in range(self.r_inputs[i]):
                    sl = slices[a]
                    if np.all(sl == sl.flat[0]):
                        return True

            else:
                # Strong canalization: all slices must be either
                # constant with the same output b or identical
                # copies of a residual function g.
                for a in range(self.r_inputs[i]):
                    g = slices[a]

                    # Identify input states whose slices differ from g.
                    differs = np.any(
                        slices != g,
                        axis=tuple(range(1, slices.ndim))
                    )

                    if not np.any(differs):
                        continue

                    # The differing slices must all be constant
                    # and have the same output.
                    canalized = slices[differs]

                    if np.all(canalized == canalized.flat[0]):
                        return True

        return False



    def is_k_canalizing(
            self,
            k: int,
            *,
            segments_only: bool = False,
    ) -> bool:
        """
        Determine whether the function is k-canalizing.

        A function is k-canalizing if it admits at least k successive
        strongly canalizing variables. For each variable, a nonempty
        proper subset S of its states forces a constant output b,
        while the function outside S is independent of that variable.
        The residual function must be (k-1)-canalizing.

        Constant functions have canalizing depth zero.

        Parameters
        ----------
        k : int
            Desired canalizing depth, with 0 <= k <= n.
            Every function is trivially 0-canalizing.
        segments_only : bool, optional
            If True, canalizing sets must be lower or upper segments
            of the input state space. Default is False.

        Returns
        -------
        bool
            Whether the function is k-canalizing.
        """
        if k < 0 or k > self.n:
            return False
        if k == 0:
            return True
        if self.constant:
            return False

        return _is_k_canalizing_recursive(self.f, self.r_inputs, k, segments_only)


    def get_layer_structure(
            self,
            *,
            segments_only: bool = False,
    ) -> dict:
        """
        Determine the strong canalizing layer structure of the function.

        Identifies all strongly canalizing variables at each layer and
        removes them simultaneously. For multistate inputs, canalizing
        inputs are represented by sets of states.

        Parameters
        ----------
        segments_only : bool, optional
            If True, canalizing input sets must be lower or upper
            segments of the input state space. Default is False.

        Returns
        -------
        dict
            Dictionary containing the following entries:

            - ``CanalizingDepth`` : int
                Total number of canalizing variables.
            - ``NumberOfLayers`` : int
                Number of canalizing layers.
            - ``CanalizingInputs`` : np.ndarray | list[set]
                Canalizing input values (Boolean) or sets (multistate).
            - ``CanalizedOutputs`` : np.ndarray
                Canalized output values.
            - ``CoreFunction`` : Function
                Remaining non-canalizing core function.
            - ``OrderOfCanalizingVariables`` : np.ndarray
                Original indices of the canalizing variables.
            - ``LayerStructure`` : list[int]
                Number of variables in each canalizing layer.

        Notes
        -----
        Results are cached in ``self.properties``. The unrestricted
        decomposition uses the existing flat cache structure, while
        the segment-restricted decomposition is stored separately.
        """
        keys = (
            "CanalizingDepth",
            "NumberOfLayers",
            "CanalizingInputs",
            "CanalizedOutputs",
            "CoreFunction",
            "OrderOfCanalizingVariables",
            "LayerStructure",
        )

        # Return cached results when available.
        if segments_only:
            if "LayerDecompositionSegments" in self.properties:
                return self.properties["LayerDecompositionSegments"]
        else:
            if "CanalizingDepth" in self.properties:
                return {key: self.properties[key] for key in keys}

        # Compute the layer decomposition.
        (
            number_layers,
            can_inputs,
            can_outputs,
            core_f,
            core_r_inputs,
            remaining_variables,
            can_order,
        ) = _get_layer_structure_recursive(
            self.f,
            self.r_inputs,
            list(range(self.n)),
            [],
            [],
            [],
            0,
            segments_only,
        )

        # Construct the remaining core function.
        core = self.__class__._from_f_unchecked(
            core_f,
            variables=self.variables[remaining_variables],
            name=self.name,
            r=self.r,
            r_inputs=core_r_inputs,
        )

        # Assemble the result.
        result = {
            "CanalizingDepth": len(can_order),
            "NumberOfLayers": number_layers,
            "CanalizingInputs": (
                np.array(can_inputs, dtype=int)
                if self.is_boolean else can_inputs
            ),
            "CanalizedOutputs": np.array(can_outputs, dtype=int),
            "CoreFunction": core,
            "OrderOfCanalizingVariables": np.array(can_order, dtype=int),
            "LayerStructure": get_layer_structure_from_canalized_outputs(
                can_outputs
            ),
        }

        # Cache the result.
        if segments_only:
            self.properties["LayerDecompositionSegments"] = result
        else:
            self.properties.update(result)

        return result


    def get_canalizing_depth(
            self,
            *,
            segments_only: bool = False,
    ) -> int:
        """
        Return the (strong) canalizing depth of the function.

        The canalizing depth is the total number of variables identified
        in the (strong)canalizing layer decomposition.

        Parameters
        ----------
        segments_only : bool, optional
            If True, restrict canalizing input sets to segments.
            Default is False.

        Returns
        -------
        int
            Canalizing depth of the function.
        """
        if segments_only:
            return self.get_layer_structure(
                segments_only=True
            )["CanalizingDepth"]

        if "CanalizingDepth" not in self.properties:
            self.get_layer_structure()

        return self.properties["CanalizingDepth"]


