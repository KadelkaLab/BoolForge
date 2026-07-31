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


class BooleanFunctionCanalizationMixin:
    @property
    def canalizing(self) -> bool:
        """Check whether the Boolean function is degenerate."""
        return self.is_canalizing()

    @property
    def nested_canalizing(self) -> bool:
        """Check whether the Boolean function is nested canalizing."""
        return self.is_k_canalizing(self.n)

    @property
    def canalizing_depth(self) -> int:
        """Determine the canalizing depth of the Boolean function."""
        return self.get_canalizing_depth()
    
    @property
    def layer_structure(self):
        if "LayerStructure" not in self.properties:
            self.get_layer_structure()
        return self.properties["LayerStructure"]
    
    def is_canalizing(self) -> bool:
        """
        Determine whether the Boolean function is canalizing.
    
        A Boolean function is canalizing if there exists at least one variable
        and a value in ``{0,1}`` such that fixing that variable to the given
        value forces the output of the function to be constant.
    
        Returns
        -------
        bool
            ``True`` if the Boolean function is canalizing, ``False`` otherwise.
        """
        indices = np.arange(2**self.n, dtype=np.uint32)
    
        for i in range(self.n):
            mask = 1 << i
            bit_is_0 = (indices & mask) == 0
            bit_is_1 = ~bit_is_0
    
            f0 = self.f[bit_is_0]
            f1 = self.f[bit_is_1]
    
            if np.all(f0 == f0[0]) or np.all(f1 == f1[0]):
                return True
    
        return False


    def is_k_canalizing(self, k: int) -> bool:
        """
        Determine whether the Boolean function is k-canalizing.
    
        A Boolean function is k-canalizing if it has a sequence of at least
        ``k`` canalizing variables. After fixing a canalizing variable to its
        canalizing value, the resulting subfunction must itself be
        (k−1)-canalizing, recursively.
    
        Parameters
        ----------
        k : int
            Desired canalizing depth, with ``0 <= k <= n``. Every Boolean
            function is trivially 0-canalizing.
    
        Returns
        -------
        bool
            ``True`` if the Boolean function is k-canalizing, ``False`` otherwise.
    
        Notes
        -----
        This method has exponential time complexity in ``n`` and is intended for
        small Boolean functions.
    
        References
        ----------
        He, Q., & Macauley, M. (2016).
            Stratification and enumeration of Boolean functions by canalizing depth.
            *Physica D: Nonlinear Phenomena*, 314, 1–8.
    
        Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022).
            Revealing the canalizing structure of Boolean functions:
            Algorithms and applications.
            *Automatica*, 146, 110630.
        """
        if k > self.n:
            return False
        if k == 0:
            return True
        if np.all(self.f == self.f[0]):
            return False
    
        indices = np.arange(2**self.n, dtype=np.uint32)
    
        for i in range(self.n):
            mask = 1 << i
            bit_is_0 = (indices & mask) == 0
            bit_is_1 = ~bit_is_0
    
            f0, f1 = self.f[bit_is_0], self.f[bit_is_1]
    
            if np.all(f0 == f0[0]):
                return True if k == 1 else self.__class__(f1).is_k_canalizing(k - 1)
    
            if np.all(f1 == f1[0]):
                return True if k == 1 else self.__class__(f0).is_k_canalizing(k - 1)
    
        return False


    def _get_layer_structure(
        self,
        can_inputs,
        can_outputs,
        can_order,
        variables,
        depth,
        number_layers
    ):
        """
        Internal recursive routine for computing the canalizing layer structure.
    
        This method identifies all canalizing variables at the current recursion
        level using bitwise operations, removes them simultaneously, and recurses
        on the resulting core function.
    
        Parameters
        ----------
        can_inputs : np.ndarray
            Accumulated canalizing input values.
        can_outputs : np.ndarray
            Accumulated canalized output values.
        can_order : np.ndarray
            Accumulated order of canalizing variables.
        variables : list[int]
            Indices of variables remaining in the current subfunction.
        depth : int
            Current canalizing depth.
        number_layers : int
            Current number of identified canalizing layers.
    
        Returns
        -------
        tuple
            A tuple containing the updated canalizing depth, number of layers,
            canalizing inputs, canalized outputs, core Boolean function, and
            canalizing variable order.
        """

        # base cases
        if np.all(self.f == self.f[0]):
            # recursion ends when function becomes constant
            return depth, number_layers, can_inputs, can_outputs, self, can_order
    
        if not variables:
            variables = list(range(self.n))
        elif isinstance(variables, np.ndarray):
            variables = variables.tolist()
    
        indices = np.arange(2**self.n, dtype=np.uint32)
    
        # candidate canalizing variables (x_i, a)
        new_canalizing_vars = []
        new_can_inputs = []
        new_can_outputs = []
        new_f = None
    
        for i in range(self.n):
            mask = 1 << (self.n-1-i)
            bit0 = (indices & mask) == 0
            bit1 = ~bit0
            f0, f1 = self.f[bit0], self.f[bit1]
    
            # check both possible canalizing directions
            if np.all(f0 == f0[0]):
                new_canalizing_vars.append(variables[i])
                new_can_inputs.append(0)
                new_can_outputs.append(int(f0[0]))
            elif np.all(f1 == f1[0]):
                new_canalizing_vars.append(variables[i])
                new_can_inputs.append(1)
                new_can_outputs.append(int(f1[0]))
    
        if not new_canalizing_vars:
            # non-canalizing core function
            return depth, number_layers, can_inputs, can_outputs, self, can_order
    
        # reduce variable list (remove canalizing ones)
        indices_new_canalizing_vars = [i for i,v in enumerate(variables) if v in new_canalizing_vars]
        remaining_vars = [v for v in variables if v not in new_canalizing_vars]
    
        # build the restricted subfunction (“core function”)
        # start with all indices, then keep those where none of the canalizing
        # variables take their canalizing inputs
        mask_keep = np.ones(2**self.n, dtype=bool)
        for var, val in zip(indices_new_canalizing_vars, new_can_inputs):
            bitmask = (indices >> (self.n-1-var)) & 1
            mask_keep &= (bitmask != val)
        new_f = self.f[mask_keep]
    

        # recurse on reduced function
        new_bf = self.__class__(list(new_f))
        return new_bf._get_layer_structure(
            np.append(can_inputs, new_can_inputs),
            np.append(can_outputs, new_can_outputs),
            np.append(can_order, new_canalizing_vars),
            remaining_vars,
            depth + len(new_canalizing_vars),
            number_layers + 1,
        )


    def get_layer_structure(self) -> dict:
        """
        Determine the canalizing layer structure of a Boolean function.
        
        This method decomposes a Boolean function into its canalizing layers
        (standard monomial form) by recursively identifying and removing
        canalizing variables. All variables that canalize the function at the
        same recursion step form one canalizing layer and are removed
        simultaneously.
        
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
        
            - ``CanalizingInputs`` : np.ndarray  
              Canalizing input value for each canalizing variable.
        
            - ``CanalizedOutputs`` : np.ndarray  
              Output value forced by each canalizing variable.
        
            - ``CoreFunction`` : BooleanFunction  
              Core Boolean function obtained after removing all canalizing
              variables.
        
            - ``OrderOfCanalizingVariables`` : np.ndarray  
              Order in which canalizing variables are identified.
        
            - ``LayerStructure`` : np.ndarray  
              Number of canalizing variables in each layer.
        
        Notes
        -----
        The result is cached in ``self.properties`` and recomputed only if the
        canalizing structure has not been computed previously.
        
        Notes
        -----
        This method has exponential time complexity in ``n`` and is intended for
        smaller Boolean functions.
        
        References
        ----------
        He, Q., & Macauley, M. (2016).
            Stratification and enumeration of Boolean functions by canalizing depth.
            *Physica D: Nonlinear Phenomena*, 314, 1–8.
        
        Dimitrova, E., Stigler, B., Kadelka, C., & Murrugarra, D. (2022).
            Revealing the canalizing structure of Boolean functions:
            Algorithms and applications.
            *Automatica*, 146, 110630.
        """
        if "CanalizingDepth" not in self.properties:
            dummy = dict(zip(["CanalizingDepth", "NumberOfLayers", "CanalizingInputs", "CanalizedOutputs", "CoreFunction", "OrderOfCanalizingVariables"],
                                            self._get_layer_structure(can_inputs=np.array([], dtype=int), can_outputs=np.array([], dtype=int),
                                                                      can_order=np.array([], dtype=int), variables=[], depth=0, number_layers=0)))
            dummy.update({'LayerStructure': get_layer_structure_from_canalized_outputs(dummy["CanalizedOutputs"])})
            self.properties.update(dummy)
            return dummy
        else:
            return {key: self.properties[key] for key in ["CanalizingDepth", "NumberOfLayers", "CanalizingInputs", "CanalizedOutputs", "CoreFunction", "OrderOfCanalizingVariables",'LayerStructure']}


    def get_canalizing_depth(self) -> int:
        """
        Return the canalizing depth of the Boolean function.
    
        The canalizing depth is the total number of canalizing variables identified
        in the canalizing layer decomposition.
    
        Returns
        -------
        int
            Canalizing depth of the Boolean function.
        """
        if "CanalizingDepth" not in self.properties:
            self.get_layer_structure()
        return self.properties["CanalizingDepth"]


