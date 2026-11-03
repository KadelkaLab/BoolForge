#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import math
import numpy as np
import warnings

from .. import utils

class BooleanFunctionCollectiveCanalizationMixin:
    def is_kset_canalizing(self, k: int) -> bool:
        """
        Determine whether the Boolean function is k-set canalizing.
    
        A Boolean function is k-set canalizing if there exists a set of ``k``
        variables such that fixing these variables to specific values forces
        the output of the function, regardless of the remaining ``n - k``
        variables.
    
        Parameters
        ----------
        k : int
            Size of the variable set, with ``0 <= k <= n``.
    
        Returns
        -------
        bool
            ``True`` if the Boolean function is k-set canalizing, ``False`` otherwise.
    
        Notes
        -----
        This method has exponential time complexity in ``n`` and is intended for
        small Boolean functions.
    
        References
        ----------
        Kadelka, C., Keilty, B., & Laubenbacher, R. (2023).
            Collectively canalizing Boolean functions.
            *Advances in Applied Mathematics*, 145, 102475.
        """
        return self.get_kset_canalizing_proportion(k) > 0


    def get_kset_canalizing_proportion(self, k : int) -> float:
        """
        Compute the proportion of k-set canalizing input sets.
    
        For a given ``k``, this method computes the probability that a randomly
        chosen set of ``k`` variables canalizes the function, i.e., fixing those
        variables to some values forces the output regardless of the remaining
        variables.
    
        Parameters
        ----------
        k : int
            Size of the variable set, with ``0 <= k <= n``.
    
        Returns
        -------
        float
            Proportion of k-set canalizing input sets.
            
        Notes
        -----
        This method has exponential time complexity in ``n`` and is intended for
        small Boolean functions.
    
        References
        ----------
        Kadelka, C., Keilty, B., & Laubenbacher, R. (2023).
            Collectively canalizing Boolean functions.
            *Advances in Applied Mathematics*, 145, 102475.
        """
        if not (type(k)==int and 0<=k<=self.n):
            raise ValueError("k must be an integer and satisfy 0 <= k <= degree n")
        
        # trivial case
        if k == 0:
            return float(self.is_constant())
        
        # precompute binary representation of all inputs
        #indices = np.arange(2**self.n, dtype=np.uint32)
        #bits = ((indices[:, None] >> np.arange(self.n)) & 1).astype(np.uint8)  # shape (2**n, n)
        left_side_of_truth_table = utils.get_left_side_of_truth_table(self.n)
        
        total_tests = 0
        canalizing_hits = 0
    
        # iterate over variable subsets of size k
        for subset in itertools.combinations(range(self.n), k):
            Xsub = left_side_of_truth_table[:, subset]  # shape (2**n, k)
            # For each possible assignment to this subset
            for assignment in itertools.product([0, 1], repeat=k):
                mask = np.all(Xsub == assignment, axis=1)
                if not np.any(mask):
                    continue
                # If all outputs equal when these vars are fixed → canalizing
                f_sub = self.f[mask]
                if np.all(f_sub == f_sub[0]):
                    canalizing_hits += 1
                total_tests += 1
    
        return canalizing_hits / total_tests if total_tests > 0 else 0.0


    def get_kset_canalizing_proportion_of_variables(self, k : int) -> float:
        """
        Compute the proportion of k-set canalizing input sets per variable.
    
        For a given ``k``, this method computes, for each variable, the proportion
        of k-variable input sets containing that variable which canalize the
        Boolean function.
    
        Parameters
        ----------
        k : int
            Size of the variable set, with ``0 <= k <= n``.
    
        Returns
        -------
        np.ndarray
            Array of length ``n`` giving the proportion of k-set canalizing input
            sets containing each variable.
            
        Notes
        -----
        This method has exponential time complexity in ``n`` and is intended for
        small Boolean functions.
    
        References
        ----------
        Kadelka, C., Keilty, B., & Laubenbacher, R. (2023).
            Collectively canalizing Boolean functions.
            *Advances in Applied Mathematics*, 145, 102475.
        """
        if not (type(k)==int and 0<=k<=self.n):
            raise ValueError("k must be an integer and satisfy 0 <= k <= degree n")
        
        # trivial case
        if k == 0:
            return float(self.is_constant())
        
        # precompute binary representation of all inputs
        #indices = np.arange(2**self.n, dtype=np.uint32)
        #bits = ((indices[:, None] >> np.arange(self.n)) & 1).astype(np.uint8)  # shape (2**n, n)
        left_side_of_truth_table = utils.get_left_side_of_truth_table(self.n)
        
        canalizing_hits = np.zeros(self.n,dtype=np.float64)
        
        # iterate over variable subsets of size k
        for subset in itertools.combinations(range(self.n), k):
            Xsub = left_side_of_truth_table[:, subset]  # shape (2**n, k)
            subset = np.array(subset)
            # For each possible assignment to this subset
            for assignment in itertools.product([0, 1], repeat=k):
                mask = np.all(Xsub == assignment, axis=1)
                if not np.any(mask):
                    continue
                # If all outputs equal when these vars are fixed → canalizing
                f_sub = self.f[mask]
                if np.all(f_sub == f_sub[0]):
                    canalizing_hits[subset] += 1
    
        return canalizing_hits / (k/self.n * math.comb(self.n,k) * 2**k)


    def get_canalizing_strength(self) -> tuple:
        """
        Compute the canalizing strength of the Boolean function.
    
        The canalizing strength is defined as a weighted average of the proportions
        of k-set canalizing inputs for ``k = 1, ..., n-1``. It equals 0 for minimally
        canalizing functions (e.g., parity functions) and 1 for maximally canalizing
        functions (e.g., nested canalizing functions with a single layer).
    
        Returns
        -------
        float
            Canalizing strength of the Boolean function.

        Notes
        -----
        This method has exponential time complexity in ``n`` and is intended for
        small Boolean functions.

        References
        ----------
        Kadelka, C., Keilty, B., & Laubenbacher, R. (2023).
            Collectively canalizing Boolean functions.
            *Advances in Applied Mathematics*, 145, 102475.
        """

        if self.n==1:
            warnings.warn(
                "Canalizing strength is only defined for Boolean functions with n > 1 inputs. "
                "Returning 1 for n == 1.",
                RuntimeWarning
            )
            return 1.0
        res = []
        for k in range(1, self.n):
            res.append(self.get_kset_canalizing_proportion(k))
        return float(np.mean(np.multiply(res, 2**np.arange(1, self.n) / (2**np.arange(1, self.n) - 1))))


    def get_canalizing_strength_of_variables(self) -> np.ndarray:
        """
        Compute the canalizing strength of each variable.
    
        The canalizing strength of a variable is defined as a weighted average of
        the proportions of k-set canalizing inputs containing that variable for
        ``k = 1, ..., n-1``.
    
        Notes
        -----
        This method has exponential time complexity in ``n`` and is intended for
        small Boolean functions.

        Returns
        -------
        np.ndarray
            Array of length ``n`` containing the canalizing strength of each
            variable.
        """
        if self.n==1:
            warnings.warn(
                "Canalizing strength is only defined for Boolean functions with n > 1 inputs. "
                "Returning 1 for n == 1.",
                RuntimeWarning
            )
            return np.ones(1,dtype=np.float64)
        res = np.zeros((self.n-1,self.n))
        for k in range(1, self.n):
            res[k-1] = self.get_kset_canalizing_proportion_of_variables(k)
        multipliers = 2**np.arange(1, self.n) / (2**np.arange(1, self.n) - 1)
        return np.mean(res * multipliers[:, np.newaxis], axis = 0)
        
    def get_input_redundancy(self) -> float:
        """
        Compute the input redundancy of the Boolean function.
    
        Input redundancy quantifies the fraction of inputs that are not required
        to determine the output. Constant functions have redundancy 1, whereas
        parity functions have redundancy 0.
    
        Returns
        -------
        float
            Normalized input redundancy in the interval ``[0, 1]``.
    
        Raises
        ------
        ImportError
            If the CANA package is not installed.
    
        Notes
        -----
        This method has exponential time complexity in ``n`` and is intended for
        small Boolean functions.

        References
        ----------
        Marques-Pita, M., & Rocha, L. M. (2013).
            Canalization and control in automata networks: body segmentation in
            *Drosophila melanogaster*. *PLoS One*, 8(3), e55946.
    
        Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018).
            CANA: a python package for quantifying control and canalization in
            Boolean networks. *Frontiers in Physiology*, 9, 1046.
        """
        utils._require_cana()
        return self.to_cana().input_redundancy()
        
    def get_edge_effectiveness(self) -> list[float]:
        """
        Compute the edge effectiveness of each input variable.
    
        Edge effectiveness measures how strongly flipping an input variable
        influences the output. Non-essential inputs have effectiveness 0, whereas
        inputs that always flip the output have effectiveness 1.
    
        Returns
        -------
        list[float]
            List of length ``n`` containing edge effectiveness values in
            ``[0, 1]``.
    
        Raises
        ------
        ImportError
            If the CANA package is not installed.

        Notes
        -----
        This method has exponential time complexity in ``n`` and is intended for
        small Boolean functions.
    
        References
        ----------
        Marques-Pita, M., & Rocha, L. M. (2013).
            Canalization and control in automata networks: body segmentation in
            *Drosophila melanogaster*. *PLoS One*, 8(3), e55946.
    
        Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018).
            CANA: a python package for quantifying control and canalization in
            Boolean networks. *Frontiers in Physiology*, 9, 1046.
        """
        utils._require_cana()
        return self.to_cana().edge_effectiveness()
    
    def get_effective_degree(self) -> float:
        """
        Compute the effective degree of the Boolean function.
    
        The effective degree is defined as the sum of the edge effectiveness
        values of all input variables.
    
        Returns
        -------
        float
            Effective degree of the Boolean function.
    
        Raises
        ------
        ImportError
            If the CANA package is not installed.

        Notes
        -----
        This method has exponential time complexity in ``n`` and is intended for
        small Boolean functions.
        
        References
        ----------
        Marques-Pita, M., & Rocha, L. M. (2013).
            Canalization and control in automata networks: body segmentation in
            *Drosophila melanogaster*. *PLoS One*, 8(3), e55946.
    
        Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018).
            CANA: a python package for quantifying control and canalization in
            Boolean networks. *Frontiers in Physiology*, 9, 1046.
        """
        utils._require_cana()
        return float(sum(self.get_edge_effectiveness()))



