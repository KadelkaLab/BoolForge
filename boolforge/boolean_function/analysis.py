#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from .. import utils
from ..backend._numba import __LOADED_NUMBA__
if __LOADED_NUMBA__:
    from ..backend.function_analysis import _is_degenerate_numba
    from ..backend.function_analysis import _get_essential_variables_numba

class BooleanFunctionAnalysisMixin:
    @property
    def hamming_weight(self) -> int:
        """Number of ones in the truth table."""
        return self.get_hamming_weight()
    
    @property
    def bias(self) -> float:
        """Fraction of ones in the truth table."""
        return self.hamming_weight / len(self.f)
    
    @property
    def absolute_bias(self) -> float:
        """Absolute bias: `|2*bias − 1|`."""
        return 2 * abs(self.bias - 0.5)

    @property
    def constant(self) -> bool:
        """Check whether the Boolean function is constant."""
        return self.is_constant()

    @property
    def degenerate(self) -> bool:
        """Check whether the Boolean function is degenerate."""
        return self.is_constant()

    @property
    def essential_degree(self) -> int:
        """Determine the number of essential variables of the Boolean function."""
        return self.get_number_of_essential_variables()
    
    @property
    def essential_variables(self) -> np.ndarray:
        """Determine the array of indices of essential variables."""
        return self.get_essential_variables()

    @property
    def input_types(self) -> np.ndarray:
        """Determine the type of each input of the Boolean function."""
        return self.get_type_of_inputs()
          
    def is_constant(self) -> bool:
        """
        Check whether the Boolean function is constant.
    
        A Boolean function is constant if all entries of its truth table are
        identical (all 0 or all 1).
    
        Returns
        -------
        bool
            ``True`` if the Boolean function is constant, ``False`` otherwise.
        """
        return bool(np.all(self.f == self.f[0]))
        
    def is_degenerate(self, use_numba: bool = True) -> bool:
        """
        Determine whether the Boolean function is degenerate.
    
        A Boolean function is degenerate if it contains at least one
        non-essential variable, i.e., a variable on which the function's
        output does not depend.
    
        Parameters
        ----------
        use_numba : bool, optional
            Whether to use Numba-accelerated computation when available.
            Default is ``True``.
    
        Returns
        -------
        bool
            ``True`` if the Boolean function contains at least one
            non-essential variable, ``False`` if all variables are essential.
        """
        if __LOADED_NUMBA__ and use_numba:
            return bool(_is_degenerate_numba(self.f, self.n))
        else:
            for i in range(self.n):
                dummy_add = 2 ** (self.n - 1 - i)
                dummy = np.arange(2**self.n) % (2 ** (self.n - i)) // dummy_add
                depends_on_i = False
                for j in range(2**self.n):
                    if dummy[j] == 1:
                        continue
                    else:
                        if self.f[j] != self.f[j + dummy_add]:
                            depends_on_i = True
                            break
                if not depends_on_i:
                    return True
            return False
        
    def is_monotonic(self) -> bool:
        """
        Determine whether the Boolean function is monotonic.
    
        A Boolean function is monotonic if it is monotonic in each variable,
        i.e., for every variable the function is either non-decreasing or
        non-increasing with respect to that variable.
    
        Returns
        -------
        bool
            ``True`` if the Boolean function is monotonic, ``False`` otherwise.
        """
        return "conditional" not in self.get_type_of_inputs()
    

    def get_hamming_weight(self) -> int:
        """
        Compute the Hamming weight of the Boolean function.
    
        The Hamming weight is the number of input states for which the function
        evaluates to ``1`` (i.e., the number of ones in the truth table).
    
        Returns
        -------
        int
            The Hamming weight of the Boolean function.
        """
        return int(self.f.sum())
        
    
    def get_essential_variables(
        self,
        as_dict: bool = False,
        use_numba: bool =True,
    ) -> dict[int, bool] | np.ndarray:
        """
        Identify essential variables of the Boolean function.
    
        A variable ``x_i`` is essential if there exists an assignment of the
        remaining variables such that flipping ``x_i`` changes the output of
        the function.
    
        Parameters
        ----------
        as_dict : bool, optional
            If True, return a dictionary mapping variable indices to booleans.
            If False (default), return an array of indices of essential variables.
        use_numba : bool, optional
            Whether to use Numba-accelerated computation when available.
            Default is ``True``.
    
        Returns
        -------
        dict[int, bool] or np.ndarray
            If ``as_dict`` is True, a dictionary indicating which variables are
            essential.
            If ``as_dict`` is False, an array of indices of essential variables.
        """
        if __LOADED_NUMBA__ and use_numba:
            is_essential = _get_essential_variables_numba(self.f,self.n)
        else:
            if len(self.f) == 0:
                is_essential = np.zeros(self.n, dtype=bool)
            else:
                is_essential = np.zeros(self.n, dtype=bool)
        
                for i in range(self.n):
                    step = 2 ** (self.n - i - 1)
        
                    for start in range(0, 2**self.n, 2 * step):
                        block0 = self.f[start : start + step]
                        block1 = self.f[start + step : start + 2 * step]
        
                        if np.any(block0 != block1):
                            is_essential[i] = True
                            break
    
        if as_dict:
            return dict(enumerate(is_essential.tolist()))
    
        return np.where(is_essential)[0]


    def get_number_of_essential_variables(self) -> int:
        """
        Count the number of essential variables of the Boolean function.
    
        Returns
        -------
        int
            The number of essential variables.
        """
        return len(self.get_essential_variables())
    
    
    def get_type_of_inputs(self) -> np.ndarray:
        """
        Classify each input variable of the Boolean function.
    
        Each variable is classified as one of:
    
        - ``'non-essential'``: flipping the variable never changes the output
        - ``'positive'``: flipping the variable from 0 to 1 never decreases the output
        - ``'negative'``: flipping the variable from 0 to 1 never increases the output
        - ``'conditional'``: flipping the variable can both increase and decrease the output
    
        The result is cached in ``self.properties['InputTypes']``.
    
        Returns
        -------
        np.ndarray
            Array of shape ``(n,)`` with dtype ``str`` giving the type of each input
            variable.
        """

        if 'InputTypes' in self.properties:
            return self.properties['InputTypes']
    
        f = np.asarray(self.f, dtype=np.int8)
        n = self.n
    
        types = np.empty(n, dtype=object)
    
        # Compute all pairwise differences for each bit position simultaneously # Each variable toggles every 2**i entries in the truth table. 
        for i in range(n): 
            period = 2 ** (i + 1) 
            half = period // 2 
            
            # Vectorized reshape pattern: consecutive blocks of 0...1 transitions 
            f_reshaped = f.reshape(-1, period) 
            diff = f_reshaped[:, half:] - f_reshaped[:, :half] 
            min_diff = diff.min() 
            max_diff = diff.max() 
            if min_diff == 0 and max_diff == 0: 
                types[i] = 'non-essential' 
            elif min_diff == -1 and max_diff == 1: 
                types[i] = 'conditional' 
            elif min_diff >= 0 and max_diff == 1: 
                types[i] = 'positive' 
            elif min_diff == -1 and max_diff <= 0: 
                types[i] = 'negative'
                
        types = np.array(types, dtype=str)[::-1] #flip because of BoolForge logic ordering
        self.properties['InputTypes'] = types
        return types 
        
    
    def get_symmetry_groups(self) -> list[list[int]]:
        """
        Identify symmetry groups of input variables.
    
        Two variables belong to the same symmetry group if swapping their values
        leaves the Boolean function invariant for all assignments of the remaining
        variables.
    
        Returns
        -------
        list[list[int]]
            A list of symmetry groups, where each group is given by a list of
            variable indices.
            
        Notes
        -----
        This method has exponential time complexity in ``n`` and is intended for
        small Boolean functions.
        """
        symmetry_groups = []
        left_to_check = np.ones(self.n)
        for i in range(self.n):
            if left_to_check[i] == 0:
                continue
            else:
                symmetry_groups.append([i])
                left_to_check[i] = 0
            for j in range(i + 1, self.n):
                diff = sum(2**np.arange(self.n - i - 2, self.n - j - 2, -1))
                for ii, x in enumerate(utils.get_left_side_of_truth_table(self.n)):
                    if x[i] != x[j] and x[i] == 0 and self.f[ii] != self.f[ii + diff]:
                        break
                else:
                    left_to_check[j] = 0
                    symmetry_groups[-1].append(j)
        return symmetry_groups
    
    def get_absolute_bias(self) -> float:
        """
        Compute the absolute bias of the Boolean function.
    
        The absolute bias is defined as
    
        ``| (H / 2**(n-1)) - 1 |``,
    
        where ``H`` is the Hamming weight of the function. It measures how far the
        output distribution deviates from being perfectly balanced.
    
        Returns
        -------
        float
            The absolute bias of the Boolean function.
        """
        return float(abs(self.get_hamming_weight() * 1.0 / 2**(self.n - 1) - 1))


    def get_activities(
        self,
        nsim: int = 10000,
        exact: bool | None = None,
        *,
        rng=None
    ) -> np.ndarray:
        """
        Compute the activities of all input variables.
    
        The activity of a variable is the probability that flipping this variable
        (while keeping all others fixed) changes the output of the Boolean function.
    
        Activities can be computed exactly by enumerating all ``2**n`` input states
        or estimated via Monte Carlo sampling.
    
        Parameters
        ----------
        nsim : int, optional
            Number of random samples used when ``exact=False`` (default: 10000).
        exact : bool, optional
            If ``True``, compute activities exactly by enumerating all input states.
            If ``False``, estimate activities via sampling.
            If ``None`` (default), BoolForge automatically chooses the method
            (exact for small functions with n<16, sampling for larger functions).
        rng : None or numpy.random.Generator, optional
            Random number generator passed to ``utils._coerce_rng``.
    
        Returns
        -------
        np.ndarray
            Array of shape ``(n,)`` containing the activities of all variables.
        """
        if exact is None:
            exact = True if self.n <= 15 else False
        size_state_space = 2**self.n
        activities = np.zeros(self.n,dtype=np.float64)
        if exact:
            # Compute all integer representations of inputs (0 .. 2^n - 1)
            X = np.arange(size_state_space, dtype=np.uint32)
        
            # For each bit position i, flipping that bit corresponds to XOR with (1 << self.n-1-i)
            for i in range(self.n):
                flipped = X ^ (1 << self.n-1-i) 
                activities[i] = np.count_nonzero(self.f != self.f[flipped])
            return activities / size_state_space
        else:
            rng = utils._coerce_rng(rng)

            random_states = rng.integers(0,size_state_space,nsim) #
            for i in range(self.n):
                flipped_random_states = random_states ^ (1 << self.n-1-i) 
                activities[i] = np.count_nonzero(self.f[random_states] != self.f[flipped_random_states])
            return activities / nsim
    
    
    def get_average_sensitivity(
        self,
        nsim: int = 10000,
        exact: bool | None = None,
        normalized: bool = True,
        *,
        rng=None
    ) -> float:
        """
        Compute the average sensitivity of the Boolean function.
    
        The (unnormalized) average sensitivity equals the sum of the activities of
        all variables. If ``normalized=True``, the result is divided by ``n``.
    
        The sensitivity can be computed exactly by enumerating all input states or
        estimated via Monte Carlo sampling.
    
        Parameters
        ----------
        nsim : int, optional
            Number of random samples used when ``exact=False`` (default: 10000).
        exact : bool, optional
            If ``True``, compute the exact activities by enumerating all input states.
            If ``False``, estimate them via sampling (default: ``False``).
            If ``None`` (default), BoolForge automatically chooses the method
            (exact for small functions with n<16, sampling for larger functions).
        normalized : bool, optional
            If ``True`` (default), return the average sensitivity divided by ``n``.
            If ``False``, return the sum of activities.
        rng : None or numpy.random.Generator, optional
            Random number generator passed to ``utils._coerce_rng``.
    
        Returns
        -------
        float
            The (optionally normalized) average sensitivity of the Boolean function.
        """      
        if exact is None:
            exact = True if self.n <= 15 else False
        activities = self.get_activities(nsim,exact,rng=rng)
        s = sum(activities)
        if normalized:
            return float(s / self.n)
        else:
            return float(s)
