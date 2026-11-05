#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from .. import utils
from ..backend._numba import __LOADED_NUMBA__
if __LOADED_NUMBA__:
    from ..backend.function_analysis import _is_degenerate_numba
    from ..backend.function_analysis import _get_essential_variables_numba

class FunctionAnalysisMixin:
    @property
    def hamming_weight(self) -> int:
        """Return the Hamming weight of the Boolean function."""
        if not self.is_boolean:
            raise NotImplementedError(
                "Hamming weight is currently only supported for Boolean functions."
            )
        return self.get_hamming_weight()

    @property
    def bias(self) -> float:
        """Return the fraction of ones in the Boolean truth table."""
        if not self.is_boolean:
            raise NotImplementedError(
                "Bias is currently only supported for Boolean functions."
            )
        return self.hamming_weight / len(self.f)

    @property
    def absolute_bias(self) -> float:
        """Return the absolute Boolean bias, |2*bias - 1|."""
        if not self.is_boolean:
            raise NotImplementedError(
                "Absolute bias is currently only supported for Boolean functions."
            )
        return 2 * abs(self.bias - 0.5)

    @property
    def constant(self) -> bool:
        """Whether the function is constant."""
        return self.is_constant()

    @property
    def degenerate(self) -> bool:
        """Whether the function has at least one nonessential input."""
        return self.is_degenerate()

    @property
    def essential_degree(self) -> int:
        """Number of essential input variables."""
        return self.get_number_of_essential_variables()

    @property
    def essential_variables(self) -> np.ndarray:
        """Indices of essential input variables."""
        return self.get_essential_variables()

    @property
    def input_types(self) -> np.ndarray:
        """Classification of input variables."""
        return self.get_type_of_inputs()
          
    def is_constant(self) -> bool:
        """Return whether all truth table entries are identical."""
        return bool(np.all(self.f == self.f[0]))

    def is_degenerate(self, use_numba: bool = True) -> bool:
        """
        Determine whether the function is degenerate.

        A function is degenerate if at least one input variable
        is nonessential, meaning that changing its value never
        changes the output.

        Parameters
        ----------
        use_numba : bool, optional
            Whether to use Numba acceleration for Boolean functions
            when available. Default is True.

        Returns
        -------
        bool
            True if at least one input variable is nonessential.
        """
        if self.is_boolean and __LOADED_NUMBA__ and use_numba:
            return bool(_is_degenerate_numba(self.f, self.n))

        table = self.f.reshape(tuple(self.r_inputs))

        for i in range(self.n):
            slices = np.moveaxis(table, i, 0)

            if np.all(slices == slices[0]):
                return True

        return False


    def is_monotonic(self) -> bool:
        """
        Determine whether the function is monotonic in every input.

        A function is monotonic if it is either non-decreasing or
        non-increasing in each input variable, holding all other
        inputs fixed.

        Returns
        -------
        bool
            True if the function is monotonic, False otherwise.
        """
        return "conditional" not in self.get_type_of_inputs()


    def get_type_of_inputs(self) -> np.ndarray:
        """
        Classify each input variable of the function.

        Each input is classified as one of:

        - 'non-essential': Changing the input never changes the output.
        - 'positive': Increasing the input never decreases the output.
        - 'negative': Increasing the input never increases the output.
        - 'conditional': Increasing the input can both increase and
        decrease the output.

        The natural ordering of input and output states is used.
        The result is cached in ``self.properties['InputTypes']``.

        Returns
        -------
        np.ndarray
            Array of length n containing the input classifications.
        """
        if "InputTypes" in self.properties:
            return self.properties["InputTypes"]

        table = self.f.reshape(tuple(self.r_inputs))
        types = np.empty(self.n, dtype=object)

        for i in range(self.n):
            slices = np.moveaxis(table, i, 0)

            # Compare consecutive states of input i.
            lower = slices[:-1]
            upper = slices[1:]

            if np.all(lower == upper):
                types[i] = "non-essential"
            elif np.all(lower <= upper):
                types[i] = "positive"
            elif np.all(lower >= upper):
                types[i] = "negative"
            else:
                types[i] = "conditional"

        types = np.asarray(types, dtype=str)
        self.properties["InputTypes"] = types

        return types


    def get_hamming_weight(self) -> int:
        """
        Compute the Hamming weight of the Boolean function.

        The Hamming weight is the number of input states for which the
        function evaluates to 1 (i.e., the number of ones in its truth table).

        Currently implemented only for Boolean functions.

        Returns
        -------
        int
            Hamming weight of the Boolean function.

        Raises
        ------
        NotImplementedError
            If the function is not Boolean.
        """
        if not self.is_boolean:
            raise NotImplementedError(
                "Hamming weight is only implemented for Boolean functions."
            )

        return int(self.f.sum())
    

    def get_essential_variables(
            self,
            as_dict: bool = False,
            use_numba: bool = True,
    ) -> dict[int, bool] | np.ndarray:
        """
        Identify the essential input variables of the function.

        An input variable is essential if changing its state can
        change the function's output while all other inputs remain fixed.

        Parameters
        ----------
        as_dict : bool, optional
            If True, return a dictionary mapping variable indices to
            their essentiality. If False (default), return an array
            containing the indices of essential variables.
        use_numba : bool, optional
            Whether to use Numba acceleration for Boolean functions
            when available. Default is True.

        Returns
        -------
        dict[int, bool] | np.ndarray
            Essentiality of each variable or indices of essential variables.
        """
        if self.is_boolean and __LOADED_NUMBA__ and use_numba:
            is_essential = _get_essential_variables_numba(self.f, self.n)
        else:
            table = self.f.reshape(tuple(self.r_inputs))
            is_essential = np.zeros(self.n, dtype=bool)

            for i in range(self.n):
                slices = np.moveaxis(table, i, 0)

                is_essential[i] = not np.all(slices == slices[0])

        if as_dict:
            return dict(enumerate(is_essential.tolist()))

        return np.flatnonzero(is_essential)

    
    def get_number_of_essential_variables(self) -> int:
        """
        Return the number of essential variables of the function.
        """
        return len(self.get_essential_variables())
    


    def get_symmetry_groups(self) -> list[list[int]]:
        """
        Identify symmetry groups of input variables.

        Two variables are symmetric if exchanging their values leaves
        the function unchanged for every input configuration.

        Only variables with the same input cardinality can be symmetric.

        Returns
        -------
        list[list[int]]
            Symmetry groups of input variables, represented by their
            indices. Every variable belongs to exactly one group.
        """
        table = self.f.reshape(tuple(self.r_inputs))
        symmetry_groups = []
        assigned = np.zeros(self.n, dtype=bool)

        for i in range(self.n):
            if assigned[i]:
                continue

            group = [i]
            assigned[i] = True

            for j in range(i + 1, self.n):
                if assigned[j] or self.r_inputs[i] != self.r_inputs[j]:
                    continue

                if np.array_equal(table, np.swapaxes(table, i, j)):
                    group.append(j)
                    assigned[j] = True

            symmetry_groups.append(group)

        return symmetry_groups

    
    def get_symmetry_groups_old(self) -> list[list[int]]:
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

        Absolute bias is defined as |2*bias - 1|, where bias is
        the fraction of ones in the truth table.

        Returns
        -------
        float
            Absolute bias in [0, 1].
        """
        if not self.is_boolean:
            raise NotImplementedError(
                "Absolute bias is currently implemented only "
                "for Boolean functions."
            )

        return float(2 * abs(self.bias - 0.5))


    def get_activities(
            self,
            nsim: int = 10000,
            exact: bool | None = None,
            *,
            rng=None,
    ) -> np.ndarray:
        """
        Compute the activities of all input variables.

        The activity of a Boolean input is the probability that
        flipping its value changes the function's output.

        Activities can be computed exactly or estimated using
        Monte Carlo sampling.

        Parameters
        ----------
        nsim : int, optional
            Number of samples when exact=False. Default is 10000.
        exact : bool | None, optional
            If True, compute exact activities. If False, estimate
            them by sampling. If None, use exact computation for
            n <= 15 and sampling otherwise.
        rng : numpy.random.Generator | None, optional
            Random number generator.

        Returns
        -------
        np.ndarray
            Array of length n containing the activities.
        """
        if not self.is_boolean:
            raise NotImplementedError(
                "Activities are currently implemented only "
                "for Boolean functions."
            )

        if exact is None:
            exact = self.n <= 15

        size_state_space = 2**self.n
        activities = np.zeros(self.n, dtype=np.float64)

        if exact:
            X = np.arange(size_state_space, dtype=np.uint32)

            for i in range(self.n):
                flipped = X ^ (1 << (self.n - 1 - i))
                activities[i] = np.count_nonzero(
                    self.f != self.f[flipped]
                )

            return activities / size_state_space

        else:
            rng = utils._coerce_rng(rng)
            random_states = rng.integers(0, size_state_space, nsim)

            for i in range(self.n):
                flipped = random_states ^ (1 << (self.n - 1 - i))
                activities[i] = np.count_nonzero(
                    self.f[random_states] != self.f[flipped]
                )

            return activities / nsim


    def get_average_sensitivity(
            self,
            nsim: int = 10000,
            exact: bool | None = None,
            normalized: bool = True,
            *,
            rng=None,
    ) -> float:
        """
        Compute the average sensitivity of the Boolean function.

        The unnormalized average sensitivity is the sum of the
        activities of all input variables. If normalized=True,
        the result is divided by n.

        Parameters
        ----------
        nsim : int, optional
            Number of Monte Carlo samples. Default is 10000.
        exact : bool | None, optional
            Whether to compute exact activities or estimate them.
            If None, the method is chosen automatically.
        normalized : bool, optional
            If True (default), divide sensitivity by n.
        rng : numpy.random.Generator | None, optional
            Random number generator.

        Returns
        -------
        float
            Average sensitivity, optionally normalized by n.
        """
        if not self.is_boolean:
            raise NotImplementedError(
                "Average sensitivity is currently implemented only "
                "for Boolean functions."
            )

        if self.n == 0:
            return 0.0

        activities = self.get_activities(
            nsim=nsim,
            exact=exact,
            rng=rng,
        )

        sensitivity = float(np.sum(activities))

        return sensitivity / self.n if normalized else sensitivity
