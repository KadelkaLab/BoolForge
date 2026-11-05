#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import math
import numpy as np
import warnings

from .. import utils


import itertools
import math
import warnings

import numpy as np

class FunctionCollectiveCanalizationMixin:
    def _get_kset_canalizing_counts(self, k: int):
        """
        Count canalizing assignments to subsets of k variables.

        Returns
        -------
        int
            Total number of canalizing assignments.
        int
            Total number of assignments tested.
        np.ndarray
            Number of canalizing assignments containing each variable.
        np.ndarray
            Total number of assignments containing each variable.
        """
        if type(k) is not int or not 0 <= k <= self.n:
            raise ValueError(
                "k must be an integer satisfying 0 <= k <= n."
            )

        hits = 0
        total = 0
        hits_per_variable = np.zeros(self.n, dtype=np.int64)
        total_per_variable = np.zeros(self.n, dtype=np.int64)

        if k == 0:
            return int(self.constant), 1, hits_per_variable, total_per_variable

        table = self.f.reshape(tuple(self.r_inputs))

        for subset in itertools.combinations(range(self.n), k):
            # Move selected variables to the first k axes.
            remaining = tuple(i for i in range(self.n) if i not in subset)
            axes = subset + remaining

            # Each row represents one assignment to the k variables.
            restricted = table.transpose(axes).reshape(
                math.prod(int(self.r_inputs[i]) for i in subset),
                -1
            )

            canalizing = np.all(
                restricted == restricted[:, :1],
                axis=1
            )

            count = int(np.count_nonzero(canalizing))
            n_assignments = len(canalizing)

            hits += count
            total += n_assignments

            hits_per_variable[list(subset)] += count
            total_per_variable[list(subset)] += n_assignments

        return hits, total, hits_per_variable, total_per_variable

    def is_kset_canalizing(self, k: int) -> bool:
        """
        Determine whether some assignment to k input variables
        forces a constant output.
        """
        hits, _, _, _ = self._get_kset_canalizing_counts(k)
        return hits > 0

    def get_kset_canalizing_proportion(self, k: int) -> float:
        """
        Return the proportion of assignments to k-variable
        subsets that force a constant output.
        """
        hits, total, _, _ = self._get_kset_canalizing_counts(k)
        return hits / total

    def get_kset_canalizing_proportion_of_variables(
            self, k: int
    ) -> np.ndarray:
        """
        Return the proportion of canalizing assignments to
        k-variable subsets containing each variable.
        """
        _, _, hits, total = self._get_kset_canalizing_counts(k)

        return np.divide(
            hits,
            total,
            out=np.zeros(self.n, dtype=float),
            where=total > 0,
        )

    def get_canalizing_strength(self) -> float:
        """
        Compute Boolean canalizing strength.

        Multistate normalization is not yet implemented.
        """
        if not self.is_boolean:
            raise NotImplementedError(
                "Canalizing strength normalization for multistate "
                "functions is not yet implemented."
            )

        if self.n == 1:
            warnings.warn(
                "Canalizing strength is only defined for n > 1. "
                "Returning 1 for n == 1.",
                RuntimeWarning,
            )
            return 1.0

        if self.n == 0:
            raise ValueError(
                "Canalizing strength is undefined for n == 0."
            )

        proportions = np.array([
            self.get_kset_canalizing_proportion(k)
            for k in range(1, self.n)
        ])

        k = np.arange(1, self.n)
        weights = 2.0**k / (2.0**k - 1)

        return float(np.mean(proportions * weights))

    def get_canalizing_strength_of_variables(self) -> np.ndarray:
        """
        Compute Boolean canalizing strength for each variable.

        Multistate normalization is not yet implemented.
        """
        if not self.is_boolean:
            raise NotImplementedError(
                "Variable-specific canalizing strength for "
                "multistate functions is not yet implemented."
            )

        if self.n == 1:
            warnings.warn(
                "Canalizing strength is only defined for n > 1. "
                "Returning 1 for n == 1.",
                RuntimeWarning,
            )
            return np.ones(1, dtype=float)

        if self.n == 0:
            raise ValueError(
                "Canalizing strength is undefined for n == 0."
            )

        proportions = np.array([
            self.get_kset_canalizing_proportion_of_variables(k)
            for k in range(1, self.n)
        ])

        k = np.arange(1, self.n)
        weights = 2.0**k / (2.0**k - 1)

        return np.mean(proportions * weights[:, None], axis=0)


def get_input_redundancy(self) -> float:
    """
    Compute the input redundancy of the Boolean function.

    Input redundancy quantifies the fraction of inputs that are not
    required to determine the output. Constant functions have
    redundancy 1, whereas parity functions have redundancy 0.

    Currently implemented only for Boolean functions using CANA.

    Returns
    -------
    float
        Normalized input redundancy in [0, 1].

    Raises
    ------
    NotImplementedError
        If the function is not Boolean.
    ImportError
        If CANA is not installed.

    References
    ----------
    Marques-Pita, M., & Rocha, L. M. (2013).
        Canalization and control in automata networks: body segmentation in
        Drosophila melanogaster. PLoS One, 8(3), e55946.

    Correia, R. B., Gates, A. J., Wang, X., & Rocha, L. M. (2018).
        CANA: a python package for quantifying control and canalization
        in Boolean networks. Frontiers in Physiology, 9, 1046.
    """
    if not self.is_boolean:
        raise NotImplementedError(
            "Input redundancy is currently implemented only "
            "for Boolean functions."
        )

    utils._require_cana()
    return self.to_cana().input_redundancy()


def get_edge_effectiveness(self) -> list[float]:
    """
    Compute the edge effectiveness of each input variable.

    Edge effectiveness quantifies the contribution of each input
    to determining the output. Nonessential inputs have
    effectiveness 0, while maximally effective inputs have
    effectiveness 1.

    Currently implemented only for Boolean functions using CANA.

    Returns
    -------
    list[float]
        Edge effectiveness of each input variable.

    Raises
    ------
    NotImplementedError
        If the function is not Boolean.
    ImportError
        If CANA is not installed.
    """
    if not self.is_boolean:
        raise NotImplementedError(
            "Edge effectiveness is currently implemented only "
            "for Boolean functions."
        )

    utils._require_cana()
    return self.to_cana().edge_effectiveness()


def get_effective_degree(self) -> float:
    """
    Compute the effective degree of the Boolean function.

    Effective degree is the sum of the edge effectiveness
    values across all input variables.

    Currently implemented only for Boolean functions using CANA.

    Returns
    -------
    float
        Effective degree of the function.

    Raises
    ------
    NotImplementedError
        If the function is not Boolean.
    ImportError
        If CANA is not installed.
    """
    if not self.is_boolean:
        raise NotImplementedError(
            "Effective degree is currently implemented only "
            "for Boolean functions."
        )

    return float(sum(self.get_edge_effectiveness()))



