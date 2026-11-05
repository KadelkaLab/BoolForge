#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import TYPE_CHECKING

from .. import utils

if TYPE_CHECKING:
    import cana.boolean_node


class FunctionInteroperabilityMixin:

    @classmethod
    def from_cana(
            cls,
            cana_BooleanNode: "cana.boolean_node.BooleanNode",
    ) -> "Function":
        """
        Construct a function from a CANA BooleanNode.

        Extracts the Boolean truth table from a CANA BooleanNode
        and returns an instance of the calling class.

        Parameters
        ----------
        cana_BooleanNode : cana.boolean_node.BooleanNode
            Boolean node from the CANA package.

        Returns
        -------
        Function
            Boolean function with the same truth table as the
            supplied CANA BooleanNode.
        """
        return cls(
            np.asarray(cana_BooleanNode.outputs, dtype=np.uint8)
        )

    def to_cana(self) -> "cana.boolean_node.BooleanNode":
        """
        Convert the Boolean function to a CANA BooleanNode.

        Currently supported only for Boolean functions.

        Returns
        -------
        cana.boolean_node.BooleanNode
            Equivalent CANA BooleanNode.

        Raises
        ------
        NotImplementedError
            If the function is not Boolean.
        ImportError
            If CANA is not installed.
        """
        if not self.is_boolean:
            raise NotImplementedError(
                "CANA interoperability is currently supported "
                "only for Boolean functions."
            )

        cana_boolean_node = utils._require_cana()

        return cana_boolean_node.BooleanNode(
            k=self.n,
            outputs=self.f,
        )

    def to_prime(self) -> list[list[dict]]:
        """
        Convert the Boolean function to a prime implicant representation.

        Returns
        -------
        list[list[dict]]
            Two lists containing input configurations producing
            outputs 0 and 1, respectively.

        Raises
        ------
        NotImplementedError
            If the function is not Boolean.
        """
        if not self.is_boolean:
            raise NotImplementedError(
                "Prime implicant conversion is currently supported "
                "only for Boolean functions."
            )

        combinations = [[], []]

        for inputs, output in zip(
            utils.get_left_side_of_truth_table(self.n),
            self.f,
        ):
            combinations[int(output)].append({
                str(k): int(v)
                for k, v in zip(self.variables, inputs)
            })

        return combinations
