#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections.abc import Sequence
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .core import Function

from .. import utils



def display_truth_table(
        *functions: "Function",
        labels: Sequence[str] | None = None,
) -> None:
    """
    Display the truth tables of one or more functions side by side.

    All functions must have identical input cardinalities.
    """
    if not functions:
        raise ValueError("Please provide at least one Function.")

    f = functions[0]

    if any(
        not np.array_equal(g.r_inputs, f.r_inputs)
        for g in functions[1:]
    ):
        raise ValueError(
            "All functions must have identical input cardinalities."
        )

    if isinstance(labels, str):
        labels = [labels]

    if labels is None:
        labels = [f"f{i}" for i in range(len(functions))]

    if len(labels) != len(functions):
        raise ValueError(
            "The number of labels must equal the number of functions."
        )

    table = f.to_truth_table()

    if not all(
        np.array_equal(f.variables, g.variables)
        for g in functions[1:]
    ):
        table.columns = [
            *(f"x{i}" for i in range(f.n)),
            table.columns[-1],
        ]

    table = table.iloc[:, :f.n].copy()

    for label, g in zip(labels, functions):
        table[label] = g.f

    print(table.to_string(index=False))


class FunctionConversionsMixin:
    def to_truth_table(
            self,
            return_output: bool = True,
            filename: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Return or save the full truth table of the function.

        Each row corresponds to an input configuration, with the final
        column containing the function output. Supports Boolean and
        multistate functions with heterogeneous input cardinalities.

        Parameters
        ----------
        return_output : bool, optional
            Whether to return the truth table as a DataFrame.
            Default is True.
        filename : str | None, optional
            Optional filename for saving the truth table.
            Supported extensions are csv, xls, and xlsx.

        Returns
        -------
        pd.DataFrame | None
            Full truth table if return_output=True, otherwise None.
        """
        columns = list(self.variables) + [self.name or "f"]

        inputs = np.array(
            list(itertools.product(*(range(r) for r in self.r_inputs))),
            dtype=np.uint8,
        ).reshape(len(self.f), self.n)

        truth_table = pd.DataFrame(
            np.column_stack((inputs, self.f)),
            columns=columns,
        )

        if filename is not None:
            extension = filename.rsplit(".", 1)[-1].lower()

            if extension == "csv":
                truth_table.to_csv(filename)
            elif extension in {"xls", "xlsx"}:
                truth_table.to_excel(filename)
            else:
                raise ValueError(
                    "filename must end in '.csv', '.xls', or '.xlsx'"
                )

        return truth_table if return_output else None


    def to_polynomial(self) -> str:
        """
        Convert the Boolean function to a polynomial representation.

        Returns the non-reduced Boolean polynomial obtained by
        summing the indicator monomials for all inputs with output 1.

        Currently implemented only for Boolean functions.

        Returns
        -------
        str
            Polynomial representation of the Boolean function.

        Raises
        ------
        NotImplementedError
            If the function is not Boolean.
        """
        if not self.is_boolean:
            raise NotImplementedError(
                "Polynomial conversion is currently implemented "
                "only for Boolean functions."
            )

        if self.n == 0:
            return str(int(self.f[0]))

        left_side = utils.get_left_side_of_truth_table(self.n)
        terms = []

        for inputs, output in zip(left_side, self.f):
            if output:
                term = " * ".join(
                    v if x == 1 else f"(1 - {v})"
                    for v, x in zip(self.variables, inputs)
                )
                terms.append(term)

        return " + ".join(terms) if terms else "0"


    def to_logical(
            self,
            and_op: str = "&",
            or_op: str = "|",
            not_op: str = "!",
            minimize_expression: bool = True,
    ) -> str:
        """
        Convert the Boolean function to a logical expression.

        If PyEDA is available and minimize_expression=True, the
        expression is minimized using the Espresso algorithm.
        Otherwise, a non-minimized disjunctive normal form is returned.

        Currently implemented only for Boolean functions.

        Parameters
        ----------
        and_op : str, optional
            Logical AND operator. Default is "&".
        or_op : str, optional
            Logical OR operator. Default is "|".
        not_op : str, optional
            Logical NOT operator. Default is "!".
        minimize_expression : bool, optional
            Whether to minimize the expression using PyEDA.
            Default is True.

        Returns
        -------
        str
            Logical expression representing the function.

        Raises
        ------
        NotImplementedError
            If the function is not Boolean.
        """
        if not self.is_boolean:
            raise NotImplementedError(
                "Logical expression conversion is currently "
                "implemented only for Boolean functions."
            )

        if self.n == 0 or self.constant:
            return str(int(self.f[0]))

        # Non-minimized expression, or fallback without PyEDA.
        def fallback():
            return (
                self.to_polynomial()
                .replace(" * ", and_op)
                .replace(" + ", or_op)
                .replace("1 - ", not_op)
            )

        if not minimize_expression:
            return fallback()

        try:
            from pyeda.inter import exprvar, Or, And, espresso_exprs
            from pyeda.boolalg.expr import OrOp, AndOp, NotOp, Complement
        except ImportError:
            return fallback()

        variables = [exprvar(str(var)) for var in self.variables]
        minterms = np.flatnonzero(self.f)

        terms = []
        for m in minterms:
            bits = [
                variables[i]
                if (int(m) >> (self.n - 1 - i)) & 1
                else ~variables[i]
                for i in range(self.n)
            ]
            terms.append(And(*bits))

        func_expr = Or(*terms).to_dnf()

        if func_expr.is_zero():
            return "0"

        if minimize_expression:
            func_expr, = espresso_exprs(func_expr)

        def pyeda_to_string(e):
            if isinstance(e, OrOp):
                return "(" + f"){or_op}(".join(
                    pyeda_to_string(arg) for arg in e.xs
                ) + ")"
            elif isinstance(e, AndOp):
                return and_op.join(pyeda_to_string(arg) for arg in e.xs)
            elif isinstance(e, NotOp):
                return f"{not_op}({pyeda_to_string(e.x)})"
            elif isinstance(e, Complement):
                return f"({not_op}{str(e)[1:]})"
            return str(e)

        return pyeda_to_string(func_expr)