#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections.abc import Sequence
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .core import BooleanFunction

from .. import utils


def display_truth_table(*functions : "BooleanFunction", labels : Sequence[str] | None = None) -> None:
    """
    Display the full truth table of one or more Boolean functions.

    Each row displays an input configuration ``(x1, ..., xn)`` together with
    the corresponding output values of the provided Boolean functions.

    Parameters
    ----------
    functions : BooleanFunction
        One or more BooleanFunction objects with the same number of input
        variables.
    labels : Sequence[str] or None, optional
        Column labels for the Boolean functions. If ``None`` (default), labels
        are generated automatically as ``f0, f1, ...``.

    Raises
    ------
    ValueError
        If no Boolean functions are provided, if the functions do not all have
        the same number of variables, or if the number of labels does not match
        the number of functions.

    Examples
    --------
    >>> f = BooleanFunction("(x1 & ~x2) | x3")
    >>> display_truth_table(f)
    """
    if not functions:
        raise ValueError("Please provide at least one BooleanFunction.")
    n = functions[0].n
    if any(f.n != n for f in functions):
        raise ValueError("All BooleanFunction objects must have the same number of variables.")
    f = functions[0]
    if isinstance(labels,str):
        labels = [labels]
    if labels is not None and len(labels)!=len(functions):
        raise ValueError("The number of labels (if provided) must equal the number of functions.")
        
    if np.all([np.all(f.variables == g.variables) for g in functions]):
        header = "\t".join([f.variables[i] for i in range(f.n)]) 
    else:
        header = "\t".join([f'x{i}' for i in range(f.n)]) 
    if labels is None:
        labels = [f"f{i}" for i in range(len(functions))]
    header += '\t|\t' + '\t'.join(labels)
    
    print(header)
    print("-" * len(header.expandtabs()))

    for inputs, outputs in zip(utils.get_left_side_of_truth_table(f.n), 
                               np.column_stack([f.f for f in functions])):
        inputs_str = "\t".join(map(str, inputs))
        outputs_str = "\t".join(map(str, outputs))
        print(f"{inputs_str}\t|\t{outputs_str}")


class BooleanFunctionConversionsMixin:
    def to_polynomial(self) -> str:
        """
        Convert the Boolean function to a polynomial representation.
    
        This method returns a polynomial representation of the Boolean function
        in non-reduced disjunctive normal form (DNF).
    
        Returns
        -------
        str
            Polynomial representation of the Boolean function in non-reduced DNF.
        """
        left_side_of_truth_table = utils.get_left_side_of_truth_table(self.n)
        num_values = 2 ** self.n
        text = []

        for i in range(num_values):
            if self.f[i]:
                monomial = ' * '.join(
                    [
                        v if entry == 1 else f'(1 - {v})'
                        for v, entry in zip(self.variables, left_side_of_truth_table[i])
                    ]
                )
                text.append(monomial)

        if text:
            return ' + '.join(text)
        return '0'

    def to_truth_table(
        self,
        return_output: bool = True,
        filename: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Return or save the full truth table of the Boolean function.
    
        The truth table is represented as a pandas DataFrame in which each row
        corresponds to an input configuration and the final column contains the
        output value of the Boolean function.
    
        Parameters
        ----------
        return_output : bool, optional
            Whether to return the truth table as a DataFrame. If ``False``, the
            truth table is only written to file when ``filename`` is provided.
            Default is ``True``.
        filename : str or None, optional
            File name (including extension) to which the truth table is saved.
            Supported formats are ``'csv'``, ``'xls'``, and ``'xlsx'``. If
            provided, the truth table is automatically written to disk.
    
        Returns
        -------
        pandas.DataFrame or None
            The full truth table if ``return_output=True``; otherwise ``None``.
    
        Notes
        -----
        The column names correspond to the input variable names followed by the
        function name if provided, or ``'f'`` otherwise. When saving to a file,
        the output format is determined by the file extension.
    
        Examples
        --------
        >>> f = BooleanFunction("(x1 & ~x2) | x3")
        >>> f.to_truth_table()
           x1  x2  x3  f
        0   0   0   0  0
        1   0   0   1  1
        2   0   1   0  0
        3   0   1   1  1
        4   1   0   0  1
        5   1   0   1  1
        6   1   1   0  0
        7   1   1   1  1
        """
        columns = np.append(self.variables, self.name if self.name != "" else "f")
        truth_table = pd.DataFrame(
            np.c_[utils.get_left_side_of_truth_table(self.n), self.f],
            columns=columns,
        )
    
        if filename is not None:
            ending = filename.split(".")[-1]
            if ending not in {"csv", "xls", "xlsx"}:
                raise ValueError("filename must end in 'csv', 'xls', or 'xlsx'")
            if ending == "csv":
                truth_table.to_csv(filename)
            else:
                truth_table.to_excel(filename)
    
        if return_output:
            return truth_table
        return None


    def to_logical(
        self,
        and_op: str = "&",
        or_op: str = "|",
        not_op: str = "!",
        minimize_expression: bool = True,
    ) -> str:
        """
        Convert the Boolean function to a logical expression.
    
        This method converts the Boolean function from its truth table
        representation to a logical expression. If the PyEDA package is
        available, the expression can optionally be minimized using the
        Espresso algorithm. Otherwise, a non-minimized expression is
        generated as a fallback.
    
        Parameters
        ----------
        and_op : str, optional
            String used to represent the logical AND operator. Default is ``"&"``.
        or_op : str, optional
            String used to represent the logical OR operator. Default is ``"|"``.
        not_op : str, optional
            String used to represent the logical NOT operator. Default is ``"!"``.
        minimize_expression : bool, optional
            Whether to minimize the logical expression using the Espresso
            algorithm (via PyEDA). If ``False``, the expression is returned
            in non-minimized disjunctive normal form. Default is ``True``.
    
        Returns
        -------
        str
            Logical expression representing the Boolean function.
    
        Notes
        -----
        If the PyEDA package is not installed, the method falls back to a
        non-minimized expression derived from the polynomial representation.
        """
        try:
            from pyeda.inter import exprvar, Or, And, espresso_exprs
            from pyeda.boolalg.expr import OrOp, AndOp, NotOp, Complement
            __LOADED_PyEDA__ = True
        except ModuleNotFoundError:
            __LOADED_PyEDA__ = False
            
        if __LOADED_PyEDA__:
            variables = [exprvar(str(var)) for var in self.variables]
            minterms = [i for i in range(2**self.n) if self.f[i]]
    
            terms = []
            for m in minterms:
                bits = [
                    variables[i]
                    if (m >> (self.n - 1 - i)) & 1
                    else ~variables[i]
                    for i in range(self.n)
                ]
                terms.append(And(*bits))
    
            func_expr = Or(*terms).to_dnf()
    
            if func_expr.is_zero():
                return "0"
    
            if minimize_expression:
                func_expr, = espresso_exprs(func_expr)
    
            def __pyeda_to_string__(e):
                if isinstance(e, OrOp):
                    return "(" + f"){or_op}(".join(
                        __pyeda_to_string__(arg) for arg in e.xs
                    ) + ")"
                elif isinstance(e, AndOp):
                    return and_op.join(__pyeda_to_string__(arg) for arg in e.xs)
                elif isinstance(e, NotOp):
                    return f"{not_op}({__pyeda_to_string__(e.x)})"
                elif isinstance(e, Complement):
                    return f"({not_op}{str(e)[1:]})"
                return str(e)
    
            return __pyeda_to_string__(func_expr)
        else:
            # Fallback without PyEDA
            return (
                self.to_polynomial()
                .replace(" * ", and_op)
                .replace(" + ", or_op)
                .replace("1 - ", not_op)
            )