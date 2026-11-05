#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Sequence

from .. import utils, utils_multistate
    
from .analysis import FunctionAnalysisMixin
from .canalization import FunctionCanalizationMixin
from .collective_canalization import FunctionCollectiveCanalizationMixin
from .conversions import FunctionConversionsMixin
from .interoperability import FunctionInteroperabilityMixin
from .parsing import f_from_expression


class Function(
        FunctionAnalysisMixin,
        FunctionCanalizationMixin,
        FunctionCollectiveCanalizationMixin,
        FunctionConversionsMixin,
        FunctionInteroperabilityMixin,
        ):
    """
    A finite-state discrete function.

    This class represents a function
    :math:`f : X_1 \\times \\cdots \\times X_n \\to Y`,
    where each input domain and the output domain is a finite set
    of consecutive nonnegative integers. Boolean functions are the
    special case where all domains contain exactly two states.

    Parameters
    ----------
    f : list[int] | np.ndarray | str
        Either:

        - A truth table containing the function outputs in
          lexicographic order, with length equal to the product
          of the input cardinalities.

        - A Boolean expression string, supported for Boolean
          functions and parsed using
          ``function.parsing.f_from_expression``.

    variables : list[str] | np.ndarray | None, optional
        Names of the input variables, given in order. Must have
        length ``n``. If ``None`` (default), variables are named
        ``x0, ..., x_{n-1}``.

    name : str, optional
        Name of the node regulated by the function.
        Default is ``""``.

    r : int, optional
        Number of possible output states. Default is 2.

    r_inputs : list[int] | np.ndarray | None, optional
        Number of possible states for each input variable, given
        in the same order as ``variables``. If ``None`` (default),
        all inputs are assumed to have ``r`` possible states.

    Attributes
    ----------
    f : np.ndarray
        One-dimensional NumPy array containing the truth table
        of the function, with entries in ``{0, ..., r-1}``.
    n : int
        Number of input variables.
    variables : np.ndarray
        One-dimensional NumPy array of length ``n`` containing
        variable names.
    name : str
        Name of the node regulated by the function.
    r : int
        Number of possible output states.
    r_inputs : np.ndarray
        Number of possible states for each input variable.
    is_boolean : bool
        Whether all input and output cardinalities equal 2.
    properties : dict
        Dictionary for dynamically computed properties of the
        function (e.g., canalizing structure, effective inputs,
        robustness measures).
    """

    __slots__ = [
        'f', 'n', 'variables', 'name', 'properties',
        'r', 'r_inputs', 'is_boolean'
    ]


    def __init__(
            self,
            f: Sequence[int] | np.ndarray | str,
            variables: Sequence[str] | np.ndarray | None = None,
            name: str = "",
            r: int = 2,
            r_inputs: Sequence[int] | np.ndarray | None = None,
    ):
        """
        Initialize a finite-state discrete function.

        Parameters
        ----------
        f : Sequence[int] | np.ndarray | str
            Truth table in lexicographic order, or a Boolean expression string.
        variables : Sequence[str] | np.ndarray | None, optional
            Input variable names. Defaults to x0, x1, ...
        name : str, optional
            Name of the function. Default is "".
        r : int, optional
            Number of output states (2 to 255). Default is 2.
        r_inputs : Sequence[int] | np.ndarray | None, optional
            Number of states for each input (2 to 255).
            If None, all inputs have r states and their number is inferred
            from the truth table length.
        """
        self.name = name

        # Validate output cardinality.
        if (isinstance(r, (bool, np.bool_))
                or not isinstance(r, (int, np.integer))
                or not 2 <= r <= 255):
            raise ValueError("r must be an integer between 2 and 255")
        self.r = int(r)

        # Validate input cardinalities.
        if r_inputs is None:
            self.r_inputs = None
        else:
            raw_inputs = np.asarray(r_inputs)
            if (raw_inputs.ndim != 1
                    or (raw_inputs.size > 0 and raw_inputs.dtype.kind not in "iu")
                    or np.any(raw_inputs < 2)
                    or np.any(raw_inputs > 255)):
                raise ValueError(
                    "r_inputs must contain integers between 2 and 255"
                )
            self.r_inputs = raw_inputs.astype(int, copy=False)

        if isinstance(f, str):
            if self.r != 2 or (
                self.r_inputs is not None
                and not np.all(self.r_inputs == 2)
            ):
                raise NotImplementedError(
                    "Expression parsing is only supported for Boolean functions."
                )

            f, self.variables = f_from_expression(f)
            self.n = len(self.variables)
            self.r_inputs = np.full(self.n, 2, dtype=int)

        else:
            if not isinstance(f, (Sequence, np.ndarray)) or isinstance(f, bytes):
                raise ValueError("f must be a sequence or numpy array")

            raw_f = np.asarray(f)
            if raw_f.ndim != 1 or raw_f.size == 0:
                raise ValueError("f must be a nonempty 1D sequence")

            if self.r_inputs is None:
                # Infer n from len(f) = r**n.
                size = len(raw_f)
                n = 0
                while size > 1 and size % self.r == 0:
                    size //= self.r
                    n += 1

                if size != 1:
                    raise ValueError(
                        "len(f) must be a power of r when r_inputs is None"
                    )

                self.n = n
                self.r_inputs = np.full(n, self.r, dtype=int)

            else:
                self.n = len(self.r_inputs)
                if len(raw_f) != int(np.prod(self.r_inputs.astype(object))):
                    raise ValueError(
                        "len(f) must equal the product of r_inputs"
                    )

            if variables is None:
                self.variables = np.array(
                    [f"x{i}" for i in range(self.n)]
                )
            else:
                self.variables = np.asarray(variables, dtype=str)
                if self.variables.ndim != 1 or len(self.variables) != self.n:
                    raise ValueError(
                        f"variables must be a 1D sequence of length {self.n}"
                    )

        # Identify Boolean functions.
        self.is_boolean = bool(
            self.r == 2 and np.all(self.r_inputs == 2)
        )

        # Validate truth table before converting to uint8.
        raw_f = np.asarray(f)

        if (raw_f.ndim != 1
                or raw_f.dtype.kind not in "biu"
                or np.any(raw_f < 0)
                or np.any(raw_f >= self.r)):
            raise ValueError(
                f"f must contain only integers from 0 to {self.r - 1}"
            )

        self.f = raw_f.astype(np.uint8, copy=False)
        self.properties = {}


    @classmethod
    def _from_f_unchecked(
            cls,
            f: Sequence[int] | np.ndarray,
            *,
            variables: Sequence[str] | np.ndarray | None = None,
            name: str = "",
            r: int = 2,
            r_inputs: Sequence[int] | np.ndarray | None = None,
    ) -> "Function":
        """
        Construct a Function directly from a truth table without validation.

        This internal constructor bypasses all safety checks and assumes
        that the truth table and cardinalities satisfy the invariants of
        Function, including the correct truth table length and output range.

        Intended for performance-critical internal operations where
        correctness is guaranteed by construction. Never use with
        untrusted input.
        """
        obj = cls.__new__(cls)

        obj.f = np.asarray(f, dtype=np.uint8)
        obj.r = r
        obj.name = name

        if r_inputs is None:
            # Infer n from len(f) = r**n.
            size = len(obj.f)
            n = 0
            while size > 1:
                size //= r
                n += 1

            obj.n = n
            obj.r_inputs = np.full(n, r, dtype=int)
        else:
            obj.r_inputs = np.asarray(r_inputs, dtype=int)
            obj.n = len(obj.r_inputs)

        obj.is_boolean = (
            obj.r == 2 and np.all(obj.r_inputs == 2)
        )

        if variables is None:
            obj.variables = np.array(
                [f"x{i}" for i in range(obj.n)]
            )
        else:
            obj.variables = np.asarray(variables, dtype=str)

        obj.properties = {}

        return obj


    ## Magic methods:
    
    def __str__(self):
        """Return the truth table as a human-readable string."""
        return str(self.f)

    def __repr__(self):
        """
        Return a concise representation of the function.

        For small truth tables, show the complete table.
        For larger functions, show the number of inputs and cardinalities.
        """
        name = f"name='{self.name}', " if self.name else ""

        if len(self.f) <= 32:
            return f"{type(self).__name__}({name}f={self.f.tolist()})"

        return (
            f"{type(self).__name__}("
            f"{name}n={self.n}, r={self.r}, "
            f"r_inputs={self.r_inputs.tolist()})"
        )

    def __len__(self):
        """Return the number of entries in the truth table."""
        return len(self.f)

    def __getitem__(self, index):
        """Return one or more truth table entries."""
        value = self.f[index]
        return int(value) if np.isscalar(value) else value

    def __setitem__(self, index, value):
        """
        Modify truth table entries and invalidate cached properties.

        Assigned values must be integers in {0, ..., r-1}.
        Supports scalar, slice, and array indexing.
        """
        values = np.asarray(value)

        if (values.dtype.kind not in "biu"
                or np.any(values < 0)
                or np.any(values >= self.r)):
            raise ValueError(
                f"Assigned values must be integers from 0 to {self.r - 1}."
            )

        self.f[index] = values
        self.properties.clear()
    
    def _binary_boolean_operation(self, value, operation):
        """
        Apply an element-wise Boolean operation.

        Supports another Boolean function with the same number of inputs,
        or a scalar in {0, 1}. Function inputs are combined positionally,
        regardless of variable names.

        Parameters
        ----------
        value : Function | int
            Boolean function or scalar operand.
        operation : callable
            Binary operation applied to the truth tables.

        Returns
        -------
        Function
            Result of the operation, preserving the class and metadata
            of the left operand.
        """
        if not self.is_boolean:
            raise NotImplementedError(
                "This operation is only implemented for Boolean functions."
            )

        if isinstance(value, (int, np.integer)):
            if value not in (0, 1):
                raise ValueError("Scalar operand must be 0 or 1.")
            other = value

        elif isinstance(value, Function):
            if not value.is_boolean:
                raise ValueError("Both functions must be Boolean.")
            if self.n != value.n:
                raise ValueError("Functions must have the same number of inputs.")
            other = value.f

        else:
            return NotImplemented

        result = operation(self.f, other)

        return self.__class__._from_f_unchecked(
            result,
            variables=self.variables,
            name=self.name,
        )


    def __mul__(self, value):
        """Element-wise Boolean multiplication (logical AND)."""
        return self._binary_boolean_operation(value, np.multiply)


    def __rmul__(self, value):
        """Right-hand Boolean multiplication."""
        return self.__mul__(value)


    def __and__(self, value):
        """Element-wise Boolean AND."""
        return self._binary_boolean_operation(value, np.bitwise_and)


    def __or__(self, value):
        """Element-wise Boolean OR."""
        return self._binary_boolean_operation(value, np.bitwise_or)


    def __xor__(self, value):
        """Element-wise Boolean XOR."""
        return self._binary_boolean_operation(value, np.bitwise_xor)


    def __invert__(self):
        """
        Element-wise Boolean negation.

        Returns a new function with all truth table entries flipped.
        """
        if not self.is_boolean:
            raise NotImplementedError(
                "Negation is only implemented for Boolean functions."
            )

        return self.__class__._from_f_unchecked(
            1 - self.f,
            variables=self.variables,
            name=self.name,
        )

    
    def __call__(
            self,
            values: Sequence[int] | np.ndarray,
            *,
            validate: bool = True,
    ) -> int:
        """
        Evaluate the function at a given input state.

        Parameters
        ----------
        values : Sequence[int] | np.ndarray
            Input state of length n, with each entry in
            {0, ..., r_inputs[i]-1}.
        validate : bool, optional
            Whether to validate input length, type, and state ranges.
            Default is True. Set to False for performance-critical
            internal computations with guaranteed valid inputs.

        Returns
        -------
        int
            Function output in {0, ..., r-1}.

        Raises
        ------
        ValueError
            If validate=True and the input is invalid.
        """
        if validate:
            values = np.asarray(values)

            if values.ndim != 1 or len(values) != self.n:
                raise ValueError(
                    f"Input must be a 1D sequence of length {self.n}."
                )

            if ((values.size > 0 and values.dtype.kind not in "biu")
                    or np.any(values < 0)
                    or np.any(values >= self.r_inputs)):
                raise ValueError(
                    "Input states are outside their allowed ranges."
                )

        if self.is_boolean:
            return int(self.f[utils.bin2dec(values)])

        return int(self.f[utils_multistate.mix2dec(values, self.r_inputs)])

    
    def summary(self, compute_all: bool = False, *, as_dict: bool = False):
        """
        Return a concise summary of the function.

        The summary includes basic structural properties and any previously
        computed properties. If compute_all=True, additional properties
        are computed, where supported.

        Parameters
        ----------
        compute_all : bool, optional
            If True, compute additional properties that may be expensive.
            Default is False.
        as_dict : bool, optional
            If True, return a dictionary instead of a formatted string.

        Returns
        -------
        str or dict
            Summary of the function.
        """
        summary = {
            "Number of variables": self.n,
            "Variables": self.variables.tolist(),
        }

        if self.is_boolean:
            summary.update({
                "Hamming Weight": self.hamming_weight,
                "Bias": self.bias,
                "Absolute bias": self.absolute_bias,
            })
        else:
            summary.update({
                "Output cardinality": self.r,
                "Input cardinalities": self.r_inputs.tolist(),
            })

        if compute_all:
            if not self.is_boolean:
                raise NotImplementedError(
                    "Full summary is not yet implemented for multistate functions."
                )

            activities = self.get_activities()
            summary["Activities"] = [f"{x:.3f}" for x in activities]
            summary["Average sensitivity"] = self.get_average_sensitivity()

            self.get_type_of_inputs()
            self.get_layer_structure()

        summary.update(self.properties)

        if as_dict:
            return summary

        title = type(self).__name__
        if self.name:
            title += f" ({self.name})"

        lines = [title, "-" * len(title)]

        special_formatting = {
            "Absolute bias": ".3f",
            "Bias": ".3f",
            "Average sensitivity": ".3f",
        }

        for key, value in summary.items():
            if key in special_formatting and isinstance(
                value, (int, float, np.number)
            ):
                value = format(value, special_formatting[key])

            lines.append(f"{key + ':':27}{value}")

        return "\n".join(lines)


class BooleanFunction(Function):
    """
    A Boolean function.

    A subclass of :class:`Function` representing a mapping
    :math:`f : \\{0,1\\}^n \\to \\{0,1\\}`.

    Inherits all methods and attributes from :class:`Function`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.is_boolean:
            raise ValueError(
                "BooleanFunction requires all input and output "
                "cardinalities to equal 2."
            )