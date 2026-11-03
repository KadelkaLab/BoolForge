#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Sequence

from .. import utils
    
from .analysis import BooleanFunctionAnalysisMixin
from .canalization import FunctionCanalizationMixin
from .collective_canalization import BooleanFunctionCollectiveCanalizationMixin
from .conversions import BooleanFunctionConversionsMixin
from .interoperability import BooleanFunctionInteroperabilityMixin
from .parsing import f_from_expression


class Function(
        BooleanFunctionAnalysisMixin,
        FunctionCanalizationMixin,
        BooleanFunctionCollectiveCanalizationMixin,
        BooleanFunctionConversionsMixin,
        BooleanFunctionInteroperabilityMixin,
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
                    or raw_inputs.dtype.kind not in "iu"
                    or np.any(raw_inputs < 2)
                    or np.any(raw_inputs > 255)):
                raise ValueError(
                    "r_inputs must contain integers between 2 and 255"
                )
            self.r_inputs = raw_inputs.astype(int, copy=False)

        # Identify Boolean functions.
        self.is_boolean = (
            self.r == 2
            and (self.r_inputs is None or np.all(self.r_inputs == 2))
        )

        if isinstance(f, str):
            if not self.is_boolean:
                raise NotImplementedError(
                    "Expression parsing is only supported for Boolean functions"
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
        """
        Return a human-readable string representation of the Boolean function.
    
        This method returns the underlying truth table as a NumPy array.
        """
        return f"{self.f}"
    
    def __repr__(self):
        """
        Return an unambiguous string representation of the BooleanFunction.
    
        For small functions (``n < 5``), the full truth table is shown.
        For larger functions, only the number of inputs is displayed to
        avoid excessive output.
        """
        name = f"name='{self.name}', " if self.name else ""
        
        if self.n < 5:
            return f"{type(self).__name__}({name}f={self.f.tolist()})"
        else:
            return f"{type(self).__name__}({name}n={self.n}, bias={self.bias:.3f})"
    
    def __len__(self):
        return 2**self.n

    def __getitem__(self, index):
        try:
            return int(self.f[index])
        except TypeError:
            return self.f[index]

    def __setitem__(self, index, value):
        self.f[index] = value
    
    def __mul__(self, value):
        """
        Element-wise Boolean multiplication (logical AND).
    
        This method implements logical AND between Boolean functions or between
        a Boolean function and a scalar value in ``{0,1}``.
    
        Parameters
        ----------
        value : BooleanFunction or int
            BooleanFunction of the same size or an integer value (0 or 1).
    
        Returns
        -------
        BooleanFunction
            Result of element-wise Boolean multiplication.
        """
        if isinstance(value, int):
            if value not in (0, 1):
                raise ValueError("Integer multiplier must be 0 or 1.")
            return self.__class__(self.f * value)
        elif isinstance(value, BooleanFunction):
            return self.__class__(self.f * value.f)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: "
                f"'BooleanFunction' and '{type(value).__name__}'"
            )


    def __rmul__(self, value):
        """
        Right-hand element-wise Boolean multiplication (logical AND).
    
        This method enables expressions of the form ``value * BooleanFunction``
        where ``value`` is an integer in ``{0,1}``.
    
        Parameters
        ----------
        value : int
            Integer value (0 or 1).
    
        Returns
        -------
        BooleanFunction
            Result of element-wise Boolean multiplication.
        """
        return self.__mul__(value)

    
    def __and__(self, value):
        """
        Element-wise logical AND.
    
        This method implements element-wise logical AND between Boolean
        functions or between a Boolean function and a scalar value in ``{0,1}``.
    
        Parameters
        ----------
        value : BooleanFunction or int
            BooleanFunction of the same size or an integer value (0 or 1).
    
        Returns
        -------
        BooleanFunction
            Result of element-wise logical AND.
        """
        if isinstance(value, int):
            if value not in (0, 1):
                raise ValueError("Integer must be 0 or 1.")
            return self.__class__(self.f & value)
        elif isinstance(value, BooleanFunction):
            return self.__class__(self.f & value.f)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for &: "
                f"'BooleanFunction' and '{type(value).__name__}'"
            )

    def __or__(self, value):
        """
        Element-wise logical OR.
    
        This method implements element-wise logical OR between Boolean
        functions or between a Boolean function and a scalar value in ``{0,1}``.
    
        Parameters
        ----------
        value : BooleanFunction or int
            BooleanFunction of the same size or an integer value (0 or 1).
    
        Returns
        -------
        BooleanFunction
            Result of element-wise logical OR.
        """
        if isinstance(value, int):
            if value not in (0, 1):
                raise ValueError("Integer must be 0 or 1.")
            return self.__class__(self.f | value)
        elif isinstance(value, BooleanFunction):
            return self.__class__(self.f | value.f)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for |: "
                f"'BooleanFunction' and '{type(value).__name__}'"
            )
    
    
    def __xor__(self, value):
        """
        Element-wise logical XOR.
    
        This method implements element-wise logical XOR between Boolean
        functions or between a Boolean function and a scalar value in ``{0,1}``.
    
        Parameters
        ----------
        value : BooleanFunction or int
            BooleanFunction of the same size or an integer value (0 or 1).
    
        Returns
        -------
        BooleanFunction
            Result of element-wise logical XOR.
        """
        if isinstance(value, int):
            if value not in (0, 1):
                raise ValueError("Integer must be 0 or 1.")
            return self.__class__(self.f ^ value)
        elif isinstance(value, BooleanFunction):
            return self.__class__(self.f ^ value.f)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for ^: "
                f"'BooleanFunction' and '{type(value).__name__}'"
            )


    def __invert__(self):
        """
        Element-wise logical negation.
    
        This method computes the logical NOT of the Boolean function by
        flipping all truth table entries.
    
        Returns
        -------
        BooleanFunction
            Result of element-wise logical negation.
        """
        return self.__class__(1 - self.f)
    
    
    def __call__(self, values: list[int] | tuple[int, ...] | np.ndarray):
        """
        Evaluate the Boolean function on a given input vector.
    
        This method makes BooleanFunction instances callable and returns the
        output value for a specified binary input configuration.
    
        Parameters
        ----------
        values : list[int] | tuple[int, ...] | np.ndarray
            Sequence of binary values (0 or 1) of length ``n``, where ``n`` is
            the number of input variables of the Boolean function.
    
        Returns
        -------
        int
            Output value of the Boolean function (0 or 1) for the specified input.
    
        Raises
        ------
        ValueError
            If the input length does not match ``n`` or if non-binary values are
            provided.
    
        Examples
        --------
        >>> f = BooleanFunction("x1 | (x2 & x3)")
        >>> f([1, 0, 1])
        1
        >>> f([0, 1, 0])
        0
        """
        if not len(values) == self.n:
            raise ValueError(f"The argument must be of length {self.n}.")
        if not set(values) <= {0, 1}:
            raise ValueError("Binary values required.")
        return self.f[utils.bin2dec(values)].item()

    
    def summary(self, compute_all: bool = False, *, as_dict: bool = False):
        """
        Return a concise summary of the Boolean function.
    
        The summary includes basic structural and statistical properties of the
        Boolean function and, optionally, additional properties that may require
        nontrivial computation.
    
        Parameters
        ----------
        compute_all : bool, optional
            If ``True``, additional properties are computed and included in the
            summary. These computations may be expensive. If ``False`` (default),
            only already available properties are included.
        as_dict : bool, optional
            If ``True``, return the summary as a dictionary. If ``False`` (default),
            return a formatted string.
    
        Returns
        -------
        str or dict
            Summary of the Boolean function, either as a formatted string or as
            a dictionary depending on the value of ``as_dict``.
        """
    
        core_summary = {
            "Number of variables": self.n,
            "Hamming Weight": self.hamming_weight,
            "Bias": self.bias,
            "Absolute bias": self.absolute_bias,
            "Variables": self.variables.tolist(),
        }
        
        special_formatting = {
            "Absolute bias" : ".3f",
            "Bias" : ".3f",
            "Average sensitivity" : ".3f"
        }
        
        summary = core_summary.copy()
    
        if compute_all:
            activities = self.get_activities()
            avg_sensitivity = self.get_average_sensitivity()
            summary['Activities'] = [f"{x:.3f}" for x in activities]
            summary['Average sensitivity'] = avg_sensitivity
            
            self.get_type_of_inputs()            
            self.get_layer_structure()
    
        summary.update(self.properties)
    
        if as_dict:
            return summary
    
        title = "BooleanFunction"
        if self.name:
            title += f" ({self.name})"
            
        lines = [title, "-" * len(title)]
        
        for key, value in summary.items():
            if key not in special_formatting:
                lines.append(f"{key+':':27}{value}")
            else:
                lines.append(f"{key+':':27}{value:{special_formatting[key]}}")
        
        return "\n".join(lines)

class BooleanFunction(Function):
    """
    A Boolean function.

    A subclass of :class:`Function` representing a mapping
    :math:`f : \\{0,1\\}^n \\to \\{0,1\\}`.

    Inherits all methods and attributes from :class:`Function`.
    """

    pass