#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from .. import utils
    
from .analysis import BooleanFunctionAnalysisMixin
from .canalization import BooleanFunctionCanalizationMixin
from .collective_canalization import BooleanFunctionCollectiveCanalizationMixin
from .conversions import BooleanFunctionConversionsMixin
from .interoperability import BooleanFunctionInteroperabilityMixin
from .parsing import f_from_expression

class BooleanFunction(
        BooleanFunctionAnalysisMixin,
        BooleanFunctionCanalizationMixin,
        BooleanFunctionCollectiveCanalizationMixin,
        BooleanFunctionConversionsMixin,
        BooleanFunctionInteroperabilityMixin,
        ):
    """
    A Boolean function.

    This class represents a Boolean function
    :math:`f : \\{0,1\\}^n \\to \\{0,1\\}` and stores its truth table together
    with variable names and optional metadata.

    Parameters
    ----------
    f : list[int] | np.ndarray | str
        Either:
            
        - a truth table of length ``2**n`` representing the outputs of a Boolean
          function with ``n`` inputs, or
        
        - a Boolean expression string that can be evaluated. Expression strings 
          are parsed using ``boolean_function.parsing.f_from_expression``.
        
    name : str, optional
        Name of the node regulated by the Boolean function. Default is ``""``.
    variables : list[str] | np.ndarray | None, optional
        Names of the input variables, given in order. Must have length ``n``.
        If ``None`` (default), variables are named ``x0, ..., x_{n-1}``.

    Attributes
    ----------
    f : np.ndarray
        NumPy array of dtype ``uint8`` and length ``2**n`` containing only the
        values 0 and 1, representing the truth table of the Boolean function.
    n : int
        Number of input variables.
    variables : np.ndarray
        One-dimensional NumPy array of length ``n`` containing variable names.
    name : str
        Name of the node regulated by the Boolean function.
    properties : dict
        Dictionary for dynamically computed properties of the Boolean function
        (e.g., canalizing structure, effective inputs, robustness measures).
    """
    
    __slots__ = ['f','n','variables','name','properties']
    
    def __init__(
            self, 
            f : list[int] | np.ndarray | str,
            variables : list[str] | np.ndarray | None = None,
            name : str = ""
    ):
        """
        Initialize a Boolean function.
        
        Parameters
        ----------
        f : list[int] or np.ndarray or str
            Truth table of the Boolean function, given as a list or array
            of length ``2**n``, or a Boolean expression as a string.
        variables : list[str] or np.ndarray[str], optional
            Names of the input variables. Must have length ``n`` if provided.
            If None, default variable names x0, x1, ... are assigned.
        name : str, optional
            Optional name of the Boolean function.
        
        Notes
        -----
        - The number of inputs ``n`` is inferred from the length of ``f``.
        """
        self.name = name
        if isinstance(f, str):
            f, self.variables = f_from_expression(f)
        else:
            if not isinstance(f, (list, np.ndarray)):
                raise ValueError("f must be a list, numpy array or interpretable string")
            if not len(f) > 0:
                raise ValueError("f cannot be empty")
            
            _n = int(np.log2(len(f)))
            if not abs(np.log2(len(f)) - _n) < 1e-9:
                raise ValueError("f must be of size 2^n, n >= 0")
            
            if variables is None:
                self.variables = np.array([f"x{i}" for i in range(_n)])
            else:
                self.variables = np.asarray(variables, dtype=str)
                if self.variables.ndim != 1:
                    raise ValueError("variables must be a 1D array of strings")
                if len(self.variables) != _n:
                    raise ValueError(
                        f"variables must have length {_n}, got {len(self.variables)}"
                    )
        
        self.n = len(self.variables)
            
        self.f = np.array(f, dtype=np.uint8)

        if not np.all((self.f == 0) | (self.f == 1)):
            raise ValueError("f must contain only the values 0 and 1.")
        
        self.properties = {}

    @classmethod
    def _from_f_unchecked(
        cls, 
        f : list[int] | np.ndarray, 
        *, 
        variables : list[str] | np.ndarray | None = None, 
        name : str =""
    ) -> "BooleanFunction":
        """
        Construct a BooleanFunction directly from a truth table *without*
        validating invariants.
    
        This internal constructor bypasses all safety checks and therefore
        assumes that the input truth table already satisfies all BooleanFunction
        invariants. In particular, it assumes that:
    
            - ``f`` has length ``2**n`` for some integer ``n``,
            - all entries of ``f`` are in ``{0,1}``,
            - ``f`` is already in a NumPy-compatible array-like format.
    
        This method exists for **performance-critical internal code paths**
        such as random Boolean function generation, bulk construction, or
        Numba-accelerated routines, where invariant checks would be prohibitively
        expensive and correctness is guaranteed by construction.
    
        **Warning**
        -------
        This method must *never* be called on untrusted input.
        """
        obj = cls.__new__(cls)
    
        # Core data
        obj.f = np.asarray(f, dtype=np.uint8)
        obj.n = int(np.log2(len(obj.f)))
    
        # Metadata (match __init__ invariants)
        obj.name = name
        if variables is None:
            obj.variables = np.array([f"x{i}" for i in range(obj.n)])
        else:
            obj.variables = np.asarray(variables)
    
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