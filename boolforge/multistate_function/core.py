#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from .. import utils_multistate
from typing import Sequence

from .analysis import MultistateFunctionAnalysisMixin
from .canalization import MultistateFunctionCanalizationMixin
from .collective_canalization import MultistateFunctionCollectiveCanalizationMixin
from .conversions import MultistateFunctionConversionsMixin
from .interoperability import MultistateFunctionInteroperabilityMixin

class MultistateFunction(
        MultistateFunctionAnalysisMixin,
        MultistateFunctionCanalizationMixin,
        MultistateFunctionCollectiveCanalizationMixin,
        MultistateFunctionConversionsMixin,
        MultistateFunctionInteroperabilityMixin,
        ):
    """
    A multistate function. CLEAN UP

    This class represents a multistate function
    :math:`f : X_1 x ... x X_n -> X_f` and stores its truth table together
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
    def __init__(
            self,
            f : Sequence[int] | str,
            r : int,
            in_degree : int,
            variables : Sequence[str] | None = None,
            name : str = ""):
        self.name = name
        if not isinstance(r, (int, np.int64)):
            raise ValueError(f"Radix r must be an integer (is {type(r)})")
        if not r > 1:
            raise ValueError(f"Radix r has a minimum value of 2. Received {r}")
        if isinstance(f, str):
            ## f, r, self.variables = utils.from_from_expression(f)
            raise NotImplementedError("Creating multistate function from string in the constructor is not permissible")
        else:
            if not isinstance(f, (list, np.ndarray)):
                raise ValueError("f must be a list, numpy array, or interpretable string")
            if not len(f) > 0:
                raise ValueError("f cannot be empty")
            #_n = ???
            #if not len(f) == expected length:
            #    raise ValueError("f must be of size ???")
            if variables is None:
                self.variables = np.array([f"x{i}" for i in range(in_degree)])
            else:
                self.variables = np.asarray(variables, str)
                if self.variables.ndim != 1:
                    raise ValueError("variables must be a 1D array of strings")
                #if len(self.variables) != _n:
                #    raise ValueError(f"variables must have length {_n}, got {len(self.variables)}")
        self.n = len(self.variables)
        self.r = r
        self.in_rs = in_degree
        self.f = np.array(f, np.uint8)
        
        if not np.all(self.f < self.r):
            raise ValueError(f"f must contain only values 0 <= v < {self.r}")
#             _n = int(np.log2(len(f)))
#             if not abs(np.log2(len(f)) - _n) < 1e-9:
#                 raise ValueError("f must be of size 2^n, n >= 0")
            
#             if variables is None:
#                 self.variables = np.array([f"x{i}" for i in range(_n)])
#             else:
#                 self.variables = np.asarray(variables, dtype=str)
#                 if self.variables.ndim != 1:
#                     raise ValueError("variables must be a 1D array of strings")
#                 if len(self.variables) != _n:
#                     raise ValueError(
#                         f"variables must have length {_n}, got {len(self.variables)}"
#                     )
        
#         self.n = len(self.variables)
            
#         self.f = np.array(f, dtype=np.uint8)

#         if not np.all((self.f == 0) | (self.f == 1)):
#             raise ValueError("f must contain only the values 0 and 1.")
        
#         self.properties = {}


    ## Magic methods
    
    def __str__(self):
        """
        Return a human-readable string representation of the Boolean function.
        
        This method returns the underlying truth table as a NumPy array.
        """
        return f"{self.f}"
    
    def __repr__(self):
        """
        Return an unambiguous string representation of a MultistateFunction.
        
        For small functions (less than 6 variables), the full truth table is shown.
        For larger functions, only the number of inputs is displayed to
        avoid excessive output.
        """
        if self.n < 6:
            return f"{type(self).__name__}(f={self.f.tolist()})"
        return f"{type(self).__name__}(n={self.n})"
    
    def __len__(self):
        return len(self.f)
    
    def __getitem__(self, index):
        try:
            return int(self.f[index])
        except TypeError:
            return self.f[index]
    
    def __setitem__(self, index, value):
        self.f[index] = value