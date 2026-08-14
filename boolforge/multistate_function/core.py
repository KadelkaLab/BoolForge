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
    """A multistate function.

    This class represents a discrete multistate function

    :math:`f : X_1 \\times \\cdots \\times X_n \\to X_f`,

    where input variable `x_i` has ``r_inputs[i]`` possible states and
    the output has ``r`` possible states. The function is stored as a truth
    table together with the input-state cardinalities, variable names, and
    optional node name.

    Parameters
    ----------
    f : Sequence[int] | str
        Truth table of the multistate function. Its length must equal
        ``np.prod(r_inputs)``, and each entry must be an integer in
        ``{0, ..., r - 1}``.

        String representations are reserved for future support and currently
        raise ``NotImplementedError``.
    r : int
        Number of possible output states. Must be at least 2. Output states
        are represented by the integers ``0, ..., r - 1``.
    r_inputs : Sequence[int]
        Number of possible states for each input variable, in variable order.
        The number of inputs ``n`` is ``len(r_inputs)``, and the truth table
        must contain ``np.prod(r_inputs)`` entries.
    variables : Sequence[str] | None, optional
        Names of the input variables, in the same order as ``r_inputs``.
        Must have length ``n``. If ``None`` (default), variables are named
        ``x0, ..., x{n-1}``.
    name : str, optional
        Name of the node or variable represented by the function.
        Default is ``""``.

    Attributes
    ----------
    f : np.ndarray
        One-dimensional NumPy array of dtype ``uint8`` containing the truth
        table. All entries lie in ``{0, ..., r - 1}``.
    n : int
        Number of input variables.
    r : int
        Number of possible output states.
    r_inputs : Sequence[int]
        Number of possible states of each input variable.
    variables : np.ndarray
        One-dimensional NumPy array of length ``n`` containing the input
        variable names.
    name : str
        Name of the node or variable represented by the function.
    properties : dict
        Dictionary for dynamically computed properties of the multistate
        function, such as canalizing structure, effective inputs, and
        robustness measures.

    Notes
    -----
    The truth table contains one output for every combination of input states,
    so its length is

    :math:`\\prod_{i=1}^n |X_i|`.

    Input and output states are represented by consecutive nonnegative
    integers beginning at zero.
    """
    def __init__(
            self,
            f : Sequence[int] | str,
            r : int,
            r_inputs : Sequence[int],
            variables : Sequence[str] | None = None,
            name : str = ""):
        self.name = name
        self.r = r
        self.r_inputs = np.array(r_inputs,dtype=int)
        if not isinstance(r, (int, np.int64)):
            raise ValueError(f"Radix r must be an integer (is {type(r)})")
        if not r > 1:
            raise ValueError(f"Radix r has a minimum value of 2. Received {r}.")
        if not min(r_inputs) > 1:
            raise ValueError(f"The radix of each input must be 2 or greater. Received {r_inputs}.")
        if isinstance(f, str):
            ## f, r, self.variables = utils.from_from_expression(f)
            raise NotImplementedError("Creating multistate function from string in the constructor is not permissible (yet)")
        else:
            if not isinstance(f, (list, np.ndarray)):
                raise ValueError("f must be a list or numpy array of integers")
            if not len(f) > 0:
                raise ValueError("f cannot be empty")
            self.n = len(r_inputs)

            if not len(f) == np.prod(r_inputs):
                raise ValueError(f"f must be of size prod_sum(r_inputs) = {np.prod(r_inputs)}")
            if variables is None:
                self.variables = np.array([f"x{i}" for i in range(self.n)])
            else:
                self.variables = np.asarray(variables, str)
                if self.variables.ndim != 1:
                    raise ValueError("variables must be a 1D array of strings")
                if len(self.variables) != self.n:
                    raise ValueError(f"variables must have length {self.n}, got {len(self.variables)}")
        self.f = np.array(f, np.uint8)
        
        if not np.all(self.f < self.r):
            raise ValueError(f"f must contain only values from 0 to {self.r-1}.")

        self.properties = {}


    ## Magic methods
    
    def __str__(self):
        """
        Return a human-readable string representation of the multistate function.
        
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

    def __call__(self, values: list[int] | tuple[int, ...] | np.ndarray):
        """
        Evaluate the multistate function on a given input vector.
    
        This method makes MultistateFunction instances callable and returns the
        output value for a specified binary input configuration.
    
        Parameters
        ----------
        values : list[int] | tuple[int, ...] | np.ndarray
            Sequence of admissable values of length ``n``, where ``n`` is
            the number of input variables of the multistate function.
    
        Returns
        -------
        int
            Output value of the multistate function in {0,...,self.r-1} for the specified input.
    
        Raises
        ------
        ValueError
            If the input length does not match ``n`` or if values out of range are
            provided.
    
        Examples
        --------
        >>> f = MultistateFunction([0,0,1,0,1,2],3,[2,3])
        >>> f([0,2])
        1
        >>> f([1,2])
        2
        """
        if not len(values) == self.n:
            raise ValueError(f"The argument must be of length {self.n}.")
        values = np.array(values)
        if np.any(values < 0) or np.any(values >= self.r_inputs):
            raise ValueError("Input states are outside their allowed ranges.")
        return self.f[utils_multistate.mix2dec(values,self.r_inputs)].item()

    
    def summary(self, compute_all: bool = False, *, as_dict: bool = False):
        """
        Return a concise summary of the multistate function.
    
        The summary includes basic structural and statistical properties of the
        multistate function and, optionally, additional properties that may require
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
            Summary of the multistate function, either as a formatted string or as
            a dictionary depending on the value of ``as_dict``.
        """
    
        core_summary = {
            "Radix": self.r,
            "Number of variables": self.n,
            "Radices of variables": self.r_inputs,
            "Variables": self.variables.tolist(),
        }
        
        special_formatting = { #Change/delete
            "Absolute bias" : ".3f",
            "Bias" : ".3f",
            "Average sensitivity" : ".3f"
        }
        
        summary = core_summary.copy()
    
        # if compute_all:
        #     activities = self.get_activities()
        #     avg_sensitivity = self.get_average_sensitivity()
        #     summary['Activities'] = [f"{x:.3f}" for x in activities]
        #     summary['Average sensitivity'] = avg_sensitivity
            
        #     self.get_type_of_inputs()            
        #     self.get_layer_structure()
    
        # summary.update(self.properties)
    
        if as_dict:
            return summary
    
        title = "MultistateFunction"
        if self.name:
            title += f" ({self.name})"
            
        lines = [title, "-" * len(title)]
        
        for key, value in summary.items():
            if key not in special_formatting:
                lines.append(f"{key+':':27}{value}")
            else:
                lines.append(f"{key+':':27}{value:{special_formatting[key]}}")
        
        return "\n".join(lines)