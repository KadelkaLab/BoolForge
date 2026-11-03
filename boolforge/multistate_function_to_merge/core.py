#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from .. import utils_multistate
from typing import Sequence

#from .analysis import MultistateFunctionAnalysisMixin
#from .canalization import MultistateFunctionCanalizationMixin
from .collective_canalization import MultistateFunctionCollectiveCanalizationMixin
#from .conversions import MultistateFunctionConversionsMixin
#from .interoperability import MultistateFunctionInteroperabilityMixin

class MultistateFunction(
        #MultistateFunctionAnalysisMixin,
        #MultistateFunctionCanalizationMixin,
        MultistateFunctionCollectiveCanalizationMixin,
        #MultistateFunctionConversionsMixin,
        #MultistateFunctionInteroperabilityMixin,
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
        if (not isinstance(r_inputs, (Sequence, np.ndarray))
                or np.ndim(r_inputs) != 1
                or any(
                    isinstance(x, (bool, np.bool_))
                    or not isinstance(x, (int, np.integer))
                    or not 2 <= x <= 255
                    for x in r_inputs
                )):
            raise ValueError(
                "r_inputs must be a sequence of integers between 2 and 255"
            )
        self.r_inputs = np.asarray(r_inputs, dtype=int)

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
        if (not isinstance(f, (Sequence, np.ndarray))
                or isinstance(f, (str, bytes))
                or np.ndim(f) != 1
                or len(f) == 0):
            raise ValueError("f must be a nonempty 1D sequence")

        if self.r_inputs is None:
            # Infer n from len(f) = r**n.
            size = len(f)
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
            if len(f) != int(np.prod(self.r_inputs.astype(object))):
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
            or any(
                isinstance(x, (bool, np.bool_))
                or not isinstance(x, (int, np.integer))
                or not 0 <= x < self.r
                for x in raw_f
            )):
        raise ValueError(
            f"f must contain only integers from 0 to {self.r - 1}"
        )

    self.f = np.asarray(raw_f, dtype=np.uint8)
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