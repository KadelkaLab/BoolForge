#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:54:16 2026

@author: ckadelka
"""

from collections.abc import Sequence
import numpy as np
import networkx as nx
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import BooleanNetwork
    try:
        import cana.boolean_network
    except ModuleNotFoundError:
        pass

from .. import utils
from ..wiring_diagram import WiringDiagram
from ..boolean_function import f_from_expression

class BooleanNetworkInteroperabilityMixin:
    @classmethod
    def from_cana(
        cls,
        cana_BooleanNetwork: "cana.boolean_network.BooleanNetwork",
        simplify_functions: bool = False,
    ) -> "BooleanNetwork":
        """
        Construct a BooleanNetwork from a ``cana.BooleanNetwork`` instance.
    
        This compatibility method converts a Boolean network defined using the
        ``cana`` package into a BoolForge ``BooleanNetwork``.
    
        Parameters
        ----------
        cana_BooleanNetwork : cana.boolean_network.BooleanNetwork
            A Boolean network instance from the ``cana`` package.
        simplify_functions : bool, optional
            If True, Boolean update functions are simplified after initialization.
            Default is False.  
            
        Returns
        -------
        BooleanNetwork
            The corresponding BoolForge BooleanNetwork.
    
        Raises
        ------
        ImportError
            If the CANA package is not installed.
        TypeError
            If the input object does not appear to be a valid CANA BooleanNetwork.
        KeyError
            If required fields are missing from the CANA logic specification.
        """
        utils._require_cana()
        
        try:
            logic = cana_BooleanNetwork.logic
        except AttributeError as e:
            raise TypeError(
                "Input must be a cana.boolean_network.BooleanNetwork instance."
            ) from e
    
        F = []
        I = []
        variables = []
    
        # Ensure deterministic ordering by node index
        for idx in sorted(logic.keys()):
            entry = logic[idx]
    
            if "name" not in entry or "in" not in entry or "out" not in entry:
                raise KeyError(
                    f"Logic entry for node {idx} must contain keys "
                    "'name', 'in', and 'out'."
                )
    
            variables.append(str(entry["name"]))
            I.append(list(entry["in"]))
            F.append(np.array(entry["out"], dtype=int))
    
        return cls(F=F, I=I, variables=variables, simplify_functions=simplify_functions)


    @classmethod
    def from_string(
        cls,
        network_string: str,
        separator: str | Sequence[str] = ",",
        max_degree: int = 24,
        allow_truncation: bool = False,
        simplify_functions: bool = False,
        ) -> "BooleanNetwork":
        """
        Construct a BooleanNetwork from a textual Boolean rule specification.
    
        This compatibility method parses a string representation of Boolean update
        rules (one rule per line) and constructs a corresponding BooleanNetwork. 
        The input format is intended for legacy or trusted sources and supports 
        logical expressions using AND/OR/NOT operators. See boolean_function.f_from_expression
        for details.
    
        .. warning::
            This method uses ``eval`` internally and MUST NOT be used on untrusted
            input.
    
        Parameters
        ----------
        network_string : str
            String encoding Boolean update rules, one per line.
        separator : str or sequence of str, optional
            Separator(s) between variable names and Boolean expressions.
        max_degree : int, optional
            Maximum allowed indegree for explicit truth-table construction.
        allow_truncation : bool, optional
            If False (default), nodes with indegree greater than ``max_degree``
            raise a ValueError. If True, such nodes are replaced by identity
            self-loops, allowing fast construction of large networks while
            ignoring high-degree functions.
        simplify_functions : bool, optional
            If True, Boolean update functions are simplified after initialization.
            Default is False.  
            
        Returns
        -------
        BooleanNetwork
            The constructed Boolean network.
    
        Raises
        ------
        ValueError
            If parsing fails or if ``allow_truncation`` is False and 
            a node exceeds ``max_degree``.
        """
        
        # --------------------------------------------
        # 1. Clean lines
        # --------------------------------------------
        lines = [
            l.strip()
            for l in network_string.splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
    
        rules = []
        for i, line in enumerate(lines):
            if separator not in line:
                raise ValueError(f"Missing separator '{separator}' in line {i+1}:\n{line}")
            lhs, rhs = line.split(separator, 1)
            rules.append((lhs.strip(), rhs.strip()))
    
        # --------------------------------------------
        # 2. Collect explicitly defined nodes
        # --------------------------------------------
        node_names = [lhs for lhs, _ in rules]
    
        # --------------------------------------------
        # 3. Parse RHS to detect all regulators
        # --------------------------------------------
        parsed_rhs = []
        all_regulators = set()
    
        for lhs, rhs in rules:
            f, regulators = f_from_expression(rhs, max_degree=max_degree)
            parsed_rhs.append((lhs, rhs, f, regulators))
            for r in regulators:
                all_regulators.add(r)
    
        # --------------------------------------------
        # 4. Add missing regulators as identity nodes
        # --------------------------------------------
        missing_nodes = sorted(all_regulators - set(node_names))
    
        # Append them deterministically (sorted for reproducibility)
        node_names_extended = node_names + missing_nodes
    
        node_index = {name: i for i, name in enumerate(node_names_extended)}
    
        # --------------------------------------------
        # 5. Build F and I
        # --------------------------------------------
        F = []
        I = []
    
        # First build defined rules
        for lhs, rhs, f, regulators in parsed_rhs:
    
            deg = len(regulators)
    
            if deg > max_degree:
                if not allow_truncation:
                    raise ValueError(
                        f"Node '{lhs}' has indegree {deg} > max_degree={max_degree}."
                    )
                idx = node_index[lhs]
                F.append(np.array([0, 1], dtype=int))
                I.append(np.array([idx], dtype=int))
                continue
    
            reg_indices = [node_index[r] for r in regulators]
    
            F.append(f.astype(int))
            I.append(np.array(reg_indices, dtype=int))
    
        # Add identity nodes for missing regulators
        for name in missing_nodes:
            idx = node_index[name]
            F.append(np.array([0, 1], dtype=int))  # identity
            I.append(np.array([idx], dtype=int))
    
        return cls(
            F,
            I,
            node_names_extended,
            simplify_functions=simplify_functions,
        )



    @classmethod
    def from_DiGraph(cls, nx_DiGraph: "nx.DiGraph") -> "WiringDiagram":
        raise NotImplementedError(
            "from_DiGraph is not supported for BooleanNetwork. "
            "Use WiringDiagram.from_DiGraph and then construct "
            "a BooleanNetwork by providing Boolean update functions."
        )
    
    def to_cana(self) -> "cana.boolean_network.BooleanNetwork":
        """
        Export the Boolean network as a ``cana.BooleanNetwork`` instance.
    
        This compatibility method converts the current BooleanNetwork into an
        equivalent representation from the ``cana`` package. The exported network
        reflects the current state of the model, including any removed constants,
        simplifications, or identity self-loops.
    
        Returns
        -------
        cana.boolean_network.BooleanNetwork
            A ``cana`` BooleanNetwork instance representing this network.
    
        Raises
        ------
        ImportError
            If the ``cana`` package is not installed.
        """
        try:
            import cana.boolean_network
        except ImportError as e:
            raise ImportError(
                "The 'cana' package is required for to_cana()."
            ) from e
    
        logic_dicts = []
        for bf, regulators, var in zip(self.F, self.I, self.variables):
            logic_dicts.append(
                {
                    "name": var,
                    "in": list(regulators),
                    "out": bf.f.tolist(),
                }
            )
    
        return cana.boolean_network.BooleanNetwork(
            Nnodes=self.N,
            logic={i: d for i, d in enumerate(logic_dicts)},
        )


    def to_string(
        self,
        separator: str = ',\t',
        as_polynomial: bool = True,
        logical_and_op: str = ' & ',
        logical_or_op: str = ' | ',
        logical_not_op:str = ' !'
    ) -> str:
        """
        Export the Boolean network in string format.
    
        This compatibility method returns a string representation of the Boolean
        network, with one line per variable of the form variable <separator> function.
        
        Parameters
        ----------
        separator : str, optional
            String used to separate the target variable from its update function.
            Default is `",\\t"`.
        as_polynomial : bool, optional
            If True (default), return Boolean functions in polynomial form.
            If False, return functions as logical expressions.
        logical_and_op : str, optional
            String used to represent the logical AND operator. Default is ``" & "``.
        logical_or_op : str, optional
            String used to represent the logical OR operator. Default is ``" | "``.
        logical_not_op : str, optional
            String used to represent the logical NOT operator. Default is ``" !"``.
            
        Returns
        -------
        str
            A string describing the network.
            
        Notes
        -----
        This method exports the reduced Boolean network, i.e. after semantic
        constants have been removed during initialization.
        """
        lines = []
    
        for i in range(self.N):
            if as_polynomial:
                function = self.F[i].to_polynomial()
            else:
                function = self.F[i].to_logical(and_op = logical_and_op, 
                                                or_op = logical_or_op,
                                                not_op = logical_not_op)
    
            lines.append(f"{self.variables[i]}{separator}{function}")
    
        return "\n".join(lines)
    
    
    def to_bnet(
        self,
        as_polynomial: bool = True,
        logical_and_op: str = ' & ',
        logical_or_op: str = ' | ',
        logical_not_op:str = ' !'
    ) -> str:
        """
        Export the Boolean network in BNET format.
    
        This compatibility method returns a string representation of the Boolean
        network in the BNET format used by tools such as BoolNet and PyBoolNet,
        with one line per variable of the form ``variable ,<tab> function``.
        
        Parameters
        ----------
        as_polynomial : bool, optional
            If True (default), return Boolean functions in polynomial form.
            If False, return functions as logical expressions.
        logical_and_op : str, optional
            String used to represent the logical AND operator. Default is ``" & "``.
        logical_or_op : str, optional
            String used to represent the logical OR operator. Default is ``" | "``.
        logical_not_op : str, optional
            String used to represent the logical NOT operator. Default is ``" !"``.
                
        Returns
        -------
        str
            A string containing the BNET representation of the network.
            
        Notes
        -----
        This method exports the reduced Boolean network, i.e. after semantic
        constants have been removed during initialization.
        """

        return self.to_string(separator = ',\t', 
                              as_polynomial=as_polynomial,
                              logical_and_op=logical_and_op,
                              logical_or_op=logical_or_op,
                              logical_not_op=logical_not_op
                              )
    
    
    def to_truth_table(
        self,
        filename: str | None = None,
    ) -> pd.DataFrame:
        """
        Construct the full synchronous truth table of the Boolean network.
    
        Each row corresponds to a network state at time ``t`` and its deterministic
        successor at time ``t+1`` under synchronous updating.
    
        Parameters
        ----------
        filename : str, optional
            If provided, the truth table is written to a file. The file extension
            determines the format and must be one of ``'csv'``, ``'xls'``, or
            ``'xlsx'``. If None (default), no file is created.

        Returns
        -------
        pandas.DataFrame
            The full truth table with shape ``(2**N, 2*N)``.
    
        Notes
        -----
        - States are enumerated in lexicographic order, consistent with
          ``utils.get_left_side_of_truth_table``.
        - This method computes and stores the synchronous state transition graph
          (``self.STG``) if it has not been computed previously.
        - Exporting to Excel requires the ``openpyxl`` package.
        """
        columns = [name + "(t)" for name in self.variables]
        columns += [name + "(t+1)" for name in self.variables]
    
        if self.STG is None:
            self.compute_synchronous_state_transition_graph()
    
        data = np.zeros((2**self.N, 2*self.N), dtype=int)
        data[:, :self.N] = utils.get_left_side_of_truth_table(self.N)
    
        for i in range(2**self.N):
            data[i, self.N:] = utils.dec2bin(self.STG[i], self.N)
    
        truth_table = pd.DataFrame(data, columns=columns)
    
        if filename is not None:
            if not isinstance(filename, str):
                raise TypeError("filename must be a string")
    
            ending = filename.split(".")[-1]
            if ending not in {"csv", "xls", "xlsx"}:
                raise ValueError("filename must end in 'csv', 'xls', or 'xlsx'")
    
            if ending == "csv":
                truth_table.to_csv(filename, index=False)
            else:
                truth_table.to_excel(filename, index=False)
    
        return truth_table