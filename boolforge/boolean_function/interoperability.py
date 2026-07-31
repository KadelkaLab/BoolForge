#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import TYPE_CHECKING

from .. import utils

if TYPE_CHECKING:
    from .core import BooleanFunction
    try:
        import cana.boolean_node
    except ModuleNotFoundError:
        pass

class BooleanFunctionInteroperabilityMixin:
    @classmethod
    def from_cana(
        cls,
        cana_BooleanNode: "cana.boolean_node.BooleanNode",
    ) -> "BooleanFunction":
        """
        Construct a BoolForge BooleanFunction from a CANA BooleanNode.
    
        This compatibility method converts a
        ``cana.boolean_node.BooleanNode`` instance into a BoolForge
        ``BooleanFunction`` by extracting its truth table representation.
    
        Parameters
        ----------
        cana_BooleanNode : cana.boolean_node.BooleanNode
            Boolean node object from the CANA library.
    
        Returns
        -------
        BooleanFunction
            A BoolForge BooleanFunction instance with the same truth table
            as the input CANA BooleanNode.
    
        Notes
        -----
        This method is intended for interoperability with the CANA package.
        """
        return cls(np.asarray(cana_BooleanNode.outputs, dtype=np.uint8))


    def to_cana(self) -> "cana.boolean_node.BooleanNode":
        """
        Convert the BooleanFunction to a CANA BooleanNode.
    
        This compatibility method constructs and returns a
        ``cana.boolean_node.BooleanNode`` instance corresponding to the
        Boolean function represented by this object.
    
        Returns
        -------
        cana.boolean_node.BooleanNode
            Boolean node object from the CANA library representing the same
            Boolean function.
    
        Raises
        ------
        ImportError
            If the CANA package is not installed.
    
        Notes
        -----
        This method requires the CANA package to be installed.
        """
        cana_boolean_node = utils._require_cana()
        return cana_boolean_node.BooleanNode(k=self.n, outputs=self.f)


