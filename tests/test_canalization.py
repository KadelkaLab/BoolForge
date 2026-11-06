#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:44:18 2026

@author: ckadelka
"""


import numpy as np
import pytest

import boolforge as bf


# ------------------------------------------------------------
# 1. Boolean backward compatibility
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "truth, depth",
    [
        ([0, 0, 0, 0], 0),
        ([0, 1, 1, 0], 0),  # XOR
        ([0, 0, 0, 1], 2),  # AND
        ([0, 1, 1, 1], 2),  # OR
        ([0, 0, 1, 1], 1),  # x0, x1 nonessential
    ],
)
def test_boolean_canalizing_depth(truth, depth):
    f = bf.BooleanFunction(truth)

    assert f.get_canalizing_depth() == depth

    for k in range(f.n + 1):
        assert f.is_k_canalizing(k) == (k <= depth)


# ------------------------------------------------------------
# 2. Constant functions
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "r, r_inputs, value",
    [
        (2, [2, 2], 0),
        (2, [2, 2], 1),
        (3, [2, 3], 2),
        (4, [3, 2], 3),
    ],
)
def test_constant_function_conventions(r, r_inputs, value):
    f = bf.Function(
        [value] * int(np.prod(r_inputs)),
        r=r,
        r_inputs=r_inputs,
    )

    assert f.is_canalizing()
    assert not f.is_canalizing(
        constant_functions_canalizing=False
    )
    assert f.get_canalizing_depth() == 0
    assert not f.is_k_canalizing(1)


def test_zero_input_function():
    f = bf.Function([2], r=3, r_inputs=[])

    assert not f.is_canalizing()
    assert f.is_k_canalizing(0)
    assert not f.is_k_canalizing(1)
    assert f.get_canalizing_depth() == 0


# ------------------------------------------------------------
# 3. Weak versus strong canalization
# ------------------------------------------------------------

def test_weak_but_not_strong_multistate_function():
    # Rows correspond to x0=0,1,2; columns to x1=0,1.
    #
    # [[0, 0],
    #  [0, 1],
    #  [1, 0]]
    #
    # x0=0 forces output 0, but the complementary
    # restrictions depend on x0.

    f = bf.Function(
        [0, 0, 0, 1, 1, 0],
        r=2,
        r_inputs=[3, 2],
    )

    assert f.is_canalizing(strong=False)
    assert not f.is_canalizing(strong=True)
    assert not f.is_k_canalizing(1)
    
def test_weak_but_not_strong_multistate_function_v2():
    # x0 in {0,1} forces output 0.
    # x0=2 leaves the residual function g(x1)=x1.

    f = bf.Function(
        [0, 0, 0,
         0, 0, 0,
         0, 1, 2],
        r=3,
        r_inputs=[3, 3],
    )

    assert f.is_canalizing(strong=True)
    assert f.is_k_canalizing(1)
    assert not f.is_k_canalizing(2)    
    assert f.get_canalizing_depth() == 1



def test_strong_multistate_function():
    # x0 in {0,1} forces output 0.
    # x0=2 leaves the residual function g(x1)=x1.

    f = bf.Function(
        [0, 0, 0,
         0, 0, 0,
         0, 1, 1],
        r=3,
        r_inputs=[3, 3],
    )

    assert f.is_canalizing(strong=True)
    assert f.is_k_canalizing(1)
    assert f.is_k_canalizing(2)    
    assert f.get_canalizing_depth() == 2
    



def test_binary_input_with_multistate_output():
    # f(x0,x1) = x0, with x0 binary and x1 ternary.
    f = bf.Function(
        [0, 0, 0, 1, 1, 1],
        r=3,
        r_inputs=[2, 3],
    )

    assert not f.is_boolean
    assert f.is_canalizing(strong=True)
    assert f.is_k_canalizing(1)
    assert not f.is_k_canalizing(2)


# ------------------------------------------------------------
# 4. Segment restrictions
# ------------------------------------------------------------

def test_nonsegment_strong_canalizing_set():
    # x0 in {0,2} forces 0; x0=1 leaves g(x1)=x1.
    #
    # The set {0,2} is not a lower or upper segment
    # of {0,1,2}.

    f = bf.Function(
        [0, 0, 0,
         0, 1, 2,
         0, 0, 0],
        r=3,
        r_inputs=[3, 3],
    )

    assert f.is_k_canalizing(1, allow_any_canalizing_input_set=True)
    assert not f.is_k_canalizing(1, allow_any_canalizing_input_set=False)


# ------------------------------------------------------------
# 5. Layer structure
# ------------------------------------------------------------

def test_boolean_single_layer():
    f = bf.BooleanFunction([0, 0, 0, 1])

    result = f.get_layer_structure()

    assert result["CanalizingDepth"] == 2
    assert result["NumberOfLayers"] == 1
    assert result["LayerStructure"] == [2]


def test_multistate_layer_structure():
    # f(x0,x1) = 0 if x0 in {0,1}, otherwise x1.
    f = bf.Function(
        [0, 0, 0,
         0, 0, 0,
         0, 1, 2],
        r=3,
        r_inputs=[3, 3],
    )

    result = f.get_layer_structure()

    assert result["CanalizingDepth"] == 1
    assert sum(result["LayerStructure"]) == 1
    assert result["NumberOfLayers"] == 1
    assert np.array_equal(
        result["CoreFunction"].f,
        [0, 1, 2],
    )


def test_noncanalizing_core():
    # x0=0 forces 0; otherwise the residual is XOR.
    f = bf.BooleanFunction(
        [0, 0, 0, 0,
         0, 1, 1, 0]
    )

    result = f.get_layer_structure()

    assert result["CanalizingDepth"] == 1
    assert result["LayerStructure"] == [1]
    assert np.array_equal(
        result["CoreFunction"].f,
        [0, 1, 1, 0],
    )


# ------------------------------------------------------------
# 6. Consistency and caching
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "r, r_inputs, truth",
    [
        (2, [2, 2], [0, 0, 0, 1]),
        (3, [3, 2], [0, 0, 0, 0, 1, 2]),
        (3, [2, 3], [0, 0, 0, 0, 1, 2]),
    ],
)
def test_layer_structure_invariants(r, r_inputs, truth):
    f = bf.Function(truth, r=r, r_inputs=r_inputs)

    result = f.get_layer_structure()

    assert result["CanalizingDepth"] == sum(
        result["LayerStructure"]
    )
    assert result["NumberOfLayers"] == len(
        result["LayerStructure"]
    )
    assert result["CanalizingDepth"] <= f.n

    for k in range(f.n + 1):
        assert f.is_k_canalizing(k) == (
            k <= result["CanalizingDepth"]
        )


def test_layer_structure_caching():
    f = bf.BooleanFunction([0, 0, 0, 1])

    first = f.get_layer_structure()

    assert "CanalizingDepth" in f.properties
    assert "LayerStructure" in f.properties

    second = f.get_layer_structure()

    assert first["CanalizingDepth"] == second["CanalizingDepth"]
    assert first["LayerStructure"] == second["LayerStructure"]


def test_segment_cache_is_separate():
    f = bf.Function(
        [0, 0, 0,
         0, 1, 2,
         0, 0, 0],
        r=3,
        r_inputs=[3, 3],
    )

    unrestricted = f.get_layer_structure(allow_any_canalizing_input_set=True)
    restricted = f.get_layer_structure(allow_any_canalizing_input_set=False)

    assert "CanalizingDepth" in f.properties
    assert "LayerDecompositionAnyInputSet" in f.properties

    assert unrestricted["CanalizingDepth"] >= (
        restricted["CanalizingDepth"]
    )
