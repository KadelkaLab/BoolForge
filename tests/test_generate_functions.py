#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import boolforge


def test_random_k_canalizing_function_basic():
    """
    Basic sanity check: generator returns a BooleanFunction
    with correct number of variables.
    """
    rng = np.random.default_rng(0)

    n = 5
    k = 2

    bf = boolforge.random_k_canalizing_function(
        n=n,
        k=k,
        rng=rng,
    )

    assert isinstance(bf, boolforge.BooleanFunction)
    assert bf.n == n


def test_random_k_canalizing_function_exact_depth():
    """
    If exact_depth=True, the returned function must have
    canalizing depth exactly k.
    """
    rng = np.random.default_rng(1)

    n = 6
    k = 3

    bf = boolforge.random_k_canalizing_function(
        n=n,
        k=k,
        exact_depth=True,
        rng=rng,
    )

    props = bf.get_layer_structure()
    assert props["CanalizingDepth"] == k


def test_random_k_canalizing_function_at_least_depth():
    """
    If exact_depth=False, the returned function must have
    canalizing depth at least k.
    """
    rng = np.random.default_rng(2)

    n = 6
    k = 3

    bf = boolforge.random_k_canalizing_function(
        n=n,
        k=k,
        exact_depth=False,
        rng=rng,
    )

    props = bf.get_layer_structure()
    assert props["CanalizingDepth"] >= k


def test_random_nested_canalizing_function():
    """
    k = n should generate a nested canalizing function.
    """
    rng = np.random.default_rng(3)

    n = 5
    k = n

    bf = boolforge.random_k_canalizing_function(
        n=n,
        k=k,
        rng=rng,
    )

    props = bf.get_layer_structure()
    assert props["CanalizingDepth"] == n
    assert props["LayerStructure"][-1] >= 2 or n == 1


def test_random_non_degenerate_function_activities():
    """
    All variables in a non-degenerate Boolean function must have
    strictly positive activity.
    """
    rng = np.random.default_rng(4)

    n = 6
    bias = 0.5

    bf = boolforge.generate.random_non_degenerate_function(
        n=n,
        bias=bias,
        rng=rng,
    )

    activities = bf.get_activities(exact=True)
    assert min(activities) > 0, (
        "Non-degenerate function has variable with zero activity"
    )


def test_random_degenerate_function_activities():
    """
    A degenerate Boolean function must have at least one variable
    with zero activity.
    """
    rng = np.random.default_rng(5)

    n = 6

    bf = boolforge.generate.random_degenerate_function(
        n=n,
        rng=rng,
    )

    activities = bf.get_activities(exact=True)
    assert min(activities) == 0, (
        "Degenerate function has no variable with zero activity"
    )


def test_random_non_canalizing_function_depth_zero():
    """
    A non-canalizing function must have canalizing depth exactly 0.
    """
    rng = np.random.default_rng(6)

    n = 6
    bias = 0.5

    bf = boolforge.generate.random_non_canalizing_function(
        n=n,
        bias=bias,
        rng=rng,
    )

    props = bf.get_layer_structure()
    assert props["CanalizingDepth"] == 0, (
        "random_non_canalizing_function produced a canalizing function"
    )
    
    
def test_random_non_canalizing_non_degenerate_function():
    """
    A non-canalizing, non-degenerate function must have canalizing
    depth 0 and strictly positive activity for all variables.
    """
    rng = np.random.default_rng(7)

    n = 6
    bias = 0.5

    bf = boolforge.generate.random_non_canalizing_non_degenerate_function(
        n=n,
        bias=bias,
        rng=rng,
    )

    props = bf.get_layer_structure()
    assert props["CanalizingDepth"] == 0, (
        "Function is canalizing but should not be"
    )

    activities = bf.get_activities(exact=True)
    assert min(activities) > 0, (
        "Non-degenerate function has variable with zero activity"
    )


def test_random_parity_function_average_sensitivity():
    """
    Parity (XOR-type) Boolean functions must have normalized
    average sensitivity exactly equal to 1.
    """
    rng = np.random.default_rng(8)

    n = 6

    bf = boolforge.generate.random_parity_function(
        n=n,
        rng=rng,
    )

    avg_sens = bf.get_average_sensitivity(exact=True)
    assert avg_sens == 1, (
        "Parity function does not have average sensitivity equal to 1"
    )
    
    
def test_ncf_layer_structure_recovery_from_hamming_weight():
    """
    For n-input nested canalizing functions (NCFs), every odd Hamming
    weight uniquely determines a canalizing layer structure. Generating
    an NCF with that structure must recover the same structure from its
    canalized outputs.
    """
    rng = np.random.default_rng(9)

    n = 6  # small enough to enumerate, large enough to be meaningful

    for w in range(1, 2 ** (n - 1), 2):
        layer_structure = (
            boolforge.utils.hamming_weight_to_ncf_layer_structure(n, w)
        )

        bf = boolforge.random_NCF(
            n=n,
            layer_structure=layer_structure,
            rng=rng,
        )

        recovered = boolforge.get_layer_structure_from_canalized_outputs(
            bf.get_layer_structure()["CanalizedOutputs"]
        )

        assert np.all(
            np.asarray(recovered) == np.asarray(layer_structure)
        ), (
            f"NCF layer structure recovery failed for n={n}, "
            f"Hamming weight w={w}, "
            f"expected {layer_structure}, got {recovered}"
        )
    

def test_random_k_canalizing_invalid_exact_depth():
    """
    exact_depth=True with k = n-1 should raise an AssertionError.
    """
    rng = np.random.default_rng(10)

    n = 5
    k = n - 1

    with pytest.raises(AssertionError):
        boolforge.random_k_canalizing_function(
            n=n,
            k=k,
            exact_depth=True,
            rng=rng,
        )
        
        
def test_random_k_canalizing_function_deterministic_seed():
    """
    Same RNG seed must produce identical functions.
    """
    seed = 42

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    bf1 = boolforge.random_k_canalizing_function(
        n=5,
        k=2,
        rng=rng1,
    )
    bf2 = boolforge.random_k_canalizing_function(
        n=5,
        k=2,
        rng=rng2,
    )

    assert np.array_equal(bf1.f, bf2.f)

