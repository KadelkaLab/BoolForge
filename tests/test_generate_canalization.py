#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:52:59 2026

@author: ckadelka
"""

import numpy as np
import pytest
import boolforge as bf


@pytest.mark.parametrize(
    "r,r_inputs,output_bias",
    [
        (2, [2, 2, 2], None),
        (2, [2, 3, 4], [0.7, 0.3]),
        (3, [2, 3, 4], [0.2, 0.5, 0.3]),
    ],
)
def test_random_non_canalizing_multistate(
        r, r_inputs, output_bias
):
    f = bf.generate.random_non_canalizing_function(
        n=len(r_inputs),
        r=r,
        r_inputs=r_inputs,
        output_bias=output_bias,
        rng=42,
    )

    assert not f.is_canalizing(strong=False)
    assert f.r == r
    assert np.array_equal(f.r_inputs, r_inputs)


def test_random_non_canalizing_non_degenerate_multistate():
    f = bf.generate.random_non_canalizing_non_degenerate_function(
        n=3,
        r=3,
        r_inputs=[2, 3, 4],
        output_bias=[0.2, 0.5, 0.3],
        rng=42,
    )

    assert not f.is_canalizing(strong=False)
    assert not f.is_degenerate()