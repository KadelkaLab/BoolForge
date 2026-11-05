"""Regression tests for the unified finite-state Function API."""

import numpy as np
import pytest
import boolforge as bf


@pytest.mark.parametrize('truth,r,r_inputs,n,is_boolean', [
    ([0, 1, 1, 0], 2, None, 2, True),
    ([0, 1, 2, 0, 1, 2], 3, [2, 3], 2, False),
    ([0, 1, 2, 0, 1, 2, 0, 1, 2], 3, None, 2, False),
    ([2], 3, [], 0, False),
    ([1], 2, [], 0, True),
])
def test_constructor_invariants(truth, r, r_inputs, n, is_boolean):
    f = bf.Function(truth, r=r, r_inputs=r_inputs)
    assert f.n == n
    assert f.r == r
    assert f.is_boolean is is_boolean
    assert len(f) == len(truth)
    assert f.f.dtype == np.uint8
    assert np.array_equal(f.f, truth)
    assert len(f.r_inputs) == n
    assert int(np.prod(f.r_inputs.astype(object))) == len(truth)


def test_sequence_inputs_and_metadata():
    f = bf.Function((0, 1, 2, 0, 1, 2), variables=('A', 'B'),
                    name='target', r=3, r_inputs=(2, 3))
    assert f.name == 'target'
    assert f.variables.tolist() == ['A', 'B']
    assert f.r_inputs.tolist() == [2, 3]


@pytest.mark.parametrize('kwargs', [
    {'r': 1}, {'r': 256}, {'r': 2.5},
    {'r_inputs': [1, 2]}, {'r_inputs': [2, 256]},
    {'r_inputs': [2.5, 2]},
])
def test_invalid_cardinalities(kwargs):
    with pytest.raises(ValueError):
        bf.Function([0, 0, 0, 0], **kwargs)


@pytest.mark.parametrize('truth,kwargs', [
    ([0, 1, 0], {}),
    ([0, 1, 2, 0], {}),
    ([0, 1, 2, 3], {'r': 3, 'r_inputs': [2, 2]}),
    ([-1, 0], {}),
    ([0.0, 1.0], {}),
    ([[0, 1], [1, 0]], {}),
    ([], {}),
])
def test_invalid_truth_tables(truth, kwargs):
    with pytest.raises(ValueError):
        bf.Function(truth, **kwargs)


def test_invalid_variable_names():
    with pytest.raises(ValueError):
        bf.Function([0, 1, 1, 0], variables=['only_one'])


def test_boolean_subclass_enforces_binary_domains():
    f = bf.BooleanFunction([0, 1, 1, 0])
    assert isinstance(f, bf.Function)
    assert f.is_boolean
    with pytest.raises(ValueError):
        bf.BooleanFunction([0, 1, 2], r=3)
    with pytest.raises(ValueError):
        bf.BooleanFunction([0, 1, 0, 1], r_inputs=[4])


def test_unchecked_constructor_preserves_class_and_attributes():
    f = bf.BooleanFunction._from_f_unchecked(np.array([0, 1], dtype=np.uint8))
    assert type(f) is bf.BooleanFunction
    assert f.is_boolean and f.n == 1 and f.r == 2
    assert f.r_inputs.tolist() == [2]
    g = bf.Function._from_f_unchecked(
        np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8),
        r=3, r_inputs=[2, 3], variables=['A', 'B'], name='out')
    assert type(g) is bf.Function
    assert not g.is_boolean and g.n == 2 and g.r == 3
    assert g.variables.tolist() == ['A', 'B']
    assert g.name == 'out'
    assert g.properties == {}


def test_call_mixed_radix_and_zero_input():
    f = bf.Function([0, 1, 2, 0, 1, 2], r=3, r_inputs=[2, 3])
    for x0 in range(2):
        for x1 in range(3):
            assert f([x0, x1]) == x1
            assert f((x0, x1), validate=False) == x1
    assert bf.Function([2], r=3, r_inputs=[])([]) == 2
    with pytest.raises(ValueError):
        f([1, 3])
    with pytest.raises(ValueError):
        f([0])
    with pytest.raises(ValueError):
        f([0, 1.5])


def test_setitem_validates_and_invalidates_cache():
    f = bf.BooleanFunction([0, 0, 0, 1])
    f.properties['test'] = 123
    f[0] = 1
    assert f[0] == 1 and f.properties == {}
    f.properties['test'] = 456
    before = f.f.copy()
    for invalid in (-1, 2, 256, 0.5):
        with pytest.raises(ValueError):
            f[0] = invalid
        assert np.array_equal(f.f, before)
        assert f.properties['test'] == 456
    f[1:3] = [1, 0]
    assert f.properties == {}


def test_boolean_operators_preserve_class_and_metadata():
    f = bf.BooleanFunction([0, 0, 0, 1], variables=['A', 'B'], name='out')
    g = bf.BooleanFunction([0, 1, 1, 1], variables=['X', 'Y'])
    for result, expected in [
        (f & g, [0, 0, 0, 1]),
        (f * g, [0, 0, 0, 1]),
        (f | g, [0, 1, 1, 1]),
        (f ^ g, [0, 1, 1, 0]),
        (~f, [1, 1, 1, 0]),
        (1 * f, [0, 0, 0, 1]),
    ]:
        assert type(result) is bf.BooleanFunction
        assert result.f.tolist() == expected
        assert result.variables.tolist() == ['A', 'B']
        assert result.name == 'out'


def test_boolean_operators_reject_incompatible_inputs():
    f = bf.BooleanFunction([0, 1])
    with pytest.raises(ValueError):
        _ = f & bf.BooleanFunction([0, 1, 1, 0])
    with pytest.raises(ValueError):
        _ = f & 2
    g = bf.Function([0, 1, 2], r=3)
    with pytest.raises(NotImplementedError):
        _ = ~g
    with pytest.raises(NotImplementedError):
        _ = g & 1


def test_boolean_expression_constructor():
    f = bf.BooleanFunction('A & B')
    assert f.variables.tolist() == ['A', 'B']
    assert f.f.tolist() == [0, 0, 0, 1]
