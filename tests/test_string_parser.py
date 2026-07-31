import numpy as np
import pytest

from boolforge.boolean_function import f_from_expression


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def evaluate(expr):
    return f_from_expression(expr)


# ------------------------------------------------------------
# 1 Unicode variable names
# ------------------------------------------------------------

def test_unicode_variable_names():
    f, vars_ = evaluate("α and β")
    assert set(vars_) == {"α", "β"}
    assert len(f) == 4  # 2^2 truth table


def test_unicode_mixed_with_ascii():
    f, vars_ = evaluate("AKT and γ")
    assert set(vars_) == {"AKT", "γ"}
    assert len(f) == 4


# ------------------------------------------------------------
# 2 Names containing /, -, +
# ------------------------------------------------------------

def test_variable_with_slash():
    f, vars_ = evaluate("ERK1/2 and AKT")
    assert set(vars_) == {"ERK1/2", "AKT"}


def test_variable_with_dash():
    f, vars_ = evaluate("IL-6 or TNF-alpha")
    assert set(vars_) == {"IL-6", "TNF-alpha"}


def test_variable_with_plus():
    f, vars_ = evaluate("CD4+ and CD8+")
    assert set(vars_) == {"CD4+", "CD8+"}


# ------------------------------------------------------------
# 3 Leading digits
# ------------------------------------------------------------

def test_variable_starting_with_digit():
    f, vars_ = evaluate("3oxo and AKT")
    assert set(vars_) == {"3oxo", "AKT"}


def test_multiple_digit_leading():
    f, vars_ = evaluate("12HETE or 5LOX")
    assert set(vars_) == {"12HETE", "5LOX"}


# ------------------------------------------------------------
# 4 Unimplemented arithmetic operators should raise error
# ------------------------------------------------------------

@pytest.mark.parametrize("expr", [
    "A / B",
    "A ** B",
])
def test_arithmetic_operators_disallowed(expr):
    with pytest.raises(ValueError):
        f_from_expression(expr)


# ------------------------------------------------------------
# 5 Boolean operator sanity checks
# ------------------------------------------------------------

def test_basic_and_truth_table():
    f, vars_ = evaluate("A and B")
    assert set(vars_) == {"A", "B"}
    # Truth table for AND
    expected = np.array([0, 0, 0, 1], dtype=np.uint8)
    assert np.array_equal(f, expected)


def test_basic_or_truth_table():
    f, vars_ = evaluate("A or B")
    expected = np.array([0, 1, 1, 1], dtype=np.uint8)
    assert np.array_equal(f, expected)


def test_not_operator():
    f, vars_ = evaluate("not A")
    expected = np.array([1, 0], dtype=np.uint8)
    assert np.array_equal(f, expected)


# ------------------------------------------------------------
# 6 Parentheses robustness
# ------------------------------------------------------------

def test_nested_parentheses():
    f, vars_ = evaluate("(A and (B or C))")
    assert set(vars_) == {"A", "B", "C"}
    assert len(f) == 8


# ------------------------------------------------------------
# 7 Whitespace robustness
# ------------------------------------------------------------

def test_whitespace_variations():
    f1, _ = evaluate("A and B")
    f2, _ = evaluate("  A    and    B  ")
    assert np.array_equal(f1, f2)
