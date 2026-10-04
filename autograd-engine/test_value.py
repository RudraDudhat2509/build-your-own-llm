"""
Stage 2 success criteria, as real pytest tests.

Black-box: only calls the public contract (Value(), +, *, .data, .grad,
.backward()) - never anything internal. Both gradient tests reuse the
exact numbers from the hand-traced examples in the Stage 2 chat, so
you already know, independently of this test file, what the right
answer is.

Run with: pytest test_value.py -v
"""
import math
import pytest
from value import Value


def test_add():
    a = Value(2)
    b = Value(3)
    c = a + b
    assert c.data == 5


def test_mul():
    a = Value(2)
    b = Value(3)
    c = a * b
    assert c.data == 6


def test_grad_starts_at_zero():
    v = Value(5)
    assert v.grad == 0


def test_backward_simple_chain():
    """
    a=2, b=-3, c=10, d=a*b+c. Hand-traced: d=4, a.grad=-3, b.grad=2,
    c.grad=1 - the first worked example from Stage 2.
    """
    a, b, c = Value(2), Value(-3), Value(10)
    d = a * b + c
    d.backward()

    assert d.data == 4
    assert a.grad == -3
    assert b.grad == 2
    assert c.grad == 1


def test_backward_accumulates_when_reused():
    """
    a=3, b=-2, c=4, L=a*b+a*c. `a` is used TWICE - this is the test
    that actually exercises accumulation. Hand-traced: L=6, a.grad=2
    (= -2 from the a*b branch + 4 from the a*c branch, NOT just one
    of them), b.grad=3, c.grad=3. Cross-checked by direct
    differentiation: L=a*(b+c), dL/da=b+c=2.
    """
    a, b, c = Value(3), Value(-2), Value(4)
    L = a * b + a * c
    L.backward()

    assert L.data == 6
    assert a.grad == 2
    assert b.grad == 3
    assert c.grad == 3


def test_tanh_forward():
    assert Value(0).tanh().data == pytest.approx(0)


def test_exp_forward():
    assert Value(0).exp().data == pytest.approx(1)


def test_log_forward():
    assert Value(1).log().data == pytest.approx(0)


def test_pow_forward():
    assert (Value(2) ** 3).data == 8


def test_tanh_backward():
    """x=2, y=tanh(x). Local derivative 1-tanh(x)^2 - verified against
    Python's own math.tanh, not hand-rounded."""
    x = Value(2)
    y = x.tanh()
    y.backward()

    expected = 1 - math.tanh(2) ** 2
    assert x.grad == pytest.approx(expected)


def test_division_via_pow():
    """
    Division isn't its own op - it's multiplication by a pow(-1).
    c = a * b**-1 = a / b = 6/3 = 2. Cross-checked by direct
    differentiation: dc/da = 1/b = 1/3, dc/db = -a/b^2 = -6/9.
    """
    a, b = Value(6), Value(3)
    c = a * (b ** -1)
    c.backward()

    assert c.data == pytest.approx(2)
    assert a.grad == pytest.approx(1 / 3)
    assert b.grad == pytest.approx(-6 / 9)
