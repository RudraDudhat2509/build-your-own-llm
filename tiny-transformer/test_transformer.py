"""
Stage 5 success criteria, as real pytest tests.

Batch 1 (init_linear, linear): deterministic where it can be (linear's
actual math), structural where weights are random (init_linear's shapes).

Run with: pytest test_transformer.py -v
"""
import random
import pytest
from transformer import init_linear, linear
from value import Value


# --------------------------------------------------------------- linear

def test_linear_worked_example():
    x = [Value(1.0), Value(0.5)]
    W = [[Value(1.0), Value(0.0)], [Value(0.0), Value(-1.0)]]
    b = [Value(0.0), Value(0.0)]
    out = linear(x, W, b)
    assert [v.data for v in out] == pytest.approx([1.0, -0.5])


def test_linear_uses_bias():
    x = [Value(1.0)]
    W = [[Value(1.0)]]
    b = [Value(5.0)]
    out = linear(x, W, b)
    assert out[0].data == pytest.approx(6.0)


def test_linear_output_length_matches_number_of_rows():
    x = [Value(1.0), Value(2.0)]
    W = [[Value(1.0), Value(0.0)], [Value(0.0), Value(1.0)], [Value(1.0), Value(1.0)]]
    b = [Value(0.0), Value(0.0), Value(0.0)]
    out = linear(x, W, b)
    assert len(out) == 3


def test_linear_backward_flows_to_weights():
    """
    Gradients must actually reach W and b, not just x - that's the
    whole point of this file existing (attention.py's version can't
    do this).
    """
    x = [Value(2.0)]
    W = [[Value(3.0)]]
    b = [Value(1.0)]
    out = linear(x, W, b)
    out[0].backward()
    assert W[0][0].grad == pytest.approx(2.0)   # d(out)/dW = x
    assert b[0].grad == pytest.approx(1.0)       # d(out)/db = 1
    assert x[0].grad == pytest.approx(3.0)       # d(out)/dx = W


def test_linear_none_or_empty_raises_value_error():
    with pytest.raises(ValueError):
        linear(None, [[Value(1.0)]], [Value(0.0)])
    with pytest.raises(ValueError):
        linear([], [[Value(1.0)]], [Value(0.0)])
    with pytest.raises(ValueError):
        linear([Value(1.0)], None, [Value(0.0)])
    with pytest.raises(ValueError):
        linear([Value(1.0)], [[Value(1.0)]], None)


def test_linear_w_b_length_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        linear([Value(1.0)], [[Value(1.0)], [Value(1.0)]], [Value(0.0)])  # 2 rows, 1 bias


def test_linear_row_length_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        linear([Value(1.0), Value(2.0)], [[Value(1.0)]], [Value(0.0)])  # row has 1, x has 2


# ---------------------------------------------------------------- init_linear

def test_init_linear_shapes():
    random.seed(0)
    W, b = init_linear(2, 3)
    assert len(W) == 3
    assert all(len(row) == 2 for row in W)
    assert len(b) == 3


def test_init_linear_biases_start_at_zero():
    W, b = init_linear(2, 3)
    assert all(bi.data == 0.0 for bi in b)


def test_init_linear_weights_are_values_not_plain_numbers():
    W, b = init_linear(2, 2)
    assert isinstance(W[0][0], Value)
    assert isinstance(b[0], Value)


def test_init_linear_different_calls_give_different_weights():
    W1, _ = init_linear(2, 2)
    W2, _ = init_linear(2, 2)
    assert [v.data for row in W1 for v in row] != [v.data for row in W2 for v in row]


def test_init_linear_bad_dims_raise_value_error():
    with pytest.raises(ValueError):
        init_linear(0, 3)
    with pytest.raises(ValueError):
        init_linear(2, 0)
    with pytest.raises(ValueError):
        init_linear(True, 3)   # bool is not a real int here
    with pytest.raises(ValueError):
        init_linear(2.0, 3)    # float is not an int either
