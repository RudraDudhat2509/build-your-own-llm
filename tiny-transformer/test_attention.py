"""
Stage 4 success criteria, as real pytest tests.

Unlike Stage 3, nothing here is random, so exact values ARE checkable.
The worked example from the attention.py docstring is the main check:
Q = [[1, 0]], K = [[1, 0], [0, 1], [1, 1]] -> weights [[0.401, 0.198, 0.401]].
The rest check the "Raise ValueError if ..." rules in each docstring.

Run with: pytest test_attention.py -v
"""
import pytest
from attention import softmax, dot, attention_weights


K1 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
INF = float("inf")
NAN = float("nan")


# ---------------------------------------------------------------- softmax

def test_softmax_basic_example():
    result = softmax([2.0, 1.0, 0.0])
    assert result == pytest.approx([0.665, 0.245, 0.090], abs=1e-3)


def test_softmax_sums_to_one():
    result = softmax([0.3, -2.0, 5.0, 1.1])
    assert sum(result) == pytest.approx(1.0)


def test_softmax_big_numbers_do_not_overflow():
    result = softmax([1000.0, 1001.0, 1002.0])
    assert result == pytest.approx([0.090, 0.245, 0.665], abs=1e-3)


def test_softmax_single_number_gets_everything():
    assert softmax([7.0]) == pytest.approx([1.0])


def test_softmax_equal_inputs_get_equal_shares():
    assert softmax([3.0, 3.0, 3.0, 3.0]) == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_softmax_does_not_change_its_input():
    x = [5.0, 6.0, 7.0]
    softmax(x)
    assert x == [5.0, 6.0, 7.0]


def test_softmax_none_raises_value_error():
    with pytest.raises(ValueError):
        softmax(None)


def test_softmax_empty_list_raises_value_error():
    with pytest.raises(ValueError):
        softmax([])


def test_softmax_nan_raises_value_error():
    with pytest.raises(ValueError):
        softmax([1.0, NAN, 2.0])


def test_softmax_positive_infinity_raises_value_error():
    with pytest.raises(ValueError):
        softmax([1.0, INF])


def test_softmax_negative_infinity_gets_exactly_zero():
    result = softmax([0.0, -INF])
    assert result[0] == pytest.approx(1.0)
    assert result[1] == 0.0


def test_softmax_all_negative_infinity_raises_value_error():
    with pytest.raises(ValueError):
        softmax([-INF, -INF])


# -------------------------------------------------------------------- dot

def test_dot_small_example():
    assert dot([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0)


def test_dot_three_numbers():
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)


def test_dot_different_lengths_raises_value_error():
    with pytest.raises(ValueError):
        dot([1.0, 2.0], [1.0, 2.0, 3.0])


def test_dot_empty_raises_value_error():
    with pytest.raises(ValueError):
        dot([], [])


def test_dot_none_raises_value_error():
    with pytest.raises(ValueError):
        dot(None, [1.0])
    with pytest.raises(ValueError):
        dot([1.0], None)


# ------------------------------------------------------ attention_weights

def test_attention_weights_worked_example():
    weights = attention_weights([[1.0, 0.0]], K1)
    assert len(weights) == 1
    assert weights[0] == pytest.approx([0.4011, 0.1978, 0.4011], abs=1e-3)


def test_attention_weights_shape_and_rows_sum_to_one():
    Q = [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]
    weights = attention_weights(Q, K1)
    assert len(weights) == 4
    for row in weights:
        assert len(row) == 3
        assert sum(row) == pytest.approx(1.0)


def test_attention_weights_zero_query_gives_equal_weights():
    weights = attention_weights([[0.0, 0.0]], K1)
    assert weights[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_attention_weights_divides_by_sqrt_d():
    # d = 4. dot(query, key) = 4 for the first key, 0 for the second.
    # 4 / sqrt(4) = 2 and 0 / sqrt(4) = 0, so softmax([2, 0]) = [0.8808, 0.1192].
    Q = [[2.0, 2.0, 0.0, 0.0]]
    K = [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    weights = attention_weights(Q, K)
    assert weights[0] == pytest.approx([0.8808, 0.1192], abs=1e-3)


def test_attention_weights_does_not_change_its_inputs():
    Q = [[1.0, 0.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    attention_weights(Q, K)
    assert Q == [[1.0, 0.0]]
    assert K == [[1.0, 0.0], [0.0, 1.0]]


def test_attention_weights_none_raises_value_error():
    with pytest.raises(ValueError):
        attention_weights(None, K1)
    with pytest.raises(ValueError):
        attention_weights([[1.0, 0.0]], None)


def test_attention_weights_empty_q_or_k_raises_value_error():
    with pytest.raises(ValueError):
        attention_weights([], K1)
    with pytest.raises(ValueError):
        attention_weights([[1.0, 0.0]], [])


def test_attention_weights_empty_vector_raises_value_error():
    with pytest.raises(ValueError):
        attention_weights([[]], [[]])


def test_attention_weights_q_and_k_disagree_on_d_raises_value_error():
    with pytest.raises(ValueError):
        attention_weights([[1.0, 0.0, 0.0]], K1)


def test_attention_weights_ragged_rows_raise_value_error():
    with pytest.raises(ValueError):
        attention_weights([[1.0, 0.0], [1.0]], K1)
    with pytest.raises(ValueError):
        attention_weights([[1.0, 0.0]], [[1.0, 0.0], [1.0]])