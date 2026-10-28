"""
Stage 4 success criteria, as real pytest tests.

Unlike Stage 3, nothing here is random, so exact values ARE checkable.
The worked example from the attention.py docstring is the main check:
Q = [[1, 0]], K = [[1, 0], [0, 1], [1, 1]] -> weights [[0.401, 0.198, 0.401]],
and with V = [[10, 0], [0, 10], [5, 5]] -> output [[6.017, 3.983]].
The rest check the "Raise ValueError if ..." rules in each docstring.

Run with: pytest test_attention.py -v
"""
import pytest
from attention import (
    softmax, dot, attention_weights, apply_attention, attention,
    causal_mask, attention_scores, apply_mask, masked_attention,
)


K1 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
V1 = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
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


# -------------------------------------------------------- apply_attention

def test_apply_attention_worked_example():
    out = apply_attention([[0.4011, 0.1978, 0.4011]], V1)
    assert len(out) == 1
    assert out[0] == pytest.approx([6.017, 3.983], abs=1e-2)


def test_apply_attention_small_hand_example():
    # 0.4*[10,0] + 0.2*[0,10] + 0.4*[5,5] = [4+0+2, 0+2+2] = [6, 4]
    out = apply_attention([[0.4, 0.2, 0.4]], V1)
    assert out[0] == pytest.approx([6.0, 4.0])


def test_apply_attention_one_hot_weights_pick_that_value_exactly():
    out = apply_attention([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], V1)
    assert out[0] == pytest.approx([0.0, 10.0])
    assert out[1] == pytest.approx([5.0, 5.0])


def test_apply_attention_equal_weights_give_the_average():
    third = 1 / 3
    out = apply_attention([[third, third, third]], V1)
    assert out[0] == pytest.approx([5.0, 5.0])


def test_apply_attention_shape_is_n_queries_by_d_v():
    V = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]          # 2 values, d_v = 3
    out = apply_attention([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], V)
    assert len(out) == 3
    assert all(len(row) == 3 for row in out)
    assert out[0] == pytest.approx([2.5, 3.5, 4.5])


def test_apply_attention_uses_weights_exactly_as_given():
    # No normalizing: weight 2.0 means "2 times value 0".
    out = apply_attention([[2.0, 0.0, 0.0]], V1)
    assert out[0] == pytest.approx([20.0, 0.0])


def test_apply_attention_does_not_change_its_inputs():
    weights = [[0.5, 0.25, 0.25]]
    V = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
    apply_attention(weights, V)
    assert weights == [[0.5, 0.25, 0.25]]
    assert V == [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]


def test_apply_attention_none_raises_value_error():
    with pytest.raises(ValueError):
        apply_attention(None, V1)
    with pytest.raises(ValueError):
        apply_attention([[1.0, 0.0, 0.0]], None)


def test_apply_attention_empty_raises_value_error():
    with pytest.raises(ValueError):
        apply_attention([], V1)
    with pytest.raises(ValueError):
        apply_attention([[1.0, 0.0, 0.0]], [])


def test_apply_attention_weights_row_must_have_one_weight_per_value():
    with pytest.raises(ValueError):
        apply_attention([[0.5, 0.5]], V1)                # 2 weights, 3 values
    with pytest.raises(ValueError):
        apply_attention([[0.2, 0.3, 0.5], [1.0]], V1)    # second row too short


def test_apply_attention_empty_value_vector_raises_value_error():
    with pytest.raises(ValueError):
        apply_attention([[1.0]], [[]])


def test_apply_attention_ragged_values_raise_value_error():
    with pytest.raises(ValueError):
        apply_attention([[0.5, 0.5]], [[1.0, 2.0], [3.0]])
# -------------------------------------------------------------- attention
 
def test_attention_worked_example():
    out, weights = attention([[1.0, 0.0]], K1, V1)
    assert weights[0] == pytest.approx([0.4011, 0.1978, 0.4011], abs=1e-3)
    assert out[0] == pytest.approx([6.017, 3.983], abs=1e-2)
 
 
def test_attention_returns_a_tuple_of_output_and_weights():
    result = attention([[1.0, 0.0]], K1, V1)
    assert isinstance(result, tuple)
    assert len(result) == 2
 
 
def test_attention_two_queries_second_row_is_the_mirror():
    out, weights = attention([[1.0, 0.0], [0.0, 1.0]], K1, V1)
    assert len(out) == 2 and len(weights) == 2
    assert weights[1] == pytest.approx([0.1978, 0.4011, 0.4011], abs=1e-3)
    assert out[1] == pytest.approx([3.983, 6.017], abs=1e-2)
 
 
def test_attention_matches_its_two_building_blocks():
    Q = [[0.3, -0.2], [1.0, 0.5]]
    out, weights = attention(Q, K1, V1)
    expected_weights = attention_weights(Q, K1)
    expected_out = apply_attention(expected_weights, V1)
    for i in range(len(Q)):
        assert weights[i] == pytest.approx(expected_weights[i])
        assert out[i] == pytest.approx(expected_out[i])
 
 
def test_attention_output_shape_follows_v_not_q():
    V = [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    out, weights = attention([[1.0, 0.0]], K1, V)
    assert len(out[0]) == 4
    assert len(weights[0]) == 3
 
 
def test_attention_needs_one_value_per_key():
    with pytest.raises(ValueError):
        attention([[1.0, 0.0]], K1, [[1.0, 2.0], [3.0, 4.0]])   # 3 keys, 2 values
 
 
def test_attention_bad_q_k_or_v_raise_value_error():
    with pytest.raises(ValueError):
        attention(None, K1, V1)
    with pytest.raises(ValueError):
        attention([[1.0, 0.0]], [], V1)
    with pytest.raises(ValueError):
        attention([[1.0, 0.0, 0.0]], K1, V1)       # Q and K disagree on d
    with pytest.raises(ValueError):
        attention([[1.0, 0.0]], K1, None)
 
 
def test_attention_does_not_change_its_inputs():
    Q = [[1.0, 0.0]]
    K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
    attention(Q, K, V)
    assert Q == [[1.0, 0.0]]
    assert K == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert V == [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]


# ----------------------------------------------------------- causal_mask

def test_causal_mask_worked_example():
    assert causal_mask(3) == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


def test_causal_mask_size_one():
    assert causal_mask(1) == [[True]]


def test_causal_mask_entries_are_real_bools():
    for row in causal_mask(2):
        for entry in row:
            assert isinstance(entry, bool)


def test_causal_mask_bool_raises_value_error():
    with pytest.raises(ValueError):
        causal_mask(True)
    with pytest.raises(ValueError):
        causal_mask(False)


def test_causal_mask_non_int_raises_value_error():
    with pytest.raises(ValueError):
        causal_mask(2.0)
    with pytest.raises(ValueError):
        causal_mask("3")
    with pytest.raises(ValueError):
        causal_mask(None)


def test_causal_mask_less_than_one_raises_value_error():
    with pytest.raises(ValueError):
        causal_mask(0)
    with pytest.raises(ValueError):
        causal_mask(-1)


# ------------------------------------------------------------ attention_scores

def test_attention_scores_worked_example():
    scores = attention_scores([[1.0, 0.0]], K1)
    assert scores[0] == pytest.approx([0.7071, 0.0, 0.7071], abs=1e-3)


def test_attention_scores_rows_do_not_sum_to_one():
    scores = attention_scores([[1.0, 0.0]], K1)
    assert sum(scores[0]) != pytest.approx(1.0)


def test_attention_scores_can_be_negative():
    scores = attention_scores([[-1.0, 0.0]], K1)
    assert scores[0][0] < 0


def test_attention_scores_does_not_change_its_inputs():
    Q = [[1.0, 0.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    attention_scores(Q, K)
    assert Q == [[1.0, 0.0]]
    assert K == [[1.0, 0.0], [0.0, 1.0]]


def test_attention_scores_none_or_empty_raises_value_error():
    with pytest.raises(ValueError):
        attention_scores(None, K1)
    with pytest.raises(ValueError):
        attention_scores([], K1)
    with pytest.raises(ValueError):
        attention_scores([[1.0, 0.0]], [])


def test_attention_scores_ragged_or_disagreeing_raises_value_error():
    with pytest.raises(ValueError):
        attention_scores([[1.0, 0.0], [1.0]], K1)
    with pytest.raises(ValueError):
        attention_scores([[1.0, 0.0, 0.0]], K1)


# ------------------------------------------------------------------- apply_mask

def test_apply_mask_worked_example():
    out = apply_mask([[1.0, 2.0], [3.0, 4.0]], [[True, False], [True, True]])
    assert out == [[1.0, float("-inf")], [3.0, 4.0]]


def test_apply_mask_blocked_is_negative_infinity_not_positive():
    out = apply_mask([[1.0]], [[False]])
    assert out[0][0] == float("-inf")


def test_apply_mask_row_fully_blocked_is_allowed():
    out = apply_mask([[1.0, 2.0]], [[False, False]])
    assert out == [[float("-inf"), float("-inf")]]


def test_apply_mask_does_not_change_its_inputs():
    scores = [[1.0, 2.0]]
    mask = [[True, False]]
    apply_mask(scores, mask)
    assert scores == [[1.0, 2.0]]
    assert mask == [[True, False]]


def test_apply_mask_none_or_empty_raises_value_error():
    with pytest.raises(ValueError):
        apply_mask(None, [[True]])
    with pytest.raises(ValueError):
        apply_mask([[1.0]], None)
    with pytest.raises(ValueError):
        apply_mask([], [])


def test_apply_mask_shape_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        apply_mask([[1.0, 2.0]], [[True]])
    with pytest.raises(ValueError):
        apply_mask([[1.0], [2.0]], [[True]])


# -------------------------------------------------------------- masked_attention

def test_masked_attention_worked_example():
    Q = K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
    mask = causal_mask(3)
    output, weights = masked_attention(Q, K, V, mask)

    assert weights[0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-3)
    assert weights[1] == pytest.approx([0.330, 0.670, 0.0], abs=1e-3)
    assert weights[2] == pytest.approx([0.248, 0.248, 0.503], abs=1e-3)

    assert output[0] == pytest.approx([10.0, 0.0], abs=1e-2)
    assert output[1] == pytest.approx([3.302, 6.698], abs=1e-2)
    assert output[2] == pytest.approx([5.0, 5.0], abs=1e-2)


def test_masked_attention_rows_still_sum_to_one():
    Q = K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
    _, weights = masked_attention(Q, K, V, causal_mask(3))
    for row in weights:
        assert sum(row) == pytest.approx(1.0)


def test_masked_attention_blocked_positions_are_exactly_zero():
    Q = K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
    _, weights = masked_attention(Q, K, V, causal_mask(3))
    assert weights[0][1] == 0.0
    assert weights[0][2] == 0.0
    assert weights[1][2] == 0.0


def test_masked_attention_query_blocked_from_every_key_raises_value_error():
    Q = K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[10.0, 0.0], [0.0, 10.0]]
    with pytest.raises(ValueError):
        masked_attention(Q, K, V, [[False, False], [True, True]])


def test_masked_attention_v_k_length_mismatch_raises_value_error():
    Q = K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[10.0, 0.0], [0.0, 10.0]]  # only 2 values, 3 keys
    with pytest.raises(ValueError):
        masked_attention(Q, K, V, causal_mask(3))


def test_masked_attention_wrong_mask_shape_raises_value_error():
    Q = K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
    with pytest.raises(ValueError):
        masked_attention(Q, K, V, causal_mask(2))  # 2x2 mask for 3 queries/keys


def test_masked_attention_does_not_change_its_inputs():
    Q = K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
    mask = causal_mask(3)
    masked_attention(Q, K, V, mask)
    assert Q == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert V == [[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
    assert mask == causal_mask(3)
 