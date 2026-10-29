"""
Stage 3 success criteria, as real pytest tests.

Batch 1 (build_vocab, softmax_cross_entropy): deterministic, exact
values checkable - same style as the attention.py tests.

Batch 2 (CharMLP): weights are randomly initialized, so exact output
values aren't checkable (and shouldn't be - identical init would never
break symmetry, per your own Xavier-init reasoning). What IS checkable:
the structural contract, and the one thing that actually matters -
does training reduce the loss.

Run with: pytest test_charmlp.py -v
"""
import random
import pytest
from charmlp import build_vocab, CharMLP, softmax_cross_entropy  # sets up sys.path for value
from value import Value


# ------------------------------------------------------------ build_vocab

def test_build_vocab_worked_example():
    stoi, itos = build_vocab("cab")
    assert stoi == {"a": 0, "b": 1, "c": 2}
    assert itos == {0: "a", 1: "b", 2: "c"}


def test_build_vocab_roundtrips():
    stoi, itos = build_vocab("hello")
    for ch, i in stoi.items():
        assert itos[i] == ch


def test_build_vocab_deterministic():
    assert build_vocab("hello") == build_vocab("hello")


def test_build_vocab_none_or_empty_raises_value_error():
    with pytest.raises(ValueError):
        build_vocab(None)
    with pytest.raises(ValueError):
        build_vocab("")


# ----------------------------------------------------- softmax_cross_entropy

def test_softmax_cross_entropy_worked_example():
    logits = [Value(0.0), Value(1.0), Value(-1.0)]
    loss = softmax_cross_entropy(logits, 1)
    assert loss.data == pytest.approx(0.408, abs=1e-3)


def test_softmax_cross_entropy_confident_correct_is_low_loss():
    logits = [Value(10.0), Value(0.0), Value(0.0)]
    loss = softmax_cross_entropy(logits, 0)
    assert loss.data < 0.01


def test_softmax_cross_entropy_confident_wrong_is_high_loss():
    logits = [Value(10.0), Value(0.0), Value(0.0)]
    loss = softmax_cross_entropy(logits, 1)
    assert loss.data > 5.0


def test_softmax_cross_entropy_none_or_empty_raises_value_error():
    with pytest.raises(ValueError):
        softmax_cross_entropy(None, 0)
    with pytest.raises(ValueError):
        softmax_cross_entropy([], 0)


def test_softmax_cross_entropy_bad_target_id_raises_value_error():
    logits = [Value(0.0), Value(1.0)]
    with pytest.raises(ValueError):
        softmax_cross_entropy(logits, 5)        # out of range
    with pytest.raises(ValueError):
        softmax_cross_entropy(logits, -1)        # negative
    with pytest.raises(ValueError):
        softmax_cross_entropy(logits, True)      # bool, not a real int
    with pytest.raises(ValueError):
        softmax_cross_entropy(logits, "0")       # not an int at all


# ------------------------------------------------------------------ CharMLP

def test_forward_returns_one_logit_per_vocab_entry():
    model = CharMLP(vocab_size=3, embedding_dim=2, hidden_dim=4, context_size=1)
    logits = model.forward([0])
    assert len(logits) == 3


def test_parameters_nonempty():
    model = CharMLP(vocab_size=3, embedding_dim=2, hidden_dim=4, context_size=1)
    params = model.parameters()
    assert len(params) > 0


def test_zero_grad_resets_every_parameter():
    model = CharMLP(vocab_size=3, embedding_dim=2, hidden_dim=4, context_size=1)
    params = model.parameters()
    for p in params:
        p.grad = 99
    model.zero_grad()
    assert all(p.grad == 0 for p in params)


def test_training_reduces_loss():
    """
    The real end-to-end proof: embedding, forward, softmax_cross_entropy,
    backward, and the weight update all have to work together correctly
    for this to pass. Tiny repeating pattern (a->b, b->c, c->a) trained
    for 200 steps - loss should drop dramatically, not just slightly.
    """
    random.seed(42)
    model = CharMLP(vocab_size=3, embedding_dim=2, hidden_dim=4, context_size=1)
    data = [(0, 1), (1, 2), (2, 0)] * 20

    losses = []
    lr = 0.1
    for _ in range(200):
        total_loss = None
        for ctx, target in data:
            logits = model.forward([ctx])
            loss = softmax_cross_entropy(logits, target)
            total_loss = loss if total_loss is None else total_loss + loss

        model.zero_grad()
        total_loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad

        losses.append(total_loss.data)

    assert losses[-1] < losses[0] * 0.1  # at least 90% reduction
