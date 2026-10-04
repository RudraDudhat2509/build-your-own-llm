"""
Stage 3 success criteria, as real pytest tests.

Unlike Stage 1/2, weights here are randomly initialized, so exact
output values aren't checkable (and shouldn't be - identical init
would never break symmetry, per your own Xavier-init reasoning). What
IS checkable: the structural contract, and the one thing that actually
matters - does training reduce the loss.

Run with: pytest test_charmlp.py -v
"""
import random
import pytest
from charmlp import build_vocab, CharMLP, softmax_cross_entropy


def test_build_vocab():
    stoi, itos = build_vocab("abc")
    assert len(stoi) == 3
    assert itos[stoi['a']] == 'a'
    assert itos[stoi['b']] == 'b'
    assert itos[stoi['c']] == 'c'


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
