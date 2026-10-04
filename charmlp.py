"""
Stage 3: basic neural network / char-level language model.

This file is a SCAFFOLD, not a solution. Fill in every function/method
body yourself.

Depends on the Value class from the autograd-engine repo (sibling
folder). Imported via a relative path for now - no packaging yet,
that's a real concern for a later stage (containerization), not this
one.

Recap of the mechanism (see the worked trace in the Stage 3 chat for
the full numeric derivation):
  - build_vocab: a character vocabulary is just every distinct
    character in the training text, each given an integer id - same
    idea as BPE's base vocab from Stage 1, just trivial here (no
    merging, no bytes, just whatever characters show up).
  - CharMLP.__init__: builds every trainable piece as Value objects -
    an embedding table (one row of `embedding_dim` Values per
    vocabulary entry), a hidden layer's weights+biases, and an output
    layer's weights+biases (one row per vocabulary entry, so the
    output has exactly vocab_size logits). Random initialization
    (random.uniform(-1, 1) is fine) matters here - identical initial
    weights never break symmetry (your own Xavier-init reasoning from
    the NLP course).
  - CharMLP.forward: embedding lookup for each id in the context ->
    concatenate them -> linear layer -> tanh -> linear layer -> that
    final layer's outputs ARE the logits (no softmax yet - softmax
    happens in the loss function, not here).
  - CharMLP.parameters: every single Value that training needs to
    update - every embedding row, every weight, every bias. Needed so
    the training loop can zero their gradients and later update them.
  - CharMLP.zero_grad: resets every parameter's .grad to 0. Needed
    between training steps - without this, gradients accumulate
    ACROSS steps (not just across branches within one step), which is
    silently wrong (see Stage 3 theory).
  - softmax_cross_entropy: exactly the worked trace's loss computation
    - exponentiate every logit, divide each by their sum (softmax),
    then take -log of the probability assigned to the correct target
    id. Built entirely from Value's exp/pow/log - no new autograd
    machinery needed, you already built everything this needs.

A note on sum(): Python's built-in sum() starts its accumulator at the
plain int 0, and Value only defines __add__, not __radd__ - so
sum([v1, v2, v3]) tries `0 + v1` first and fails. Either pass an
explicit Value(0.0) start (sum(values, Value(0.0))), or just accumulate
with a plain loop.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "autograd-engine"))
from value import Value  # noqa: E402


def build_vocab(text):
    """
    Return (stoi, itos): stoi maps each distinct character in `text`
    to a unique integer id, itos is the reverse mapping. Ordering
    just needs to be consistent between the two - sorted() is a
    reasonable, deterministic choice.
    """
    raise NotImplementedError


class CharMLP:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size):
        """
        Build every trainable Value this model needs:
          - an embedding table: vocab_size rows, embedding_dim Values each
          - a hidden layer: hidden_dim rows of (embedding_dim * context_size)
            weights each, plus hidden_dim biases
          - an output layer: vocab_size rows of hidden_dim weights each,
            plus vocab_size biases
        Random-initialize weights (random.uniform(-1, 1) is fine);
        biases can start at 0.
        """
        raise NotImplementedError

    def parameters(self):
        """
        Return a flat list of every trainable Value in this model -
        every embedding entry, every weight, every bias. The training
        loop needs this to zero gradients and apply updates.
        """
        raise NotImplementedError

    def zero_grad(self):
        """Reset every parameter's .grad to 0."""
        raise NotImplementedError

    def forward(self, context_ids):
        """
        context_ids: a list of character ids, length == self.context_size.
        Returns: a list of vocab_size Value objects (the logits - NOT
        yet softmax-normalized; that happens in the loss function).
        """
        raise NotImplementedError


def softmax_cross_entropy(logits, target_id):
    """
    logits: list of Value objects (one per vocabulary entry).
    target_id: the integer id of the correct next character.
    Returns: a single Value - the loss - built from exp, the sum of
    exps, pow(-1) for the division, and log, exactly like the worked
    trace.
    """
    raise NotImplementedError
