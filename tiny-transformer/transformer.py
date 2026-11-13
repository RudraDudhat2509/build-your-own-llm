"""
Stage 5: the trainable transformer block.

This file is a SCAFFOLD, not a solution. Fill in every function body
yourself. New functions get added to this file as the stage grows.

Why this is a separate file from attention.py: attention.py works on
plain floats (no gradients, forward math only - see its own docstring).
This file reuses the SAME proven formulas but rebuilds them on top of
Value, so gradients flow through Wq/Wk/Wv and everything else actually
gets trained. attention.py stays as-is, untouched, still the pure-math
reference whose 72 tests prove the formulas are correct.

Every function below comes with a spec in the style of an interview
question: exact signature, exact output, and exact rules for bad input.
Every error in this file is a ValueError (with a short, clear message),
same house convention as attention.py and charmlp.py.

Batch 1 (the linear layer): one small, reusable building block.

  Q, K, and V projections are all the exact same shape: take a vector,
  dot it with each row of a weight matrix, add that row's bias. That's
  also exactly charmlp.py's hidden-layer shape. Rather than write the
  same loop three times (once each for Q, K, V) and again later for the
  feed-forward layer, build it once:

    input = [1.0, 0.5]
    Wq = [[1, 0], [0, -1]]       (2 rows -> output has 2 numbers)
    bq = [0.0, 0.0]

    row 0: dot([1.0,0.5], [1,0])  + 0 = 1.0
    row 1: dot([1.0,0.5], [0,-1]) + 0 = -0.5
    Q = [1.0, -0.5]

  - init_linear: build a fresh weight matrix + bias vector, randomly
    initialized (same reasoning as charmlp.py's Xavier-init point -
    identical weights never break symmetry).
  - linear: apply an already-built weight matrix + bias to one input
    vector, returning the output vector. This is the function Q, K, V,
    and (later) the feed-forward layer will all be built from.
"""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "autograd-engine"))
from value import Value  # noqa: E402


def init_linear(in_dim: int, out_dim: int) -> tuple[list[list[Value]], list[Value]]:
    """Return a random weight matrix (out_dim x in_dim) and a zero bias vector of length out_dim."""
    if isinstance(in_dim,bool) or not isinstance(in_dim,int) or in_dim<1:
        raise ValueError("in_dim is not positive int")
    if isinstance(out_dim,bool) or not isinstance(out_dim,int) or out_dim<1:
            raise ValueError("out_dim is not positive int") 
    weight_matrix=[[Value(random.uniform(-1,1)) for _ in range(in_dim) ] for _ in range(out_dim)]
    bias_vector=[Value(0.0) for _ in range(out_dim)]
    return (weight_matrix,bias_vector)


def dot(a: list[float], b: list[float]) -> float:
    """Return the dot product of a and b."""
    if not a or not b:
        raise ValueError("A or B is None")
    if len(a)!=len(b):
        raise ValueError("vectors dont have equal lengths")
    return sum(xi*yi for xi,yi in zip(a,b))

def linear(x: list[Value], W: list[list[Value]], b: list[Value]) -> list[Value]:
    """Apply one linear layer: output[i] = dot(W[i], x) + b[i]."""
    if not x:
      raise ValueError("X should not be None")
    if not W or not b or len(W)!=len(b):
      raise ValueError("W or b mismatch")
    output=[]
    for w,b in zip(W,b):
        if len(w)!=len(x):
            raise ValueError("w and x mismatch")
        output.append(dot(x,w)+b)
    return output
