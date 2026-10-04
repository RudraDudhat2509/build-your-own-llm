# autograd-engine

A backpropagation engine, built from scratch — no `torch.autograd`.
Part of a larger from-scratch AI stack:
[bpe-tokenizer](https://github.com/RudraDudhat2509/bpe-tokenizer),
[tiny-transformer](https://github.com/RudraDudhat2509/tiny-transformer),
[vector-db](https://github.com/RudraDudhat2509/vector-db),
[llm-agent](https://github.com/RudraDudhat2509/llm-agent).

## What it does

A tiny scalar-valued computational graph: every op (`+`, `*`, `tanh`, …)
records what produced it, so calling `.backward()` on the output walks
the graph in reverse and computes the gradient of every value that fed
into it, via the chain rule — the mechanism every deep learning
framework is built on top of.

## Status

Scaffold + test suite in place (`value.py`, `test_value.py`).
Implementation in progress.

## Usage

```python
from value import Value

a, b, c = Value(2), Value(-3), Value(10)
d = a * b + c
d.backward()
print(d.data, a.grad, b.grad, c.grad)  # 4 -3 2 1

# also supports tanh, exp, log, and pow (division = a * b**-1)
x = Value(2).tanh()
```

## Running tests

```
pytest test_value.py -v
```
