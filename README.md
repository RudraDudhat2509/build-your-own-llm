# tiny-transformer

A GPT-style transformer language model, built from scratch — architecture,
attention, training loop, and serving, all hand-written. Toy/nanoGPT
scale: a few million parameters, trains in minutes-to-hours on CPU or a
single consumer GPU. Part of a larger from-scratch AI stack:
[bpe-tokenizer](https://github.com/RudraDudhat2509/bpe-tokenizer),
[autograd-engine](https://github.com/RudraDudhat2509/autograd-engine),
[vector-db](https://github.com/RudraDudhat2509/vector-db),
[llm-agent](https://github.com/RudraDudhat2509/llm-agent).

## What it covers

- Basic neural network / char-level language model (embeddings, MLP,
  training loop) before attention is introduced
- Scaled dot-product attention and multi-head attention, from raw
  matrix ops
- Full transformer block (attention + MLP + layernorm + residuals) →
  GPT architecture, using `bpe-tokenizer`'s vocabulary
- Training loop + optimizer (own AdamW), producing real trained weights
- A small curated training data pipeline
- A minimal inference/serving engine (`generate()` + HTTP server)

## Status

**Stage 3 (basic char-level MLP) scaffolded** — `charmlp.py` (interface,
no implementation) + `test_charmlp.py` (5 tests, including an
end-to-end "does training reduce the loss" check). Depends on
`autograd-engine` via a relative import (sibling folder) - no proper
packaging yet. Stages 4+ (attention, full transformer, training data,
serving) not started.

## Running tests

```
pytest test_charmlp.py -v
```
