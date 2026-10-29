# build-your-own-llm

An AI agent stack built entirely from scratch, stage by stage: tokenizer,
autograd engine, neural network, transformer, training loop, training
data, serving engine, vector database, and finally the agent that ties
it all together — deployed to the cloud, not just running on a laptop.

you might ask why, even idk, ig im a masochist

Each component lives in its own subfolder, independently runnable
(each has its own tests). Originally built as separate repos
(`bpe-tokenizer`, `autograd-engine`, `tiny-transformer`) and merged
here on 2026-10-04 with full commit history preserved via
`git subtree` 

## Components

- [x] `bpe-tokenizer` -> byte-pair encoding (BPE), from scratch. Done,
  9/9 tests passing.
- [x] `autograd-engine` -> backprop engine (`Value` class: `+`, `*`,
  `tanh`, `exp`, `log`, `pow`/division). Done, 11/11 tests passing.
- [ ] `tiny-transformer` -> basic char-level MLP done, attention
  mechanism done (both tested separately) → full transformer block
  (with RoPE) → training loop → training data → serving (real
  sampling + KV-cache + quantization + simulated tensor/pipeline
  parallelism, added to scope 2026-10-29), still to go. Depends on
  `autograd-engine` via a relative import (sibling folder within
  this repo) - no packaging yet.
- [ ] `vector-db` -> brute-force → graph search → HNSW. Not started.
- [ ] `llm-agent` -> ties everything together + owns cloud/CI-CD/
  observability. Not started.

## Benchmarks

Measured results for the finished components are in [BENCHMARKS.md](BENCHMARKS.md): the tokenizer round-trips 1000 of 1000 random unicode strings and reaches 2.92 bytes per token on held-out text (GPT-2: 3.09, on in-domain Shakespeare only), the autograd engine matches PyTorch gradients to 7e-15 across 200 random expression graphs and trains an MLP identically to PyTorch step for step (25x to 66x slower on 100 points), attention.py passes 72/72 tests, and charmlp.py trains a toy char-level MLP from loss 74.8 down to 0.02 (99.97% reduction) over 200 steps.

## Running tests

Each component's tests run from within its own folder:

```
cd bpe-tokenizer && pytest test_tokenizer.py -v
cd autograd-engine && pytest test_value.py -v
cd tiny-transformer && pytest test_charmlp.py -v
```
