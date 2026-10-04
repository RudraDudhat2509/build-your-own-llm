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
- [ ] `tiny-transformer` -> basic char-level MLP (scaffolded, not yet
  implemented) → attention → full transformer → training loop →
  training data → serving. Depends on `autograd-engine` via a
  relative import (sibling folder within this repo) - no packaging
  yet.
- [ ] `vector-db` -> brute-force → graph search → HNSW. Not started.
- [ ] `llm-agent` -> ties everything together + owns cloud/CI-CD/
  observability. Not started.

## Running tests

Each component's tests run from within its own folder:

```
cd bpe-tokenizer && pytest test_tokenizer.py -v
cd autograd-engine && pytest test_value.py -v
cd tiny-transformer && pytest test_charmlp.py -v
```
