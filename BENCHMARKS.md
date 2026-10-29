# build-your-own-llm benchmarks

Measured 2026-10-08, one Windows machine, Python 3.12.7, torch 2.10.0, numpy 1.26.4. Raw outputs are in `benchmarks/raw/`. Rows marked "measured 2026-10-29" below were added later, same machine, no torch/numpy comparison run for them (pure-Python-only code, nothing to compare against).

**Implemented and benchmarked:** `bpe-tokenizer`, `autograd-engine`, `tiny-transformer` (`attention.py` and `charmlp.py` both fully implemented and tested as of 2026-10-29; no full transformer block yet, just the attention mechanism and a basic char-level MLP in isolation).
**Not implemented, so not benchmarked:** `vector-db`, `llm-agent`. There is no full transformer, so still no validation loss or perplexity to report for real text generation.

To reproduce, first fetch the corpus: `curl -sL -o benchmarks/raw/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` (not committed, 1.1 MB).

| Metric | Value | Baseline | n | Command | Reproduced? |
|---|---|---|---|---|---|
| bpe-tokenizer tests | 9 / 9 pass, 100% line coverage | n/a | 9 | `cd bpe-tokenizer && pytest --cov=.` | yes |
| autograd-engine tests | 11 / 11 pass, 97% line coverage | n/a | 11 | `cd autograd-engine && pytest --cov=.` | yes |
| BPE round trip (decode(encode(s)) == s) | 1000 / 1000 exact (100%) | n/a | 1000 random strings, 0-200 chars, ASCII + Latin + Greek + CJK + Devanagari + emoji | `python benchmarks/bench_bpe.py` | yes |
| BPE compression, vocab 5000 | 2.922 bytes/token | GPT-2 (50,257 vocab): 3.094 bytes/token | 100 KB held-out | same | yes |
| BPE compression, vocab 1000 | 2.227 bytes/token | none at this size | 100 KB held-out | same | yes |
| BPE training speed | vocab 1000 in 29.9 s (24.9 merges/s); vocab 5000 in 106.9 s (44.4 merges/s) | n/a | 100 KB train text | same | yes |
| Autograd gradient check vs PyTorch | max abs diff 7.1e-15; 200 / 200 within 1e-6 (and within 1e-9); max relative grad error 9.3e-16 | torch.autograd float64 | 200 random expression graphs (226 sampled, 26 rejected for leaving the domain) | `python benchmarks/bench_autograd.py` | yes |
| MLP training, XOR | 100% accuracy on 2 of 3 init seeds, 75% on 1 seed. Identical to PyTorch: max per-step loss difference 1.3e-13 | PyTorch MLP, same init and hyperparameters | 3 seeds | same | yes |
| MLP training, make_moons | 94% / 95% / 87% train accuracy over 3 seeds. Identical to PyTorch: max per-step loss difference 1.1e-15 | PyTorch MLP, same init and hyperparameters | 3 seeds, n=100 | same | yes |
| Step time, make_moons (100 points) | 27 to 36 ms per step | PyTorch 0.5 to 1.1 ms: engine is 25x to 66x slower | 3 seeds | same | yes |
| Step time, XOR (4 points) | about 1.0 to 1.3 ms per step | PyTorch 0.8 to 1.0 ms: ratio 1x to 2x | 3 seeds | same | yes |
| attention.py tests (measured 2026-10-29) | 72 / 72 pass | n/a | 72 | `cd tiny-transformer && pytest test_attention.py -v` | yes |
| charmlp.py tests (measured 2026-10-29) | 13 / 13 pass | n/a | 13 | `cd tiny-transformer && pytest test_charmlp.py -v` | yes |
| CharMLP training, toy 3-char cycle (measured 2026-10-29) | loss 74.80 -> 0.0246 over 200 steps (99.97% reduction) | n/a (no comparable PyTorch run built for this) | 1 run, seed 42, vocab_size=3, embedding_dim=2, hidden_dim=4, context_size=1 | see test_charmlp.py::test_training_reduces_loss | yes |

## Compression by vocab size (held-out 100 KB, 100,000 bytes)

| Vocab | Tokens | Bytes/token |
|---|---|---|
| 256 (raw bytes) | 100,000 | 1.000 |
| 512 | 54,374 | 1.839 |
| 1000 | 44,905 | 2.227 |
| 2000 | 38,814 | 2.576 |
| 5000 | 34,221 | 2.922 |
| GPT-2 baseline (50,257) | 32,324 | 3.094 |

## Caveats a skeptical engineer would raise

1. **Compression is not an apples-to-apples win.** The tokenizer was trained on 100 KB of Shakespeare and tested on a different 100 KB of the same book, so it is tuned to this text. GPT-2's tokenizer was trained on web text and has 10x the vocabulary. The honest framing is that a 5,000-token vocab trained on 100 KB reaches 94% of GPT-2's compression on in-domain text. It says nothing about other kinds of text.
2. **The training and test text come from one file** (Tiny Shakespeare, the first 100 KB for training and the last 100 KB held out, no overlap). I did not test a different corpus.
3. **Gradient check scope.** The 200 graphs reuse subexpressions (so gradient accumulation is exercised) and use all six ops, but only 3 inputs and 3 to 12 ops each. Inputs are in [0.3, 2.0]. Graphs where a value left the valid domain (log of a non-positive number, exp over 20, overflow) were rejected, 26 of 226. The engine's `log` backward has an odd branch for zero input that this check never hits.
4. **MLP accuracy is on the training set**, not held-out data, and the networks are tiny (2-8-1, full-batch gradient descent). The point is that the engine matches PyTorch step for step (loss differences at floating-point rounding level), not that the model is good. XOR seed 1 gets stuck at 75% in both implementations, which is a local minimum of this setup and not an engine bug.
5. **Speed.** The 25x to 66x slowdown is on 100 points. On 4 points PyTorch's fixed per-call overhead hides the gap (ratio about 1x). Both are single-thread CPU, float64, and I did not tune either.
6. **(Fixed 2026-10-29, was true at measurement time)** The engine was originally missing operators, which shaped the test code written against it at the time: there was no subtraction, negation, division operator, or constant mixing (`Value + 3` failed). Subtraction was built from `* Value(-1.0)` as a workaround. The engine now has `__neg__`, `__sub__`, and constant-mixing (`__radd__`/`__rmul__`, plain numbers auto-wrap in `__add__`/`__mul__`) - see `autograd-engine/test_value.py`'s arithmetic-completeness tests. The gradient-check and MLP numbers above were measured before this fix and were not rerun after it; they're unaffected since neither benchmark script used the new operators.
7. **BPE encode is slow** (62.6 s to encode 100 KB at vocab 5000, since it replays every merge over the whole text). I did not compare speed against tiktoken, which is orders of magnitude faster.
8. No source file was modified. New files are only in `benchmarks/`.

## Not run

- `vector-db`, `llm-agent`: not started.
- A full transformer block (attention + MLP + layernorm + residuals combined) - `attention.py` and `charmlp.py` are tested separately, not yet wired together.
- Any GPU or multi-threaded run.
