# bpe-tokenizer

A byte-pair encoding (BPE) tokenizer, built from scratch (MANUALLY CODED LOL) no `tiktoken`,
no `sentencepiece`, no `transformers`. Part of a larger from-scratch AI
stack: [autograd-engine](https://github.com/RudraDudhat2509/autograd-engine),
[tiny-transformer](https://github.com/RudraDudhat2509/tiny-transformer),
[vector-db](https://github.com/RudraDudhat2509/vector-db), and
[llm-agent](https://github.com/RudraDudhat2509/llm-agent) tie the rest
together.

## What it does

Turns text into a list of integer token ids, and back, using byte-level
BPE (the same family of algorithm GPT-2/GPT-4 use): start from raw
UTF-8 bytes (256 possible values, so any text in any language or emoji
is representable with zero "unknown token" cases), then learn merges
by repeatedly combining whichever adjacent pair of tokens is most
frequent in the training text.

## Status

Scaffold + test suite in place (`tokenizer.py`, `test_tokenizer.py`).
Implementation in progress.

## Usage (once implemented)

```python
from tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train("some training text", vocab_size=500)
ids = tok.encode("some new text")
text = tok.decode(ids)
```

## Running tests

```
pytest test_tokenizer.py -v
```
