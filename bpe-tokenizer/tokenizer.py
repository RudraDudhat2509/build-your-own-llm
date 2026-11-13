"""
Stage 1: Byte-Pair Encoding (BPE) tokenizer, built from scratch.

Recap of the mechanism (see Stage 1 theory for the full derivation):
  - Start from raw UTF-8 bytes (0-255) as the base vocabulary. That's
    "byte-level" BPE — text.encode("utf-8") gets you there.
  - TRAIN: repeatedly find the most frequent adjacent pair of token ids
    in the sequence, invent a new token id for it, and replace every
    non-overlapping occurrence, left to right. Do this
    (vocab_size - 256) times. Record each merge, IN ORDER — order
    matters, because a later merge can be built on top of an earlier
    one (recall X = ZY from the worked trace).
  - ENCODE: never re-learn anything. Take new text, turn it into bytes,
    and replay the learned merges in the exact order they were learned.
  - DECODE: reverse the mapping — for each token id, look up which byte
    sequence it stands for, concatenate them all, then
    bytes.decode("utf-8") back to a string.

Tie-break rule: when two pairs are tied for "most frequent" during
training, whichever one Python's max() encounters first while scanning
the sequence wins (the natural behavior of max(dict, key=dict.get)).
"""


class BPETokenizer:
    """Byte-pair encoding tokenizer that learns merges from text, then encodes and decodes with them."""

    def __init__(self):
        self.ids = []
        self.merges = {}
        self.vocab = {}
        self.counter = 256
        self.adj_freq = {}

    def count_freq(self, ids):
        """Count every adjacent pair of ids in ids, keyed by the pair's values."""
        count = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            count[pair] = count.get(pair, 0) + 1
        return count

    def replace(self, ids, new_id, x, y):
        """Return a new list with each non-overlapping occurrence of the pair (x, y) replaced by new_id."""
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == x and ids[i + 1] == y:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int) -> None:
        """Learn (vocab_size - 256) merges from text and store them in self.merges and self.vocab."""
        self.ids = list(text.encode('utf-8'))
        self.merges = {}
        self.counter = 256
        self.vocab = {i: bytes([i]) for i in range(256)}
        for _ in range(vocab_size - 256):
            self.adj_freq = self.count_freq(self.ids)
            winning_pair = max(self.adj_freq, key=self.adj_freq.get)
            x, y = winning_pair
            self.ids = self.replace(self.ids, self.counter, x, y)
            self.merges[(x, y)] = self.counter
            self.vocab[self.counter] = self.vocab[x] + self.vocab[y]
            self.counter += 1

    def encode(self, text: str) -> list[int]:
        """Turn a string into token ids using the learned merges. Works on any text; raises if train() has not run."""
        ids = list(text.encode('utf-8'))
        for x, y in self.merges:
            ids = self.replace(ids, self.merges[(x, y)], x, y)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into the original string. Raises if train() has not run."""
        result_bytes = bytes()
        for token_id in ids:
            result_bytes += self.vocab[token_id]
        return result_bytes.decode('utf-8')
