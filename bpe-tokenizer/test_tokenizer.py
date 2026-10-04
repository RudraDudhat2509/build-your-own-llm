"""
Stage 1 success criteria, as real pytest tests.

These are black-box: they only call train()/encode()/decode() - the
public contract from tokenizer.py - never anything internal. That's
deliberate: a test that reaches into your internals breaks the moment
you refactor them, even if behavior is still correct.

Run with: pytest 01_tokenizer/test_tokenizer.py -v
"""
import pytest
from tokenizer import BPETokenizer


ROUNDTRIP_CASES = [
    "aaabdaaabac",
    "cat sat cat ran cat sat",
    "hello world, this text was never seen during training!",
    "café",  # accented character not in the training text below
    "cat 🙂",  # emoji - definitely not in training text
]


@pytest.fixture
def trained_tok():
    tok = BPETokenizer()
    tok.train("cat sat cat ran cat sat", vocab_size=258)  # 2 merges
    return tok


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_roundtrip(trained_tok, text):
    """decode(encode(x)) == x must hold for ANY string, trained-on or not."""
    ids = trained_tok.encode(text)
    assert trained_tok.decode(ids) == text


def test_no_unknown_token_crash_on_unseen_unicode(trained_tok):
    """
    Byte-level BPE's core guarantee: text containing characters that
    never appeared during training still encodes without error,
    because the base vocabulary is all 256 possible byte values, not
    a fixed set of "known" characters.
    """
    ids = trained_tok.encode("日本語 emoji test 🎉 café")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert trained_tok.decode(ids) == "日本語 emoji test 🎉 café"


def test_tiny_example_reaches_expected_compression():
    """
    Ground truth from the hand-traced example: "aaabdaaabac" (11 bytes)
    with exactly 3 merges learned (vocab_size 259) must compress to
    exactly 5 tokens - this is true regardless of which pair wins any
    tie along the way (verified independently both ways).
    """
    tok = BPETokenizer()
    tok.train("aaabdaaabac", vocab_size=259)
    ids = tok.encode("aaabdaaabac")
    assert len(ids) == 5


def test_cat_example_reaches_expected_compression():
    """
    Ground truth from the hand-traced worked application: 23 bytes,
    2 merges (vocab_size 258, no ties involved in this one) -> 14
    tokens.
    """
    tok = BPETokenizer()
    tok.train("cat sat cat ran cat sat", vocab_size=258)
    ids = tok.encode("cat sat cat ran cat sat")
    assert len(ids) == 14


def test_vocab_size_256_means_zero_merges():
    """
    Edge case worth thinking about on purpose: vocab_size=256 means
    "no merges at all" - the tokenizer should just be raw bytes.
    encode() on a plain-ASCII string should equal its raw byte values.
    """
    tok = BPETokenizer()
    tok.train("hello", vocab_size=256)
    ids = tok.encode("hello")
    assert ids == list("hello".encode("utf-8"))
