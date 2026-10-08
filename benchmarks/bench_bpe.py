"""BPE tokenizer benchmark. Train on first 100 KB of Tiny Shakespeare, evaluate on the LAST 100 KB (held out, disjoint)."""
import random, sys, time, pathlib
root = pathlib.Path(__file__).parent
sys.path.insert(0, str(root.parent / "bpe-tokenizer"))
from tokenizer import BPETokenizer

text = (root / "raw" / "shakespeare.txt").read_text(encoding="utf-8")
train, held = text[:100_000], text[-100_000:]
held_bytes = len(held.encode("utf-8")); print(f"train {len(train.encode())} B, held-out {held_bytes} B (disjoint: {len(text) > 200_000})")

# 1. round trip: 1000 random strings, unicode + emoji + edge cases
tok = BPETokenizer(); tok.train(train, 512)
random.seed(0)
pools = [range(32, 127), range(0xA0, 0x250), range(0x370, 0x3FF), range(0x4E00, 0x4F00), range(0x1F300, 0x1F6FF), range(0x0900, 0x097F)]
ok = 0; N = 1000
for _ in range(N):
    s = "".join(chr(random.choice(random.choice(pools))) for _ in range(random.randint(0, 200)))
    ok += tok.decode(tok.encode(s)) == s
print(f"round-trip exact match: {ok}/{N}")

# 2. training speed
for V in (1000, 5000):
    t = BPETokenizer(); s = time.perf_counter(); t.train(train, V); dt = time.perf_counter() - s
    print(f"train to vocab {V}: {dt:.1f}s  ({(V-256)/dt:.1f} merges/s)", flush=True)
big = t  # vocab 5000

# 3. compression on held-out text at several vocab sizes (truncate the 5000-merge list: first k merges == a k-vocab tokenizer)
def truncated(full, V):
    n = BPETokenizer(); n.merges = dict(list(full.merges.items())[:V - 256]); n.vocab = {k: v for k, v in full.vocab.items() if k < V}; return n
print("vocab  tokens  bytes/token  (held-out 100 KB)")
for V in (256, 512, 1000, 2000, 5000):
    n = truncated(big, V); s = time.perf_counter(); ids = n.encode(held); dt = time.perf_counter() - s
    assert n.decode(ids) == held
    print(f"{V:5d} {len(ids):7d}  {held_bytes/len(ids):.3f}   (encode took {dt:.1f}s)", flush=True)

# 4. baseline: GPT-2 tokenizer on the same held-out text
try:
    import tiktoken
    g = tiktoken.get_encoding("gpt2"); n = len(g.encode(held)); name = "tiktoken gpt2 (50257)"
except Exception as e:
    from transformers import GPT2TokenizerFast
    g = GPT2TokenizerFast.from_pretrained("gpt2"); n = len(g.encode(held)); name = "hf gpt2 (50257)"
print(f"{name}: {n} tokens, {held_bytes/n:.3f} bytes/token")
