"""Autograd benchmark: (1) gradient check vs PyTorch on 200 random expression DAGs (shared subexpressions included),
(2) MLP training on XOR and make_moons, same init/hyperparameters/full-batch GD as an equivalent PyTorch MLP, (3) step time ratio."""
import math, random, sys, time, pathlib
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "autograd-engine"))
from value import Value
sys.setrecursionlimit(10000)

# ---------- 1. gradient check ----------
OPS = ["+", "*", "tanh", "exp", "log", "pow"]
def make_program(rng, n_ops):
    """Program = list of (op, arg indices, k). Nodes 0..2 are inputs; each op picks earlier nodes, so subexpressions get reused (a DAG)."""
    prog = []
    for i in range(n_ops):
        op = rng.choice(OPS); n = 3 + i
        prog.append((op, [rng.randrange(n), rng.randrange(n)], rng.choice([2, 3, -1, 0.5])))
    return prog

def run(prog, xs, lib):
    nodes = list(xs)
    for op, a, k in prog:
        x, y = nodes[a[0]], nodes[a[1]]
        if op == "+": nodes.append(x + y)
        elif op == "*": nodes.append(x * y)
        elif op == "tanh": nodes.append(x.tanh() if lib == "mine" else torch.tanh(x))
        elif op == "exp": nodes.append(x.exp() if lib == "mine" else torch.exp(x))
        elif op == "log": nodes.append(x.log() if lib == "mine" else torch.log(x))
        else: nodes.append(x ** k)
    return nodes

def valid(prog, vals):
    """Reject programs that leave the real domain or overflow, judged by torch float64 values of every node."""
    nodes = [torch.tensor(v, dtype=torch.float64) for v in vals]
    for op, a, k in prog:
        x, y = nodes[a[0]], nodes[a[1]]
        if op == "log" and not x > 1e-3: return False
        if op == "pow" and x <= 0 and (k != int(k) or k < 0): return False
        if op == "pow" and x == 0: return False
        if op == "exp" and x > 20: return False
        nodes.append({"+": lambda: x + y, "*": lambda: x * y, "tanh": lambda: torch.tanh(x), "exp": lambda: torch.exp(x),
                      "log": lambda: torch.log(x), "pow": lambda: x ** k}[op]())
        if not torch.isfinite(nodes[-1]) or nodes[-1].abs() > 1e8: return False
    return True

rng = random.Random(0); diffs = []; tried = 0; ops_used = {o: 0 for o in OPS}; rel = []
while len(diffs) < 200:
    tried += 1
    prog = make_program(rng, rng.randint(3, 12)); vals = [rng.uniform(0.3, 2.0) for _ in range(3)]
    if not valid(prog, vals): continue
    mine = [Value(v) for v in vals]; out = run(prog, mine, "mine")[-1]; out.backward()
    ts = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in vals]
    tout = run(prog, ts, "torch")[-1]; tout.backward()
    tg = [0.0 if t.grad is None else t.grad.item() for t in ts]
    d = [abs(m.grad - g) for m, g in zip(mine, tg)]
    diffs.append(max(d + [abs(out.data - tout.item())]))
    rel.append(max(abs(m.grad - g) / (abs(g) + 1e-12) for m, g in zip(mine, tg)))
    for op, _, _ in prog: ops_used[op] += 1
diffs = np.array(diffs)
print(f"[grad check] 200 valid random DAGs (sampled {tried}, rejected {tried-200} out-of-domain/overflow); ops used {ops_used}")
print(f"  max |abs diff| over value+grads: {diffs.max():.3e}   within 1e-6: {(diffs < 1e-6).mean():.1%}   within 1e-9: {(diffs < 1e-9).mean():.1%}")
print(f"  max relative grad error {max(rel):.3e}")

# ---------- 2. learning check ----------
from sklearn.datasets import make_moons
def init(h, seed): 
    r = np.random.default_rng(seed); return r.uniform(-1, 1, (h, 2)), r.uniform(-1, 1, h), r.uniform(-1, 1, h), float(r.uniform(-1, 1))

def train_mine(X, y, h, lr, epochs, seed):
    W1, b1, W2, b2 = init(h, seed)
    w1 = [[Value(float(W1[i][j])) for j in range(2)] for i in range(h)]; bb1 = [Value(float(v)) for v in b1]
    w2 = [Value(float(v)) for v in W2]; bb2 = Value(b2)
    params = [p for r in w1 for p in r] + bb1 + w2 + [bb2]; losses = []; neg = Value(-1.0); inv = Value(1 / len(X)); times = []
    for _ in range(epochs):
        s = time.perf_counter()
        for p in params: p.grad = 0
        total = None
        for xi, yi in zip(X, y):
            xs = [Value(float(xi[0])), Value(float(xi[1]))]; acc = bb2
            for i in range(h):
                hid = (w1[i][0] * xs[0] + w1[i][1] * xs[1] + bb1[i]).tanh(); acc = acc + w2[i] * hid
            out = acc.tanh(); err = out + Value(float(yi)) * neg; sq = err ** 2
            total = sq if total is None else total + sq
        loss = total * inv; loss.backward(); losses.append(loss.data)
        for p in params: p.data -= lr * p.grad
        times.append(time.perf_counter() - s)
    def predict(xi):
        acc = bb2
        for i in range(h):
            acc = acc + w2[i] * (w1[i][0] * Value(float(xi[0])) + w1[i][1] * Value(float(xi[1])) + bb1[i]).tanh()
        return acc.tanh().data
    return losses, np.array([predict(xi) for xi in X]), float(np.mean(times))

def train_torch(X, y, h, lr, epochs, seed):
    W1, b1, W2, b2 = init(h, seed); d = torch.float64
    w1 = torch.tensor(W1, dtype=d, requires_grad=True); bb1 = torch.tensor(b1, dtype=d, requires_grad=True)
    w2 = torch.tensor(W2, dtype=d, requires_grad=True); bb2 = torch.tensor(b2, dtype=d, requires_grad=True)
    Xt = torch.tensor(X, dtype=d); yt = torch.tensor(y, dtype=d); losses = []; times = []
    for _ in range(epochs):
        s = time.perf_counter()
        out = torch.tanh(torch.tanh(Xt @ w1.T + bb1) @ w2 + bb2); loss = ((out - yt) ** 2).mean()
        for p in (w1, bb1, w2, bb2): p.grad = None
        loss.backward(); losses.append(loss.item())
        with torch.no_grad():
            for p in (w1, bb1, w2, bb2): p -= lr * p.grad
        times.append(time.perf_counter() - s)
    with torch.no_grad(): pred = torch.tanh(torch.tanh(Xt @ w1.T + bb1) @ w2 + bb2).numpy()
    return losses, pred, float(np.mean(times))

tasks = {"XOR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float), np.array([-1, 1, 1, -1], float), 8, 0.3, 300),
         "make_moons(n=100,noise=0.1)": None}
Xm, ym = make_moons(n_samples=100, noise=0.1, random_state=0); tasks["make_moons(n=100,noise=0.1)"] = (Xm, 2.0 * ym - 1, 8, 0.1, 300)
print("\n[learning] full-batch GD, 2-8-1 tanh MLP, MSE, identical init, float64; 3 init seeds")
for name, (X, y, h, lr, ep) in tasks.items():
    for seed in (0, 1, 2):
        lm, pm, tm = train_mine(X, y, h, lr, ep, seed); lt, pt, tt = train_torch(X, y, h, lr, ep, seed)
        am = (np.sign(pm) == y).mean(); at = (np.sign(pt) == y).mean()
        print(f"  {name:28s} seed {seed}: mine loss {lm[0]:.4f}->{lm[-1]:.4f} acc {am:.1%} | torch loss {lt[0]:.4f}->{lt[-1]:.4f} acc {at:.1%} | max per-step loss diff {max(abs(a-b) for a,b in zip(lm,lt)):.2e} | step time mine {tm*1000:.1f} ms, torch {tt*1000:.3f} ms, ratio {tm/tt:.0f}x", flush=True)
