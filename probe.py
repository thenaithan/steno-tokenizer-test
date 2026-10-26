#!/usr/bin/env python3
"""
Does a model that only ever saw stroke IDs discover the chord structure?

Fits one linear probe per steno key from the learned stroke embeddings
(steno_id run) to that key's bit, with a held-out set of strokes.
Reports per-key test accuracy against two baselines: majority class, and
the same probe on row-shuffled embeddings (destroys the stroke<->vector
pairing but keeps the distribution).

  python probe.py runs/steno_id.pt data/steno
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from stenoconv import STENO_ORDER

ckpt = torch.load(sys.argv[1], map_location="cpu")
data = Path(sys.argv[2] if len(sys.argv) > 2 else "data/steno")
sd = ckpt["state_dict"]
assert "wte.weight" in sd, "probe needs a --embed table run"
E = sd["wte.weight"].float().numpy()
K = np.load(data / "keys.npy").astype(np.float64)
vocab = json.load(open(data / "meta.json"))["vocab"]
E, K, vocab = E[1:], K[1:], vocab[1:]  # drop <unk>
E = (E - E.mean(0)) / (E.std(0) + 1e-6)
rng = np.random.default_rng(0)
perm = rng.permutation(len(E))
ntest = len(E) // 5
te, tr = perm[:ntest], perm[ntest:]


def fit_logreg(X, y, steps=300, lr=0.5, l2=1e-3):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(steps):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / len(X) + l2 * w)
        b -= lr * g.mean()
    return w, b


def acc(w, b, X, y):
    return (((X @ w + b) > 0) == (y > 0.5)).mean()


print(f"{'key':>4} {'majority':>9} {'shuffled':>9} {'learned':>9}")
rows = []
Eshuf = E[rng.permutation(len(E))]
for k, name in enumerate(STENO_ORDER):
    y = K[:, k]
    maj = max(y[te].mean(), 1 - y[te].mean())
    w, b = fit_logreg(E[tr], y[tr])
    a = acc(w, b, E[te], y[te])
    w2, b2 = fit_logreg(Eshuf[tr], y[tr])
    a2 = acc(w2, b2, Eshuf[te], y[te])
    rows.append((name, maj, a2, a))
    print(f"{name:>4} {maj:9.3f} {a2:9.3f} {a:9.3f}")
m = np.array([[r[1], r[2], r[3]] for r in rows])
print(f"{'mean':>4} {m[:,0].mean():9.3f} {m[:,1].mean():9.3f} {m[:,2].mean():9.3f}")
# excess accuracy over majority, averaged: the number to quote
print(f"\nmean excess over majority: learned {np.mean(m[:,2]-m[:,0]):.3f}  shuffled {np.mean(m[:,1]-m[:,0]):.3f}")

# nearest neighbours for a few strokes, by cosine
En = E / np.linalg.norm(E, axis=1, keepdims=True)
idx = {s: i for i, s in enumerate(vocab)}
for s in ["KAT", "TKOG", "HOUS", "-G", "-D", "TP-PL", "#ST", "KPA"]:
    if s in idx:
        sims = En @ En[idx[s]]
        nn_ = [vocab[j] for j in np.argsort(-sims)[1:9]]
        print(f"{s:>6}: {' '.join(nn_)}")
