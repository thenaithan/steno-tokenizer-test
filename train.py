#!/usr/bin/env python3
"""
Small GPT trained on one token stream, with switchable token representation.

  --embed table         ordinary V x d lookup (BPE and steno-as-IDs)
  --embed factored      embedding = keys @ Wk + b          (linear in the 23 key bits)
  --embed factored_mlp  embedding = MLP(keys)              (compositional, nonlinear)
  --head  softmax       standard V-way next-token softmax
  --head  factored      23 independent key logits (BCE); p(stroke) = prod p(bit)

All conditions: same body (layers, width, context), same batch, same number of
optimizer steps, so compute per run is matched. The score that is comparable
across tokenizers is bits per byte of the ORIGINAL text (nats / ln2 / bytes).

Usage (see README):
  python train.py --data data/bpe   --embed table        --head softmax  --name bpe
  python train.py --data data/steno --embed table        --head softmax  --name steno_id
  python train.py --data data/steno --embed factored_mlp --head softmax  --name steno_fact_in
  python train.py --data data/steno --embed factored_mlp --head factored --name steno_fact_both
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--name", required=True)
p.add_argument("--embed", default="table", choices=["table", "factored", "factored_mlp"])
p.add_argument("--head", default="softmax", choices=["softmax", "factored"])
p.add_argument("--n_layer", type=int, default=6)
p.add_argument("--n_head", type=int, default=6)
p.add_argument("--n_embd", type=int, default=384)
p.add_argument("--block", type=int, default=512)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--wd", type=float, default=0.1)
p.add_argument("--eval_every", type=int, default=500)
p.add_argument("--eval_batches", type=int, default=40)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--compile", action="store_true")
p.add_argument("--out", default="runs")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
ctx = torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.autocast("cpu", enabled=False)

DATA = Path(args.data)
meta = json.load(open(DATA / "meta.json"))
V = meta["vocab_size"]
keys = None
if (DATA / "keys.npy").exists():
    keys = torch.tensor(np.load(DATA / "keys.npy"), dtype=torch.float32, device=device)  # V x 23
if args.embed != "table" or args.head != "softmax":
    assert keys is not None, "factored embed/head need keys.npy (steno data only)"
NK = keys.shape[1] if keys is not None else 0

train = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
vals = {reg: np.memmap(DATA / f"val_{reg}.bin", dtype=np.uint16, mode="r") for reg in meta["val"]}


def get_batch(arr):
    ix = torch.randint(len(arr) - args.block - 1, (args.batch,))
    x = torch.stack([torch.from_numpy(arr[i:i + args.block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(arr[i + 1:i + 1 + args.block].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)
        self.h = h

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(B, T, C))
        x = x + self.fc2(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        d = args.n_embd
        if args.embed == "table":
            self.wte = nn.Embedding(V, d)
        elif args.embed == "factored":
            self.wk = nn.Linear(NK, d)                      # keys @ Wk + b
        else:
            self.wk = nn.Sequential(nn.Linear(NK, d), nn.GELU(), nn.Linear(d, d))
        self.wpe = nn.Embedding(args.block, d)
        self.blocks = nn.ModuleList([Block(d, args.n_head) for _ in range(args.n_layer)])
        self.lnf = nn.LayerNorm(d)
        if args.head == "softmax":
            self.head = nn.Linear(d, V, bias=False)
        else:
            self.head = nn.Linear(d, NK)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def embed_table(self):
        if args.embed == "table":
            return self.wte.weight
        return self.wk(keys)                                # V x d, recomputed each step

    def forward(self, x, y=None):
        B, T = x.shape
        tab = self.embed_table()
        h = tab[x] + self.wpe(torch.arange(T, device=x.device))
        for b in self.blocks:
            h = b(h)
        h = self.lnf(h)
        logits = self.head(h)
        if y is None:
            return logits, None
        if args.head == "softmax":
            loss = F.cross_entropy(logits.view(-1, V).float(), y.view(-1), reduction="none")
        else:
            tgt = keys[y]                                   # B x T x 23
            loss = F.binary_cross_entropy_with_logits(logits.float(), tgt, reduction="none").sum(-1).view(-1)
        return logits, loss                                 # per-position nats


model = GPT().to(device)
n_body = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("wte", "wk", "head")))
n_all = sum(p.numel() for p in model.parameters())
print(f"params: body {n_body/1e6:.2f}M, total {n_all/1e6:.2f}M, vocab {V}, device {device}, dtype {dtype}")
raw_model = model
if args.compile and device == "cuda":
    model = torch.compile(model)

decay = [p for n, p in raw_model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in raw_model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.95), fused=(device == "cuda"))


def lr_at(it):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    prog = (it - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.1 + 0.5 * (args.lr - args.lr * 0.1) * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def quick_eval():
    model.eval()
    out = {}
    for reg, arr in vals.items():
        tot = 0.0
        n = 0
        for _ in range(args.eval_batches):
            x, y = get_batch(arr)
            with ctx:
                _, loss = model(x, y)
            tot += loss.sum().item()
            n += loss.numel()
        out[reg] = tot / n
    model.train()
    return out


@torch.no_grad()
def full_eval(arr, bytes_total):
    """Deterministic pass over the whole val stream in non-overlapping blocks."""
    model.eval()
    T = args.block
    n = (len(arr) - 1) // T
    tot_nats = 0.0
    correct = 0
    valid = 0
    count = 0
    key_set = None
    if args.head == "factored":
        key_set = {tuple(int(b) for b in row) for row in keys.cpu().numpy().astype(int)}
    for s in range(0, n, args.batch):
        idx = list(range(s, min(n, s + args.batch)))
        x = torch.stack([torch.from_numpy(arr[i * T:i * T + T].astype(np.int64)) for i in idx]).to(device)
        y = torch.stack([torch.from_numpy(arr[i * T + 1:i * T + T + 1].astype(np.int64)) for i in idx]).to(device)
        with ctx:
            logits, loss = model(x, y)
        tot_nats += loss.sum().item()
        count += loss.numel()
        if args.head == "softmax":
            correct += (logits.argmax(-1) == y).sum().item()
        else:
            pred = (logits > 0).int()
            tgt = keys[y].int()
            correct += (pred == tgt).all(-1).sum().item()
            for row in pred.view(-1, NK).cpu().numpy():
                valid += tuple(int(b) for b in row) in key_set
    model.train()
    res = {"nats_per_token": tot_nats / count, "bits_per_byte": tot_nats / math.log(2) / bytes_total,
           "tokens": count, "bytes": bytes_total, "top1_acc": correct / count}
    if args.head == "factored":
        res["pred_is_valid_stroke"] = valid / count
    return res


log = []
t0 = time.time()
model.train()
for it in range(args.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    if it % args.eval_every == 0:
        ev = quick_eval()
        positions = it * args.batch * args.block
        rec = {"step": it, "positions": positions, "time": time.time() - t0, **{f"val_{k}_nats": v for k, v in ev.items()}}
        for k, v in ev.items():
            rec[f"val_{k}_bpb"] = v / math.log(2) * meta["val"][k]["tokens"] / meta["val"][k]["bytes"]
        log.append(rec)
        print(json.dumps(rec), flush=True)
    if it == args.steps:
        break
    x, y = get_batch(train)
    with ctx:
        _, loss = model(x, y)
    loss.mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

final = {reg: full_eval(arr, meta["val"][reg]["bytes"]) for reg, arr in vals.items()}
out = Path(args.out)
out.mkdir(exist_ok=True)
result = {"name": args.name, "args": vars(args), "vocab_size": V, "params_body": n_body, "params_total": n_all,
          "train_tokens_seen": args.steps * args.batch * args.block, "train_bytes_available": meta["train_bytes"],
          "train_tokens_available": meta["train_tokens"], "final": final, "log": log, "seconds": time.time() - t0}
json.dump(result, open(out / f"{args.name}.json", "w"), indent=1)
torch.save({"state_dict": raw_model.state_dict(), "args": vars(args)}, out / f"{args.name}.pt")
print("FINAL", json.dumps({k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in final.items()}, indent=1))
