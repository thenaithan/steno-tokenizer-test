#!/usr/bin/env python3
"""
Build matched token streams for the representation experiment.

Conditions share the same normalised text, the same document-level
train/val split and the same vocabulary size:
  data/steno/   strokes from Plover's main.json  (+ keys.npy: V x 23 key bits)
  data/bpe/     byte-level BPE trained on the training text, vocab = |strokes|

Outputs per condition: train.bin, val_fiction.bin, val_court.bin (uint16),
meta.json with vocab, byte counts (for bits-per-byte) and token counts.

Pure Python + numpy. Run once; takes a few minutes.
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpe import BPEEncoder, train_bpe  # noqa: E402
from stenoconv import NKEYS, StenoConverter, parse_stroke  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
DATA = HERE / "data"
MAIN_JSON = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "dl/plover/plover/assets/main.json"
VAL_FRACTION = 0.08
BPE_TRAIN_BYTES = 30_000_000
SEED = 1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalise(s):
    s = (s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
         .replace("—", "--").replace("–", "-").replace("\r", ""))
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" ?\n ?", "\n", s)
    s = re.sub(r"\n{2,}", "\n\n", s)
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)   # single newlines -> space
    s = re.sub(r" {2,}", " ", s)
    return s.strip() + "\n\n"


def load_docs(register):
    docs = []
    for f in sorted((RAW / register).glob("*.txt")):
        t = normalise(f.read_text(encoding="utf-8", errors="replace"))
        if len(t) > 2000:
            docs.append((f.stem, t))
    return docs


def split(docs):
    rnd = random.Random(SEED)
    docs = list(docs)
    rnd.shuffle(docs)
    nval = max(2, int(len(docs) * VAL_FRACTION))
    return docs[nval:], docs[:nval]


def main():
    DATA.mkdir(exist_ok=True)
    splits = {}
    for reg in ("fiction", "court"):
        docs = load_docs(reg)
        tr, va = split(docs)
        splits[reg] = (tr, va)
        log(f"{reg}: {len(docs)} docs, {sum(len(t) for _, t in docs)/1e6:.1f} MB, val docs {len(va)}")
    train_text = [t for reg in splits for _, t in splits[reg][0]]
    rnd = random.Random(SEED)
    rnd.shuffle(train_text)
    val_text = {reg: [t for _, t in splits[reg][1]] for reg in splits}

    # ---------------- steno ----------------
    conv = StenoConverter(MAIN_JSON)
    log("converting to steno...")
    steno_train = []
    st_tot = {"words": 0, "oov_words": 0, "strokes": 0}
    for i, t in enumerate(train_text):
        s, st = conv.convert(t)
        steno_train.append(s)
        for k in st_tot:
            st_tot[k] += st[k]
        if i % 100 == 0:
            log(f"  steno train doc {i}/{len(train_text)}")
    steno_val = {}
    for reg, texts in val_text.items():
        steno_val[reg] = [conv.convert(t)[0] for t in texts]
    log(f"steno train: {st_tot}, oov rate {st_tot['oov_words']/st_tot['words']:.3%}")

    from collections import Counter
    cnt = Counter(s for doc in steno_train for s in doc)
    # keep every stroke seen at least twice in training; rarer ones map to UNK
    vocab = ["<unk>"] + [s for s, c in cnt.most_common() if c >= 2]
    sid = {s: i for i, s in enumerate(vocab)}
    V = len(vocab)
    log(f"steno vocab: {V} strokes (of {len(cnt)} distinct in train)")
    keys = np.zeros((V, NKEYS), dtype=np.uint8)
    for i, s in enumerate(vocab):
        if i:
            keys[i] = parse_stroke(s)
    assert V < 65535
    sd = DATA / "steno"
    sd.mkdir(exist_ok=True)

    def enc_steno(docs):
        return np.array([sid.get(s, 0) for doc in docs for s in doc], dtype=np.uint16)

    arr_train = enc_steno(steno_train)
    arr_train.tofile(sd / "train.bin")
    meta = {"vocab": vocab, "vocab_size": V, "nkeys": NKEYS, "train_tokens": int(len(arr_train)),
            "train_bytes": sum(len(t.encode()) for t in train_text), "val": {}}
    for reg in val_text:
        a = enc_steno(steno_val[reg])
        a.tofile(sd / f"val_{reg}.bin")
        meta["val"][reg] = {"tokens": int(len(a)), "bytes": sum(len(t.encode()) for t in val_text[reg]),
                            "unk_rate": float((a == 0).mean())}
    np.save(sd / "keys.npy", keys)
    json.dump(meta, open(sd / "meta.json", "w"))
    log(f"steno written: train {len(arr_train)} tokens; val {meta['val']}")

    # ---------------- bpe (matched vocab) ----------------
    log(f"training BPE with vocab {V} on {BPE_TRAIN_BYTES/1e6:.0f} MB sample...")
    sample, n = [], 0
    for t in train_text:
        sample.append(t)
        n += len(t)
        if n >= BPE_TRAIN_BYTES:
            break
    bvocab, merges = train_bpe("".join(sample), V, log=log)
    enc = BPEEncoder(bvocab, merges)
    bd = DATA / "bpe"
    bd.mkdir(exist_ok=True)
    enc.save(bd / "tokenizer.json")
    log("encoding train text with BPE...")
    ids = []
    for i, t in enumerate(train_text):
        ids.extend(enc.encode(t))
        if i % 100 == 0:
            log(f"  bpe train doc {i}/{len(train_text)}")
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(bd / "train.bin")
    meta_b = {"vocab_size": len(bvocab), "train_tokens": int(len(arr)), "train_bytes": meta["train_bytes"], "val": {}}
    for reg in val_text:
        a = np.array([i for t in val_text[reg] for i in enc.encode(t)], dtype=np.uint16)
        a.tofile(bd / f"val_{reg}.bin")
        meta_b["val"][reg] = {"tokens": int(len(a)), "bytes": meta["val"][reg]["bytes"]}
    json.dump(meta_b, open(bd / "meta.json", "w"))
    log(f"bpe written: train {len(arr)} tokens; val {meta_b['val']}")

    summary = {
        "vocab_size": V,
        "train_bytes": meta["train_bytes"],
        "steno_train_tokens": meta["train_tokens"],
        "bpe_train_tokens": meta_b["train_tokens"],
        "bytes_per_token": {"steno": meta["train_bytes"] / meta["train_tokens"], "bpe": meta["train_bytes"] / meta_b["train_tokens"]},
        "steno_oov_rate_train": st_tot["oov_words"] / st_tot["words"],
        "val": {reg: {"bytes": meta["val"][reg]["bytes"], "steno_tokens": meta["val"][reg]["tokens"], "bpe_tokens": meta_b["val"][reg]["tokens"],
                      "steno_unk_rate": meta["val"][reg]["unk_rate"]} for reg in val_text},
    }
    json.dump(summary, open(DATA / "summary.json", "w"), indent=1)
    log(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
