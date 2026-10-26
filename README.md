# Steno strokes as tokens: representation experiment

Question: given that stenotype strokes do not reduce the number of positions
a model must decode (see ../results.txt), does the chord *structure* of a
stroke (23 keys: `#STKPWHRAO*EUFRPBLGTSDZ`) make a better token
representation than an opaque ID, per unit of compute?

Four runs, same model body, same context, same batch, same number of
optimizer steps, same training text, same vocabulary size (18,598):

| name            | data  | token enters as                 | token leaves as              |
|-----------------|-------|---------------------------------|------------------------------|
| bpe             | BPE   | lookup table                    | softmax over vocab           |
| steno_id        | steno | lookup table                    | softmax over vocab           |
| steno_fact_in   | steno | MLP over its 23 key bits        | softmax over vocab           |
| steno_fact_both | steno | MLP over its 23 key bits        | 23 independent key logits    |

`steno_fact_both` is the "chord logic" condition: the model never sees a
stroke ID at all, in or out. Its output is a proper distribution over all
2^23 key combinations (product of Bernoullis), so its bits-per-byte is
directly comparable; `summarize.py` also reports how often its argmax is a
real stroke.

## Metric

Bits per byte of the original (normalised) text on the full held-out stream,
per register (fiction / court). This is the only loss that is comparable
across tokenizers. Per-token accuracy is reported but only comparable
between the three steno runs.

## Requirements

Python 3.10+, numpy, torch >= 2.1 with CUDA. Nothing else.

## Run

    # data: reuses ../dl (Plover main.json) and raw/ (already populated)
    python prepare_data.py                 # ~15 min CPU, once

    python train.py --data data/bpe   --embed table        --head softmax  --name bpe
    python train.py --data data/steno --embed table        --head softmax  --name steno_id
    python train.py --data data/steno --embed factored_mlp --head softmax  --name steno_fact_in
    python train.py --data data/steno --embed factored_mlp --head factored --name steno_fact_both

    python summarize.py                    # table from runs/*.json
    python probe.py runs/steno_id.pt       # does the ID model discover the keys on its own?

Defaults: 6 layers, 6 heads, d=384 (10.6M body params), context 512,
batch 32, 3000 steps (~50M positions, ~3.5 epochs of the 68 MB corpus),
AdamW 6e-4 cosine, bf16. On a 12 GB 4070 Ti expect roughly 6-10 minutes per
run. If you hit OOM, `--batch 16 --steps 6000` keeps the compute matched.
Add `--compile` for a ~1.5x speedup once you know a run works.

Optional fifth run, the linear version of the factored embedding
(`--embed factored`): embeddings then live in a 24-dim subspace, which is
the strongest possible compositional prior and probably too strong.

## How to read the result

- `steno_id` vs `bpe`: same vocab size, different segmentation. Tells you
  whether syllable-level phonetic units are a better or worse *unit* than
  learned subwords at matched compute. (Expect bpe to win on bpb; steno
  carries slightly less information per byte, see caveats.)
- `steno_fact_in` vs `steno_id`: the actual question. Same units, the only
  change is that the embedding is built from the chord. If it wins, the
  compositional prior helps; if it loses, 18k free parameters per stroke
  beat the prior.
- `steno_fact_both`: whether a 23-way factored head can replace an 18k-way
  softmax. Watch `pred_is_valid_stroke`: if it is low the head is hedging
  between homophones, which was the failure mode flagged in the steno blog post.
- `probe.py`: if the ID model's embeddings already predict the key bits
  well above the shuffled baseline, the model recovers chord structure by
  itself and the prior adds nothing it could not learn.

## Caveats (all favour steno)

- Steno is slightly lossy: capitalisation of dictionary-exact entries,
  homophone choice and some punctuation spacing are resolved by the
  dictionary rather than carried in the stream. Less information to predict
  means a slightly lower bpb for the same model quality. Keep that in mind if
  steno runs come within a few percent of bpe.
- Out-of-dictionary words (1.25%) are fingerspelled, one stroke per letter.
- Briefs are applied optimally (DP), which a human writer would not manage.

## Files

- `stenoconv.py`   text -> strokes (DP over briefs), stroke -> 23 key bits
- `bpe.py`         pure-Python byte-level BPE trainer/encoder (matched vocab)
- `prepare_data.py` builds data/{steno,bpe}/{train,val_fiction,val_court}.bin + meta
- `train.py`       the model and both switches; writes runs/<name>.json and .pt
- `probe.py`       linear probes from learned ID embeddings to key bits
- `summarize.py`   comparison table
- `raw/`           normalised source text, one file per document

## Rebuilding the corpus from scratch

    cd .. && mkdir -p dl && cd dl
    git clone --depth 1 --filter=blob:none --sparse https://github.com/openstenoproject/plover.git
    (cd plover && git sparse-checkout set plover/assets)
    git clone --depth 1 --filter=blob:none --sparse https://github.com/walkerdb/supreme_court_transcripts.git
    (cd supreme_court_transcripts && git ls-tree -r --name-only HEAD \
       | grep -E 'oyez/cases/20(1[0-9]|2[0-4])\..*-t0[1-9]\.json' | sort | head -700 \
       | git sparse-checkout set --no-cone --stdin)
    for r in Great-Expectations_1400 A-Tale-of-Two-Cities_98 Emma_158 ... ; do   # see raw/fiction for the list
      git clone --depth 1 --filter=blob:none --sparse https://github.com/GITenberg/$r.git
      (cd $r && n=${r##*_} && git sparse-checkout set --no-cone "$n-0.txt" "$n.txt")
    done
    cd ../steno_lm && python build_raw.py && python prepare_data.py

## Status

`prepare_data.py` was run end to end (CPU, pure Python) and the shipped
`data/` is its output. `train.py`, `probe.py` and `summarize.py` were
syntax-checked and reviewed but have NOT been executed, because the machine
that built this had no torch. Expect at most a minor fix on first run.
