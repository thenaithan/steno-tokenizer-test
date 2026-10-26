# steno-tokenizer-test

Two measurements of stenotype strokes as language-model tokens.

The hunch: court reporters write at 200+ words per minute by pressing
chords on a 22-key stenotype, one stroke per syllable, with "briefs" that
collapse whole phrases into a single chord. If a model's tokens were
strokes, (a) it might need fewer positions per word than BPE, which is
the cost that matters at decode time, and (b) the chord structure of a
stroke (which keys are down) might be a better token representation than
an opaque ID.

Both were tested. Both came out negative.

1. **Strokes per word.** Against production-size BPE vocabularies, strokes
   need more positions, not fewer. Steno beats word-boundary BPE by 5-9% on
   its home register (spoken transcripts) and loses to SuperBPE by 34-50%
   on every register.
2. **Representation.** At matched compute, strokes as opaque IDs tie BPE on
   bits per byte; building the embedding from the chord's key bits costs
   4%; a factored 23-key output head fails outright. The ID model's learned
   embeddings carry no linear trace of the chord structure at all.

Single seed, ~11M-parameter models, English only. See Caveats.

## Experiment 1: positions per word

`steno_count.py` converts text to stroke outlines with Plover's main
dictionary and counts positions against three tokenizers on the same text.
Every assumption is in steno's favour (see Method).

Positions per word (lower is cheaper to decode):

| corpus    | words   | OOV  | Llama 3 (128k) | BPE-200k | SuperBPE-t180k | steno (Plover) | steno (ideal dict) |
|-----------|---------|------|----------------|----------|----------------|----------------|--------------------|
| court     | 122,973 | 0.8% | 1.255          | 1.313    | **0.843**      | 1.196          | 1.167              |
| fiction   | 335,900 | 1.7% | 1.393          | 1.356    | **1.021**      | 1.367          | 1.289              |
| technical | 146,021 | 5.3% | 1.700          | 1.873    | **1.544**      | 2.324          | 2.065              |

Steno positions relative to each tokenizer (above 1.0 means steno needs
more positions):

| corpus    | vs Llama 3 | vs BPE-200k | vs SuperBPE |
|-----------|-----------:|------------:|------------:|
| court     | 0.953      | 0.911       | 1.419       |
| fiction   | 0.981      | 1.008       | 1.339       |
| technical | 1.367      | 1.241       | 1.505       |

"Ideal dict" charges out-of-dictionary words one stroke per syllable
instead of one per letter, i.e. a dictionary with perfect coverage; it
closes a few points and changes nothing.

Corpora: Pride and Prejudice + Moby-Dick (fiction), 12 Supreme Court
oral-argument transcripts from the 2019 term (court), the CPython
tutorial and howto docs (technical). Court transcripts are steno's home
register; technical prose has no phonetic form for code, paths and
identifiers, so there it loses to everything. SuperBPE's lead over its
matched BPE-200k baseline comes out at 18-36% here, consistent with the
paper's "up to 33%".

One nuance from the second experiment: at an *equal* vocabulary of 18,598,
steno is 4.5% denser than BPE (4.52 vs 4.33 bytes per position on the
training text; 8% fewer positions on the court validation set). The
decode-cost result above holds because production tokenizers run 128k to
200k vocabularies, where learned superwords find four times as many
multi-word units as a human-curated brief dictionary.

Full tables: `runs/strokes_per_word.txt`.

## Experiment 2: representation at matched compute

`prepare_data.py` builds two token streams from the same 68 MB of
normalised text (26 novels, 676 oral-argument transcripts; 8% of
documents held out per register) with the same vocabulary size, 18,598:
Plover strokes, and a byte-level BPE trained on the training text.
`train.py` trains the same 10.8M-parameter GPT body (6 layers, d=384,
context 512, batch 32, 3,000 steps, ~49M positions) on each, with two
switches: how a token enters (lookup table, or an MLP over its 23 key
bits) and how it leaves (softmax over the vocabulary, or 23 independent
key logits whose product is a proper distribution over all 2^23 chords).

| run             | token enters as         | token leaves as          | params | bpb fiction | bpb court | top-1 fiction | top-1 court |
|-----------------|-------------------------|--------------------------|-------:|------------:|----------:|--------------:|------------:|
| bpe             | lookup table            | softmax over vocab       | 25.1M  | **1.681**   | **1.174** | 0.221         | 0.315       |
| steno_id        | lookup table            | softmax over vocab       | 25.1M  | 1.682       | 1.186     | 0.230         | 0.284       |
| steno_fact_in   | MLP over 23 key bits    | softmax over vocab       | 18.1M  | 1.743       | 1.241     | 0.207         | 0.261       |
| steno_fact_both | MLP over 23 key bits    | 23 independent key logits| 11.0M  | 3.026       | 2.446     | 0.058         | 0.096       |

bpb is bits per byte of the original text over the full held-out stream,
the one loss that is comparable across tokenizers. Top-1 is exact
next-token accuracy and is only comparable between the steno runs. For
`steno_fact_both`, the argmax key pattern is a real dictionary stroke
99.4% (fiction) and 98.7% (court) of the time, so the head learned which
chords exist; it cannot represent "either of two homophone strokes"
without spreading mass over every mixture of their keys.

Probe (`probe.py`): a linear probe per key from the `steno_id` model's
learned embeddings to that key's bit, on held-out strokes. Mean test
accuracy 0.736 against a majority-class baseline of 0.738 and a
row-shuffled baseline of 0.732. Mean excess over majority: learned
-0.002, shuffled -0.006. The embeddings do not linearly encode which keys
a stroke uses.

Nearest neighbours by cosine in the same embeddings:

```
   KAT: TKUFP TEUPL KA*T TKHRAP KOBGS KHAP PWOPB TKPWA
  TKOG: PHRAPL TPHERS TKR TRUGS TP*EL PHAPS TKPWRUB PWAOBLG
  HOUS: RAOPLS TKPWA*RPB TPH*PB SHOP HO*US RAOPL TOUPB TAFRPB
    -G: -BLG TPAOEUG PREFG ERG -D K-G -LG -PG
    -D: ERD -LD -G -PBD -PLTD PHRAOEUD PHARBGD RUPTD
 TP-PL: TK-PL PH-RPB SHAO*EUS SPHAEPBS KEBG T-RBGS TPHAOERL TPAEU
   #ST: #SP #SL #SSEU #TO #HHEU #SO #SH #SA
```

`KAT` sits next to its homophone-variant `KA*T`, `HOUS` next to `HO*US`
and `SHOP`, the suffix strokes cluster with each other, and the number
strokes with each other: grouping by function and meaning, not by shared
keys.

Full logs with the training curves: `runs/*.json`.

## Method, and what is tilted towards steno

- Text to strokes (`stenoconv.py`): Plover's `main.json` (147k entries).
  Multi-word briefs are applied optimally by dynamic programming over the
  sentence, not greedily; a human writer would not manage that. Suffix and
  prefix folding with orthographic repair (running = run + `-G`). Spaces
  are free. Sentence-initial capitals are free; other capitals not in the
  dictionary cost one stroke (`KPA`). Two digits per number-bar stroke.
  Punctuation one stroke each, as in Plover. Out-of-dictionary words are
  fingerspelled, one stroke per letter (1.25% of training words).
- Each stroke is parsed into its 23 keys, `#STKPWHRAO*EUFRPBLGTSDZ`.
- Steno is slightly lossy: capitalisation of dictionary-exact entries,
  homophone choice and some punctuation spacing are resolved by the
  dictionary rather than carried in the stream, so the steno runs have
  marginally less information to predict per byte. This flatters their
  bits per byte.
- Experiment 1 tokenizers: Llama 3 (128k), OLMo2 BPE-200k (the SuperBPE
  paper's matched baseline) and SuperBPE-t180k, all run through a
  heap-based merge engine with HF-tokenizers / tiktoken semantics. The only
  approximation is `\p{L}` as `[^\W\d_]` in the pretokeniser regex, which
  is exact for English.
- Experiment 2: matched body, context, batch, optimizer steps, training
  text and vocabulary size. Parameter counts differ only in the embedding
  and head, which is the thing under test. AdamW 6e-4, cosine, bf16.
  `prepare_data.py` is seeded, so a rebuild reproduces the byte and token
  counts in `data/*/meta.json` exactly.

## Caveats

- One seed per run. Differences under about 1% bpb should be read as ties.
- 11M-parameter models and 49M training positions; whether the 4% gap for
  the factored embedding closes at scale is untested.
- English only. The onset/rime/tone factorisation has a published positive
  result for Vietnamese and Chinese (see below); tonal languages are a
  different question.
- Experiment 1's corpora are small (3.7 MB); the ratios are stable across
  registers but a wider sample could move them by a few points.

## Reproduce

Python 3.10+ and numpy for everything; torch >= 2.1 with CUDA for
`train.py`. No other packages: the BPE trainer and encoders are pure Python.

Fetch the dictionary and tokenizers (all from GitHub; nothing is vendored
here because Plover's dictionary and the tokenizer files carry their own
licences):

    mkdir -p dl && cd dl
    git clone --depth 1 --filter=blob:none --sparse https://github.com/openstenoproject/plover.git
    (cd plover && git sparse-checkout set plover/assets)                 # main.json
    git clone --depth 1 --filter=blob:none --sparse https://github.com/meta-llama/llama-models.git
    (cd llama-models && git sparse-checkout set models/llama3)           # tokenizer.model
    git clone --depth 1 https://github.com/PythonNut/superbpe.git        # tokenizer_json/*
    cd ..

Experiment 1:

    cd dl
    git clone --depth 1 https://github.com/GITenberg/Pride-and-Prejudice_1342.git
    git clone --depth 1 https://github.com/GITenberg/Moby-Dick--Or-The-Whale_2701.git
    git clone --depth 1 --filter=blob:none --sparse https://github.com/walkerdb/supreme_court_transcripts.git
    (cd supreme_court_transcripts && git ls-tree -r --name-only HEAD \
       | grep -E 'oyez/cases/2019\..*-t01\.json' | sort | head -12 \
       | git sparse-checkout set --no-cone --stdin)
    git clone --depth 1 --filter=blob:none --sparse https://github.com/python/cpython.git
    (cd cpython && git sparse-checkout set Doc/tutorial Doc/howto)
    cd ..
    python build_count_corpus.py
    python steno_count.py corpus              # ~15 s; writes runs/strokes_per_word.json

Experiment 2 (the shipped `data/*/meta.json`, `tokenizer.json` and
`keys.npy` are outputs of this; the `.bin` streams are regenerated):

    cd dl
    (cd supreme_court_transcripts && git ls-tree -r --name-only HEAD \
       | grep -E 'oyez/cases/20(1[0-9]|2[0-4])\..*-t0[1-9]\.json' | sort | head -700 \
       | git sparse-checkout set --no-cone --stdin)
    for r in Great-Expectations_1400 A-Tale-of-Two-Cities_98 Emma_158 \
             The-Picture-of-Dorian-Gray_174 Dracula_345 The-Adventures-of-Tom-Sawyer_74 \
             Adventures-of-Huckleberry-Finn_76 Middlemarch_145 Treasure-Island_120 \
             Little-Women_514 The-Count-of-Monte-Cristo_1184 War-and-Peace_2600 \
             Anna-Karenina_1399 Sense-and-Sensibility_161 Persuasion_105 \
             David-Copperfield_766 Crime-and-Punishment_2554 \
             The-Adventures-of-Sherlock-Holmes_1661 Heart-of-Darkness_219 The-Jungle_140 \
             Northanger-Abbey_121 Oliver-Twist_730 Vanity-Fair_599 The-Moonstone_155 \
             Pride-and-Prejudice_1342 Moby-Dick--Or-The-Whale_2701; do
      git clone --depth 1 --filter=blob:none --sparse https://github.com/GITenberg/$r.git
      (cd $r && n=${r##*_} && git sparse-checkout set --no-cone "$n-0.txt" "$n.txt")
    done
    cd ..
    python build_raw.py                        # raw/{fiction,court}/*.txt
    python prepare_data.py                     # ~15 min CPU; data/{steno,bpe}/*.bin

    python train.py --data data/bpe   --embed table        --head softmax  --name bpe
    python train.py --data data/steno --embed table        --head softmax  --name steno_id
    python train.py --data data/steno --embed factored_mlp --head softmax  --name steno_fact_in
    python train.py --data data/steno --embed factored_mlp --head factored --name steno_fact_both
    python summarize.py                        # table from runs/*.json
    python probe.py runs/steno_id.pt           # key-bit probes and nearest neighbours

Each training run took 2-4 minutes on a 12 GB RTX 4070 Ti. If you hit OOM,
`--batch 16 --steps 6000` keeps the compute matched. `--embed factored`
gives the linear version of the factored embedding (a 24-dimensional
embedding subspace), which was not run.

## Files

- `steno_count.py`        experiment 1: strokes per word vs three tokenizers
- `build_count_corpus.py` builds its three corpora from `dl/`
- `stenoconv.py`          text -> strokes (DP over briefs); stroke -> 23 key bits
- `bpe.py`                pure-Python byte-level BPE trainer and encoder
- `build_raw.py`          per-document corpus for experiment 2 from `dl/`
- `prepare_data.py`       matched-vocabulary token streams, `data/`
- `train.py`              the model and both switches; writes `runs/<name>.json` and `.pt`
- `probe.py`              linear probes from learned ID embeddings to key bits
- `summarize.py`          comparison table
- `runs/`                 results of all runs (checkpoints are not committed)
- `data/`                 vocabularies, key bits and byte/token counts (streams are not committed)

## Related work

- [SuperBPE: Space Travel for Language Models](https://arxiv.org/abs/2503.13423) (Liu et al., 2025): superword tokens that cross whitespace, up to 33% fewer tokens than BPE at 200k. The learned version of briefs, and the tokenizer that wins experiment 1.
- [From Babble to Words](https://arxiv.org/abs/2410.22906) (Goriely et al., 2024): pretraining on continuous phoneme streams; slightly lower scores on orthographic benchmarks.
- [How Tokenization Limits Phonological Knowledge Representation in Language Models](https://icml.cc/virtual/2025/47753) (Liao and Shi, ICML 2025 TokShop).
- Phonemic Tokenizer for Vietnamese and Chinese (Nguyen et al., 2026): onset, rime and tone factorised into one position; the positive result for tonal languages.
- [Coining Words with a Stenotype](https://bala.is-a.dev/writing/steno): a phoneme-to-stroke model whose output hedges between homophones, the failure mode `steno_fact_both` reproduces.
- [Plover](https://github.com/openstenoproject/plover) and its main dictionary, which made the conversion possible.

## Conclusion

_(to be written)_

## Acknowledgement

The experiments and analysis were built with Claude (Anthropic) as a
pair-programming partner. The questions, the framing and the conclusions
are mine.
