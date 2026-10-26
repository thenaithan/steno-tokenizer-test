#!/usr/bin/env python3
"""
Strokes-per-word test: does a stenotype-style tokenizer produce fewer
positions than modern BPE tokenizers?

Compares, on the same text:
  - chars / bytes
  - Llama 3 tokenizer (128k, tiktoken format)
  - OLMo2 BPE 200k      (the SuperBPE paper's matched baseline)
  - SuperBPE-t180k      (the paper's headline tokenizer)
  - Plover steno strokes, two ways:
      steno_plover : real Plover main.json; OOV words are fingerspelled
                     (one stroke per letter), which is what a stenographer does
      steno_ideal  : same, but OOV words are charged one stroke per syllable,
                     i.e. a hypothetical dictionary with full coverage

Steno is given every advantage: optimal (DP) use of multi-word briefs,
suffix/prefix folding with orthographic repair, free spaces, free
sentence-initial capitalisation.

Pure Python, no third-party packages (PyPI was unreachable when this was
written). The BPE encoder reproduces HF `tokenizers` / tiktoken merge
semantics; the only approximation is Unicode classes in the pretokeniser
regex (\\p{L} -> [^\\W\\d_]), which is exact for English text.
"""
import base64
import heapq
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DL = ROOT / "dl"

# ----------------------------------------------------------------------------
# Generic BPE merge engine (heap based, HF/tiktoken semantics)
# ----------------------------------------------------------------------------

def bpe_merge(symbols, pair_rank):
    """symbols: list of atomic symbols (str or bytes). pair_rank(a,b)->int|None.
    Returns number of tokens after merging (we only need the count)."""
    n = len(symbols)
    if n < 2:
        return n
    sym = list(symbols)
    nxt = list(range(1, n)) + [-1]
    prv = [-1] + list(range(0, n - 1))
    alive = [True] * n
    heap = []
    for i in range(n - 1):
        r = pair_rank(sym[i], sym[i + 1])
        if r is not None:
            heap.append((r, i, sym[i], sym[i + 1]))
    heapq.heapify(heap)
    count = n
    while heap:
        r, i, a, b = heapq.heappop(heap)
        if not alive[i] or sym[i] != a:
            continue
        j = nxt[i]
        if j == -1 or sym[j] != b:
            continue
        # merge i and j
        sym[i] = a + b
        alive[j] = False
        k = nxt[j]
        nxt[i] = k
        if k != -1:
            prv[k] = i
            rr = pair_rank(sym[i], sym[k])
            if rr is not None:
                heapq.heappush(heap, (rr, i, sym[i], sym[k]))
        p = prv[i]
        if p != -1:
            rr = pair_rank(sym[p], sym[i])
            if rr is not None:
                heapq.heappush(heap, (rr, p, sym[p], sym[i]))
        count -= 1
    return count


# ----------------------------------------------------------------------------
# Byte-level helpers (GPT-2 style)
# ----------------------------------------------------------------------------

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

BYTE2UNI = _bytes_to_unicode()

def to_bytelevel(s):
    return "".join(BYTE2UNI[b] for b in s.encode("utf-8"))


# Unicode-class approximations for Python's `re` (exact for English text)
L = r"[^\W\d_]"      # \p{L}
NOT_L_N = r"(?![^\W\d_]|\d)"  # negative lookahead: not letter, not digit


class HFByteBPE:
    """OLMo2 / SuperBPE tokenizer.json (ByteLevel BPE with pair merges)."""

    def __init__(self, path, name):
        t = json.load(open(path))
        self.name = name
        self.pair_rank_map = {tuple(m): i for i, m in enumerate(t["model"]["merges"])}
        pre = t["pre_tokenizer"]["pretokenizers"]
        self.has_digits = any(p["type"] == "Digits" for p in pre)
        split = [p for p in pre if p["type"] == "Split"]
        pat = split[0]["pattern"]["Regex"] if split else None
        if pat == "(?=(\\d{3})+(?!\\d))":
            self.mode = "nopretok"
        elif pat and pat.startswith("(?=(\\d{3})+(?!\\d))| ?\\p{L}+"):
            self.mode = "pretok"
        else:
            raise ValueError(f"unexpected pretokenizer pattern: {pat}")
        # OLMo2 word pattern, with \p{L}/\p{N} approximated
        self.word_re = re.compile(rf" ?{L}+| ?(?:{NOT_L_N}\S)+|\s+(?!\S)|\s+")
        self.cache = {}

    @staticmethod
    def _split_digits(text):
        """HF Digits(individual_digits=False) then Split on (?=(\\d{3})+(?!\\d))."""
        out = []
        for m in re.finditer(r"\d+|[^\d]+", text):
            s = m.group()
            if s[0].isdigit():
                # split into groups of 3 from the right
                first = len(s) % 3
                if first:
                    out.append(s[:first])
                for i in range(first, len(s), 3):
                    out.append(s[i:i + 3])
            else:
                out.append(s)
        return out

    def pretokenize(self, text):
        pieces = []
        for chunk in self._split_digits(text):
            if chunk[0].isdigit() or self.mode == "nopretok":
                pieces.append(chunk)
            else:
                pieces.extend(m.group() for m in self.word_re.finditer(chunk))
        return pieces

    def _rank(self, a, b):
        return self.pair_rank_map.get((a, b))

    def count(self, text):
        total = 0
        for piece in self.pretokenize(text):
            c = self.cache.get(piece)
            if c is None:
                c = bpe_merge(list(to_bytelevel(piece)), self._rank)
                if len(piece) < 64:
                    self.cache[piece] = c
            total += c
        return total


class TiktokenBPE:
    """Llama 3 tokenizer.model (base64 token -> rank)."""

    PAT = re.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        rf"|(?:{NOT_L_N}[^\r\n])?{L}+"
        r"|\d{1,3}"
        rf"| ?(?:{NOT_L_N}\S)+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    )

    def __init__(self, path, name):
        self.name = name
        self.ranks = {}
        for line in open(path):
            tok, rank = line.split()
            self.ranks[base64.b64decode(tok)] = int(rank)
        self.cache = {}

    def _rank(self, a, b):
        return self.ranks.get(a + b)

    def count(self, text):
        total = 0
        for m in self.PAT.finditer(text):
            piece = m.group()
            c = self.cache.get(piece)
            if c is None:
                b = piece.encode("utf-8")
                if b in self.ranks:
                    c = 1
                else:
                    c = bpe_merge([bytes([x]) for x in b], self._rank)
                self.cache[piece] = c
            total += c
        return total


# ----------------------------------------------------------------------------
# Steno converter
# ----------------------------------------------------------------------------

VOWELS = set("aeiouy")

def syllables(word):
    w = word.lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 1
    groups = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and groups > 1:
        groups -= 1
    return max(1, groups)


class Steno:
    MAX_PHRASE = 8

    def __init__(self, path):
        d = json.load(open(path))
        self.word = {}          # lowercase word -> min strokes
        self.word_exact = {}    # exact-case word -> min strokes
        self.phrase = {}        # tuple(lowercase words) -> min strokes
        self.suffix = {}        # 'ing' -> strokes
        self.prefix = {}        # 'un' -> strokes
        self.punct = {}         # ',' -> strokes
        self.letters = set()
        meta_re = re.compile(r"\{[^}]*\}")

        def better(m, k, v):
            if k not in m or v < m[k]:
                m[k] = v

        for outline, tr in d.items():
            n = outline.count("/") + 1
            if "{" in tr:
                metas = meta_re.findall(tr)
                rest = meta_re.sub("", tr).strip()
                if rest:
                    continue  # mixed entries: ignore for simplicity
                if len(metas) == 1:
                    m = metas[0][1:-1]
                    if m.startswith("^") and m.endswith("^") and len(m) > 2:
                        better(self.punct, m[1:-1], n)
                    elif m.startswith("^") and len(m) > 1:
                        s = m[1:]
                        if re.fullmatch(r"[a-z']+", s):
                            better(self.suffix, s, n)
                        elif re.fullmatch(r"[^\w\s]+", s):
                            better(self.punct, s, n)
                    elif m.endswith("^") and len(m) > 1:
                        s = m[:-1]
                        if re.fullmatch(r"[a-z]+", s):
                            better(self.prefix, s, n)
                    elif m.startswith("&") and len(m) == 2 and m[1].isalpha():
                        self.letters.add(m[1].lower())
                    elif re.fullmatch(r"[^\w\s]+", m):
                        better(self.punct, m, n)
                continue
            tr = tr.strip()
            if not tr:
                continue
            words = tr.split()
            if len(words) == 1:
                if re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", tr):
                    better(self.word, tr.lower(), n)
                    better(self.word_exact, tr, n)
                else:
                    better(self.word_exact, tr, n)
            else:
                key = tuple(w.lower() for w in words)
                if all(re.fullmatch(r"[a-z][a-z'\-]*", w) for w in key) and len(key) <= self.MAX_PHRASE:
                    better(self.phrase, key, n)
        for p in ".,?!:;":
            self.punct.setdefault(p, 1)
        for p in ['"', "'", "(", ")", "-", "--", "...", "/", "[", "]"]:
            self.punct.setdefault(p, 1)
        self.cache = {}

    # --- word cost -------------------------------------------------------
    def _lookup(self, w):
        """strokes for a bare word (no case handling), or None"""
        c = self.word.get(w)
        if c is not None:
            return c
        # apostrophe forms: court's, students'
        m = re.fullmatch(r"([a-z\-]+)('s|')", w)
        if m and m.group(1) in self.word:
            return self.word[m.group(1)] + self.suffix.get(m.group(2), 1)
        # hyphenated
        if "-" in w:
            parts = w.split("-")
            costs = [self._lookup(p) for p in parts if p]
            if all(c is not None for c in costs):
                return sum(costs) + self.punct.get("-", 1) * (len(parts) - 1)
        # suffix folding with orthographic repair
        best = None
        for s, sc in self.suffix.items():
            if len(s) >= len(w) or not w.endswith(s):
                continue
            stem = w[: -len(s)]
            cands = {stem, stem + "e"}
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                cands.add(stem[:-1])              # running -> run
            if stem.endswith("i"):
                cands.add(stem[:-1] + "y")        # tried -> try
            for c in cands:
                if c in self.word:
                    v = self.word[c] + sc
                    if best is None or v < best:
                        best = v
        if best is not None:
            return best
        for p, pc in self.prefix.items():
            if len(p) < len(w) and w.startswith(p) and w[len(p):] in self.word:
                v = pc + self.word[w[len(p):]]
                if best is None or v < best:
                    best = v
        return best

    def word_cost(self, w, sentence_start):
        """returns (plover_strokes, ideal_strokes, is_oov)"""
        key = (w, sentence_start)
        r = self.cache.get(key)
        if r is not None:
            return r
        lw = w.lower()
        c = self._lookup(lw)
        if c is not None:
            extra = 0
            if w != lw and not sentence_start and w not in self.word_exact:
                extra = 1  # capitalisation stroke for a proper noun not in dict
            r = (c + extra, c + extra, False)
        else:
            letters = sum(1 for ch in w if ch.isalpha())
            r = (max(1, letters), syllables(w), True)
        self.cache[key] = r
        return r

    # --- sequence cost (DP over briefs) -----------------------------------
    TOK = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*(?:-[A-Za-z]+(?:['’][A-Za-z]+)*)*'?|\d+(?:[.,]\d+)*|--|\.\.\.|[^\w\s]")

    def count(self, text):
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("—", "--")
        toks = [m.group() for m in self.TOK.finditer(text)]
        n = len(toks)
        INF = float("inf")
        dp_p = [INF] * (n + 1)
        dp_i = [INF] * (n + 1)
        dp_p[0] = dp_i[0] = 0.0
        oov_words = 0
        oov_letters = 0
        words = 0
        sentence_start = True
        starts = []
        for t in toks:
            starts.append(sentence_start)
            if t[0].isalpha():
                sentence_start = False
            elif t in (".", "?", "!"):
                sentence_start = True
        # precompute per-token single costs
        single = []
        for idx, t in enumerate(toks):
            if t[0].isalpha():
                words += 1
                p, i, oov = self.word_cost(t, starts[idx])
                if oov:
                    oov_words += 1
                    oov_letters += len(t)
                single.append((p, i))
            elif t[0].isdigit():
                digits = sum(ch.isdigit() for ch in t)
                c = (digits + 1) // 2 + t.count(",") + t.count(".")
                single.append((c, c))
            else:
                c = self.punct.get(t, 1)
                single.append((c, c))
        lower = [t.lower() if t[0].isalpha() else None for t in toks]
        for i in range(n):
            if dp_p[i] == INF:
                continue
            p, q = single[i]
            if dp_p[i] + p < dp_p[i + 1]:
                dp_p[i + 1] = dp_p[i] + p
            if dp_i[i] + q < dp_i[i + 1]:
                dp_i[i + 1] = dp_i[i] + q
            if lower[i] is None:
                continue
            key = []
            for j in range(i, min(n, i + self.MAX_PHRASE)):
                if lower[j] is None:
                    break
                key.append(lower[j])
                if len(key) >= 2:
                    c = self.phrase.get(tuple(key))
                    if c is not None:
                        if dp_p[i] + c < dp_p[j + 1]:
                            dp_p[j + 1] = dp_p[i] + c
                        if dp_i[i] + c < dp_i[j + 1]:
                            dp_i[j + 1] = dp_i[i] + c
        return {
            "words": words,
            "steno_plover": int(dp_p[n]),
            "steno_ideal": int(dp_i[n]),
            "oov_words": oov_words,
            "oov_rate": oov_words / max(1, words),
        }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(corpus_dir):
    steno = Steno(DL / "plover/plover/assets/main.json")
    toks = [
        TiktokenBPE(DL / "llama-models/models/llama3/tokenizer.model", "llama3_128k"),
        HFByteBPE(DL / "superbpe/tokenizer_json/olmo2_p99_truncate_pretok_10G_200K/tokenizer.json", "bpe_200k"),
        HFByteBPE(DL / "superbpe/tokenizer_json/olmo2_p99_truncate_10G_180K_extend_200K_mw4_colon/tokenizer.json", "superbpe_t180k"),
    ]
    files = sorted(Path(corpus_dir).glob("*.txt"))
    rows = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        r = {"corpus": f.stem, "chars": len(text), "bytes": len(text.encode("utf-8"))}
        r.update(steno.count(text))
        for t in toks:
            r[t.name] = t.count(text)
        rows.append(r)
        print(f"done {f.stem}", file=sys.stderr)

    cols = ["llama3_128k", "bpe_200k", "superbpe_t180k", "steno_plover", "steno_ideal"]
    print()
    print("Positions per word (lower is cheaper to decode)")
    print(f"{'corpus':<12}{'words':>9}{'oov%':>7}" + "".join(f"{c:>16}" for c in cols))
    for r in rows:
        print(f"{r['corpus']:<12}{r['words']:>9}{100*r['oov_rate']:>6.1f}%" + "".join(f"{r[c]/r['words']:>16.3f}" for c in cols))
    print()
    print("Bytes per position (higher is denser)")
    print(f"{'corpus':<12}{'bytes':>9}" + "".join(f"{c:>16}" for c in cols))
    for r in rows:
        print(f"{r['corpus']:<12}{r['bytes']:>9}" + "".join(f"{r['bytes']/r[c]:>16.2f}" for c in cols))
    print()
    print("Steno positions relative to each BPE tokenizer (>1.0 = steno needs MORE positions)")
    print(f"{'corpus':<12}" + "".join(f"{'plover/'+c:>22}" for c in cols[:3]) + "".join(f"{'ideal/'+c:>22}" for c in cols[:3]))
    for r in rows:
        print(f"{r['corpus']:<12}" + "".join(f"{r['steno_plover']/r[c]:>22.3f}" for c in cols[:3]) + "".join(f"{r['steno_ideal']/r[c]:>22.3f}" for c in cols[:3]))
    json.dump(rows, open(ROOT / "runs" / "strokes_per_word.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "corpus")
