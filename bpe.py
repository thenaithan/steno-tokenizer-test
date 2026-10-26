"""
Minimal byte-level BPE (GPT-2 style), pure Python: trainer + encoder.
Used so the BPE condition has a vocabulary of exactly the same size as
the steno stroke vocabulary, trained on the same text.
"""
import heapq
import json
import re
from collections import Counter, defaultdict

L = r"[^\W\d_]"
NOT_L_N = r"(?![^\W\d_]|\d)"
# GPT-2 pretokeniser with \p{L}/\p{N} approximated (exact for English)
PRETOK = re.compile(rf"'s|'t|'re|'ve|'m|'ll|'d| ?{L}+| ?\d+| ?(?:{NOT_L_N}\S)+|\s+(?!\S)|\s+")


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
UNI2BYTE = {v: k for k, v in BYTE2UNI.items()}


def to_bytelevel(s):
    return "".join(BYTE2UNI[b] for b in s.encode("utf-8"))


def bpe_merge(symbols, pair_rank):
    """Heap-based merge with HF/tiktoken semantics. Returns merged symbol list."""
    n = len(symbols)
    if n < 2:
        return list(symbols)
    sym = list(symbols)
    nxt = list(range(1, n)) + [-1]
    prv = [-1] + list(range(0, n - 1))
    alive = [True] * n
    heap = []
    for i in range(n - 1):
        r = pair_rank.get((sym[i], sym[i + 1]))
        if r is not None:
            heap.append((r, i, sym[i], sym[i + 1]))
    heapq.heapify(heap)
    while heap:
        r, i, a, b = heapq.heappop(heap)
        if not alive[i] or sym[i] != a:
            continue
        j = nxt[i]
        if j == -1 or sym[j] != b:
            continue
        sym[i] = a + b
        alive[j] = False
        k = nxt[j]
        nxt[i] = k
        if k != -1:
            prv[k] = i
            rr = pair_rank.get((sym[i], sym[k]))
            if rr is not None:
                heapq.heappush(heap, (rr, i, sym[i], sym[k]))
        p = prv[i]
        if p != -1:
            rr = pair_rank.get((sym[p], sym[i]))
            if rr is not None:
                heapq.heappush(heap, (rr, p, sym[p], sym[i]))
    return [sym[i] for i in range(n) if alive[i]]


def train_bpe(text, vocab_size, log=None):
    """Return (vocab list, merges list). vocab = 256 byte symbols + merges."""
    counts = Counter(m.group() for m in PRETOK.finditer(text))
    words = []
    freqs = []
    for w, f in counts.items():
        words.append(list(to_bytelevel(w)))
        freqs.append(f)
    pair_count = defaultdict(int)
    where = defaultdict(set)
    for wi, w in enumerate(words):
        f = freqs[wi]
        for a, b in zip(w, w[1:]):
            pair_count[(a, b)] += f
            where[(a, b)].add(wi)
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)
    merges = []
    target = vocab_size - 256
    while len(merges) < target and heap:
        negc, pair = heapq.heappop(heap)
        if pair_count.get(pair, 0) != -negc or -negc == 0:
            continue
        a, b = pair
        ab = a + b
        merges.append(pair)
        changed = set()
        for wi in list(where[pair]):
            w = words[wi]
            f = freqs[wi]
            # remove old pair contributions of this word
            for x, y in zip(w, w[1:]):
                pair_count[(x, y)] -= f
                changed.add((x, y))
            nw = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    nw.append(ab)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            words[wi] = nw
            for x, y in zip(nw, nw[1:]):
                pair_count[(x, y)] += f
                where[(x, y)].add(wi)
                changed.add((x, y))
        del where[pair]
        pair_count.pop(pair, None)
        for p in changed:
            c = pair_count.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
        if log and len(merges) % 2000 == 0:
            log(f"  bpe merges {len(merges)}/{target}")
    vocab = [BYTE2UNI[i] for i in range(256)] + [a + b for a, b in merges]
    return vocab, merges


class BPEEncoder:
    def __init__(self, vocab, merges):
        self.vocab = list(vocab)
        self.id = {t: i for i, t in enumerate(self.vocab)}
        self.pair_rank = {tuple(m): i for i, m in enumerate(merges)}
        self.cache = {}

    def encode(self, text):
        out = []
        for m in PRETOK.finditer(text):
            piece = m.group()
            ids = self.cache.get(piece)
            if ids is None:
                ids = [self.id[s] for s in bpe_merge(list(to_bytelevel(piece)), self.pair_rank)]
                if len(piece) < 32:
                    self.cache[piece] = ids
            out.extend(ids)
        return out

    def decode(self, ids):
        s = "".join(self.vocab[i] for i in ids)
        return bytes(UNI2BYTE[ch] for ch in s).decode("utf-8", errors="replace")

    def save(self, path):
        json.dump({"vocab": self.vocab, "merges": [list(m) for m in self.pair_rank]}, open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["vocab"], d["merges"])
