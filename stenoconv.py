"""
Text -> stenotype outlines, using Plover's main.json.

Every choice is made in steno's favour: optimal (DP) use of multi-word
briefs, suffix/prefix folding with orthographic repair, free spaces,
free sentence-initial capitals. Each stroke is also parsed into its
23 keys (#STKPWHRAO*EUFRPBLGTSDZ) so a model can use the chord structure.
"""
import json
import re

STENO_ORDER = "#STKPWHRAO*EUFRPBLGTSDZ"   # 23 keys
NKEYS = len(STENO_ORDER)
# number bar: digit -> key letter (left bank for 1-5/0 by position)
DIGIT_KEY = {"1": "S", "2": "T", "3": "P", "4": "H", "5": "A", "0": "O", "6": "F", "7": "P", "8": "L", "9": "T"}

_STROKE_RE = re.compile(r"^(#?)(S?T?K?P?W?H?R?)(A?O?)(\*?)(-?)(E?U?)(F?R?P?B?L?G?T?S?D?Z?)$")

# Standard Plover fingerspelling (lowercase: letter+*, capital: letter+*P)
FINGERSPELL = {
    "a": "A*", "b": "PW*", "c": "KR*", "d": "TK*", "e": "E*", "f": "TP*", "g": "TKPW*", "h": "H*",
    "i": "EU*", "j": "SKWR*", "k": "K*", "l": "HR*", "m": "PH*", "n": "TPH*", "o": "O*", "p": "P*",
    "q": "KW*", "r": "R*", "s": "S*", "t": "T*", "u": "U*", "v": "SR*", "w": "W*", "x": "KP*",
    "y": "KWR*", "z": "STKPW*",
}
CAP_STROKE = "KPA"        # {-|} capitalise next
PARA_STROKE = "R-R"       # new paragraph
HYPHEN_STROKE = "H-PB"    # {^-^}
UNKNOWN_STROKE = "*"      # anything we cannot write
PUNCT_DEFAULT = {
    ".": "TP-PL", ",": "KW-BG", "?": "STPH", "!": "SKHRAPL", ":": "STPH-FPLT", ";": "STPH*FPLT",
    '"': "KW-GS", "'": "AE", "-": "H-PB", "--": "TK-RB", "...": "SW-BS", "(": "PREPB", ")": "PR*PB",
    "[": "PWR-BGT", "]": "PWR*BGT", "/": "OEU", "&": "SP-PBD", "$": "TK-PL", "%": "P*ERS", "*": "STA*R",
    "_": "RUPBD", "@": "SKWRAT", "#": "HAERB",
}


def parse_stroke(s):
    """Return a 23-bit key tuple for a stroke in Plover notation (digits allowed)."""
    t = s
    if any(ch.isdigit() for ch in t):
        t = "#" + "".join(DIGIT_KEY.get(ch, ch) for ch in t if ch != "#")
    m = _STROKE_RE.match(t)
    if not m:
        # fallback: greedy left-to-right along steno order
        bits = [0] * NKEYS
        pos = 0
        for ch in t:
            if ch == "-":
                pos = max(pos, STENO_ORDER.index("*") + 1)
                continue
            j = STENO_ORDER.find(ch, pos)
            if j == -1:
                j = STENO_ORDER.find(ch)
                if j == -1:
                    continue
            bits[j] = 1
            pos = j + 1
        return tuple(bits)
    bits = [0] * NKEYS
    hashk, left, vowel, star, _, eu, right = m.groups()
    if hashk:
        bits[0] = 1
    for ch in left:
        bits[1 + "STKPWHR".index(ch)] = 1
    for ch in vowel:
        bits[8 + "AO".index(ch)] = 1
    if star:
        bits[10] = 1
    for ch in eu:
        bits[11 + "EU".index(ch)] = 1
    for ch in right:
        bits[13 + "FRPBLGTSDZ".index(ch)] = 1
    return tuple(bits)


def syllables(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    g = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and g > 1:
        g -= 1
    return max(1, g)


class StenoConverter:
    MAX_PHRASE = 8
    TOK = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*(?:-[A-Za-z]+(?:'[A-Za-z]+)*)*'?|\d+(?:[.,]\d+)*|--|\.\.\.|\n\n|[^\w\s]")

    def __init__(self, main_json):
        d = json.load(open(main_json))
        self.word = {}        # lowercase -> (nstrokes, outline)
        self.word_exact = {}  # exact case -> (nstrokes, outline)
        self.phrase = {}      # tuple(lowercase) -> (nstrokes, outline)
        self.suffix = {}
        self.prefix = {}
        self.punct = {k: (1, v) for k, v in PUNCT_DEFAULT.items()}
        meta_re = re.compile(r"\{[^}]*\}")

        def better(m, k, outline):
            n = outline.count("/") + 1
            cur = m.get(k)
            if cur is None or n < cur[0] or (n == cur[0] and len(outline) < len(cur[1])):
                m[k] = (n, outline)

        for outline, tr in d.items():
            if "{" in tr:
                metas = meta_re.findall(tr)
                if meta_re.sub("", tr).strip() or len(metas) != 1:
                    continue
                m = metas[0][1:-1]
                if m.startswith("^") and m.endswith("^") and len(m) > 2:
                    better(self.punct, m[1:-1], outline)
                elif m.startswith("^") and len(m) > 1:
                    s = m[1:]
                    if re.fullmatch(r"[a-z']+", s):
                        better(self.suffix, s, outline)
                    elif re.fullmatch(r"[^\w\s]+", s):
                        better(self.punct, s, outline)
                elif m.endswith("^") and len(m) > 1 and re.fullmatch(r"[a-z]+", m[:-1]):
                    better(self.prefix, m[:-1], outline)
                elif re.fullmatch(r"[^\w\s]+", m):
                    better(self.punct, m, outline)
                continue
            tr = tr.strip()
            if not tr:
                continue
            words = tr.split()
            if len(words) == 1:
                if re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", tr):
                    better(self.word, tr.lower(), outline)
                    better(self.word_exact, tr, outline)
            else:
                key = tuple(w.lower() for w in words)
                if len(key) <= self.MAX_PHRASE and all(re.fullmatch(r"[a-z][a-z'\-]*", w) for w in key):
                    better(self.phrase, key, outline)
        self.cache = {}

    # ---- single word -> (nstrokes, outline, oov) ---------------------------
    def _lookup(self, w):
        c = self.word.get(w)
        if c is not None:
            return c
        m = re.fullmatch(r"([a-z\-]+)('s|')", w)
        if m and m.group(1) in self.word:
            base = self.word[m.group(1)]
            suf = self.suffix.get(m.group(2), (1, "AES"))
            return (base[0] + suf[0], base[1] + "/" + suf[1])
        if "-" in w:
            parts = [p for p in w.split("-") if p]
            looked = [self._lookup(p) for p in parts]
            if parts and all(x is not None for x in looked):
                n = sum(x[0] for x in looked) + (len(parts) - 1)
                outline = ("/" + HYPHEN_STROKE + "/").join(x[1] for x in looked)
                return (n, outline)
        best = None
        for s, (sc, so) in self.suffix.items():
            if len(s) >= len(w) or not w.endswith(s):
                continue
            stem = w[: -len(s)]
            cands = {stem, stem + "e"}
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                cands.add(stem[:-1])
            if stem.endswith("i"):
                cands.add(stem[:-1] + "y")
            for c in cands:
                if c in self.word:
                    v = (self.word[c][0] + sc, self.word[c][1] + "/" + so)
                    if best is None or v[0] < best[0]:
                        best = v
        if best is not None:
            return best
        for p, (pc, po) in self.prefix.items():
            if len(p) < len(w) and w.startswith(p) and w[len(p):] in self.word:
                v = (pc + self.word[w[len(p):]][0], po + "/" + self.word[w[len(p):]][1])
                if best is None or v[0] < best[0]:
                    best = v
        return best

    def word_outline(self, w, sentence_start):
        key = (w, sentence_start)
        r = self.cache.get(key)
        if r is not None:
            return r
        lw = w.lower()
        c = self._lookup(lw)
        if c is not None:
            n, outline = c
            if w != lw and not sentence_start and w not in self.word_exact:
                n, outline = n + 1, CAP_STROKE + "/" + outline
            r = (n, outline, False)
        else:
            strokes = []
            for i, ch in enumerate(w):
                lc = ch.lower()
                if lc in FINGERSPELL:
                    s = FINGERSPELL[lc]
                    if ch.isupper() and not (sentence_start and i == 0):
                        s += "P"
                    strokes.append(s)
                elif ch == "-":
                    strokes.append(HYPHEN_STROKE)
                elif ch == "'":
                    strokes.append(self.punct.get("'", (1, "AE"))[1])
                else:
                    strokes.append(UNKNOWN_STROKE)
            r = (len(strokes), "/".join(strokes), True)
        self.cache[key] = r
        return r

    @staticmethod
    def number_outline(t):
        strokes = []
        for part in re.split(r"([.,])", t):
            if part in (".", ","):
                strokes.append(PUNCT_DEFAULT[part])
                continue
            for i in range(0, len(part), 2):
                pair = part[i:i + 2]
                keys = [DIGIT_KEY[d] for d in pair]
                if len(pair) == 2:
                    a, b = (STENO_ORDER.index(k) for k in keys)
                    s = "#" + "".join(keys) + ("EU" if a >= b else "")
                else:
                    s = "#" + keys[0]
                strokes.append(s)
        return strokes

    # ---- full text -> list of strokes (DP over briefs) ----------------------
    def convert(self, text):
        text = (text.replace("’", "'").replace("‘", "'").replace("“", '"')
                .replace("”", '"').replace("—", "--").replace("–", "-"))
        toks = [m.group() for m in self.TOK.finditer(text)]
        n = len(toks)
        starts = []
        ss = True
        for t in toks:
            starts.append(ss)
            if t[0].isalpha():
                ss = False
            elif t in (".", "?", "!", "\n\n"):
                ss = True
        single = []
        stats = {"words": 0, "oov_words": 0}
        for i, t in enumerate(toks):
            if t[0].isalpha():
                stats["words"] += 1
                nn, outline, oov = self.word_outline(t, starts[i])
                stats["oov_words"] += oov
                single.append((nn, outline.split("/")))
            elif t[0].isdigit():
                s = self.number_outline(t)
                single.append((len(s), s))
            elif t == "\n\n":
                single.append((1, [PARA_STROKE]))
            else:
                nn, o = self.punct.get(t, (1, UNKNOWN_STROKE))
                single.append((nn, o.split("/")))
        lower = [t.lower() if t[0].isalpha() else None for t in toks]
        INF = float("inf")
        dp = [INF] * (n + 1)
        back = [None] * (n + 1)
        dp[0] = 0
        for i in range(n):
            if dp[i] == INF:
                continue
            nn, strokes = single[i]
            if dp[i] + nn < dp[i + 1]:
                dp[i + 1] = dp[i] + nn
                back[i + 1] = (i, strokes)
            if lower[i] is None:
                continue
            key = []
            for j in range(i, min(n, i + self.MAX_PHRASE)):
                if lower[j] is None:
                    break
                key.append(lower[j])
                if len(key) >= 2:
                    c = self.phrase.get(tuple(key))
                    if c is not None and dp[i] + c[0] < dp[j + 1]:
                        dp[j + 1] = dp[i] + c[0]
                        back[j + 1] = (i, c[1].split("/"))
        out = []
        i = n
        while i > 0:
            j, strokes = back[i]
            out.append(strokes)
            i = j
        out.reverse()
        strokes = [s for group in out for s in group]
        stats["strokes"] = len(strokes)
        return strokes, stats


if __name__ == "__main__":
    import sys
    conv = StenoConverter(sys.argv[1])
    s, st = conv.convert(sys.stdin.read())
    print(" ".join(s))
    print(st, file=sys.stderr)
