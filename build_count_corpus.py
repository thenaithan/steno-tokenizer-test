#!/usr/bin/env python3
"""
Build the three small corpora used by steno_count.py into corpus/:
  fiction.txt    Pride and Prejudice + Moby-Dick (GITenberg, headers stripped)
  court.txt      the first 12 oral-argument transcripts of the 2019 term
  technical.txt  CPython Doc/tutorial/*.rst + Doc/howto/*.rst
Needs the clones listed in the README under dl/.
"""
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DL = HERE / "dl"
OUT = HERE / "corpus"
OUT.mkdir(exist_ok=True)


def gutenberg(path):
    s = open(path, encoding="utf-8-sig").read()
    a, b = s.find("*** START OF"), s.rfind("*** END OF")
    return s[s.find("\n", a) + 1:b]


fiction = gutenberg(DL / "Pride-and-Prejudice_1342/1342-0.txt") + "\n" + gutenberg(DL / "Moby-Dick--Or-The-Whale_2701/2701-0.txt")
(OUT / "fiction.txt").write_text(fiction, encoding="utf-8")

blocks = []
files = sorted(glob.glob(str(DL / "supreme_court_transcripts/oyez/cases/2019.*-t01.json")))[:12]
for f in files:
    t = json.load(open(f))["transcript"]
    for sec in t["sections"]:
        for turn in sec["turns"]:
            for b in turn["text_blocks"]:
                blocks.append(b["text"].strip())
(OUT / "court.txt").write_text("\n".join(blocks), encoding="utf-8")

tech = "".join(open(f, encoding="utf-8").read() for f in sorted(glob.glob(str(DL / "cpython/Doc/tutorial/*.rst")) + glob.glob(str(DL / "cpython/Doc/howto/*.rst"))))
(OUT / "technical.txt").write_text(tech, encoding="utf-8")
for f in OUT.glob("*.txt"):
    print(f.name, f.stat().st_size, "bytes")
