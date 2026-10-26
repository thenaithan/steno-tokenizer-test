#!/usr/bin/env python3
"""
Rebuild raw/{fiction,court}/*.txt from the GitHub clones under ../dl:
  GITenberg/<Title>_<id>            (one repo per novel; <id>-0.txt or <id>.txt)
  walkerdb/supreme_court_transcripts (oyez/cases/*-t0?.json)
Only needed if raw/ is missing; the shipped raw/ was built with this.
"""
import glob
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DL = HERE.parent / "dl"
out = HERE / "raw"
(out / "fiction").mkdir(parents=True, exist_ok=True)
(out / "court").mkdir(parents=True, exist_ok=True)

for d in sorted(glob.glob(str(DL / "*_[0-9]*"))):
    if not os.path.isdir(d):
        continue
    files = sorted(glob.glob(d + "/*-0.txt")) or sorted(glob.glob(d + "/*.txt"))
    files = [f for f in files if re.search(r"/\d+(-0)?\.txt$", f)]
    if not files:
        continue
    raw = open(files[0], "rb").read()
    try:
        s = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        s = raw.decode("latin-1")
    a, b = s.find("*** START OF"), s.rfind("*** END OF")
    if a < 0 or b < 0:
        print("no Gutenberg markers, skipped:", d)
        continue
    s = s[s.find("\n", a) + 1:b]
    (out / "fiction" / f"{Path(d).name}.txt").write_text(s, encoding="utf-8")

for f in sorted(glob.glob(str(DL / "supreme_court_transcripts/oyez/cases/*-t0*.json"))):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    t = d.get("transcript")
    if not t:
        continue
    paras = []
    for sec in t["sections"]:
        for turn in sec["turns"]:
            txt = " ".join(b["text"].strip() for b in turn["text_blocks"])
            if txt:
                paras.append(txt)
    (out / "court" / (Path(f).stem + ".txt")).write_text("\n\n".join(paras), encoding="utf-8")
print("fiction docs:", len(list((out / "fiction").glob("*.txt"))), "court docs:", len(list((out / "court").glob("*.txt"))))
