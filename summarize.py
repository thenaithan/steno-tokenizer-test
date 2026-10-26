#!/usr/bin/env python3
"""Print a comparison table from runs/*.json."""
import glob
import json
import sys

runs = [json.load(open(f)) for f in sorted(glob.glob((sys.argv[1] if len(sys.argv) > 1 else "runs") + "/*.json"))]
if not runs:
    sys.exit("no runs found")
regs = list(runs[0]["final"].keys())
print(f"{'run':<18}{'embed':<14}{'head':<10}{'params':>9}" + "".join(f"{'bpb_'+r:>13}{'acc_'+r:>10}" for r in regs) + f"{'bytes/tok':>11}{'min':>7}")
for r in runs:
    f = r["final"]
    bpt = r["train_bytes_available"] / r["train_tokens_available"]
    print(f"{r['name']:<18}{r['args']['embed']:<14}{r['args']['head']:<10}{r['params_total']/1e6:>8.1f}M"
          + "".join(f"{f[g]['bits_per_byte']:>13.4f}{f[g]['top1_acc']:>10.3f}" for g in regs)
          + f"{bpt:>11.2f}{r['seconds']/60:>7.1f}")
print("\nbpb = bits per byte of the original text on the full held-out stream (lower is better; comparable across tokenizers)")
print("acc = exact next-token top-1 accuracy (NOT comparable across tokenizers; compare steno runs with each other)")
for r in runs:
    for g in regs:
        if "pred_is_valid_stroke" in r["final"][g]:
            print(f"{r['name']} {g}: argmax key pattern is a real stroke {r['final'][g]['pred_is_valid_stroke']:.3f} of the time")
