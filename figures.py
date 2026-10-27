"""Figures for the README, from the committed run records.

    python3 figures.py            # writes figures/*.png

Every number plotted comes from runs/strokes_per_word.json (Experiment 1) and
runs/<name>.json (Experiment 2); nothing is retyped.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# palette: one hue per tokenizer family, fixed order
TOK = {"Llama 3 (128k)": "#52514e", "BPE-200k": "#9a9893", "SuperBPE-t180k": "#eb6834",
       "steno (Plover)": "#2a78d6", "steno (ideal dict)": "#8fb8ea"}
RUN = {"bpe": "#52514e", "steno_id": "#2a78d6", "steno_fact_in": "#1baf7a", "steno_fact_both": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
})


def positions_per_word():
    rows = json.load(open("runs/strokes_per_word.json"))
    keys = {"Llama 3 (128k)": "llama3_128k", "BPE-200k": "bpe_200k", "SuperBPE-t180k": "superbpe_t180k",
            "steno (Plover)": "steno_plover", "steno (ideal dict)": "steno_ideal"}
    corpora = [r["corpus"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    n = len(keys)
    w = 0.8 / n
    for i, (label, k) in enumerate(keys.items()):
        xs = [j + (i - (n - 1) / 2) * w for j in range(len(rows))]
        ys = [r[k] / r["words"] for r in rows]
        ax.bar(xs, ys, width=w * 0.92, color=TOK[label], label=label, edgecolor="white", linewidth=0.6)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(corpora)
    ax.set_ylabel("positions per word (lower is cheaper to decode)")
    ax.set_ylim(0, 2.6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Experiment 1: positions per word, three tokenizers against stenotype strokes")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/positions.png")


def training_curves():
    names = ["bpe", "steno_id", "steno_fact_in", "steno_fact_both"]
    runs = {n: json.load(open(f"runs/{n}.json")) for n in names}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, split in zip(axes, ["fiction", "court"]):
        for n in names:
            log = runs[n]["log"]
            xs = [e["positions"] / 1e6 for e in log]
            ys = [e[f"val_{split}_bpb"] for e in log]
            ax.plot(xs, ys, color=RUN[n], lw=1.6, label=n)
        ax.set_title(f"held-out {split}", fontsize=10)
        ax.set_xlabel("training positions (millions)")
        ax.set_ylim(1.0, 3.6)
        ax.set_xlim(0, 50)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("bits per byte")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment 2: validation bits per byte at matched compute", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/bpb.png")


positions_per_word()
training_curves()
print("wrote", sorted(os.listdir(OUT)))
