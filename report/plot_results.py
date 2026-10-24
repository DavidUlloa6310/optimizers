"""Draws the REPORT.md figures from the .npz/.md files written by optim.benchmarks.bench."""

import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "SGD": "#2a78d6",
    "SGD_Momentum": "#eb6834",
    "Nesterov_SGD": "#1baf7a",
    "Adagrad": "#eda100",
    "RMSProp": "#e87ba4",
    "Adam": "#008300",
    "Muon w/ Adam": "#6250d6",
    "Shampoo": "#e34948",
}
NAMES = {
    "SGD_Momentum": "SGD + Momentum",
    "Nesterov_SGD": "SGD + Nesterov",
    "Adagrad": "AdaGrad",
    "Muon w/ Adam": "Muon + Adam",
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def main(npz, md, spe, title, out, ylim):
    d = np.load(npz)
    test = {}
    with open(md) as f:
        lines = f.readlines()
    for line in lines:
        m = re.match(r"\| (.+?) \| [\d.]+ \| ([\d.]+) \| ([\d.]+) \|", line)
        if m:
            test[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    fig.patch.set_facecolor("white")
    for o, c in COLORS.items():
        L = np.asarray(d[f"{o}/loss"], float)
        n = len(L) // spe
        ep = L[: n * spe].reshape(n, spe).mean(1)
        a1.plot(np.arange(1, n + 1), ep, color=c, lw=2, label=NAMES.get(o, o))
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Training loss (epoch mean)")
    a1.set_title("Training loss", loc="left", fontsize=11)
    a1.set_ylim(*ylim)
    a1.grid(True, color=GRID, lw=0.8)
    a1.set_axisbelow(True)
    a1.legend(frameon=False, ncol=2, fontsize=9)
    order = sorted(test, key=lambda o: test[o][1])
    y = np.arange(len(order))
    acc = [test[o][1] * 100 for o in order]
    lo = min(acc) - 1.5
    a2.hlines(y, lo, acc, color=GRID, lw=2)
    a2.scatter(
        acc,
        y,
        s=64,
        color=[COLORS[o] for o in order],
        zorder=3,
        edgecolor="white",
        linewidth=2,
    )
    for yi, v in zip(y, acc):
        a2.text(v + 0.08, yi, f"{v:.2f}%", va="center", fontsize=9, color=INK)
    a2.set_yticks(y, [NAMES.get(o, o) for o in order])
    a2.set_xlim(lo, max(acc) + 0.9)
    a2.set_xlabel("Test accuracy (%)")
    a2.set_title("Test accuracy", loc="left", fontsize=11)
    a2.grid(True, axis="x", color=GRID, lw=0.8)
    a2.set_axisbelow(True)
    for a in (a1, a2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    # python report/plot_results.py <metrics.npz> <results.md> <steps_per_epoch> <title> <out.png> <ymin> <ymax>
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]), a[3], a[4], (float(a[5]), float(a[6])))
