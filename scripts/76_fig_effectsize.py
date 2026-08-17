#!/usr/bin/env python
"""Effect sizes of real Parkinson's risk variants against what the method needs."""
import os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clinical_style import *          # noqa
import matplotlib.pyplot as plt
import paths as _P
warnings.filterwarnings("ignore")

RES = str(_P.RESULTS)


def load_grid():
    d = pd.read_csv(f"{RES}/09_simulation_grid.csv")
    for c in ["beta", "mean_recall_small", "mean_recall_large", "mean_overlap"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def main():
    al = pd.read_csv(f"{RES}/19b_risk_vs_onset_aligned.csv")
    orr = np.exp(al.risk_beta_target.abs().values)
    lo, hi = np.percentile(orr, [10, 90])
    med = np.median(orr)
    d = load_grid()

    fig = plt.figure(figsize=(7.0, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.15], hspace=0.46,
                          left=0.175, right=0.955, top=0.930, bottom=0.095)

    ax = fig.add_subplot(gs[0])
    bins = np.arange(1.0, 2.3, 0.04)
    ax.hist(orr, bins=bins, color=FILL, edgecolor=MID, lw=0.8, zorder=3)
    ax.axvspan(lo, hi, color=GREY_XL, zorder=1)
    ax.axvline(med, color=NAVY, lw=1.6, zorder=5)
    ax.text(med + 0.02, ax.get_ylim()[1] * 0.94, f"median {med:.2f}",
            color=NAVY, fontsize=8.6, va="top")
    ax.annotate("middle 80% of variants", xy=(hi, ax.get_ylim()[1] * 0.30),
                xytext=(1.42, ax.get_ylim()[1] * 0.62), fontsize=8.2,
                color=GREY, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GREY_L, lw=0.8,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(1.0, 2.3)
    ax.set_xlabel("effect of the variant on Parkinson's risk (odds ratio)")
    ax.set_ylabel("known risk\nvariants")

    ax2 = fig.add_subplot(gs[1])
    ax2.axvspan(lo, hi, color=GREY_XL, zorder=1)
    shades = [PALE, "#a9c2d6", MID, NAVY]
    sizes = sorted(d.n_small.unique())
    for n, c in zip(sizes, shades):
        s = d[(d.n_small == n) & (d.p == 162)].groupby(
            "beta").mean_recall_small.mean()
        ax2.plot(np.exp(s.index), s.values * 100, "o-", ms=5.5, lw=1.8,
                 color=c, markeredgecolor="white", markeredgewidth=0.8,
                 zorder=4)
    s = d[d.p == 162].groupby("beta").mean_recall_large.mean()
    ax2.plot(np.exp(s.index), s.values * 100, "s--", ms=5.5, lw=1.8,
             color=WARM, markeredgecolor="white", markeredgewidth=0.8,
             zorder=5)

    ends = [(np.exp(s.index[-1]), s.values[-1] * 100, "3,386 patients", WARM)]
    for n, c in zip(sizes, shades):
        ss = d[(d.n_small == n) & (d.p == 162)].groupby(
            "beta").mean_recall_small.mean()
        ends.append((np.exp(ss.index[-1]), ss.values[-1] * 100,
                     f"{n:,} patients", c))
    ends.sort(key=lambda t: t[1])
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < 5.5:
            ends[i] = (ends[i][0], ends[i - 1][1] + 5.5, ends[i][2], ends[i][3])
    for xe, ye, lab, c in ends:
        ax2.text(xe + 0.015, ye, lab, fontsize=7.8, va="center", ha="left",
                 color=c)

    ax2.set_xlim(1.0, 1.92); ax2.set_ylim(-3, 108)
    ax2.set_xlabel("effect of the variant on Parkinson's risk (odds ratio)")
    ax2.set_ylabel("causal variants the model\nrecovers (%)")
    ax2.text((lo + hi) / 2, 96, "where real Parkinson's\nrisk variants sit",
             ha="center", va="top", fontsize=8.2, color=GREY, linespacing=1.35)

    inband = d[(d.p == 162) & (np.exp(d.beta) >= lo) & (np.exp(d.beta) <= hi)]
    rec = inband.mean_recall_small.mean() * 100
    ax2.annotate(f"{rec:.0f}% of causal variants recovered\nin that range",
                 xy=(hi, rec), xytext=(1.30, 62), fontsize=8.4, color=INK,
                 ha="left", va="center", linespacing=1.35,
                 arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8,
                                 shrinkA=2, shrinkB=4))

    for a, s_ in zip([ax, ax2], "ab"):
        PL(fig, a, s_)
    save(fig, "S1_effect_sizes")


if __name__ == "__main__":
    main()
