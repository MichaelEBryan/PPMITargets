#!/usr/bin/env python
"""Age at onset in PPMI, by genetic cause and by common-variant burden."""
import os, sys, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clinical_style import *          # noqa
import matplotlib.pyplot as plt
import paths as _P
warnings.filterwarnings("ignore")

RES = str(_P.RESULTS)
DATA = str(_P.DATA)

ROUTE_ORDER = ["PRKN or PINK1 carrier", "SNCA carrier", "GBA carrier",
               "LRRK2 carrier", "no known variant"]
ROUTE_LABEL = {"PRKN or PINK1 carrier": "PRKN or PINK1",
               "SNCA carrier": "SNCA", "GBA carrier": "GBA",
               "LRRK2 carrier": "LRRK2",
               "no known variant": "no known variant"}


def main():
    d = pd.read_parquet(f"{DATA}/ppmi_aao.parquet")
    allc = d[(d.COHORT_DEFINITION == "Parkinson's Disease") & d.AAO.notna()]
    byroute = pd.read_csv(f"{RES}/191_cases_by_route.csv")
    summ = pd.read_csv(f"{RES}/190_onset_by_route.csv").set_index("route")
    curves = pd.read_csv(f"{RES}/150_onset_curves_by_tertile.csv", index_col=0)
    S = json.load(open(f"{RES}/152_clinical_summary.json"))

    fig = plt.figure(figsize=(7.0, 8.7))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.82, 1.05, 1.20, 0.76],
                          hspace=0.58, left=0.215, right=0.955, top=0.960,
                          bottom=0.060)

    # a. when patients develop Parkinson's disease
    ax = fig.add_subplot(gs[0])
    bins = np.arange(15, 96, 2.5)
    ax.hist(allc.AAO, bins=bins, color=FILL, edgecolor=MID, lw=0.7, zorder=3)
    med = allc.AAO.median()
    ax.axvline(med, color=NAVY, lw=1.6, zorder=5)
    ax.text(med + 2.0, ax.get_ylim()[1] * 0.96, f"median {med:.0f} years",
            color=NAVY, fontsize=8.6, va="top")
    ax.axvspan(15, 50, color=GREY_L, alpha=0.35, zorder=1)
    ax.text(32, ax.get_ylim()[1] * 0.92, "onset before 50", color=GREY,
            fontsize=8.4, ha="center", va="top")
    ax.set_xlabel("age at first motor symptom (years)")
    ax.set_ylabel("patients")
    ax.set_xlim(15, 95)

    # b. the same disease, from single genes to none at all
    ax2 = fig.add_subplot(gs[1])
    rng = np.random.default_rng(5)
    shades = {"PRKN or PINK1 carrier": DEEP, "SNCA carrier": NAVY,
              "GBA carrier": "#5583ab", "LRRK2 carrier": MID,
              "no known variant": PALE}
    y = np.arange(len(ROUTE_ORDER))[::-1]
    for yi, r in zip(y, ROUTE_ORDER):
        v = byroute[byroute.route == r].AAO.values
        if not len(v):
            continue
        col = shades[r]
        ax2.scatter(v, yi + rng.normal(0, 0.10, len(v)), s=9, color=col,
                    alpha=0.40 if len(v) > 200 else 0.75, edgecolors="none",
                    zorder=3)
        q1, m_, q3 = np.percentile(v, [25, 50, 75])
        ax2.plot([q1, q3], [yi, yi], color=col, lw=3.0, alpha=0.55,
                 solid_capstyle="round", zorder=5)
        ax2.plot([m_], [yi], "o", ms=7, color=col,
                 markeredgecolor="white" if r != "no known variant" else GREY,
                 markeredgewidth=1.1, zorder=6)
    ax2.axvline(float(summ.loc["no known variant", "median"]), color=GREY_L,
                lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{ROUTE_LABEL[r]}\nn = {int(summ.loc[r, 'n'])}"
                         for r in ROUTE_ORDER], fontsize=8.2)
    for t, r in zip(ax2.get_yticklabels(), ROUTE_ORDER):
        if r != "no known variant":
            t.set_style("italic")
    ax2.set_xlim(15, 95)
    ax2.set_ylim(-0.7, len(ROUTE_ORDER) - 0.02)
    ax2.set_xlabel("age at first motor symptom (years)")
    bare_y(ax2)
    lead(ax2, "25 years earlier than patients\nwith no known variant",
         xy=(37.0, len(ROUTE_ORDER) - 1), xytext=(58.0, len(ROUTE_ORDER) - 0.30),
         color=DEEP)

    # c. cumulative onset by polygenic third
    ax3 = fig.add_subplot(gs[2])
    styles = [("lowest third", PALE), ("middle third", MID),
              ("highest third", NAVY)]
    for name, col in styles:
        if name in curves.columns:
            ax3.step(curves.index, curves[name] * 100, where="post", color=col,
                     lw=2.6, zorder=4)
    LABEL_AT = {"highest third": (52.0, 44.0, 82.0),
                "middle third": (54.5, 44.0, 71.0),
                "lowest third": (57.0, 44.0, 60.0)}
    for name, col in styles:
        if name not in curves.columns:
            continue
        xa, xt, yt = LABEL_AT[name]
        idx = curves.index[curves.index <= xa]
        yv = float(curves[name].reindex(idx).iloc[-1] * 100)
        ax3.annotate(name, xy=(xa, yv), xytext=(xt, yt), color=col,
                     fontsize=8.8, va="center", ha="right",
                     arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                                     shrinkA=1, shrinkB=3))
    for name, col in styles:
        t = [r for r in S["tertiles"] if r["tertile"] == name]
        if not t:
            continue
        m_ = t[0]["median"]
        ax3.plot([m_, m_], [0, 50], color=col, lw=0.9, ls=(0, (3, 3)), zorder=2)
        ax3.plot([m_], [50], "o", ms=6, color=col, markeredgecolor="white",
                 markeredgewidth=1.0, zorder=6)
    ax3.annotate("", xy=(58.0, 50), xytext=(63.8, 50),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1,
                                 shrinkA=4, shrinkB=4))
    ax3.text(67.0, 50.0, f"  {S['median_diff_years']:.1f} years earlier",
             ha="left", va="center", fontsize=9, color=INK)
    ax3.axhline(50, color=GREY_L, lw=0.9, zorder=1)
    ax3.set_xlabel("age (years)")
    ax3.set_ylabel("patients with Parkinson's\ndisease (%)")
    ax3.set_xlim(25, 92); ax3.set_ylim(0, 104)
    ax3.set_yticks([0, 25, 50, 75, 100])

    # d. onset before 50, by third
    ax4 = fig.add_subplot(gs[3])
    names = [t["tertile"] for t in S["tertiles"]]
    vals = [t["pct_before_50"] for t in S["tertiles"]]
    ns = [t["n"] for t in S["tertiles"]]
    y = np.arange(len(names))[::-1]
    ax4.barh(y, vals, height=0.52, color=[PALE, MID, NAVY], zorder=3)
    for yi, (v, n) in zip(y, zip(vals, ns)):
        ax4.text(v + 0.7, yi, f"{v:.1f}%", va="center", fontsize=8.8,
                 color=INK)
    ax4.set_yticks(y)
    ax4.set_yticklabels([f"{nm}\nn = {n}" for nm, n in zip(names, ns)])
    ax4.set_xlabel("patients with onset before age 50 (%)")
    ax4.set_xlim(0, 27)
    bare_y(ax4)

    for a, s in zip([ax, ax2, ax3, ax4], "abcd"):
        PL(fig, a, s)
    save(fig, "C1_when_patients_develop_pd")


if __name__ == "__main__":
    main()
