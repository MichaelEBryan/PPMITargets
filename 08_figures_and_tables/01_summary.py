import sys
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.style import *
from lib import paths

warnings.filterwarnings("ignore")

RES = str(paths.RESULTS)
DATA = str(paths.DATA)


def band(fig, ax, letter, label, dy=0.016, dx=0.048):
    p = ax.get_position()
    fig.text(0.012, p.y1 + dy, letter, fontsize=11.5, fontweight="bold",
             va="bottom", ha="left", color=INK)
    fig.text(0.012 + dx, p.y1 + dy + 0.0015, label, fontsize=9.2,
             va="bottom", ha="left", color=GREY)


def main():
    summ = pd.read_csv(f"{RES}/190_onset_by_route.csv").set_index("route")
    S = json.load(open(f"{RES}/152_clinical_summary.json"))
    d = pd.read_parquet(f"{DATA}/ppmi_aao.parquet")
    allc = d[(d.COHORT_DEFINITION == "Parkinson's Disease") & d.AAO.notna()]

    fig = plt.figure(figsize=(7.2, 9.4))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.55, 0.85, 0.80],
                             hspace=0.62, left=0.235, right=0.925, top=0.928,
                             bottom=0.052)
    ax = fig.add_subplot(outer[0])
    AGE0, AGE1 = 26, 84

    ROWS = [
        ("PRKN or PINK1 carrier", "PRKN or PINK1", DEEP, True),
        ("SNCA carrier", "SNCA", NAVY, True),
        ("GBA carrier", "GBA", "#4a76a2", True),
        ("LRRK2 carrier", "LRRK2", MID, True),
        (None, None, None, None),
        ("__hi__", "highest third", NAVY, False),
        ("__mid__", "middle third", "#6e97ba", False),
        ("__lo__", "lowest third", PALE, False),
    ]
    tert = {t["tertile"]: t for t in S["tertiles"]}
    TQ = {"__hi__": tert["highest third"], "__mid__": tert["middle third"],
          "__lo__": tert["lowest third"]}

    y = np.arange(len(ROWS))[::-1].astype(float)
    ref = float(summ.loc["no known variant", "median"])
    ax.axvline(ref, color=GREY_L, lw=1.0, ls=(0, (4, 3)), zorder=1)

    for yi, (key, lab, col, ital) in zip(y, ROWS):
        if key is None:
            continue
        if key.startswith("__"):
            r = TQ[key]
            q1, m_, q3, n = r["q1"], r["median"], r["q3"], r["n"]
        else:
            q1 = float(summ.loc[key, "q1"]); q3 = float(summ.loc[key, "q3"])
            m_ = float(summ.loc[key, "median"]); n = int(summ.loc[key, "n"])
        ax.plot([q1, q3], [yi, yi], color=col, lw=5.0, alpha=0.34,
                solid_capstyle="round", zorder=3)
        ax.plot([m_], [yi], "o", ms=8.5, color=col, mec="white", mew=1.5,
                zorder=6)
        ax.text(AGE0 - 1.4, yi, lab, ha="right", va="center", fontsize=8.6,
                color=INK, style="italic" if ital else "normal")
        ax.text(AGE1 + 1.0, yi, f"{m_:.0f}", ha="left", va="center",
                fontsize=8.4, color=INK)
        ax.text(AGE1 + 6.0, yi, f"{n:,}", ha="left", va="center",
                fontsize=8.0, color=GREY)

    ax.text(AGE1 + 1.0, y[0] + 0.72, "median", ha="left", va="bottom",
            fontsize=7.8, color=GREY)
    ax.text(AGE1 + 6.0, y[0] + 0.72, "n", ha="left", va="bottom",
            fontsize=7.8, color=GREY)

    for y0, y1_, txt in [(y[0], y[3], "one causal\nvariant"),
                         (y[5], y[7], "no known causal variant,\n"
                                      "by third of common-\nvariant burden")]:
        xb = AGE0 - 16.0
        ax.plot([xb, xb], [y1_ - 0.32, y0 + 0.32], color=GREY_L, lw=1.2,
                clip_on=False, zorder=2)
        ax.text(xb - 1.4, (y0 + y1_) / 2, txt, ha="right", va="center",
                fontsize=8.0, color=GREY, linespacing=1.45, clip_on=False)

    hi_, lo_ = TQ["__hi__"]["median"], TQ["__lo__"]["median"]
    yb = y[7] - 0.88
    ax.annotate("", xy=(hi_, yb), xytext=(lo_, yb),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0,
                                shrinkA=1, shrinkB=1))
    ax.text((hi_ + lo_) / 2, yb - 0.26, f"{S['median_diff_years']:.1f} years",
            ha="center", va="top", fontsize=8.6, color=INK)

    xs = np.linspace(AGE0, AGE1, 400)
    k = st.gaussian_kde(allc.AAO)
    dens = k(xs); dens = dens / dens.max() * 1.05
    base = y[7] - 2.60
    ax.fill_between(xs, base, base + dens, color=FILL, lw=0, zorder=1)
    ax.plot(xs, base + dens, color=PALE, lw=1.0, zorder=2)
    ax.text(AGE0 - 1.4, base + 0.42, "all patients", ha="right", va="center",
            fontsize=8.0, color=GREY)
    ax.text(AGE1 + 6.0, base + 0.42, "2,107", ha="left", va="center",
            fontsize=8.0, color=GREY)

    ax.set_xlim(AGE0, AGE1); ax.set_ylim(base - 0.28, y[0] + 1.05)
    ax.set_yticks([]); ax.set_xticks([30, 40, 50, 60, 70, 80])
    ax.set_xlabel("age at first motor symptom (years)")
    for s_ in ("left", "right", "top"):
        ax.spines[s_].set_visible(False)
    g1 = outer[1].subgridspec(1, 3, wspace=0.42)
    axb, verdicts = [], []

    a1 = fig.add_subplot(g1[0])
    pv = pd.read_csv(f"{RES}/53_pervariant_AAO.csv")
    o = -np.log10(pv.cc_p.clip(lower=1e-12).sort_values().values)
    e = -np.log10((np.arange(1, len(o) + 1) - 0.5) / len(o))
    thr = -np.log10(0.05 / len(o))
    a1.plot([0, thr], [0, thr], color=GREY_L, lw=1.0, zorder=2)
    a1.scatter(e, o, s=15, color=MID, edgecolors="none", zorder=4)
    a1.axhline(thr, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=3)
    a1.set_xlim(0, thr * 0.92); a1.set_ylim(0, thr * 1.16)
    a1.set_xticks([]); a1.set_yticks([])
    a1.set_xlabel("individual variants", fontsize=8.6)
    axb.append(a1); verdicts.append("0 of 86 reach the threshold")

    a2 = fig.add_subplot(g1[1])
    mn = np.load(f"{RES}/110_maxnull_adj.npy")
    obs = pd.read_csv(f"{RES}/110_geneset_maxT.csv").t_onset_adj.max()
    a2.hist(mn, bins=60, color=FILL, edgecolor=MID, lw=0.4, density=True,
            zorder=3)
    p95 = np.percentile(mn, 95)
    yl = a2.get_ylim()[1]
    a2.axvline(p95, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=4)
    a2.axvline(obs, color=WARM, lw=1.8, zorder=5)
    a2.set_xlim(2.6, 8.6); a2.set_ylim(0, yl * 1.16)
    a2.set_xticks([]); a2.set_yticks([])
    a2.set_xlabel("pathways", fontsize=8.6)
    axb.append(a2); verdicts.append("0 of 6,259 reach the threshold")

    a3 = fig.add_subplot(g1[2])
    cv = pd.read_csv(f"{RES}/91_ml_vs_additive.csv").set_index("model")
    nice = [("additive linear", "linear"), ("elastic net", "penalised"),
            ("gradient boosting", "boosting")]
    yv = np.arange(len(nice))[::-1]
    for yi, (kk, nm) in zip(yv, nice):
        if kk in cv.index:
            a3.barh(yi, cv.loc[kk, "r2"], height=0.52,
                    color=WARM if kk == "gradient boosting" else MID, zorder=3)
    a3.axvline(0, color=INK, lw=1.1, zorder=4)
    a3.set_yticks(yv); a3.set_yticklabels([n for _, n in nice], fontsize=7.8)
    a3.set_xlim(-0.62, 0.06); a3.set_xticks([])
    a3.set_ylim(-0.7, len(nice) - 0.20)
    a3.set_xlabel("prediction for one patient", fontsize=8.6)
    hide_y_spine(a3)
    axb.append(a3); verdicts.append("0 of 3 beat the group average")
    g2 = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.14], wspace=0.80)
    a4 = fig.add_subplot(g2[0])
    dirn = pd.read_csv(f"{RES}/182_direction_final.csv")
    call = dirn[dirn.drug_would_need_to != "unknown"].copy()
    call["ord"] = call.drug_would_need_to.map({"lower": 0, "raise": 1})
    call = call.sort_values(["ord", "gene"])
    AGENT_DIR = {"SNCA": "lower", "MAPT": "lower", "LRRK2": "lower",
                 "GPNMB": "lower", "GBA1": "raise", "IDUA": "raise"}
    x = np.arange(len(call))
    for xi, (_, r) in zip(x, call.iterrows()):
        up = r.drug_would_need_to == "raise"
        col = NAVY if up else MID
        a4.add_patch(FancyArrow(xi, 0.0, 0, 0.52 if up else -0.52,
                                width=0.085, head_width=0.28, head_length=0.18,
                                color=col, length_includes_head=True,
                                zorder=3))
        if AGENT_DIR.get(r.gene) == r.drug_would_need_to:
            a4.plot([xi + 0.32], [0.66 if up else -0.66], "o", ms=6.5,
                    color=col, mec="white", mew=1.0, zorder=5)
        a4.text(xi - 0.10, -0.90, r.gene, ha="right", va="top", fontsize=7.6,
                style="italic", rotation=38, rotation_mode="anchor",
                color=WARM if r.gene in ("GALC", "LRRC37A2") else INK)
    a4.axhline(0, color=GREY_L, lw=1.0, zorder=1)
    a4.set_xlim(-2.2, len(call) + 0.15); a4.set_ylim(-1.50, 1.20)
    a4.set_xticks([]); a4.set_yticks([])
    for s2 in a4.spines.values():
        s2.set_visible(False)
    a4.text(-2.15, 0.50, "raise", ha="left", va="center", fontsize=8.4,
            color=INK)
    a4.text(-2.15, -0.50, "lower", ha="left", va="center", fontsize=8.4,
            color=INK)
    a4.text(-2.15, 1.06, "a drug would have to", ha="left", va="center",
            fontsize=7.8, color=GREY)

    a5 = fig.add_subplot(g2[1])
    dos = pd.read_csv(f"{RES}/170_target_dossier.csv").set_index("gene")
    ga = pd.read_csv(f"{RES}/85_target_annotation.csv").dropna(
        subset=["onset_min_p"]).set_index("gene")
    tot = len(ga)
    inga = dos.index.isin(ga.index)
    steps = [("act on age at onset", int((ga.onset_min_p < 1e-3).sum())),
             ("have a confident fold",
              int(((dos.mean_plddt >= 70) & inga).sum())),
             ("have a bound ligand", int(((dos.n_pdb_ligand > 0) & inga).sum())),
             ("have an agent in the clinic",
              len({"SNCA", "MAPT", "GPNMB", "IDUA"} & set(ga.index)))]
    yb2 = np.arange(len(steps))[::-1]
    for yi, (nm, v) in zip(yb2, steps):
        a5.barh(yi, tot, height=0.48, color=GREY_XL, zorder=2)
        a5.barh(yi, v, height=0.48, color=NAVY, zorder=3)
        a5.text(tot + 0.5, yi, f"{v}", va="center", fontsize=8.4, color=INK)
    a5.set_yticks(yb2)
    a5.set_yticklabels([n for n, _ in steps], fontsize=8.0)
    a5.set_xlim(0, tot + 2.2); a5.set_xticks([])
    a5.set_ylim(-0.7, len(steps) - 0.20)
    a5.text(0, len(steps) - 0.48, f"of the {tot} implicated genes",
            fontsize=7.6, color=GREY, va="center")
    hide_y_spine(a5)

    fig.canvas.draw()
    for a, v in zip(axb, verdicts):
        p = a.get_position()
        fig.text((p.x0 + p.x1) / 2, p.y0 - 0.038, v, ha="center", va="top",
                 fontsize=8.0, color=WARM)
    band(fig, ax, "a", "Age at first motor symptom, by genetic cause")
    band(fig, axb[0], "b", "Tests for a difference in mechanism", dy=0.030)
    band(fig, a4, "c", "Therapeutic direction and tractability", dy=0.022)
    save(fig, "fig01_summary")


if __name__ == "__main__":
    main()
