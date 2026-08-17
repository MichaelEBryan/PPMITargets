#!/usr/bin/env python
"""What the classifier attends to, and how stable that is."""
import os, sys, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clinical_style import *          # noqa
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import paths as _P
warnings.filterwarnings("ignore")

RES = str(_P.RESULTS)

BLOCK_NAME = {"design": "how the patient entered the study",
              "genetics_core": "genetic sub-study record",
              "geno": "genotype", "prs": "polygenic score",
              "pcs": "genetic ancestry",
              "family": "family history", "age": "age",
              "demo": "demographic", "motor": "gait and wearable",
              "other": "other"}
BLOCK_COL = {"design": WARM, "genetics_core": "#c9a55f",
             "geno": NAVY, "prs": DEEP, "pcs": MID,
             "family": SAGE, "demo": "#8fa88f", "age": GREY,
             "motor": PALE, "other": GREY_L}
NICE = {"ENRLSRDC": "enrolled in the sporadic arm",
        "APPRDX": "the diagnosis code entered at screening",
        "PATHVAR_COUNT": "pathogenic variants recorded by the study",
        "RNASEQ_VIS": "visits with RNA sequencing",
        "ANYFAMPD1": "any relative with Parkinson's disease",
        "PATAU1": "father affected",
        "KIDSNUM1": "number of children",
        "SE_EDUCYRS1": "years of education",
        "ENRLPINK1": "enrolled in the PINK1 arm",
        "ENRLPRKN": "enrolled in the PRKN arm",
        "ENRLLRRK2": "enrolled in the LRRK2 arm",
        "ENRLGBA": "enrolled in the GBA arm",
        "ENRLSNCA": "enrolled in the SNCA arm",
        "ENRLRBD": "enrolled in the REM-sleep arm",
        "ENRLHPSM": "enrolled in the hyposmia arm",
        "ENRLNORM": "enrolled as a healthy control",
        "ENROLL_AGE": "age at enrolment",
        "SCREENEDAM": "screened in the amendment",
        "DATELIG": "eligible for the imaging sub-study",
        "GAITSTDY": "in the gait sub-study",
        "TAUSTDY": "in the tau sub-study",
        "AV133STDY": "in the AV-133 sub-study",
        "PISTDY": "in the PI-2620 sub-study",
        "SV2ASTDY": "in the SV2A sub-study",
        "NXTAUSTDY": "in the next tau sub-study"}


def label(f):
    if f in NICE:
        return NICE[f]
    if f.startswith("miss_"):
        return f"whether {f[5:]} was measured"
    if f.startswith("AGE_AT_VISIT"):
        return f"age at visit {f[12:]}"
    if f.startswith("MG_Genetic_PRS_PC"):
        return f"ancestry component {f[17:]}"
    if f.startswith("Genetic_PRS_PC"):
        return f"ancestry component {f[14:]}"
    if "_rs" in f:
        return "genotype at " + f.split("_rs")[-1].join(["rs", ""])
    if f.startswith("MG_rs"):
        return "genotype at " + f[3:].split("_")[0]
    return f


def main():
    imp = pd.read_csv(f"{RES}/160_shap_importance.csv")
    boot = pd.read_csv(f"{RES}/161_shap_rank_bootstrap.csv")
    share = pd.read_csv(f"{RES}/162_shap_block_share.csv")
    dep = pd.read_csv(f"{RES}/163_shap_dependence.csv")
    S = "LOPD_ge60"

    fig = plt.figure(figsize=(7.0, 8.4))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.50, 0.58, 1.00],
                             hspace=0.46, left=0.365, right=0.965,
                             top=0.955, bottom=0.062)

    # a. the twenty features the model leans on most
    ax = fig.add_subplot(outer[0])
    m = imp[(imp.stratum == S) & (imp.model == "XGBoost")].copy()
    m = m.sort_values("shap", ascending=False).head(20)[::-1]
    y = np.arange(len(m))
    cols = [BLOCK_COL.get(b, GREY_L) for b in m.block]
    ax.barh(y, m.shap.values, height=0.62, color=cols, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([label(f) for f in m.feature],
                                         fontsize=7.8)
    for t, b in zip(ax.get_yticklabels(), m.block):
        if b in ("design", "genetics_core"):
            t.set_color(WARM)
    ax.set_xlabel("how much the model leans on the feature\n"
                  "(mean absolute Shapley value)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    bare_y(ax)
    used = [b for b in ["design", "genetics_core", "family", "age", "demo",
                        "pcs", "geno", "prs", "motor"]
            if b in set(imp[imp.stratum == S].block)]
    h = [Rectangle((0, 0), 1, 1, fc=BLOCK_COL.get(b, GREY_L)) for b in used]
    ax.legend(h, [BLOCK_NAME.get(b, b) for b in used], loc="lower right",
              bbox_to_anchor=(1.0, 0.0), ncol=1, fontsize=7.6)

    # b. where the total attribution goes
    ax2 = fig.add_subplot(outer[1])
    sh = share[(share.model == "XGBoost")]
    order = ["design", "genetics_core", "family", "age", "demo", "motor",
             "pcs", "geno", "prs", "other"]
    strata = [("EOPD_lt50", "patients enrolled\nbefore 50"),
              ("LOPD_ge60", "patients enrolled\nat 60 or later")]
    y = np.arange(len(strata))[::-1]
    for yi, (st, nm) in zip(y, strata):
        s = sh[sh.stratum == st].set_index("block").share
        left = 0.0
        for b in order:
            v = float(s.get(b, 0.0))
            if v <= 0:
                continue
            ax2.barh(yi, v * 100, left=left * 100, height=0.52,
                     color=BLOCK_COL.get(b, GREY_L), zorder=3)
            if v > 0.09:
                ax2.text((left + v / 2) * 100, yi, f"{v*100:.0f}%",
                         ha="center", va="center", fontsize=8.0,
                         color="white" if b in ("design", "geno", "prs",
                                                "pcs")
                         else INK)
            left += v
    ax2.set_yticks(y); ax2.set_yticklabels([n for _, n in strata])
    ax2.set_ylim(-0.62, len(strata) - 0.38)
    ax2.set_xlim(0, 100); ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("share of everything the model uses (%)")
    bare_y(ax2)

    # c. how far the ranking moves when the patients change
    ax3 = fig.add_subplot(outer[2])
    b = boot[boot.stratum == S].sort_values("rank_point").head(15)[::-1]
    y = np.arange(len(b))
    for yi, (_, r) in zip(y, b.iterrows()):
        c = BLOCK_COL.get(r.block, GREY_L)
        ax3.plot([r.rank_q05, r.rank_q95], [yi, yi], color=c, lw=2.6,
                 alpha=0.35, solid_capstyle="round", zorder=2)
        ax3.plot([r.rank_median], [yi], "o", ms=5.5, color=c, zorder=4,
                 markeredgecolor="white", markeredgewidth=0.8)
    ax3.axvline(20, color=GREY_L, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax3.set_xscale("log")
    ax3.set_xticks([1, 3, 10, 30, 100])
    ax3.set_xticklabels(["1st", "3rd", "10th", "30th", "100th"])
    ax3.set_xlim(0.85, max(45, b.rank_q95.max() * 1.35))
    ax3.set_yticks(y)
    ax3.set_yticklabels([label(f) for f in b.feature], fontsize=7.6)
    for t, bl in zip(ax3.get_yticklabels(), b.block):
        if bl in ("design", "genetics_core"):
            t.set_color(WARM)
    ax3.set_xlabel("where the feature ranks when the patients are resampled")
    bare_y(ax3)
    ax3.text(20, len(b) - 0.35, "top 20", ha="center", va="bottom",
             fontsize=8.0, color=GREY)

    for a, s_ in [(ax, "a"), (ax2, "b"), (ax3, "c")]:
        PL(fig, a, s_)
    save(fig, "C4_what_the_model_learns")

    # ------------------------------------------------------------------ ED
    fig2 = plt.figure(figsize=(7.0, 7.4))
    g = fig2.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.58,
                          wspace=0.46, left=0.150, right=0.965, top=0.945,
                          bottom=0.085)

    ax = fig2.add_subplot(g[0, 0])
    for st, col, nm in [("EOPD_lt50", MID, "enrolled before 50"),
                        ("LOPD_ge60", NAVY, "enrolled at 60 or later")]:
        s = boot[boot.stratum == st]
        s = s[s.rank_point <= 60].sort_values("rank_point")
        ax.plot(s.rank_point, s.p_in_top20, "o", ms=3.6, color=col, alpha=0.8,
                markeredgecolor="none", zorder=4)
    ax.axhline(1.0, color=GREY_L, lw=1.0)
    ax.set_xlabel("where the feature ranks\nin the full sample")
    ax.set_ylabel("how often it stays in the\ntop twenty when resampled")
    ax.set_ylim(-0.03, 1.06)

    ax2 = fig2.add_subplot(g[0, 1])
    gs_ = boot[boot.stratum == S].groupby("block")
    rows = []
    for bl, s in gs_:
        if len(s) < 3:
            continue
        rows.append((bl, float(s.p_in_top20.mean()),
                     float((s.rank_q95 - s.rank_q05).median())))
    r = pd.DataFrame(rows, columns=["block", "p", "spread"]).sort_values("p")
    y = np.arange(len(r))
    ax2.barh(y, r.p, height=0.6,
             color=[BLOCK_COL.get(b, GREY_L) for b in r.block], zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels([BLOCK_NAME.get(b, b) for b in r.block], fontsize=7.8)
    ax2.set_xlabel("average chance a feature of this\nkind lands in the top twenty")
    bare_y(ax2)

    ax3 = fig2.add_subplot(g[1, :])
    d = dep[dep.stratum == S] if len(dep) else dep
    if len(d):
        rng = np.random.default_rng(3)
        cats = [(-999.0, "not genotyped", GREY_L), (0.0, "no copies", PALE),
                (1.0, "one copy", MID), (2.0, "two copies", NAVY)]
        ax3.axhline(0, color=INK, lw=1.0, zorder=3)
        for i, (v, nm, col) in enumerate(cats):
            sub = d[np.isclose(d.value, v)]
            if not len(sub):
                continue
            ax3.scatter(i + rng.normal(0, 0.075, len(sub)), sub.shap, s=8,
                        color=col, alpha=0.35, edgecolors="none", zorder=4)
            ax3.plot([i - 0.30, i + 0.30], [sub.shap.median()] * 2, color=INK,
                     lw=2.0, zorder=6)
            ax3.text(i, -0.60, f"n = {len(sub):,}", ha="center", va="top",
                     fontsize=7.8, color=GREY)
        ax3.set_xticks(range(len(cats)))
        ax3.set_xticklabels([nm for _, nm, _ in cats], fontsize=8.2)
        ax3.set_xlim(-0.6, len(cats) - 0.4); ax3.set_ylim(-0.72, 0.52)
        ax3.set_xlabel("the strongest single variant the model uses, in the "
                       "late stratum")
        ax3.set_ylabel("effect on the model's\noutput for that patient")

    for a, s_ in [(ax, "a"), (ax2, "b"), (ax3, "c")]:
        PL(fig2, a, s_, x=a.get_position().x0 - 0.125)
    save(fig2, "E14_model_stability")


if __name__ == "__main__":
    main()
