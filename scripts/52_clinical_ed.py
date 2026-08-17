#!/usr/bin/env python
"""Extended Data figures, on the same clinical template as the main set.

This is where the method work now lives: how the ranking rule behaves on data
whose architecture is known, the full ablation, the gene-set null, and the
structural comparison for GALC.
"""
import os, sys, json, warnings
from math import comb
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clinical_style import *          # noqa
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import paths as _P
warnings.filterwarnings("ignore")

RES = str(_P.RESULTS)


def load_grid():
    d = pd.read_csv(f"{RES}/81_method_grid.csv", keep_default_na=False,
                    na_values=[""])
    for c in ["n_small", "n_large", "p", "k", "m_causal"]:
        d[c] = pd.to_numeric(d[c])
    for c in ["beta", "ld", "naive_calls_distinct", "formal_calls_distinct",
              "mean_overlap", "mean_recall_small", "mean_recall_large"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def ed1():
    """What the ranking rule concludes when the truth is known."""
    d = load_grid()
    core = d[(d.p == 162) & (d.k == 10) & (d.n_small == 230)]
    scen = ["null", "dose", "partial", "disjoint"]
    lab = ["identical", "same variants,\nhalf the effect",
           "half the variants\nshared", "no variant\nshared"]
    x = np.arange(4)
    series = [(0.05, PALE, "o"), (0.10, MID, "s"), (0.40, NAVY, "^")]
    names = ["odds ratio 1.05", "odds ratio 1.11", "odds ratio 1.49"]

    fig = plt.figure(figsize=(7.0, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.05], hspace=0.62,
                          left=0.175, right=0.965, top=0.945, bottom=0.075)
    axes = []
    for row, (col, what) in enumerate([
            ("naive_calls_distinct", "the ranking rule"),
            ("formal_calls_distinct", "a formal test")]):
        ax = fig.add_subplot(gs[row])
        for (b, c, mk), nm in zip(series, names):
            v = [core[(core.scenario == s) & (core.beta == b)][col].mean()
                 for s in scen]
            ax.plot(x, v, mk + "-", color=c, ms=6.5, lw=1.8, label=nm,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=4)
        ax.axhline(0.05, color=WARM, lw=1.1, ls=(0, (4, 3)), zorder=2)
        ax.text(3.42, 0.062, "0.05", color=WARM, fontsize=8.2, va="bottom")
        ax.set_xticks(x); ax.set_xticklabels(lab)
        ax.set_xlim(-0.35, 3.62)
        ax.set_ylim(-0.03, 0.86 if row else 0.62)
        ax.set_ylabel(f"how often {what}\nconcludes the two groups differ")
        if row == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02))
        if row == 1:
            ax.set_xlabel("how the two groups were built")
        axes.append(ax)

    ax = fig.add_subplot(gs[2])
    ks = d[(d.n_small == 230) & (d.p == 162)
           & (d.scenario == "null")].groupby("k")
    kk = sorted(ks.groups)
    v = [ks.get_group(k).naive_calls_distinct.mean() for k in kk]
    ax.plot(kk, v, "o-", color=NAVY, ms=7, lw=2.0, markeredgecolor="white",
            markeredgewidth=1.0, zorder=4)
    ax.axhline(0.05, color=WARM, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(32.4, 0.062, "0.05", color=WARM, fontsize=8.2, va="bottom", ha="right")
    ax.set_xlabel("number of features compared between the two lists")
    ax.set_ylabel("how often the ranking rule\nconcludes they differ, when\n"
                  "they were built identically")
    ax.set_xticks(kk); ax.set_xlim(2.5, 33.5); ax.set_ylim(-0.05, 0.88)
    axes.append(ax)
    for a, s in zip(axes, "abc"):
        PL(fig, a, s)
    save(fig, "E1_ranking_rule_behaviour")


def ed2():
    """Where the ranking rule fails, across sample size and effect size."""
    d = load_grid()
    n = d[(d.scenario == "null") & (d.p == 162) & (d.k == 10)]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.4),
                             gridspec_kw=dict(hspace=0.42, left=0.215,
                                              right=0.885, top=0.930,
                                              bottom=0.080))
    for ax, col, what in [(axes[0], "naive_calls_distinct", "the ranking rule"),
                          (axes[1], "formal_calls_distinct", "a formal test")]:
        piv = n.groupby(["n_small", "beta"])[col].mean().unstack()
        im = ax.imshow(piv.values, cmap=SEQ, vmin=0, vmax=0.5, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"{np.exp(b):.2f}" for b in piv.columns])
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([f"{int(i):,}" for i in piv.index])
        ax.set_xlabel("effect of each variant (odds ratio)")
        ax.set_ylabel("patients in the\nsmaller group")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8.2, color="white" if v > 0.28 else INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cb = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
        cb.set_label(f"how often {what}\nwrongly concludes they differ",
                     fontsize=8.2)
        cb.ax.tick_params(labelsize=7.8)
        cb.outline.set_visible(False)
    for a, s in zip(axes, "ab"):
        PL(fig, a, s)
    save(fig, "E2_where_the_rule_fails")


def ed3():
    """How many features two lists share when nothing differs between them."""
    n2 = pd.read_csv(f"{RES}/21_null_splithalf.csv")
    n2 = n2[n2.overlap.notna()]
    obs = pd.read_csv(f"{RES}/04_observed_overlap.csv")
    donor = n2[n2.donor == "LOPD_ge60"]

    fig = plt.figure(figsize=(7.0, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.46,
                          left=0.185, right=0.965, top=0.945, bottom=0.085)
    ax = fig.add_subplot(gs[0])
    mx = int(max(6, donor.overlap.max() + 1))
    cnt = donor.overlap.value_counts(normalize=True).reindex(range(mx),
                                                            fill_value=0)
    ax.bar(cnt.index, cnt.values, width=0.68, color=FILL, edgecolor=MID,
           lw=1.0, zorder=3)
    p_snp, KA, KB = 162, 10, 13
    ks = list(range(0, min(KA, KB) + 1))
    pmf = [comb(KA, j) * comb(p_snp - KA, KB - j) / comb(p_snp, KB) for j in ks]
    ax.plot(ks, pmf, "o--", color=GREY, ms=5.5, lw=1.3, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8)
    for v in sorted(set(obs.overlap)):
        ax.axvline(v, color=WARM, lw=1.6, ls=(0, (5, 3)), zorder=5)
    obsv = sorted(set(obs.overlap))
    lead(ax, "the two lists in the real data\nshared " +
         (", ".join(str(v) for v in obsv[:-1]) + f" and {obsv[-1]}"
          if len(obsv) > 1 else str(obsv[0])) + " variants",
         xy=(obsv[-1], 0.245), xytext=(3.1, 0.375), color=WARM)
    ax.set_xlabel("variants shared between the two lists")
    ax.set_ylabel("proportion of 3,600 repeats\nof a single group split in two")
    ax.set_xticks(range(mx)); ax.set_ylim(0, 0.44); ax.set_xlim(-0.7, mx - 0.3)

    ax2 = fig.add_subplot(gs[1])
    grp = donor.groupby("model").overlap.agg(
        p0=lambda x: (x == 0).mean()).reindex(
        ["XGBoost", "LightGBM", "CatBoost"])
    x = np.arange(len(grp))
    ax2.bar(x, grp.p0, width=0.52, color=[NAVY, MID, PALE], zorder=3)
    ax2.axhline(0.05, color=WARM, lw=1.2, ls=(0, (5, 3)), zorder=4)
    ax2.text(len(grp) - 0.40, 0.062, "0.05", color=WARM, fontsize=8.2,
             ha="right")
    ax2.set_xticks(x); ax2.set_xticklabels(grp.index)
    ax2.set_ylabel("how often the two lists share\nnothing at all, when they\n"
                   "came from the same patients")
    ax2.set_ylim(0, 0.40)
    for a, s in zip([ax, ax2], "ab"):
        PL(fig, a, s)
    save(fig, "E3_shared_features_null")


def ed4():
    """Every ablation of the released feature table."""
    d = pd.read_csv(f"{RES}/22_ablation_summary.csv")
    order = ["full", "minus_design", "minus_genetics_core", "minus_motor",
             "no_leakage", "only_design", "only_genetics_core", "only_motor",
             "only_family", "only_geno", "only_prs", "only_pcs",
             "genetics_only", "genetics_demo_age", "missingness_only"]
    label = {"full": "everything, as published",
             "minus_design": "without the enrolment design",
             "minus_genetics_core": "without the carrier calls",
             "minus_motor": "without gait and wearable measures",
             "no_leakage": "without design, carrier calls or motor",
             "only_design": "the enrolment design alone",
             "only_genetics_core": "the carrier calls alone",
             "only_motor": "gait and wearable measures alone",
             "only_family": "family history alone",
             "only_geno": "genotypes alone",
             "only_prs": "polygenic scores alone",
             "only_pcs": "ancestry components alone",
             "genetics_only": "genotypes, scores and ancestry",
             "genetics_demo_age": "genetics, demographics and age",
             "missingness_only": "which measurements exist, no values"}
    GROUP = {"full": "all", "minus_design": "drop", "minus_genetics_core": "drop",
             "minus_motor": "drop", "no_leakage": "drop", "only_design": "design",
             "only_genetics_core": "design", "only_motor": "pheno",
             "only_family": "pheno", "only_geno": "gen", "only_prs": "gen",
             "only_pcs": "gen", "genetics_only": "gen", "genetics_demo_age": "gen",
             "missingness_only": "miss"}
    COL = {"all": INK, "drop": GREY, "design": WARM, "pheno": SAGE,
           "gen": NAVY, "miss": MID}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 7.4), sharey=True,
                             gridspec_kw=dict(wspace=0.06, left=0.395,
                                              right=0.978, top=0.915,
                                              bottom=0.160))
    for ax, strat, sname in zip(axes, ["EOPD_lt50", "LOPD_ge60"],
                                ["patients enrolled before 50\nn = 230",
                                 "patients enrolled at 60 or later\nn = 3,386"]):
        s = d[d.stratum == strat]
        base = s[s.model == "MajorityBaseline"].f1w.mean()
        avail = [a for a in order if len(s[s.arm == a])]
        ax.axvline(base, color=GREY_L, lw=1.4, zorder=1)
        for i, arm in enumerate(reversed(avail)):
            mods = s[(s.arm == arm) & (s.model != "MajorityBaseline")]
            vals = [mods[mods.model == m].f1w.iloc[0] for m in
                    ["XGBoost", "LightGBM", "CatBoost"]
                    if len(mods[mods.model == m])]
            if not vals:
                continue
            c = COL[GROUP[arm]]
            ax.plot([min(vals), max(vals)], [i, i], color=c, lw=2.6, alpha=0.30,
                    solid_capstyle="round", zorder=2)
            ax.scatter(vals, [i] * len(vals), s=32, color=c, zorder=4,
                       edgecolors="white", linewidths=0.8)
        ax.set_ylim(-0.8, len(avail) - 0.2)
        ax.set_xlim(0.36, 1.05); ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.text(base, len(avail) - 0.30, "chance", fontsize=8.0, color=GREY,
                ha="center", va="bottom")
        ax.text(0.5, 1.055, sname, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.6, color=INK)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([label[a] for a in reversed(order)], fontsize=8.2)
    h = [plt.Line2D([], [], marker="o", ls="", ms=6, color=COL[k],
                    markeredgecolor="white")
         for k in ["design", "pheno", "gen", "miss"]]
    axes[0].legend(h, ["study design", "clinical measure", "genetic",
                       "measurement pattern"],
                   loc="upper center", bbox_to_anchor=(1.03, -0.070), ncol=4,
                   columnspacing=1.4)
    fig.text(0.69, 0.015, "how well the model separates the four clinical groups",
             ha="center", fontsize=9, color=INK)
    PL(fig, axes[0], "a", dy=0.040)
    PL(fig, axes[1], "b", x=axes[1].get_position().x0 - 0.032, dy=0.040)
    save(fig, "E4_full_ablation")


def ed5():
    """Gene sets, and whether any model beats simple addition."""
    fig = plt.figure(figsize=(7.0, 8.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.62,
                          left=0.300, right=0.965, top=0.945, bottom=0.075)

    ax = fig.add_subplot(gs[0])
    s = pd.read_csv(f"{RES}/110_geneset_maxT.csv")
    mn = np.load(f"{RES}/110_maxnull_adj.npy")
    obs = s.t_onset_adj.max()
    ax.hist(mn, bins=80, color=FILL, edgecolor=MID, lw=0.5, density=True,
            zorder=3)
    p95 = np.percentile(mn, 95)
    ax.axvline(p95, color=GREY, lw=1.4, zorder=4)
    ax.axvline(obs, color=WARM, lw=1.8, zorder=5)
    top = ax.get_ylim()[1]
    lead(ax, "the strongest of the 6,259 gene sets\nin the real data",
         xy=(obs, top * 0.50), xytext=(3.30, top * 0.86), color=WARM)
    lead(ax, "the level a set would have had to reach\nto count as a finding",
         xy=(p95, top * 0.22), xytext=(6.6, top * 0.62), color=GREY)
    ax.set_xlim(2.6, 9.4)
    ax.set_xlabel("strongest gene-set signal anywhere in the genome")
    ax.set_ylabel("20,000 rotations\nof the genome")
    ax.set_yticks([]); bare_y(ax)

    ax2 = fig.add_subplot(gs[1])
    cv = pd.read_csv(f"{RES}/91_ml_vs_additive.csv")
    order = ["additive linear", "elastic net", "gradient boosting", "mean only"]
    nice = {"additive linear": "plain linear regression",
            "elastic net": "penalised regression",
            "gradient boosting": "gradient boosting",
            "mean only": "guessing the average age"}
    cv = cv.set_index("model").reindex([o for o in order if o in set(cv.model)])
    y = np.arange(len(cv))[::-1]
    cols = [WARM if i == "gradient boosting" else MID for i in cv.index]
    ax2.barh(y, cv.r2.values, xerr=cv.r2_sd.values, color=cols, height=0.55,
             error_kw=dict(elinewidth=1.1, capsize=0, ecolor=GREY), zorder=3)
    ax2.axvline(0, color=INK, lw=1.1, zorder=4)
    ax2.set_yticks(y); ax2.set_yticklabels([nice[i] for i in cv.index])
    ax2.set_xlabel("variation in age at onset explained in patients the model\n"
                   "has not seen (0 means no better than the group average)")
    ax2.set_xlim(-0.85, 0.06)
    ax2.set_xticks([-0.6, -0.4, -0.2, 0])
    bare_y(ax2)

    ax3 = fig.add_subplot(gs[2])
    lc = pd.read_csv(f"{RES}/140_learning_curve.csv")
    curves = [("additive PRS", MID, "adding up known risk variants"),
              ("gradient boosting", WARM, "gradient boosting")]
    for mdl, col, nm in curves:
        sub = lc[lc.model == mdl].groupby("n").r2.mean().sort_index()
        if not len(sub):
            continue
        ax3.plot(sub.index, sub.values, "o-", color=col, ms=5.5, lw=1.8,
                 markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    sub = lc[lc.model == "additive PRS"].groupby("n").r2.mean().sort_index()
    lead(ax3, "adding up known\nrisk variants", xy=(342, float(sub.loc[342])),
         xytext=(285, 0.055), color=MID, ha="right")
    sub = lc[lc.model == "gradient boosting"].groupby("n").r2.mean().sort_index()
    lead(ax3, "gradient boosting", xy=(282, float(sub.loc[282])),
         xytext=(232, -0.185), color=WARM, ha="right")
    ax3.axhline(0, color=INK, lw=1.0, zorder=2)
    ax3.set_xlim(70, 430); ax3.set_ylim(-0.21, 0.09)
    ax3.set_xticks([100, 200, 300, 400])
    ax3.set_xlabel("patients used to train")
    ax3.set_ylabel("variation in age at\nonset explained")
    for a, s_ in zip([ax, ax2, ax3], "abc"):
        PL(fig, a, s_)
    save(fig, "E5_geneset_and_models")


def ed6():
    """Two structural methods asked the same question about GALC."""
    mc = pd.read_csv(f"{RES}/132_galc_method_comparison.csv").dropna(
        subset=["AUC"]).sort_values("AUC")
    short = {"distance to catalytic pair (AlphaFold geometry)":
             "distance to the catalytic pair,\non the AlphaFold model",
             "ESMFold Ca RMSD, mutant vs wild type":
             "backbone shift in the predicted\nmutant structure",
             "ESMFold change in mean pLDDT":
             "change in overall predicted confidence",
             "ESMFold change in pLDDT at the residue":
             "change in confidence at the residue",
             "AlphaFold pLDDT at the residue":
             "predicted confidence at the residue"}
    fig, ax = plt.subplots(figsize=(7.0, 4.4),
                           gridspec_kw=dict(left=0.395, right=0.965, top=0.925,
                                            bottom=0.215))
    y = np.arange(len(mc))
    cols = [NAVY if a >= 0.8 else MID if a >= 0.7 else PALE for a in mc.AUC]
    ax.barh(y, mc.AUC, color=cols, height=0.55, zorder=3)
    ax.axvline(0.5, color=INK, lw=1.2, zorder=4)
    ax.text(0.5, -0.95, "no better than chance", fontsize=8.2, color=GREY,
            ha="center", va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([short.get(m, m) for m in mc.method], fontsize=8.2)
    ax.set_xlim(0.2, 1.0); ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(-0.7, len(mc) - 0.3)
    ax.set_xlabel("how well the method separates variants known to cause\n"
                  "Krabbe disease from common harmless ones")
    bare_y(ax)
    save(fig, "E6_galc_structure_methods")


def ed7():
    """The 17q21.31 haplotype the second nominated gene sits on."""
    prof = pd.read_csv(f"{RES}/33_LRRC37A2_variant_LD_profile.csv")
    GENES = {"PLEKHM1": (45.460, 45.524), "LRRC37A2": (45.619, 45.674),
             "CRHR1": (45.784, 45.895), "MAPT": (45.894, 46.028),
             "KANSL1": (46.030, 46.225), "ARL17B": (46.197, 46.275),
             "ARL17A": (46.274, 46.357), "LRRC37A": (46.360, 46.420),
             "NSF": (46.596, 46.762)}
    eqtl = {}
    if os.path.exists(f"{RES}/36_gtex_haplotype_genes.csv"):
        eqtl = pd.read_csv(f"{RES}/36_gtex_haplotype_genes.csv",
                           index_col=0).tissues.to_dict()

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 3, 1.45],
                                              hspace=0.22, left=0.175,
                                              right=0.965, top=0.945,
                                              bottom=0.085))
    mb = prof.pos38 / 1e6
    for ax, col, lab, thr in [
            (axes[0], "r2", "how closely each marker\ntracks the variant", 0.5),
            (axes[1], "Dprime", "whether each marker sits on\nthe same haplotype",
             0.9)]:
        ax.scatter(mb, prof[col], s=7, color=PALE, edgecolors="none", zorder=3)
        ax.axhline(thr, color=GREY_L, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.axvline(45.666837, color=NAVY, lw=1.2, zorder=4)
        ax.axvline(45.917282, color=WARM, lw=1.2, zorder=4)
        ax.set_ylabel(lab); ax.set_ylim(-0.05, 1.12)
        ax.set_yticks([0, 0.5, 1.0])
    lead(axes[0], "rs62053943, in LRRC37A2", xy=(45.667, 1.02),
         xytext=(44.95, 1.09), color=NAVY)
    lead(axes[0], "rs17649553, in MAPT", xy=(45.917, 0.55),
         xytext=(46.30, 0.92), color=WARM)

    ax = axes[2]
    # two rows of gene boxes, then greedy packing of labels into three tiers
    boxrow, occupied = {}, [[], []]
    for g, (s0, e0) in sorted(GENES.items(), key=lambda kv: kv[1][0]):
        r = 0 if not occupied[0] or s0 > occupied[0][-1] + 0.075 else 1
        occupied[r].append(e0)
        boxrow[g] = r
        col = NAVY if g == "LRRC37A2" else WARM if g == "MAPT" else MID
        ax.add_patch(Rectangle((s0, 0.60 if r == 0 else 0.06),
                               max(e0 - s0, 0.014), 0.17, color=col, zorder=3))
    # labels alternate between two heights within each row so none collide
    for row in (0, 1):
        genes = [g for g, _ in sorted(GENES.items(), key=lambda kv: kv[1][0])
                 if boxrow[g] == row]
        for i, g in enumerate(genes):
            s0, e0 = GENES[g]
            nt = eqtl.get(g, 0)
            txt = f"{g}  {int(nt)}/13" if nt else g
            yy = (0.80 if row == 0 else 0.28) + (i % 2) * 0.175
            ax.text((s0 + e0) / 2, yy, txt, fontsize=7.8, ha="center",
                    va="bottom", color=INK, style="italic")
    ax.set_ylim(0, 1.16); ax.set_yticks([])
    ax.set_xlabel("position on chromosome 17 (millions of bases)")
    ax.spines["left"].set_visible(False)
    for a, s in zip(axes, "abc"):
        PL(fig, a, s)
    save(fig, "E7_17q21_haplotype")


def ed8():
    """How the study arms were assigned, and how well that alone predicts."""
    fig = plt.figure(figsize=(7.0, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.10, 1.0], hspace=0.58,
                          left=0.300, right=0.965, top=0.945, bottom=0.085)

    ax = fig.add_subplot(gs[0])
    r = pd.read_csv(f"{RES}/10_ascertainment_route_by_stratum.csv", index_col=0)
    r = r.drop(index=[i for i in r.index if str(i) == "All"],
               columns=[c for c in r.columns if str(c) == "All"])
    r = r.T
    r = r.div(r.sum(axis=1), axis=0) * 100
    nicerow = {"EO(<50)": "onset before 50", "LO(>=60)": "onset at 60 or later",
               "mid(50-59)": "onset between 50 and 59"}
    order = list(r.columns)
    shades = [NAVY, "#5583ab", MID, "#a9c2d6", PALE, GREY_L][:len(order)]
    left = np.zeros(len(r))
    y = np.arange(len(r))[::-1]
    for c, col in zip(order, shades):
        ax.barh(y, r[c].values, left=left, height=0.52, color=col, zorder=3,
                label=str(c))
        left = left + r[c].values
    ax.set_yticks(y)
    ax.set_yticklabels([nicerow.get(str(i), str(i)) for i in r.index])
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of patients recruited by each route (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.46, -0.30), ncol=3)
    bare_y(ax)

    ax2 = fig.add_subplot(gs[1])
    t = pd.read_csv(f"{RES}/00_enrl_pattern_table.csv", index_col=0)
    M = t.values.astype(float)
    tot = M.sum()
    rule = M.max(axis=1).sum() / tot
    base = M.sum(axis=0).max() / tot
    loo = 0.0
    for row in M:
        for c in range(len(row)):
            if row[c] == 0:
                continue
            held = row.copy(); held[c] -= 1
            if int(np.argmax(held)) == c:
                loo += row[c]
    loo /= tot
    vals = [rule, loo, base]
    nm = ["a fixed rule read off\nthe enrolment flags",
          "the same rule, scored on\npatients held out one at a time",
          "always guessing the\ncommonest group"]
    y = np.arange(3)[::-1]
    ax2.barh(y, [v * 100 for v in vals], height=0.50,
             color=[NAVY, MID, GREY_L], zorder=3)
    for yi, v in zip(y, vals):
        ax2.text(v * 100 + 1.4, yi, f"{v*100:.1f}%", va="center", fontsize=8.6,
                 color=INK)
    ax2.set_yticks(y); ax2.set_yticklabels(nm)
    ax2.set_xlim(0, 105); ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("patients placed in the right clinical group (%)")
    bare_y(ax2)
    for a, s in zip([ax, ax2], "ab"):
        PL(fig, a, s)
    save(fig, "E8_how_the_arms_were_assigned")


if __name__ == "__main__":
    print("extended data figures")
    for f in (ed1, ed2, ed3, ed4, ed5, ed6, ed7, ed8):
        try:
            f()
        except Exception:
            import traceback
            print(f"  [fail] {f.__name__}"); traceback.print_exc(limit=3)
