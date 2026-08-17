#!/usr/bin/env python
"""Extended Data figures 9 to 13, on the clinical template.

The simulation detail, the robustness diagnostics, the claim-by-claim audit,
the structural predictions for GALC, and the same analyses run on the age
proxy the published study used in place of onset age.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clinical_style import *          # noqa
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import paths as _P
warnings.filterwarnings("ignore")

RES = str(_P.RESULTS)


def ed9():
    """What the simulation recovers, and how often the two lists share nothing."""
    d = pd.read_csv(f"{RES}/09_simulation_grid.csv")
    fig = plt.figure(figsize=(7.0, 8.0))
    gs = fig.add_gridspec(3, 1, hspace=0.58, left=0.205, right=0.885,
                          top=0.945, bottom=0.070)

    ax = fig.add_subplot(gs[0])
    piv = d[d.p == 162].groupby(["n_small", "beta"]).p_zero_overlap.mean().unstack()
    im = ax.imshow(piv.values, cmap=SEQ, vmin=0, vmax=0.45, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{np.exp(b):.2f}" for b in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{int(i):,}" for i in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.2,
                    color="white" if v > 0.26 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("how often the two lists\nshare nothing at all", fontsize=8.2)
    cb.ax.tick_params(labelsize=7.8); cb.outline.set_visible(False)
    ax.set_xlabel("effect of each variant (odds ratio)")
    ax.set_ylabel("patients in the\nsmaller group")

    ax2 = fig.add_subplot(gs[1])
    shades = [PALE, "#a9c2d6", MID, NAVY]
    for n, c in zip(sorted(d.n_small.unique()), shades):
        s = d[(d.n_small == n) & (d.p == 162)].groupby(
            "beta").mean_recall_small.mean()
        ax2.plot(np.exp(s.index), s.values, "o-", ms=5.5, lw=1.8, color=c,
                 markeredgecolor="white", markeredgewidth=0.8,
                 label=f"{n:,} patients", zorder=4)
    s = d[d.p == 162].groupby("beta").mean_recall_large.mean()
    ax2.plot(np.exp(s.index), s.values, "s--", ms=5.5, lw=1.8, color=WARM,
             markeredgecolor="white", markeredgewidth=0.8,
             label="3,386 patients", zorder=5)
    ax2.axvspan(1.045, 1.16, color=GREY_XL, zorder=1)
    ax2.text(1.10, 1.05,
             "the range in which real Parkinson's\nrisk variants actually sit",
             fontsize=8.0, ha="center", va="bottom", color=GREY,
             linespacing=1.3)
    ax2.set_xlabel("effect of each variant (odds ratio)")
    ax2.set_ylabel("share of the variants that\nreally matter that the model finds")
    ax2.set_ylim(0, 1.30)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99))

    ax3 = fig.add_subplot(gs[2])
    s = d[(d.n_small == 230) & (d.n_large == 3386) & (d.p == 162)]
    for m, c, mk, nm in [(5, NAVY, "o", "5 variants really matter"),
                         (15, MID, "s", "15 variants really matter")]:
        ss = s[s.m_causal == m].groupby("beta").mean_overlap.mean()
        ax3.plot(np.exp(ss.index), ss.values, mk + "-", ms=5.5, lw=1.8, color=c,
                 markeredgecolor="white", markeredgewidth=0.8, label=nm, zorder=4)
    ax3.axhline(0, color=GREY_L, lw=1.0)
    ax3.set_xlabel("effect of each variant (odds ratio)")
    ax3.set_ylabel("variants shared between\nthe two lists")
    ax3.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99))

    for a, s_ in zip([ax, ax2, ax3], "abc"):
        PL(fig, a, s_)
    save(fig, "E9_simulation_detail")


def ed10():
    """Six checks that could have broken the results in this work."""
    fig = plt.figure(figsize=(7.0, 8.8))
    gs = fig.add_gridspec(3, 2, hspace=0.66, wspace=0.52, left=0.135,
                          right=0.965, top=0.940, bottom=0.062)

    ax = fig.add_subplot(gs[0, 0])
    m = 6259
    ns = np.logspace(2, 6, 200)
    ax.loglog(ns, 1 / (ns + 1), color=NAVY, lw=2.0, zorder=3)
    ax.axhline(0.05 / m, color=WARM, lw=1.3, ls=(0, (5, 3)), zorder=4)
    ax.plot([2000], [1 / 2001], "o", ms=8, color=WARM, zorder=6,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.plot([20000], [1 / 20001], "o", ms=8, color=NAVY, zorder=6,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.set_xlabel("shuffles used")
    ax.set_ylabel("smallest result the\ntest could return")
    ax.set_ylim(1e-7, 3e-2)

    ax2 = fig.add_subplot(gs[0, 1])
    g = pd.read_csv(f"{RES}/70_genestats_aao.csv")
    pv = np.clip(g.calib_p.values, 1e-9, 1 - 1e-9)
    o = -np.log10(np.sort(pv))
    e = -np.log10((np.arange(1, len(o) + 1) - 0.5) / len(o))
    sub = np.unique(np.linspace(0, len(o) - 1, 3000).astype(int))
    mx = max(e.max(), o.max()) * 1.03
    ax2.plot([0, mx], [0, mx], color=GREY_L, lw=1.2, zorder=2)
    ax2.plot(e[sub], o[sub], "o", ms=2.6, color=MID, alpha=0.6, zorder=3)
    ax2.set_xlabel("expected by chance")
    ax2.set_ylabel("observed, gene by gene")
    ax2.set_xlim(0, mx); ax2.set_ylim(0, mx)

    ax3 = fig.add_subplot(gs[1, 0])
    d = pd.read_csv(f"{RES}/81_method_grid.csv", keep_default_na=False,
                    na_values=[""])
    for cc in ["ld", "beta", "formal_calls_distinct"]:
        d[cc] = pd.to_numeric(d[cc], errors="coerce")
    n = d[d.scenario == "null"]
    b = sorted(n.beta.unique())
    for ld, col, mk, lb in [(0.0, NAVY, "o", "variants independent"),
                            (0.6, MID, "s", "variants correlated")]:
        v = [n[(n.ld == ld) & (n.beta == x)].formal_calls_distinct.mean()
             for x in b]
        ax3.plot([np.exp(x) for x in b], v, mk + "-", color=col, ms=5.5, lw=1.8,
                 markeredgecolor="white", markeredgewidth=0.8, label=lb, zorder=4)
    ax3.axhline(0.05, color=WARM, lw=1.3, ls=(0, (5, 3)), zorder=3)
    ax3.set_xlabel("effect of each variant\n(odds ratio)")
    ax3.set_ylabel("how often the formal test\nfires when it should not")
    ax3.set_ylim(0, 0.13)
    ax3.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99))

    ax4 = fig.add_subplot(gs[1, 1])
    nl = json.load(open(f"{RES}/94b_interaction_nulls.json"))
    u, f_ = nl["unfiltered"], nl["filtered"]
    bins = np.linspace(2.5, 11, 42)
    ax4.hist(u["null"], bins=bins, histtype="step", edgecolor=WARM, lw=0.9,
             density=True, zorder=3)
    ax4.hist(f_["null"], bins=bins, color=FILL, edgecolor=NAVY, lw=0.9,
             density=True, zorder=4)
    ax4.axvline(u["obs"], color=WARM, lw=1.5, ls=(0, (5, 3)), zorder=5)
    ax4.axvline(f_["obs"], color=NAVY, lw=1.5, ls=(0, (5, 3)), zorder=5)
    lead(ax4, f"all {u['n_pairs']:,} pairs", xy=(8.4, 0.10), xytext=(9.9, 0.42),
         color=WARM, ha="right")
    lead(ax4, f"{f_['n_pairs']:,} pairs with\nenough carriers",
         xy=(4.6, 0.42), xytext=(6.4, 0.72), color=NAVY)
    ax4.set_xlabel("strongest interaction found\nafter shuffling")
    ax4.set_ylabel("shuffles")
    ax4.set_xlim(2.5, 11); ax4.set_yticks([]); bare_y(ax4)

    ax5 = fig.add_subplot(gs[2, 0])
    s = pd.read_csv(f"{RES}/51_prs_vs_AAO.csv").sort_values("beta_years_per_SD")
    x = np.arange(len(s))
    ax5.errorbar(x, -s.beta_years_per_SD,
                 yerr=[s.ci_hi - s.beta_years_per_SD,
                       s.beta_years_per_SD - s.ci_lo],
                 fmt="none", elinewidth=0.9, color=GREY_L, zorder=2)
    ax5.scatter(x, -s.beta_years_per_SD, s=30, zorder=4, edgecolors="white",
                linewidths=0.7,
                color=[NAVY if v < 0.05 else PALE for v in s.p])
    ax5.axhline(0, color=INK, lw=1.1)
    ax5.axhline(-s.beta_years_per_SD.median(), color=WARM, lw=1.5, zorder=3)
    ax5.set_xlabel("each way of estimating\nthe effect, in order")
    ax5.set_ylabel("years of onset brought\nforward per standard deviation")

    ax6 = fig.add_subplot(gs[2, 1])
    lo = pd.read_csv(f"{RES}/95_lyso_leave_one_out.csv").sort_values("p")
    y = np.arange(len(lo))[::-1]
    ax6.barh(y, lo.p.values, color=[NAVY if v < 0.05 else PALE for v in lo.p],
             height=0.62, zorder=3)
    ax6.axvline(0.05, color=WARM, lw=1.3, ls=(0, (5, 3)), zorder=4)
    lab = [g.split(";")[0] if isinstance(g, str) and g else r
           for g, r in zip(lo.gene, lo.dropped)]
    ax6.set_yticks(y); ax6.set_yticklabels(lab, fontsize=7.6, style="italic")
    ax6.set_xlabel("result once that one gene\nis taken out")
    ax6.set_xlim(0, 0.155); ax6.set_xticks([0, 0.05, 0.10, 0.15])
    bare_y(ax6)

    for a, s_ in zip([ax, ax2, ax3, ax4, ax5, ax6], "abcdef"):
        PL(fig, a, s_, x=a.get_position().x0 - 0.115)
    save(fig, "E10_robustness_checks")


def ed11():
    """Every claim in this work against every check that could break it."""
    CLAIMS = [
        ("Ranking rule cannot distinguish shared from disjoint architecture",
         ["pass", "pass", "pass", "na", "pass", "pass"]),
        ("Its error rate is set by the choice of list length",
         ["pass", "na", "pass", "na", "na", "pass"]),
        ("Formal test dominates the ranking rule",
         ["pass", "part", "pass", "na", "na", "part"]),
        ("PPMI enrolment flags predict cohort at 93%",
         ["pass", "na", "pass", "na", "na", "pass"]),
        ("Reported accuracy is carried by study design",
         ["pass", "na", "pass", "na", "na", "pass"]),
        ("Nonlinear modelling adds nothing over an additive score",
         ["pass", "na", "pass", "pass", "na", "part"]),
        ("Polygenic burden brings onset forward (direction)",
         ["pass", "na", "pass", "pass", "pass", "pass"]),
        ("...and by 1.97 years per SD (magnitude)",
         ["fail", "na", "fail", "part", "fail", "na"]),
        ("Risk alleles bring onset forward across 86 loci",
         ["pass", "pass", "pass", "pass", "pass", "pass"]),
        ("Strata differ in per-variant architecture",
         ["fail", "na", "part", "fail", "na", "pass"]),
        ("Lysosomal loci carry excess onset signal",
         ["fail", "na", "fail", "fail", "fail", "pass"]),
        ("No gene set carries onset signal beyond risk",
         ["pass", "pass", "pass", "pass", "na", "pass"]),
        ("GALC activity hypothesis holds on an independent test",
         ["pass", "pass", "na", "na", "pass", "pass"]),
        ("GALC and LRRC37A2 are not onset-specific targets",
         ["pass", "pass", "pass", "na", "pass", "na"]),
    ]
    CHECKS = ["reproduces\nelsewhere", "robust to\ncorrelation",
              "robust to\ndropping one", "survives\nmultiplicity",
              "external\nevidence", "test is\ncalibrated"]
    COL = {"pass": NAVY, "part": MID, "fail": WARM, "na": GREY_XL}

    fig, ax = plt.subplots(figsize=(7.0, 6.4),
                           gridspec_kw=dict(left=0.400, right=0.985, top=0.800,
                                            bottom=0.085))
    nR, nC = len(CLAIMS), len(CHECKS)
    for i, (claim, res) in enumerate(CLAIMS):
        yy = nR - 1 - i
        for j, v in enumerate(res):
            ax.add_patch(Rectangle((j + 0.08, yy + 0.10), 0.84, 0.80,
                                   facecolor=COL[v], edgecolor="white", lw=1.0))
    ax.set_xlim(0, nC); ax.set_ylim(0, nR)
    ax.set_xticks(np.arange(nC) + 0.5)
    ax.set_xticklabels(CHECKS, fontsize=7.2)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(nR) + 0.5)
    ax.set_yticklabels([c[0] for c in CLAIMS][::-1], fontsize=8.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    h = [Rectangle((0, 0), 1, 1, facecolor=COL[k])
         for k in ("pass", "part", "fail", "na")]
    ax.legend(h, ["holds", "holds in part", "fails", "not applicable"],
              loc="upper center", bbox_to_anchor=(0.5, -0.020), ncol=4)
    save(fig, "E11_claim_audit")


def ed12():
    """What a structure predictor sees when GALC is mutated."""
    d = pd.read_csv(f"{RES}/131_galc_variants_geometry.csv")
    d = d.copy()
    d["delta_mean_plddt"] = d["delta_mean_plddt"] * 100
    d["delta_local_plddt"] = d["delta_local_plddt"] * 100
    order = ["ClinVar pathogenic", "gnomAD common (tolerated)",
             "random substitution"]
    short = ["known to cause\nKrabbe disease", "common and\nharmless",
             "picked at\nrandom"]
    cols = {order[0]: NAVY, order[1]: MID, order[2]: PALE}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.8),
                             gridspec_kw=dict(hspace=0.56, wspace=0.46,
                                              left=0.155, right=0.965,
                                              top=0.940, bottom=0.095))
    specs = [("rmsd_ca", "how far the backbone moves\nwhen the residue changes (Å)"),
             ("delta_mean_plddt",
              "change in the predictor's\noverall confidence (points)"),
             ("delta_local_plddt",
              "change in confidence at\nthe changed residue (points)"),
             ("dist_active", "distance from the residue to\nthe catalytic pair (Å)")]
    for ax, (met, lab) in zip(axes.ravel(), specs):
        for i, g in enumerate(order):
            v = d[d.group == g][met].dropna()
            if not len(v):
                continue
            xj = np.random.default_rng(i).normal(i, 0.075, len(v))
            ax.scatter(xj, v, s=26, color=cols[g], alpha=0.85,
                       edgecolors="white", linewidths=0.6, zorder=4)
            ax.hlines(v.median(), i - 0.26, i + 0.26, color=INK, lw=1.8, zorder=5)
        ax.set_xticks(range(3)); ax.set_xticklabels(short, fontsize=7.8)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylabel(lab)
        if met != "dist_active":
            ax.axhline(0, color=GREY_L, lw=1.0, zorder=2)
    for a, s_ in zip(axes.ravel(), "abcd"):
        PL(fig, a, s_, x=a.get_position().x0 - 0.125)
    save(fig, "E12_galc_structure_detail")


def ed13():
    """The same analyses run on age at enrolment instead of age at onset."""
    fig = plt.figure(figsize=(7.0, 7.0))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.50, left=0.150,
                          right=0.965, top=0.940, bottom=0.085)

    ax = fig.add_subplot(gs[0, 0])
    m = pd.read_csv(f"{RES}/13_ppmi_vs_ipdgc_aao.csv")
    s = json.load(open(f"{RES}/15_corrected_analysis_summary.json"))[
        "replication_AAO"]
    ax.axhline(0, color=GREY_L, lw=1.0); ax.axvline(0, color=GREY_L, lw=1.0)
    ax.errorbar(m.aao_beta_aligned, m.beta_years, yerr=m.se, fmt="o", ms=4,
                color=MID, elinewidth=0.6, alpha=0.85, markeredgecolor="white",
                markeredgewidth=0.5, zorder=4)
    lim = float(np.nanmax(np.abs(m.aao_beta_aligned))) * 1.15
    xs = np.linspace(-lim, lim, 10)
    ax.plot(xs, s["wls_slope"] * xs, color=WARM, lw=2.0, zorder=5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 2.0, lim * 2.0)
    ax.set_xlabel("years of onset per allele,\npatients elsewhere")
    ax.set_ylabel("years of enrolment age\nper allele, PPMI")

    ax2 = fig.add_subplot(gs[0, 1])
    pv = pd.read_csv(f"{RES}/12_pervariant_onsetage.csv")
    o = -np.log10(pv.p.clip(lower=1e-12).sort_values().values)
    e = -np.log10((np.arange(1, len(o) + 1) - 0.5) / len(o))
    mx = max(e.max(), o.max()) * 1.05
    ax2.plot([0, mx], [0, mx], color=GREY_L, lw=1.2, zorder=2)
    ax2.plot(e, o, "o", ms=4.5, color=MID, markeredgecolor="white",
             markeredgewidth=0.5, zorder=4)
    ax2.set_xlabel("expected by chance")
    ax2.set_ylabel("observed, variant by variant")
    ax2.set_xlim(0, mx); ax2.set_ylim(0, mx)

    ax3 = fig.add_subplot(gs[1, :])
    pw = pd.read_csv(f"{RES}/14_power.csv")
    real = np.abs(m.aao_beta_aligned)
    maf = np.minimum(m.maf_ppmi, 1 - m.maf_ppmi)
    ax3.scatter(maf, real, s=28, color=GREY_L, zorder=3, edgecolors="white",
                linewidths=0.6)
    for lab, c, ls, nm in [("nominal", NAVY, "-", "detectable in PPMI"),
                           ("Bonferroni", MID, (0, (5, 3)),
                            "detectable after correction")]:
        s2 = pw[pw.alpha_label == lab]
        ax3.plot(s2.maf, s2.mde_years_per_allele, ls=ls, color=c, lw=1.8,
                 marker="o", ms=5, markeredgecolor="white", markeredgewidth=0.8,
                 zorder=5, label=nm)
    ax3.plot([], [], "o", color=GREY_L, ms=5, markeredgecolor="white",
             label="effect the variant really has")
    ax3.set_yscale("log"); ax3.set_xlim(0, 0.52); ax3.set_ylim(1e-3, 40)
    ax3.set_xlabel("how common the variant is")
    ax3.set_ylabel("effect on onset\n(years per allele)")
    ax3.legend(loc="lower left")

    for a, s_ in zip([ax, ax2, ax3], "abc"):
        PL(fig, a, s_, x=a.get_position().x0 - 0.125)
    save(fig, "E13_enrolment_proxy")


if __name__ == "__main__":
    print("extended data figures 9-13")
    for f in (ed9, ed10, ed11, ed12, ed13):
        try:
            f()
        except Exception:
            import traceback
            print(f"  [fail] {f.__name__}"); traceback.print_exc(limit=3)
