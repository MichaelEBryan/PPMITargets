#!/usr/bin/env python
"""Main figures 2 to 6, clinical framing, on the approved Figure 1 template."""
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clinical_style import *          # noqa
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import paths as _P
warnings.filterwarnings("ignore")

RES = str(_P.RESULTS)
DATA = str(_P.DATA)


def fig2():
    """How much inherited burden shifts onset, and whether it holds up."""
    fig = plt.figure(figsize=(7.0, 8.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.30, 1.05, 1.05], hspace=0.66,
                          wspace=0.50, left=0.285, right=0.965, top=0.945,
                          bottom=0.070)

    # a. the effect, across every reasonable way of estimating it
    ax = fig.add_subplot(gs[0, :])
    d = pd.read_csv(f"{RES}/51_prs_vs_AAO.csv")
    d = d[d.adjustment == "sex + 10 PCs"]
    sc = {"META5_PGS": ("all known risk variants", NAVY),
          "META5_excl_LRRK2_GBA_PGS": ("excluding LRRK2 and GBA", MID),
          "GP2_PGS": ("an independent score", PALE)}
    nice = {"all cases with PRS": "all patients",
            "sporadic only": "sporadic patients",
            "EUR only": "European ancestry",
            "EUR + sporadic": "European and sporadic"}
    subs = list(dict.fromkeys(d.subset))
    yp, lb, i = [], [], 0
    for s in subs:
        for score, (nm, c) in sc.items():
            r = d[(d.subset == s) & (d.score == score)]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.errorbar(-r.beta_years_per_SD, i,
                        xerr=[[r.ci_hi - r.beta_years_per_SD],
                              [r.beta_years_per_SD - r.ci_lo]],
                        fmt="o", ms=6, color=c, elinewidth=1.5, capsize=0,
                        markeredgecolor="white", markeredgewidth=0.9, zorder=4)
            i += 1
        yp.append(i - 2)
        lb.append(f"{nice.get(s, s)}\nn = {int(r.n)}")
        i += 1.1
    ax.axvline(0, color=INK, lw=1.0, zorder=2)
    ax.set_yticks(yp); ax.set_yticklabels(lb)
    ax.invert_yaxis()
    ax.set_xlim(-0.9, 3.6); ax.set_ylim(i - 0.5, -3.6)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("years of onset brought forward, per standard deviation "
                  "of inherited risk")
    bare_y(ax)
    for nm, c in sc.values():
        ax.plot([], [], "o", color=c, ms=6, markeredgecolor="white", label=nm)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.035), ncol=3,
              columnspacing=1.4, handletextpad=0.5)

    # b. the same variants that raise risk also bring onset forward
    ax2 = fig.add_subplot(gs[1, :])
    al = pd.read_csv(f"{RES}/19b_risk_vs_onset_aligned.csv")
    ax2.axhline(0, color=GREY_L, lw=1.0, zorder=1)
    ax2.axvline(0, color=GREY_L, lw=1.0, zorder=1)
    ax2.scatter(al.risk_beta_target, al.aao_beta_aligned, s=30, color=NAVY,
                alpha=0.65, edgecolors="white", linewidths=0.6, zorder=4)
    xs = np.linspace(al.risk_beta_target.min(), al.risk_beta_target.max(), 10)
    b = np.polyfit(al.risk_beta_target, al.aao_beta_aligned, 1)
    ax2.plot(xs, np.polyval(b, xs), color=WARM, lw=2.0, zorder=5)
    ors = np.array([0.8, 1.0, 1.25, 1.6, 2.0])
    ax2.set_xticks(np.log(ors)); ax2.set_xticklabels([f"{o:g}" for o in ors])
    ax2.set_xlim(np.log(0.72), np.log(2.25))
    ax2.set_xlabel("risk of Parkinson's disease carried by the variant "
                   "(odds ratio)")
    ax2.set_ylabel("years of onset\nadded by the variant")

    # c. the effect reproduces in an independent patient set
    ax3 = fig.add_subplot(gs[2, 0])
    m = pd.read_csv(f"{RES}/54_ppmi_AAO_vs_ipdgc.csv")
    s = json.load(open(f"{RES}/56_aao_core_summary.json"))["replication"]
    ax3.axhline(0, color=GREY_L, lw=1.0); ax3.axvline(0, color=GREY_L, lw=1.0)
    ax3.errorbar(m.ipdgc_beta, m.aao_beta, yerr=m.aao_se, fmt="o", ms=4,
                 color=MID, elinewidth=0.6, alpha=0.85, markeredgecolor="white",
                 markeredgewidth=0.5, zorder=4)
    lim = float(np.nanmax(np.abs(m.ipdgc_beta))) * 1.15
    xs = np.linspace(-lim, lim, 10)
    ax3.plot(xs, s["wls_slope"] * xs, color=WARM, lw=2.0, zorder=5)
    ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim * 2.5, lim * 2.5)
    ax3.set_xticks([-1, 0, 1]); ax3.set_yticks([-2, 0, 2])
    ax3.set_xlabel("years of onset per allele,\n28,568 patients elsewhere")
    ax3.set_ylabel("years of onset\nper allele, PPMI")

    # d. onset by fifth of inherited risk
    ax4 = fig.add_subplot(gs[2, 1])
    c = pd.read_csv(f"{RES}/151_cases_for_clinical_fig.csv")
    q = pd.qcut(c.META5_PGS, 5, labels=False)
    med = c.groupby(q).AAO.median()
    x = np.arange(len(med)) + 1
    cols = [PALE, "#a9c2d6", MID, "#5583ab", NAVY]
    ax4.bar(x, med.values - 45, bottom=45, width=0.62, color=cols, zorder=3)
    bfit = np.polyfit(x, med.values, 1)
    ax4.plot(x, np.polyval(bfit, x), color=WARM, lw=2.0, zorder=5)
    ax4.set_ylim(45, 70); ax4.set_yticks([50, 55, 60, 65, 70])
    ax4.set_xticks(x)
    ax4.set_xlabel("fifth of inherited risk\n(1 = lowest)")
    ax4.set_ylabel("median age at onset\n(years)")

    for a, s_ in [(ax, "a"), (ax2, "b"), (ax3, "c")]:
        PL(fig, a, s_)
    PL(fig, ax4, "d", x=0.615)
    save(fig, "C2_inherited_risk_and_onset")


def fig3():
    """Early and late onset: the same disease carried at a different dose."""
    fig = plt.figure(figsize=(7.0, 8.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.05, 1.15], hspace=0.66,
                          left=0.285, right=0.965, top=0.945, bottom=0.075)

    # a. inherited burden in the earliest and the latest patients
    ax = fig.add_subplot(gs[0])
    d = pd.read_parquet(f"{DATA}/ppmi_aao.parquet")
    c = d[(d.COHORT_DEFINITION == "Parkinson's Disease") & d.AAO.notna()
          & (d.Genetic_PRS_InfPop == "EUR") & (~d.gen_asc)].dropna(
        subset=["AAO", "META5_PGS"])
    mu, sd = c.META5_PGS.mean(), c.META5_PGS.std()
    ze = (c[c.AAO < 50].META5_PGS - mu) / sd
    zl = (c[c.AAO >= 60].META5_PGS - mu) / sd
    xs = np.linspace(-3.0, 3.0, 300)
    for v, col in [(zl, PALE), (ze, NAVY)]:
        k = stats.gaussian_kde(v)
        ax.fill_between(xs, k(xs), color=col, alpha=0.55, lw=0, zorder=3)
        ax.plot(xs, k(xs), color=col, lw=1.8, zorder=4)
        ax.plot([v.mean()] * 2, [0, float(k(v.mean()))], color=col, lw=1.1,
                ls=(0, (3, 3)), zorder=5)
    ke, kl = stats.gaussian_kde(ze), stats.gaussian_kde(zl)
    lead(ax, f"onset before 50\nn = {len(ze)}", xy=(1.30, float(ke(1.30))),
         xytext=(1.80, 0.34), color=NAVY)
    lead(ax, f"onset at 60 or later\nn = {len(zl)}", xy=(-1.75, float(kl(-1.75))),
         xytext=(-2.55, 0.30), color=GREY)
    ax.set_xlim(-3.0, 3.0); ax.set_ylim(0, 0.58)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xlabel("inherited risk score (standard deviations)")
    ax.set_ylabel("patients")
    ax.set_yticks([]); bare_y(ax)

    # b. no single variant separates the two
    ax2 = fig.add_subplot(gs[1])
    pv = pd.read_csv(f"{RES}/53_pervariant_AAO.csv")
    o = -np.log10(pv.cc_p.clip(lower=1e-12).sort_values().values)
    e = -np.log10((np.arange(1, len(o) + 1) - 0.5) / len(o))
    thr = -np.log10(0.05 / len(o))
    mx = max(e.max(), o.max(), thr) * 1.10
    ax2.plot([0, mx], [0, mx], color=GREY_L, lw=1.2, zorder=2)
    ax2.scatter(e, o, s=32, color=MID, edgecolors="white", linewidths=0.7,
                zorder=4)
    ax2.axhline(thr, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax2.text(0.05, thr + 0.07, "threshold for 86 variants", fontsize=8.2,
             color=GREY, va="bottom")
    ax2.set_xlim(0, mx * 0.82); ax2.set_ylim(0, mx)
    ax2.set_xlabel("difference expected by chance alone")
    ax2.set_ylabel("difference observed between\nearly and late onset")

    # c. no pathway separates them either
    ax3 = fig.add_subplot(gs[2])
    cur = pd.read_csv(f"{RES}/60_pathway_curated.csv").sort_values("z")
    y = np.arange(len(cur))
    ax3.barh(y, cur.z.values, height=0.55,
             color=[MID if zz > 0 else PALE for zz in cur.z], zorder=3)
    ax3.axvline(0, color=INK, lw=1.0, zorder=4)
    for v in (-1.96, 1.96):
        ax3.axvline(v, color=GREY_L, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax3.set_yticks(y)
    ax3.set_yticklabels([f"{p.replace(' / ', ' and ')}\n{int(n)} loci"
                         for p, n in zip(cur.pathway, cur.n_loci)])
    ax3.set_xlim(-2.9, 2.9); ax3.set_xticks([-1.96, 0, 1.96])
    ax3.set_xticklabels(["against", "no effect", "for"])
    ax3.set_xlabel("evidence that the pathway acts on when the disease begins,\n"
                   "over and above its effect on whether it begins")
    bare_y(ax3)

    for a, s_ in [(ax, "a"), (ax2, "b"), (ax3, "c")]:
        PL(fig, a, s_)
    save(fig, "C3_same_disease_different_dose")


def fig4():
    """Which genes carry onset signal, and what could be given to a patient."""
    ga = pd.read_csv(f"{RES}/85_target_annotation.csv").dropna(
        subset=["onset_min_p"]).sort_values("onset_min_p")
    NOM = {"GALC", "LRRC37A2"}

    def sm(r):
        t = str(r.sm_tractability)
        if "High-Quality Ligand" in t or "Druggable Family" in t:
            return 2
        if "Structure with Ligand" in t:
            return 1
        return 0

    AGENT = {"SNCA": "antisense oligonucleotide, phase 1",
             "MAPT": "antisense oligonucleotide, phase 2",
             "GPNMB": "antibody-drug conjugate, phase 2",
             "IDUA": "laronidase, approved in Hurler syndrome"}

    fig, ax = plt.subplots(figsize=(7.0, 6.0),
                           gridspec_kw=dict(left=0.150, right=0.520, top=0.870,
                                            bottom=0.115))
    n = len(ga)
    y = np.arange(n)[::-1]
    v = -np.log10(ga.onset_min_p.values)
    col = [WARM if g in NOM else (NAVY if p < 5e-8 else
                                  MID if p < 1e-3 else PALE)
           for g, p in zip(ga.gene, ga.onset_min_p)]
    ax.barh(y, v, height=0.58, color=col, zorder=3)
    ax.axvline(-np.log10(5e-8), color=INK, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(ga.gene)
    for t, g in zip(ax.get_yticklabels(), ga.gene):
        t.set_style("italic")
        if g in NOM:
            t.set_color(WARM)
    ax.set_xlim(0, 9.4); ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("evidence that the gene acts on age at onset")
    ax.text(-np.log10(5e-8), n - 0.25, "conventional\nthreshold  ",
            fontsize=8.2, color=GREY, ha="right", va="bottom")
    bare_y(ax)

    # a target table to the right of the bars, drawn in data coordinates
    cx = [10.9, 14.0, 17.1]
    head = ["small\nmolecule", "antibody\naccessible", "agent in\nthe clinic"]
    for x0, h in zip(cx, head):
        ax.text(x0, n - 0.25, h, fontsize=8.2, ha="center", va="bottom",
                color=INK, linespacing=1.35, clip_on=False)
    for yi, (_, r) in zip(y, ga.iterrows()):
        vals = [sm(r),
                1 if "Advanced Clinical" in str(r.ab_tractability) else 0,
                1 if AGENT.get(r.gene) else 0]
        for x0, val in zip(cx, vals):
            fc = {0: "white", 1: MID, 2: NAVY}[val]
            ax.scatter([x0], [yi], s=54, facecolor=fc, edgecolor=GREY_L,
                       linewidths=0.9, zorder=6, clip_on=False)
        if AGENT.get(r.gene):
            ax.text(18.6, yi, AGENT[r.gene], fontsize=8.2, va="center",
                    color=INK, clip_on=False)
    save(fig, "E15_candidate_genes")


def fig5():
    """The GALC locus: the regional signal, and where GALC is under control."""
    fig = plt.figure(figsize=(7.0, 5.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.62,
                             left=0.145, right=0.965, top=0.930, bottom=0.095)

    # a. the locus, with the genes beneath it
    g0 = outer[0].subgridspec(2, 1, height_ratios=[6, 1.4], hspace=0.24)
    ax = fig.add_subplot(g0[0])
    axg = fig.add_subplot(g0[1], sharex=ax)
    r = pd.read_csv(f"{RES}/120_galc_region_risk.csv")
    r = r[r.base_pair_location.between(88_360_000, 88_580_000)]
    ax.scatter(r.base_pair_location / 1e6, -np.log10(r.p_value), s=8,
               color=PALE, edgecolors="none", zorder=3)
    ld = r[r.base_pair_location == 88_464_264]
    if len(ld):
        ax.scatter(ld.base_pair_location / 1e6, -np.log10(ld.p_value), s=64,
                   color=NAVY, zorder=6, edgecolors="white", linewidths=1.1)
        lead(ax, "rs979812", xy=(88.4643, -np.log10(float(ld.p_value.iloc[0]))),
             xytext=(88.507, 3.35), color=NAVY)
    ax.set_ylim(0, 4.0); ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel("association with\nParkinson's disease\n"
                  r"($-\log_{10}P$)")
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.spines["bottom"].set_visible(False)
    for s0, e0, nm, xoff in [(88.399357, 88.460009, "GALC", 0.0),
                             (88.471478, 88.481155, "GPR65", 0.013)]:
        axg.add_patch(Rectangle((s0, 0.62), e0 - s0, 0.34, color=MID, zorder=5))
        axg.text((s0 + e0) / 2 + xoff, 0.46, nm, fontsize=8.4, ha="center",
                 va="top", color=INK, style="italic")
    axg.set_ylim(0, 1.0); axg.set_yticks([])
    axg.set_xlim(88.355, 88.580)
    axg.set_xticks([88.375, 88.425, 88.475, 88.525, 88.575])
    axg.set_xlabel("position on chromosome 14 (millions of bases)", labelpad=3)
    for sp in ("left", "bottom"):
        axg.spines[sp].set_visible(False)
    axg.tick_params(axis="x", length=0, pad=2)

    # b. where in the brain GALC is under genetic control
    ax6 = fig.add_subplot(outer[1])
    eq = pd.read_csv(f"{RES}/121_galc_eqtl_brain.csv")
    g = eq[eq.gene == "GALC"].groupby("tissue").size()
    short = {"Brain_Anterior_cingulate_cortex_BA24": "anterior cingulate",
             "Brain_Cerebellar_Hemisphere": "cerebellar hemisphere",
             "Brain_Cerebellum": "cerebellum",
             "Brain_Nucleus_accumbens_basal_ganglia": "nucleus accumbens"}
    ALL13 = ["amygdala", "anterior cingulate", "caudate", "cerebellar hemisphere",
             "cerebellum", "cortex", "frontal cortex", "hippocampus",
             "hypothalamus", "nucleus accumbens", "putamen", "spinal cord",
             "substantia nigra"]
    counts = {t: 0 for t in ALL13}
    for k, val in g.items():
        counts[short.get(k, k)] = val
    ser = pd.Series(counts).sort_values()
    ax6.barh(np.arange(len(ser)), ser.values, color=NAVY, height=0.66, zorder=3)
    ax6.set_yticks(np.arange(len(ser)))
    ax6.set_yticklabels(ser.index, fontsize=7.8)
    for t, val in zip(ax6.get_yticklabels(), ser.values):
        if val == 0:
            t.set_color(GREY)
    ax6.set_xlim(0, 560); ax6.set_xticks([0, 250, 500])
    ax6.set_xlabel("markers controlling GALC in that region")
    bare_y(ax6)

    PL(fig, ax, "a")
    PL(fig, ax6, "b")
    save(fig, "E16_galc_locus")


def fig6():
    """What the published classifier was actually separating."""
    d = pd.read_csv(f"{RES}/22_ablation_summary.csv")
    order = ["full", "only_design", "minus_design", "missingness_only",
             "genetics_only", "only_geno"]
    label = {"full": "everything in the released table",
             "only_design": "which study arm the patient joined",
             "minus_design": "everything except the study arm",
             "missingness_only": "which tests the patient happened to have",
             "genetics_only": "genotypes, risk scores and ancestry",
             "only_geno": "the patient's genotypes"}
    COL = {"full": INK, "only_design": WARM, "minus_design": GREY,
           "missingness_only": MID, "genetics_only": NAVY, "only_geno": NAVY}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.2), sharey=True,
                             gridspec_kw=dict(wspace=0.06, left=0.345,
                                              right=0.975, top=0.860,
                                              bottom=0.230))
    for ax, strat, sname in zip(axes, ["EOPD_lt50", "LOPD_ge60"],
                                ["patients enrolled before 50",
                                 "patients enrolled at 60 or later"]):
        s = d[d.stratum == strat]
        base = s[s.model == "MajorityBaseline"].f1w.mean()
        ax.axvline(base, color=GREY_L, lw=1.4, zorder=1)
        ax.text(base, -0.62, "chance", fontsize=8.2, color=GREY, ha="center",
                va="top")
        for i, arm in enumerate(reversed(order)):
            mods = s[(s.arm == arm) & (s.model != "MajorityBaseline")]
            vals = [mods[mods.model == m].f1w.iloc[0] for m in
                    ["XGBoost", "LightGBM", "CatBoost"]
                    if len(mods[mods.model == m])]
            if not vals:
                continue
            c = COL[arm]
            ax.plot([min(vals), max(vals)], [i, i], color=c, lw=3.0, alpha=0.30,
                    solid_capstyle="round", zorder=2)
            ax.scatter(vals, [i] * len(vals), s=38, color=c, zorder=4,
                       edgecolors="white", linewidths=0.9)
        ax.set_xlim(0.40, 1.03); ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_ylim(-0.7, len(order) - 0.25)
        ax.tick_params(axis="y", length=0)
        ax.text(0.5, 1.035, sname, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.6, color=INK)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([label[a] for a in reversed(order)])
    for a in axes:
        a.spines["left"].set_visible(False)
    fig.text(0.66, 0.055, "how well the model separates the four clinical groups",
             ha="center", fontsize=9, color=INK)
    PL(fig, axes[0], "a", dy=0.030)
    PL(fig, axes[1], "b", x=axes[1].get_position().x0 - 0.030, dy=0.030)
    save(fig, "C7_what_the_model_separated")


if __name__ == "__main__":
    print("clinical main figures")
    for f in (fig2, fig3, fig4, fig5, fig6):
        try:
            f()
        except Exception:
            import traceback
            print(f"  [fail] {f.__name__}"); traceback.print_exc(limit=3)
