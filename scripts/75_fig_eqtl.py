#!/usr/bin/env python
"""Effect of the Parkinson's risk allele on gene expression in brain."""
import os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clinical_style import *          # noqa
import matplotlib.pyplot as plt
import paths as _P
warnings.filterwarnings("ignore")

RES = str(_P.RESULTS)

TISSUE_ORDER = ["Substantia nigra", "Putamen basal ganglia",
                "Caudate basal ganglia", "Nucleus accumbens basal ganglia",
                "Hypothalamus", "Amygdala", "Hippocampus",
                "Anterior cingulate cortex BA24", "Frontal Cortex BA9",
                "Cortex", "Cerebellum", "Cerebellar Hemisphere",
                "Spinal cord cervical c-1"]
SHORT = {"Substantia nigra": "substantia nigra",
         "Putamen basal ganglia": "putamen",
         "Caudate basal ganglia": "caudate",
         "Nucleus accumbens basal ganglia": "nucleus accumbens",
         "Hypothalamus": "hypothalamus", "Amygdala": "amygdala",
         "Hippocampus": "hippocampus",
         "Anterior cingulate cortex BA24": "anterior cingulate",
         "Frontal Cortex BA9": "frontal cortex", "Cortex": "cortex",
         "Cerebellum": "cerebellum",
         "Cerebellar Hemisphere": "cerebellar hemisphere",
         "Spinal cord cervical c-1": "spinal cord"}


def main():
    t = pd.read_csv(f"{RES}/185_eqtl_by_tissue.csv")
    e = pd.read_csv(f"{RES}/186_brain_expression.csv")
    v = pd.read_csv(f"{RES}/184_eqtl_variants.csv")

    order = (t.groupby("gene").beta.median().sort_values().index.tolist())
    fig = plt.figure(figsize=(7.2, 10.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.55, 0.90, 0.72], hspace=0.50,
                          left=0.245, right=0.945, top=0.940, bottom=0.058)

    # per-tissue effect, aligned to the risk allele
    ax = fig.add_subplot(gs[0])
    ax.axvline(0, color=INK, lw=1.1, zorder=5)
    yy, ylab, ycol = [], [], []
    tick_y, tick_lab = [], []
    row = 0
    for gene in order:
        s = t[t.gene == gene].copy()
        s["ord"] = s.tissue.map({x: i for i, x in enumerate(TISSUE_ORDER)})
        s = s.sort_values("ord")
        col = WARM if gene in ("LRRC37A2", "MAPT") else NAVY
        for _, r in s.iterrows():
            ax.plot([r.lo, r.hi], [row, row], color=col, lw=1.3, alpha=0.75,
                    zorder=3, solid_capstyle="round")
            ax.plot([r.beta], [row], "o", ms=4.6, color=col, mec="white",
                    mew=0.8, zorder=6)
            tick_y.append(row); tick_lab.append(SHORT.get(r.tissue, r.tissue))
            row += 1.0
        ymid = row - len(s) / 2 - 0.5
        yy.append(ymid); ylab.append(gene); ycol.append(col)
        row += 1.5

    ax.set_yticks(tick_y); ax.set_yticklabels(tick_lab, fontsize=6.8)
    for t_ in ax.get_yticklabels():
        t_.set_color(GREY)
    for ym, lab, col in zip(yy, ylab, ycol):
        ax.text(-0.30, ym, lab, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8.6, style="italic",
                color=col, clip_on=False)
    ax.set_ylim(row - 1.4, -1.0)
    ax.set_xlim(-2.35, 1.35)
    ax.set_xlabel("effect of the risk allele on expression "
                  "(normalised units, 95% CI)")
    bare_y(ax)
    ax.text(-1.15, -0.55, "risk allele lowers the gene", ha="center",
            va="center", fontsize=8.0, color=GREY)
    ax.text(0.62, -0.55, "raises it", ha="center", va="center", fontsize=8.0,
            color=GREY)

    # expression level, for context
    ax2 = fig.add_subplot(gs[1])
    genes2 = order
    x = np.arange(len(TISSUE_ORDER))
    ends = []
    for gi, gene in enumerate(genes2):
        sub = e[e.gene == gene].set_index("tissue").tpm
        vals = [sub.get(tt, np.nan) for tt in TISSUE_ORDER]
        col = WARM if gene in ("LRRC37A2", "MAPT") else MID
        ax2.plot(x, vals, "o-", ms=4.2, lw=1.2, color=col, alpha=0.85,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        last = next((v_ for v_ in reversed(vals) if not np.isnan(v_)), None)
        if last is not None:
            ends.append([np.log10(last), gene, col])
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 0.19:
            ends[i][0] = ends[i - 1][0] + 0.19
    for ly, gene, col in ends:
        ax2.text(x[-1] + 0.30, 10 ** ly, gene, fontsize=7.4, va="center",
                 ha="left", style="italic", color=col)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([SHORT[tt] for tt in TISSUE_ORDER], rotation=45,
                        ha="right", fontsize=7.0)
    ax2.set_xlim(-0.5, len(x) + 2.4)
    ax2.set_ylabel("expression\n(median TPM)")

    # how many genes each locus lead variant controls in brain
    ax3 = fig.add_subplot(gs[2])
    lc = pd.read_csv(f"{RES}/187_locus_gene_count.csv")
    lc = lc.drop_duplicates(subset=["chr", "pos"]).copy()
    lc["label"] = lc.apply(
        lambda r: "LRRC37A2 / MAPT" if r.chr == 17 else
        ("TMEM175 / GAK / DGKQ / IDUA" if r.pos == 951947 else r.gene), axis=1)
    lc = lc.sort_values("n_genes_controlled")
    yb = np.arange(len(lc))
    cols = [WARM if r.chr == 17 else MID for _, r in lc.iterrows()]
    ax3.barh(yb, lc.n_genes_controlled, height=0.60, color=cols, zorder=3)
    for yi, n in zip(yb, lc.n_genes_controlled):
        ax3.text(n + 0.5, yi, str(int(n)), va="center", fontsize=7.6,
                 color=INK)
    ax3.set_yticks(yb)
    ax3.set_yticklabels(lc.label, fontsize=7.4)
    for t_, r_ in zip(ax3.get_yticklabels(), lc.chr):
        t_.set_style("italic")
        if r_ == 17:
            t_.set_color(WARM)
    ax3.set_xlim(0, 32); ax3.set_xticks([0, 10, 20, 30])
    ax3.set_xlabel("genes in brain whose expression the locus lead variant "
                   "controls")
    bare_y(ax3)

    for a, s_ in zip([ax, ax2, ax3], "abc"):
        PL(fig, a, s_, x=0.012)
    save(fig, "F3_eqtl_evidence")


if __name__ == "__main__":
    main()
