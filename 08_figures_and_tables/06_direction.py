import sys
from pathlib import Path
import warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.pocket import read_pdb, find_pockets
from lib import protein_render as pr
from lib.style import *
from lib import paths

warnings.filterwarnings("ignore")

ROOT = str(paths.ROOT)
RES = str(paths.RESULTS)
STR = str(paths.STRUCTURES)
PLDDT_CM = LinearSegmentedColormap.from_list(
    "plddt", ["#c9c9c3", "#b9c9d6", PALE, MID, NAVY, DEEP])

AGENT_DIR = {
    "SNCA": "lower", "MAPT": "lower", "LRRK2": "lower", "GPNMB": "lower",
    "GBA1": "raise", "IDUA": "raise",
}


def main():
    d = pd.read_csv(f"{RES}/182_direction_final.csv")
    pt = pd.read_csv(f"{RES}/183_direction_per_tissue.csv")

    fig = plt.figure(figsize=(7.0, 8.9))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.50, 1.30],
                             hspace=0.30, left=0.075, right=0.965,
                             top=0.950, bottom=0.058)

    ax = fig.add_subplot(outer[0])
    ax.set_position(ax.get_position())
    have = d[d.n_tissues > 0].sort_values("median_slope")
    y = np.arange(len(have))
    rng = np.random.default_rng(11)
    ax.axvline(0, color=INK, lw=1.1, zorder=4)
    for yi, (_, r) in zip(y, have.iterrows()):
        s = pt[pt.gene == r.gene]
        col = NAVY if r.risk_allele_effect == "raises" else \
            MID if r.risk_allele_effect == "lowers" else GREY_L
        if r.gene in ("GALC", "LRRC37A2"):
            col = WARM
        ax.plot([s.median_slope.min(), s.median_slope.max()], [yi, yi],
                color=col, lw=2.4, alpha=0.30, solid_capstyle="round",
                zorder=2)
        ax.scatter(s.median_slope, yi + rng.normal(0, 0.055, len(s)), s=26,
                   color=col, edgecolors="white", linewidths=0.6, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}   {int(n)} brain region" + ("s" if n != 1 else "")
                        for g, n in zip(have.gene, have.n_tissues)],
                       fontsize=8.2)
    for t, g in zip(ax.get_yticklabels(), have.gene):
        t.set_style("italic")
        if g in ("GALC", "LRRC37A2"):
            t.set_color(WARM)
    ax.set_xlim(-2.0, 1.4)
    ax.set_xlabel("what the Parkinson's risk allele does to the gene in brain")
    ax.set_ylim(-0.8, len(have) - 0.2)
    hide_y_spine(ax)
    ax.text(-1.35, len(have) - 0.35, "lowers it", ha="center", va="bottom",
            fontsize=8.4, color=GREY)
    ax.text(0.75, len(have) - 0.35, "raises it", ha="center", va="bottom",
            fontsize=8.4, color=GREY)

    ax2 = fig.add_subplot(outer[1])
    call = d[d.drug_would_need_to != "unknown"].copy()
    call["ord"] = call.drug_would_need_to.map({"lower": 0, "raise": 1})
    call = call.sort_values(["ord", "gene"])
    x = np.arange(len(call))
    for xi, (_, r) in zip(x, call.iterrows()):
        up = r.drug_would_need_to == "raise"
        col = NAVY if up else MID
        ax2.add_patch(FancyArrow(xi, 0.0, 0, 0.58 if up else -0.58,
                                 width=0.085, head_width=0.26,
                                 head_length=0.20, color=col,
                                 length_includes_head=True, zorder=3))
        exists = AGENT_DIR.get(r.gene) == r.drug_would_need_to
        ax2.plot([xi + 0.30], [0.72 if up else -0.72], "o", ms=7,
                 color=col if exists else "white", mec=col, mew=1.3, zorder=5)
        ax2.text(xi, -1.12, r.gene, ha="center", va="top", fontsize=8.2,
                 style="italic",
                 color=WARM if r.gene in ("GALC", "LRRC37A2") else INK)
        ax2.text(xi, -1.40, r.confidence, ha="center", va="top", fontsize=7.4,
                 color=GREY)
    ax2.axhline(0, color=GREY_L, lw=1.0, zorder=1)
    ax2.set_xlim(-1.9, len(call) - 0.1); ax2.set_ylim(-1.95, 0.95)
    ax2.set_xticks([]); ax2.set_yticks([])
    for s_ in ax2.spines.values():
        s_.set_visible(False)
    ax2.text(-1.85, 0.55, "raise it", ha="left", va="center", fontsize=8.4,
             color=INK)
    ax2.text(-1.85, -0.55, "lower it", ha="left", va="center", fontsize=8.4,
             color=INK)
    ax2.text(-1.85, 0.98, "a drug would have to", ha="left", va="center",
             fontsize=8.0, color=GREY)
    ax2.text(len(call) - 0.15, -1.62, "filled: such an agent exists",
             ha="right", va="top", fontsize=7.4, color=GREY)

    g2 = outer[2].subgridspec(1, 3, width_ratios=[1.0, 1.05, 1.05],
                              wspace=0.52)

    ax3 = fig.add_subplot(g2[0])
    xyz, plddt, resid, resn, elem = read_pdb(f"{STR}/AF-P54803-F1.pdb")
    pk = find_pockets(xyz, resn, resid, elem, min_psp=4, min_volume=50,
                      n_report=3)
    P, rid, idx = pr.ca_trace(xyz, resid)
    c, R = pr.view_matrix(P, focus=np.array(pk[0]["centre"]))
    P3 = pr.project(P, c, R)
    Q3 = pr.project(pk[0]["points"], c, R)
    pr.draw_backbone(ax3, P3, values=(np.clip(plddt[idx], 45, 95) - 45) / 50.0,
                     cmap=PLDDT_CM, lw=(0.8, 2.6))
    pr.draw_cloud(ax3, Q3[:, :2], color=WARM, alpha=0.10, size=10)
    for r_, nm, dx, dy in [(198, "Glu198", -0.16, 0.20),
                           (274, "Glu274", -0.20, -0.10)]:
        j = np.where(rid == r_)[0]
        if len(j):
            px, py = P3[j[0], 0], P3[j[0], 1]
            ax3.plot(px, py, "o", ms=5.5, color=WARM, mec="white", mew=1.1,
                     zorder=12)
            span = np.ptp(P3[:, :2])
            ax3.annotate(nm, xy=(px, py),
                         xytext=(px + dx * span, py + dy * span),
                         fontsize=7.8, color=WARM,
                         ha="right" if dx < 0 else "left", va="center",
                         arrowprops=dict(arrowstyle="-", color=WARM, lw=0.7,
                                         shrinkA=1, shrinkB=3), zorder=12)
    pr.frame(ax3, P3)
    ax3.text(0.5, -0.03, f"catalytic pocket, {pk[0]['volume']:.0f} "
                         r"$\mathrm{\AA}^3$",
             transform=ax3.transAxes, ha="center", va="top", fontsize=7.8,
             color=GREY)

    ax4 = fig.add_subplot(g2[1])
    steps = [("the risk variant\nraises the risk", np.exp(0.061),
              np.exp(0.043), np.exp(0.079), NAVY),
             ("it raises enzyme\nactivity in blood", 1.205, 1.09, 1.32, MID),
             ("higher activity\nraises the risk", np.exp(0.025),
              np.exp(0.0113), np.exp(0.0387), NAVY)]
    yy = np.arange(3)[::-1]
    for i, (nm, est, lo, hi, col) in enumerate(steps):
        span = max(hi - 1.0, 1.0 - lo) * 1.9
        xc = (est - 1.0) / span
        xl = (lo - 1.0) / span
        xh = (hi - 1.0) / span
        ax4.plot([xl, xh], [yy[i]] * 2, color=col, lw=1.6, zorder=4)
        ax4.plot([xc], [yy[i]], "o", ms=6.5, color=col, mec="white", mew=1.0,
                 zorder=5)
        ax4.text(-0.30, yy[i], nm, ha="right", va="center", fontsize=7.8,
                 color=INK)
        ax4.text(xh + 0.05, yy[i],
                 f"{est:.2f}" if est < 1.1 else f"{est:.2f}",
                 ha="left", va="center", fontsize=7.6, color=GREY)
    ax4.axvline(0, color=GREY_L, lw=1.1, zorder=2)
    ax4.set_xlim(-0.30, 0.86); ax4.set_ylim(-0.7, 2.7)
    ax4.set_xticks([0]); ax4.set_xticklabels(["no effect"], fontsize=7.8)
    ax4.set_yticks([])
    for s_ in ax4.spines.values():
        s_.set_visible(False)
    ax4.text(0.30, 2.45, "each step on its own scale", ha="center",
             va="bottom", fontsize=7.4, color=GREY)

    ax5 = fig.add_subplot(g2[2])
    P_ = pd.read_csv(f"{RES}/124_galc_eqtl_ld_all.csv")
    hi_ld = P_[P_.r2 > 0.6]
    ax5.axhline(0, color=INK, lw=1.0, zorder=4)
    ax5.scatter(P_.r2, P_.slope_on_risk_allele, s=13, color=GREY_L,
                edgecolors="none", zorder=3)
    ax5.scatter(hi_ld.r2, hi_ld.slope_on_risk_allele, s=50, color=WARM,
                zorder=5, edgecolors="white", linewidths=1.0)
    leader(ax5, f"all {len(hi_ld)} closest\nmarkers raise GALC",
         xy=(0.985, 0.220), xytext=(0.92, 0.80), color=WARM, ha="right")
    ax5.set_xlim(-0.03, 1.10); ax5.set_ylim(-0.55, 1.05)
    ax5.set_xticks([0, 0.5, 1.0]); ax5.set_yticks([-0.5, 0, 0.5, 1.0])
    ax5.set_xlabel("how closely the marker\ntracks the risk variant")
    ax5.set_ylabel("effect of the risk allele\non GALC in brain")

    fig.canvas.draw()
    for a, s_, x in [(ax, "a", 0.012), (ax2, "b", 0.012),
                     (ax3, "c", 0.012), (ax4, "d", 0.355),
                     (ax5, "e", 0.665)]:
        panel_label(fig, a, s_, x=x)
    save(fig, "supp03_direction_of_effect")


if __name__ == "__main__":
    main()
