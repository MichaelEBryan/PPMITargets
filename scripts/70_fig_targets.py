#!/usr/bin/env python
"""What makes a target targetable: structures, pockets, and a benchmark."""
import os, sys, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clinical_style import *          # noqa
from pocket import read_pdb, find_pockets
import protein_render as pr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import paths as _P
warnings.filterwarnings("ignore")

ROOT = str(_P.ROOT)
RES = str(_P.RESULTS)
STR = str(_P.STRUCTURES)

PLDDT_CM = LinearSegmentedColormap.from_list(
    "plddt", ["#c9c9c3", "#b9c9d6", PALE, MID, NAVY, DEEP])

SHOW = [
    ("GBA1", "P04062", [274, 379], "already carries drugs"),
    ("GALC", "P54803", [198, 274], "a pocket, no chemistry"),
    ("SNCA", "P37840", [], "no pocket to bind"),
    ("LRRC37A2", "A6NM11", [], "no folded structure"),
]


def panel_structure(ax, gene, acc, sites, cache):
    path = f"{STR}/AF-{acc}-F1.pdb"
    if gene in cache:
        P3, pl, rid, Q3 = cache[gene]
    else:
        xyz, plddt, resid, resn, elem = read_pdb(path)
        pk = find_pockets(xyz, resn, resid, elem, min_psp=4, min_volume=50,
                          n_report=3)
        P, rid, idx = pr.ca_trace(xyz, resid)
        pl = plddt[idx]
        focus = np.array(pk[0]["centre"]) if pk else None
        c, R = pr.view_matrix(P, focus=focus)
        P3 = pr.project(P, c, R)
        Q3 = pr.project(pk[0]["points"], c, R) if pk else None
        cache[gene] = (P3, pl, rid, Q3)
    v = (np.clip(pl, 45, 95) - 45) / 50.0
    pr.draw_backbone(ax, P3, values=v, cmap=PLDDT_CM, lw=(0.8, 2.6))
    if Q3 is not None:
        pr.draw_cloud(ax, Q3[:, :2], color=WARM, alpha=0.09, size=9)
    for r_ in sites:
        j = np.where(rid == r_)[0]
        if len(j):
            ax.plot(P3[j[0], 0], P3[j[0], 1], "o", ms=5.5, color=WARM,
                    mec="white", mew=1.1, zorder=12)
    pr.frame(ax, P3)
    ax.text(0.5, -0.045, gene, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, style="italic", color=INK)
    return float(np.nanmean(pl))


def main():
    d = pd.read_csv(f"{RES}/170_target_dossier.csv").set_index("gene")
    b = pd.read_csv(f"{RES}/172_pocket_benchmark.csv")
    auc = json.load(open(f"{RES}/173_pocket_benchmark_auc.json"))
    ga = pd.read_csv(f"{RES}/85_target_annotation.csv").dropna(
        subset=["onset_min_p"]).set_index("gene")

    fig = plt.figure(figsize=(7.0, 9.4))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.78, 0.95, 1.55],
                             hspace=0.42, left=0.075, right=0.965, top=0.955,
                             bottom=0.055)

    # a. four structures, four kinds of target
    g0 = outer[0].subgridspec(1, 4, wspace=0.10)
    cache, axes_a = {}, []
    for j, (gene, acc, sites, tag) in enumerate(SHOW):
        ax = fig.add_subplot(g0[j])
        panel_structure(ax, gene, acc, sites, cache)
        ax.text(0.5, 1.02, tag, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.0, color=GREY)
        axes_a.append(ax)
    # a small confidence key under the first panel
    cax = axes_a[0].inset_axes([0.06, -0.36, 0.72, 0.050])
    cax.imshow(np.linspace(0, 1, 200)[None, :], aspect="auto", cmap=PLDDT_CM)
    cax.set_xticks([0, 199]); cax.set_xticklabels(["low", "high"], fontsize=7.4)
    cax.set_yticks([]); cax.tick_params(length=0, pad=1)
    for s in cax.spines.values():
        s.set_visible(False)
    cax.text(1.10, 0.5, "confidence in the predicted structure",
             transform=cax.transAxes, fontsize=7.6, va="center", color=GREY)

    # b. does pocket shape predict whether a protein gets drugged?
    g1 = outer[1].subgridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.40)
    ax = fig.add_subplot(g1[0])
    rng = np.random.default_rng(7)
    for lab, yy, col, nm in [(1, 1.0, NAVY, "carry an approved\nsmall-molecule drug"),
                             (0, 0.0, PALE, "long-standing\nhard targets")]:
        s = b[b.has_approved_sm_drug == lab].dropna(subset=["pocket_enclosure"])
        ax.scatter(s.pocket_enclosure, yy + rng.normal(0, 0.055, len(s)), s=42,
                   color=col, edgecolors="white", linewidths=0.8, zorder=4)
        ax.text(0.545, yy, nm, ha="right", va="center", fontsize=8.0,
                color=INK, linespacing=1.35)
    cut = 0.705
    ax.axvline(cut, color=GREY_L, lw=1.1, ls=(0, (4, 3)), zorder=2)
    cand = d.dropna(subset=["pocket_enclosure"])
    ax.scatter(cand.pocket_enclosure, np.full(len(cand), -0.85), s=30,
               color=MID, edgecolors="white", linewidths=0.7, zorder=4)
    for g, tx, ty, ha in [("GALC", 0.600, -1.40, "center"),
                          ("GBA1", 0.700, -1.40, "center")]:
        if g in cand.index:
            ax.plot([cand.loc[g, "pocket_enclosure"]], [-0.85], "o", ms=7,
                    color=WARM, mec="white", mew=1.0, zorder=6)
            ax.annotate(g, xy=(cand.loc[g, "pocket_enclosure"], -0.85),
                        xytext=(tx, ty), ha=ha, fontsize=8.0, color=WARM,
                        style="italic",
                        arrowprops=dict(arrowstyle="-", color=WARM, lw=0.7,
                                        shrinkA=1, shrinkB=3))
    ax.text(0.545, -0.85, "the candidate genes", ha="right", va="center",
            fontsize=8.0, color=INK)
    ax.set_xlim(0.545, 0.90); ax.set_ylim(-1.75, 1.45)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9])
    ax.set_yticks([])
    ax.set_xlabel("how enclosed the largest pocket is")
    bare_y(ax)
    ax.text(cut, 1.42, f"AUC {auc['pocket_enclosure']:.2f}", ha="center",
            va="bottom", fontsize=8.4, color=GREY)

    # c. is it expressed where it matters, and is it safe to hit?
    ax2 = fig.add_subplot(g1[1])
    s = d.dropna(subset=["tpm_substantia_nigra", "loeuf"])
    sz = 20 + 90 * (s.pocket_volume.fillna(0).clip(0, 3000) / 3000)
    col = [WARM if g in ("GALC", "LRRC37A2") else
           (NAVY if s.loc[g, "n_pdb_ligand"] > 0 else PALE) for g in s.index]
    ax2.axvline(0.35, color=GREY_L, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax2.scatter(s.loeuf, s.tpm_substantia_nigra, s=sz, color=col,
                edgecolors="white", linewidths=0.8, zorder=4)
    ax2.set_yscale("log")
    for g, dx, dy, ha in [("GALC", 0.14, 1.55, "left"),
                          ("LRRC37A2", -0.14, 2.10, "right"),
                          ("APOE", -0.10, 0.40, "right"),
                          ("SNCA", -0.13, 1.9, "right")]:
        if g in s.index:
            ax2.annotate(g, xy=(s.loc[g, "loeuf"], s.loc[g, "tpm_substantia_nigra"]),
                         xytext=(s.loc[g, "loeuf"] + dx,
                                 s.loc[g, "tpm_substantia_nigra"] * dy),
                         fontsize=7.8, style="italic", ha=ha, va="center",
                         color=WARM if g in ("GALC", "LRRC37A2") else GREY,
                         arrowprops=dict(arrowstyle="-", lw=0.6, shrinkA=1,
                                         shrinkB=3,
                                         color=WARM if g in ("GALC", "LRRC37A2")
                                         else GREY_L))
    ax2.set_xlim(0.25, 1.85)
    ax2.set_xlabel("how well healthy people tolerate\nlosing the gene")
    ax2.set_ylabel("expression in the\nsubstantia nigra")

    # d. the target table
    ax3 = fig.add_subplot(outer[2])
    order = [g for g in ga.sort_values("onset_min_p").index if g in d.index]
    CRIT = [
        ("acts on\nage at onset", lambda g: ga.loc[g, "onset_min_p"] < 1e-3),
        ("confidently\npredicted fold", lambda g: d.loc[g, "mean_plddt"] >= 70),
        ("has a\npocket", lambda g: (d.loc[g, "pocket_volume"] >= 300)
         and (d.loc[g, "mean_plddt"] >= 70)),
        ("pocket is the\ncatalytic site",
         lambda g: (d.loc[g, "site_pocket_rank"] == 1)
         and (d.loc[g, "mean_plddt"] >= 70)),
        ("a ligand has\nbeen bound", lambda g: d.loc[g, "n_pdb_ligand"] > 0),
        ("present in the\nsubstantia nigra",
         lambda g: d.loc[g, "tpm_substantia_nigra"] >= 5),
        ("loss is\ntolerated", lambda g: d.loc[g, "loeuf"] >= 0.6),
        ("agent in the\nclinic", lambda g: g in AGENT),
    ]
    AGENT = {"SNCA", "MAPT", "GPNMB", "IDUA", "GBA1", "LRRK2"}
    y = np.arange(len(order))[::-1]
    for yi, g in zip(y, order):
        for j, (_, fn) in enumerate(CRIT):
            try:
                v = bool(fn(g))
            except Exception:
                v = False
            ax3.add_patch(Rectangle((j + 0.10, yi - 0.36), 0.80, 0.72,
                                    facecolor=NAVY if v else GREY_XL,
                                    edgecolor="white", lw=1.0, zorder=3))
    ax3.set_xlim(0, len(CRIT)); ax3.set_ylim(-0.75, len(order) - 0.25)
    ax3.set_yticks(y); ax3.set_yticklabels(order, fontsize=8.2)
    for t, g in zip(ax3.get_yticklabels(), order):
        t.set_style("italic")
        if g in ("GALC", "LRRC37A2"):
            t.set_color(WARM)
    ax3.set_xticks(np.arange(len(CRIT)) + 0.5)
    ax3.set_xticklabels([c[0] for c in CRIT], fontsize=7.6)
    ax3.xaxis.set_ticks_position("top")
    ax3.tick_params(length=0)
    for s_ in ax3.spines.values():
        s_.set_visible(False)

    for a, s_, x in [(axes_a[0], "a", 0.012), (ax, "b", 0.012),
                     (ax2, "c", 0.615), (ax3, "d", 0.012)]:
        PL(fig, a, s_, x=x)
    save(fig, "C5_what_makes_a_target")


if __name__ == "__main__":
    main()
