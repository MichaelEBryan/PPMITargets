#!/usr/bin/env python
"""
Can a single SNP nominate LRRC37A2 over MAPT?

The paper's early-onset target is LRRC37A2, chosen because the variant that
ranked highest falls inside that gene and the gene sits "near the MAPT locus".
17q21.31 carries a ~900 kb polymorphic inversion (H1/H2) in which recombination
is suppressed, so every common variant across the block tags the same haplotype.
If that is so, no single-variant association can distinguish LRRC37A2 from MAPT,
KANSL1, ARL17A/B, NSF or CRHR1 -- they are one statistical unit.

We measure it directly: pairwise r^2 across the block in 1000 Genomes
(GRCh38, 30x), in each superpopulation, and specifically between the panel's
LRRC37A2-annotated variant and the panel's MAPT-annotated variant.
"""
import os, json, gzip, numpy as np, pandas as pd
import paths as _P

BASE = str(_P.ROOT)
VCF = f"{BASE}/external/1kg/chr17_MAPT.vcf.gz"
OUT = f"{BASE}/results"

# GRCh38 coordinates
LRRC37A2_SNP = ("rs62053943", 45666837)   # inside LRRC37A2 -> the paper's target
MAPT_SNP = ("rs17649553", 45917282)       # PPMI panel's MAPT-annotated variant
# 17q21.31 inversion breakpoints, GRCh38 (approximate, Steinberg 2012 / Boettger 2012)
INV = (45_600_000, 46_200_000)
GENES38 = {  # GRCh38 spans
    "KANSL1":    (46_029_916, 46_225_403),
    "MAPT":      (45_894_382, 46_028_334),
    "MAPT-AS1":  (45_875_000, 45_897_000),
    "LRRC37A":   (46_360_000, 46_420_000),
    "LRRC37A2":  (45_619_000, 45_674_000),
    "ARL17A":    (46_274_000, 46_357_000),
    "ARL17B":    (46_197_000, 46_275_000),
    "NSF":       (46_596_000, 46_762_000),
    "CRHR1":     (45_784_000, 45_895_000),
    "SPPL2C":    (45_878_000, 45_884_000),
    "PLEKHM1":   (45_460_000, 45_524_000),
}


def read_vcf(path):
    ids, pos, gts = [], [], []
    samples = None
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            p = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                samples = p[9:]
                continue
            fmt_gt = p[8].split(":").index("GT")
            row = np.empty(len(p) - 9, dtype=np.int8)
            bad = False
            for i, s in enumerate(p[9:]):
                g = s.split(":")[fmt_gt]
                a = g.replace("|", "/").split("/")
                try:
                    row[i] = int(a[0]) + int(a[1])
                except Exception:
                    row[i] = -1
                    bad = True
            ids.append(p[2]); pos.append(int(p[1])); gts.append(row)
    G = np.vstack(gts)
    return np.array(ids), np.array(pos), G, samples


def main():
    ids, pos, G, samples = read_vcf(VCF)
    print(f"loaded {G.shape[0]} biallelic SNPs x {G.shape[1]} samples "
          f"({pos.min():,}-{pos.max():,} GRCh38)", flush=True)

    ped = pd.read_csv(f"{BASE}/external/1kg/ped.txt", sep=r"\s+")
    pops = dict(zip(ped.SampleID, ped.Superpopulation))
    unrel = set(ped.loc[(ped.FatherID.astype(str) == "0") &
                        (ped.MotherID.astype(str) == "0"), "SampleID"])

    def r2(a, b, keep):
        x, y = G[a][keep].astype(float), G[b][keep].astype(float)
        m = (x >= 0) & (y >= 0)
        if m.sum() < 30 or x[m].std() == 0 or y[m].std() == 0:
            return np.nan
        return float(np.corrcoef(x[m], y[m])[0, 1] ** 2)

    def nearest(target_pos):
        return int(np.argmin(np.abs(pos - target_pos)))

    sp = np.array([pops.get(s, "NA") for s in samples])
    keep_unrel = np.array([s in unrel for s in samples])
    groups = {"ALL_unrelated": keep_unrel}
    for p in ("EUR", "AFR", "EAS", "SAS", "AMR"):
        g = (sp == p) & keep_unrel
        if g.sum() > 50:
            groups[p] = g
    print("sample groups:", {k: int(v.sum()) for k, v in groups.items()}, flush=True)

    # the specific question
    iA = np.where(ids == LRRC37A2_SNP[0])[0]
    iB = np.where(ids == MAPT_SNP[0])[0]
    iA = int(iA[0]) if len(iA) else nearest(LRRC37A2_SNP[1])
    iB = int(iB[0]) if len(iB) else nearest(MAPT_SNP[1])
    res = {"lrrc37a2_variant": dict(requested=LRRC37A2_SNP[0], used=str(ids[iA]),
                                    pos38=int(pos[iA])),
           "mapt_variant": dict(requested=MAPT_SNP[0], used=str(ids[iB]), pos38=int(pos[iB])),
           "r2_between_them": {k: r2(iA, iB, v) for k, v in groups.items()},
           "distance_bp": int(abs(pos[iA] - pos[iB]))}
    print(f"\nr2({ids[iA]} [LRRC37A2] , {ids[iB]} [MAPT]) "
          f"= {res['r2_between_them']}", flush=True)

    # how far does that LD extend?
    keep = groups.get("EUR", groups["ALL_unrelated"])
    Gk = np.where(G[:, keep] >= 0, G[:, keep], np.nan).astype(float)
    af = np.nanmean(Gk, axis=1) / 2
    common = np.where((af > 0.05) & (af < 0.95))[0]
    r2A = np.array([r2(iA, j, keep) for j in common])
    tag = common[r2A > 0.8]
    res["n_common_snps_in_window"] = int(len(common))
    res["n_snps_r2_gt_0.8_with_LRRC37A2_variant"] = int(len(tag))
    res["r2_gt_0.8_span_bp"] = int(pos[tag].max() - pos[tag].min()) if len(tag) else 0
    res["r2_gt_0.8_span"] = (f"chr17:{pos[tag].min():,}-{pos[tag].max():,}"
                             if len(tag) else None)

    genes_tagged = []
    if len(tag):
        lo, hi = pos[tag].min(), pos[tag].max()
        for g, (s, e) in GENES38.items():
            if not (e < lo or s > hi):
                genes_tagged.append(g)
    res["genes_inside_r2_gt_0.8_block"] = sorted(genes_tagged)
    print(f"\n{len(tag)} of {len(common)} common SNPs are at r2>0.8 with the "
          f"LRRC37A2 variant, spanning {res['r2_gt_0.8_span']} "
          f"({res['r2_gt_0.8_span_bp']/1e3:.0f} kb)", flush=True)
    print(f"protein-coding genes inside that block: {', '.join(genes_tagged)}", flush=True)

    # LD matrix for the figure
    sel = common[::max(1, len(common) // 400)]
    M = np.full((len(sel), len(sel)), np.nan)
    for a in range(len(sel)):
        for b in range(a, len(sel)):
            v = r2(sel[a], sel[b], keep)
            M[a, b] = M[b, a] = v
    np.save(f"{OUT}/30_ld_matrix.npy", M)
    np.save(f"{OUT}/30_ld_positions.npy", pos[sel])
    pd.DataFrame(dict(rsid=ids[sel], pos38=pos[sel])).to_csv(f"{OUT}/30_ld_snps.csv", index=False)
    json.dump(res, open(f"{OUT}/31_mapt_ld_summary.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/31_mapt_ld_summary.json", flush=True)


if __name__ == "__main__":
    main()
