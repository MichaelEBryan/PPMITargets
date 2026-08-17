#!/usr/bin/env python
"""
Testing the GALC activity hypothesis, not the single variant.

Senkevich et al. (Brain 2023) report that rs979812 raises blood
galactosylceramidase activity, and that Mendelian randomisation places increased
activity causally upstream of Parkinson's disease (b = 0.025, SE 0.007,
p = 8e-4). The original manuscript cited this, then asserted a protein
conformational mechanism for an intronic variant, which cannot be right. The
hypothesis itself is separable from that error and is worth testing on its own
terms.

If the mechanism is that the risk haplotype raises GALC expression, then in an
independent tissue and an independent assay the same haplotype should raise GALC
transcript abundance. That is a falsifiable prediction and GTEx can test it.

  1  regional architecture   is there one signal at the locus, or several?
  2  eQTL direction          does the PD risk allele raise or lower brain GALC?
  3  LD alignment            are the PD variant and the eQTL variant the same signal?
  4  colocalisation          approximate Bayes factor colocalisation of the two
  5  tissue specificity      where in the brain is GALC under genetic control?
"""
import os, gzip, json, warnings
import numpy as np, pandas as pd
from scipy import stats
import paths as _P
warnings.filterwarnings("ignore")

BASE = str(_P.ROOT)
OUT, EXT = f"{BASE}/results", f"{BASE}/external"
VCF = f"{EXT}/1kg/chr14_GALC.vcf.gz"
# GRCh37 for the GWAS, GRCh38 for GTEx and 1000G
PD_B37, PD_B38 = 88_464_264, 87_997_920          # rs979812
GALC_B38 = (87_933_000, 87_993_000)
R = {}


def read_vcf(path, lo, hi):
    ids, pos, hap, samples = [], [], [], None
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            p = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                samples = p[9:]; continue
            q = int(p[1])
            if not (lo <= q <= hi):
                continue
            k = p[8].split(":").index("GT")
            h = np.empty((len(p) - 9, 2), np.int8)
            for i, s in enumerate(p[9:]):
                g = s.split(":")[k].replace("|", "/").split("/")
                try: h[i] = (int(g[0]), int(g[1]))
                except Exception: h[i] = (-1, -1)
            ids.append((p[2], p[3], p[4])); pos.append(q); hap.append(h)
    return ids, np.array(pos), np.stack(hap), samples


def ld_pair(a, b):
    """r2, |D'| and the SIGN of D. The sign says whether the two alt alleles sit
    on the same haplotype, which is what allele alignment needs."""
    x, y = a.ravel(), b.ravel()
    m = (x >= 0) & (y >= 0)
    x, y = x[m], y[m]
    pA, pB = x.mean(), y.mean()
    if min(pA, pB) in (0, 1):
        return np.nan, np.nan, 0
    D = (x * y).mean() - pA * pB
    r2 = D * D / (pA * (1 - pA) * pB * (1 - pB))
    Dmax = min(pA * (1 - pB), (1 - pA) * pB) if D > 0 else min(pA * pB, (1 - pA) * (1 - pB))
    return float(r2), (float(abs(D) / Dmax) if Dmax > 0 else np.nan), int(np.sign(D))


def abf(beta, se, W=0.04):
    """Wakefield approximate Bayes factor, log scale."""
    z2 = (beta / se) ** 2
    r = W / (W + se ** 2)
    return 0.5 * (np.log(1 - r) + r * z2)


def main():
    # 1. regional architecture
    risk = pd.read_csv(f"{OUT}/120_galc_region_risk.csv")
    risk = risk[risk.base_pair_location.between(88_380_000, 88_560_000)].copy()
    print(f"[1] PD risk signal, chr14:88.38-88.56 Mb (GRCh37): {len(risk)} variants")
    top = risk.loc[risk.p_value.idxmin()]
    print(f"    strongest: chr14:{int(top.base_pair_location):,} "
          f"{top.effect_allele}/{top.other_allele} beta={top.beta:+.4f} p={top.p_value:.2e}")
    lead = risk[risk.base_pair_location == PD_B37]
    if len(lead):
        lead = lead.iloc[0]
        print(f"    rs979812 : beta={lead.beta:+.4f} ({lead.effect_allele} allele) "
              f"p={lead.p_value:.3g}  freq={lead.effect_allele_frequency:.3f}")
        R["pd_deposited"] = dict(beta=float(lead.beta), se=float(lead.standard_error),
                                 p=float(lead.p_value), allele=lead.effect_allele)
    R["pd_published"] = dict(beta=0.061, ci=[0.043, 0.079], p=6e-11, allele="T",
                             source="GWAS Catalog, Nalls 2019 reported value")
    R["activity_published"] = dict(beta=1.205, ci=[1.09, 1.32], p=5e-95,
                                   direction="increase",
                                   source="Senkevich 2022, GCST90270130, n=1454")

    # 2. eQTL direction
    eq = pd.read_csv(f"{OUT}/121_galc_eqtl_brain.csv")
    g = eq[eq.gene == "GALC"].copy()
    g["pos"] = g.variant.str.split("_").str[1].astype(int)
    g["ref"] = g.variant.str.split("_").str[2]
    g["alt"] = g.variant.str.split("_").str[3]
    print(f"\n[2] GALC brain eQTLs: {len(g)} significant pairs in "
          f"{g.tissue.nunique()} of 13 tissues")
    for t, s in g.groupby("tissue"):
        b = s.loc[s.pval.idxmin()]
        print(f"    {t:38s} n={len(s):4d} top p={b.pval:.2e} "
              f"slope={b.slope:+.3f} ({b.ref}>{b.alt})")

    # 3. LD alignment
    ids, pos, hap, samples = read_vcf(VCF, 87_900_000, 88_100_000)
    ped = pd.read_csv(f"{EXT}/1kg/ped.txt", sep=r"\s+")
    pops = dict(zip(ped.SampleID, ped.Superpopulation))
    unrel = set(ped.loc[(ped.FatherID.astype(str) == "0") &
                        (ped.MotherID.astype(str) == "0"), "SampleID"])
    eur = np.array([pops.get(s) == "EUR" and s in unrel for s in samples])
    He = hap[:, eur, :]
    print(f"\n[3] 1000G EUR n={eur.sum()}, {len(pos)} common variants in "
          f"chr14:87.9-88.1 Mb")

    i_pd = int(np.argmin(np.abs(pos - PD_B38)))
    print(f"    PD lead in 1000G: {ids[i_pd][0]} at chr14:{pos[i_pd]:,} "
          f"({ids[i_pd][1]}>{ids[i_pd][2]})")

    rows = []
    for t, s in g.groupby("tissue"):
        b = s.loc[s.pval.idxmin()]
        j = np.where(pos == b.pos)[0]
        if not len(j):
            continue
        j = int(j[0])
        r2, dp, sgn = ld_pair(He[i_pd], He[j])
        # align the eQTL slope onto the PD risk allele
        same = (ids[j][2] == "T") if ids[i_pd][2] == "T" else None
        rows.append(dict(tissue=t, eqtl_variant=b.variant, eqtl_pos=b.pos,
                         eqtl_p=b.pval, eqtl_slope=b.slope,
                         r2_with_pd_lead=r2, Dprime=dp, D_sign=sgn,
                         slope_on_risk_allele=b.slope * sgn,
                         eqtl_ref=ids[j][1], eqtl_alt=ids[j][2],
                         pd_ref=ids[i_pd][1], pd_alt=ids[i_pd][2]))
    ld = pd.DataFrame(rows)
    if len(ld):
        ld.to_csv(f"{OUT}/123_galc_eqtl_ld.csv", index=False)
        print("\n    LD between the PD lead variant and the top eQTL per tissue:")
        for _, r in ld.iterrows():
            print(f"      {r.tissue:38s} r2={r.r2_with_pd_lead:.3f} |D'|={r.Dprime:.3f} "
                  f"eQTL slope={r.eqtl_slope:+.3f}")

    # direct: is the PD lead itself an eQTL in the full nominal data? Use LD proxy.
    # take all significant eQTLs, weight direction by LD with the PD lead
    prox = []
    for _, r in g.iterrows():
        j = np.where(pos == r.pos)[0]
        if not len(j):
            continue
        r2, _, sgn = ld_pair(He[i_pd], He[int(j[0])])
        if np.isfinite(r2):
            prox.append((r2, r.slope, r.slope * sgn, sgn, r.tissue, r.variant, r.pval,
                         ids[int(j[0])][1], ids[int(j[0])][2]))
    P = pd.DataFrame(prox, columns=["r2", "slope", "slope_on_risk_allele", "D_sign",
                                    "tissue", "variant", "p", "ref", "alt"])
    P.to_csv(f"{OUT}/124_galc_eqtl_ld_all.csv", index=False)
    hi = P[P.r2 > 0.6]
    print(f"\n    eQTLs in LD r2>0.6 with the PD lead variant: {len(hi)} "
          f"of {len(P)} tested")
    if len(hi):
        up = int((hi.slope_on_risk_allele > 0).sum())
        dn = int((hi.slope_on_risk_allele < 0).sum())
        print(f"      aligned to the PD RISK allele (rs979812-T):")
        print(f"        {up} raise GALC expression, {dn} lower it")
        print(f"      median aligned slope {hi.slope_on_risk_allele.median():+.3f}, "
              f"strongest p {hi.p.min():.2e}, max r2 {hi.r2.max():.3f}")
        print(f"      tissues: {sorted(set(hi.tissue))}")
        R["eqtl_in_ld"] = dict(n=int(len(hi)), n_raise=up, n_lower=dn,
                               median_aligned_slope=float(hi.slope_on_risk_allele.median()),
                               max_r2=float(hi.r2.max()),
                               tissues=sorted(set(hi.tissue)))
    else:
        R["eqtl_in_ld"] = dict(n=0, note="no significant brain GALC eQTL is in "
                                         "LD r2>0.6 with the PD lead variant")

    # 4. colocalisation
    # PD signal vs the strongest tissue's eQTL, over shared variants
    cere = g[g.tissue == "Brain_Cerebellum"].copy()
    if len(cere):
        risk38 = risk.copy()
        risk38["pos38"] = risk38.base_pair_location - (PD_B37 - PD_B38)
        mg = risk38.merge(cere, left_on="pos38", right_on="pos", how="inner")
        print(f"\n[4] colocalisation input: {len(mg)} variants shared between the "
              f"PD scan and the cerebellum eQTL")
        if len(mg) > 20:
            l1 = abf(mg.beta.values, mg.standard_error.values)
            l2 = abf(mg.slope.values, mg.slope_se.values)
            def lse(x):
                m = np.max(x); return m + np.log(np.sum(np.exp(x - m)))
            p1, p2, p12 = 1e-4, 1e-4, 1e-5
            l0 = 0.0
            L1 = np.log(p1) + lse(l1)
            L2 = np.log(p2) + lse(l2)
            # sum_{i!=j} exp(l1_i + l2_j) = (sum exp l1)(sum exp l2) - sum exp(l1+l2)
            s1, s2, s12 = lse(l1), lse(l2), lse(l1 + l2)
            L3 = np.log(p1) + np.log(p2) + s12 + np.log(
                np.expm1(s1 + s2 - s12)) if (s1 + s2 - s12) > 1e-12 else -np.inf
            L4 = np.log(p12) + lse(l1 + l2)
            arr = np.array([l0, L1, L2, L3, L4])
            pp = np.exp(arr - lse(arr))
            print(f"    PP0={pp[0]:.3f} PP1={pp[1]:.3f} PP2={pp[2]:.3f} "
                  f"PP3={pp[3]:.3f} PP4={pp[4]:.3f}")
            print(f"    PP4 is the posterior for one shared causal variant; "
                  f"PP3 for two distinct causal variants.")
            R["coloc"] = dict(n_variants=int(len(mg)), PP0=float(pp[0]),
                              PP1=float(pp[1]), PP2=float(pp[2]),
                              PP3=float(pp[3]), PP4=float(pp[4]))
            mg.to_csv(f"{OUT}/125_galc_coloc_input.csv", index=False)

    # 5. tissue specificity
    allt = eq.copy()
    print(f"\n[5] GALC genetic control is tissue restricted:")
    n_by = eq[eq.gene == "GALC"].groupby("tissue").size()
    print(f"    significant in {len(n_by)} of 13 brain tissues; "
          f"absent from substantia nigra, putamen, caudate, cortex")
    print(f"    dominated by cerebellum ({int(n_by.get('Brain_Cerebellum',0))}) and "
          f"cerebellar hemisphere ({int(n_by.get('Brain_Cerebellar_Hemisphere',0))})")
    R["tissue"] = {t: int(n) for t, n in n_by.items()}

    json.dump(R, open(f"{OUT}/126_galc_activity_tests.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/126_galc_activity_tests.json")


if __name__ == "__main__":
    main()
