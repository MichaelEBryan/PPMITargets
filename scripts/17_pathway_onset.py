#!/usr/bin/env python
"""
Do specific biological pathways govern onset age rather than risk?

This is the testable form of the claim that early- and late-onset Parkinson's
disease respond to different mechanisms. If the pathways carrying the onset
signal are the same ones carrying the risk signal, then onset is a dose effect on
one architecture and there is no separate target set. If some pathways carry
onset signal over and above their risk signal, those pathways are onset-specific
mechanisms and their druggable members are candidate targets.

Design. The 90 Nalls loci are independent by construction, so there is no LD
problem. For each locus we have a risk effect (META5, 1.5M participants) and an
onset effect (IPDGC age-at-onset GWAS, 28,568 cases). We assign each locus to
pathways, then run a competitive test: is the onset chi-square of loci in a
pathway larger than that of loci outside it, holding the risk chi-square fixed?
Significance comes from permuting pathway labels across loci, which preserves the
joint risk-onset distribution exactly.
"""
import os, re, json, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import paths as _P
warnings.filterwarnings("ignore")

BASE = str(_P.ROOT)
OUT, EXT = f"{BASE}/results", f"{BASE}/external"
NPERM = 20000
rng = np.random.default_rng(20260816)

CURATED = {
    "lysosomal / autophagy": [
        "GBA", "TMEM175", "SCARB2", "CTSB", "GALC", "ASAH1", "GRN", "VPS13C",
        "ATP6V0A1", "LAMP3", "TMEM163", "NEU1", "SMPD1", "CTSK", "GUSB", "HEXB",
        "NAGLU", "GPR65"],
    "endolysosomal trafficking": [
        "RAB7L1", "RAB29", "VPS35", "DNAJC6", "SYNJ1", "SH3GL2", "LRRK2", "BIN3",
        "VPS13C", "RIT2", "SNCA", "GAK"],
    "mitochondrial": [
        "PINK1", "PRKN", "PARK7", "NDUFAF2", "MCCC1", "CHCHD2", "ELOVL7",
        "FBXO7", "VPS13C", "MICU3"],
    "immune / inflammatory": [
        "HLA-DRB5", "HLA-DRB1", "HLA_DBQ1", "HLA-DQB1", "BST1", "LRRK2", "IL1R2",
        "MAP4K4", "ZNF184", "NOTCH4", "TNFSF9", "CD38", "P2RY12"],
    "synaptic / alpha-synuclein": [
        "SNCA", "RIT2", "SYT11", "SH3GL2", "BIN3", "STK39", "SYT4",
        "SCN3A", "SCN2A", "CAMK2D", "ANK2", "NSF"],
    "tau / microtubule": ["MAPT", "KANSL1", "CRHR1", "LRRC37A2", "LRRC37A", "SPPL2C",
                          "ARL17A", "ARL17B", "STH", "WNT3"],
    "dopamine metabolism": ["COMT", "GCH1", "DDC", "SLC6A3", "TH", "MAOB", "ALDH1A1"],
}


def load_loci():
    m = pd.read_csv(f"{OUT}/13_ppmi_vs_ipdgc_aao.csv")
    meta = pd.read_csv(f"{EXT}/risk_gwas/GCST009325.tsv", sep="\t",
                       usecols=["chromosome", "base_pair_location", "effect_allele",
                                "other_allele", "beta", "standard_error", "p_value"])
    meta["key"] = meta.chromosome.astype(str) + ":" + meta.base_pair_location.astype(str)
    m["key2"] = m.key.str.replace("chr", "", regex=False)
    j = m.merge(meta, left_on="key2", right_on="key", how="inner", suffixes=("", "_m5"))
    flip = j.effect_allele.str.upper() != j.Allele1.str.upper()
    j["risk_beta"] = np.where(flip, -j.beta, j.beta)
    j["risk_se"] = j.standard_error
    j["onset_beta"] = j.aao_beta_aligned
    j["onset_se"] = j.aao_se
    j["risk_chi2"] = (j.risk_beta / j.risk_se) ** 2
    j["onset_chi2"] = (j.onset_beta / j.onset_se) ** 2
    return j


def annotate_genes(j, window=100_000):
    """Positional annotation: every RefSeq gene whose body falls within `window`
    of the locus, plus the nearest gene. Complete for all loci, unlike a
    hand-curated label set."""
    g = pd.read_csv(f"{BASE}/data/genes_hg19.csv", dtype={"chr": str})
    near, within = [], []
    for _, r in j.iterrows():
        if pd.isna(r.pos37) or pd.isna(r.chr37):
            near.append(None); within.append(set()); continue
        gc = g[g.chr == str(int(r.chr37))]
        pos = int(r.pos37)
        hit = gc[(gc.start - window <= pos) & (gc.end + window >= pos)]
        within.append(set(hit.gene))
        d = np.minimum(np.abs(gc.start - pos), np.abs(gc.end - pos))
        d = np.where((gc.start <= pos) & (gc.end >= pos), 0, d)
        near.append(gc.gene.iloc[int(np.argmin(d))] if len(gc) else None)
    j["gene"] = near
    j["genes_in_window"] = [";".join(sorted(w)) for w in within]
    j["_within"] = within
    print(f"  annotated {j.gene.notna().sum()} of {len(j)} loci "
          f"(median {np.median([len(w) for w in within]):.0f} genes within "
          f"{window//1000} kb)")
    return j


def load_genesets():
    sets = {}
    for lib in ["KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2023",
                "GO_Cellular_Component_2023", "WikiPathway_2023_Human"]:
        p = f"{EXT}/genesets/{lib}.txt"
        if not os.path.exists(p):
            continue
        for line in open(p):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = {g.split(",")[0].strip() for g in parts[2:] if g.strip()}
            if 5 <= len(genes) <= 800:
                sets[f"{lib}|{name}"] = genes
    return sets


def competitive_test(j, member, label):
    y = np.log1p(j.onset_chi2.values)
    x = np.log1p(j.risk_chi2.values)
    g = member.astype(float)
    if g.sum() < 3 or g.sum() > len(g) - 3:
        return None
    obs = sm.OLS(y, sm.add_constant(np.column_stack([g, x]))).fit().params[1]
    Xc = sm.add_constant(np.column_stack([np.zeros_like(g), x]))
    null = np.empty(NPERM)
    for i in range(NPERM):
        Xc[:, 1] = g[rng.permutation(len(g))]
        null[i] = sm.OLS(y, Xc).fit().params[1]
    p = (np.sum(null >= obs) + 1) / (NPERM + 1)
    mb = member.astype(bool)
    return dict(pathway=label, n_loci=int(mb.sum()), beta=float(obs),
                perm_p=float(p), z=float((obs - null.mean()) / null.std()),
                onset_chi2_in=float(j.onset_chi2[mb].mean()),
                onset_chi2_out=float(j.onset_chi2[~mb].mean()),
                risk_chi2_in=float(j.risk_chi2[mb].mean()),
                risk_chi2_out=float(j.risk_chi2[~mb].mean()))


def match(j, gs):
    return np.array([bool(w & gs) for w in j._within])


def main():
    j = annotate_genes(load_loci())
    print(f"{len(j)} independent loci with aligned risk and onset effects")
    print(f"  onset chi2 median {j.onset_chi2.median():.2f} max {j.onset_chi2.max():.1f}")
    print(f"  risk  chi2 median {j.risk_chi2.median():.2f} max {j.risk_chi2.max():.1f}\n")

    rows = []
    for name, genes in CURATED.items():
        mb = match(j, set(genes))
        r = competitive_test(j, mb, name)
        if r:
            r["genes"] = ";".join(sorted({x for w in j._within[mb] for x in (w & set(genes))}))
            rows.append(r)
            print(f"{name:30s} n={r['n_loci']:3d} beta={r['beta']:+.3f} z={r['z']:+.2f} "
                  f"p={r['perm_p']:.4f}  onset chi2 in/out "
                  f"{r['onset_chi2_in']:.2f}/{r['onset_chi2_out']:.2f}", flush=True)
    cur = pd.DataFrame(rows)
    if len(cur):
        cur["fdr"] = sm.stats.multipletests(cur.perm_p, method="fdr_bh")[1]
        cur.sort_values("perm_p").to_csv(f"{OUT}/60_pathway_curated.csv", index=False)

    sets = load_genesets()
    print(f"\n{len(sets)} public gene sets loaded", flush=True)
    present = {x for w in j._within for x in w}
    rows = []
    for name, gs in sets.items():
        if len(gs & present) < 4:
            continue
        mb = match(j, gs)
        if mb.sum() < 4:
            continue
        r = competitive_test(j, mb, name)
        if r:
            r["genes"] = ";".join(sorted({x for w in j._within[mb] for x in (w & gs)})[:25])
            rows.append(r)
    sw = pd.DataFrame(rows)
    if len(sw):
        sw["fdr"] = sm.stats.multipletests(sw.perm_p, method="fdr_bh")[1]
        sw = sw.sort_values("perm_p")
        sw.to_csv(f"{OUT}/61_pathway_sweep.csv", index=False)
        print(f"tested {len(sw)} gene sets with >=4 panel loci; top hits:")
        for _, r in sw.head(12).iterrows():
            nm = r.pathway.split("|", 1)[1]
            print(f"  {nm[:56]:56s} n={int(r.n_loci):2d} z={r.z:+.2f} "
                  f"p={r.perm_p:.4f} fdr={r.fdr:.3f}")
        print(f"\nsets surviving FDR 0.05: {int((sw.fdr < 0.05).sum())}")

    j.drop(columns=["_within"]).to_csv(f"{OUT}/62_loci_risk_onset.csv", index=False)
    json.dump(dict(n_loci=int(len(j)),
                   n_sets_tested=int(len(sw)) if len(sw) else 0,
                   n_sets_fdr05=int((sw.fdr < 0.05).sum()) if len(sw) else 0,
                   curated=cur.sort_values("perm_p").to_dict("records") if len(cur) else []),
              open(f"{OUT}/63_pathway_summary.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/63_pathway_summary.json")


if __name__ == "__main__":
    main()
