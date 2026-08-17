#!/usr/bin/env python
"""
Genome-wide gene-based analysis of onset age, and of risk, then the contrast.

The 86-locus test is limited to loci already known to affect risk. This arm uses
the whole age-at-onset GWAS (6.8M variants) and the whole risk GWAS, computes a
gene statistic for every RefSeq gene, calibrates it against genes with the same
number of variants, and then runs a competitive gene-set test. It can find
pathways with no representative among the 90 risk loci.

Gene statistic is the minimum p in the window. Because min-p depends strongly on
how many variants a gene contains, it is calibrated empirically: genes are binned
by variant count and each gene's min-p is converted to its rank within its bin.
That removes the gene-size confound without needing an LD reference.
"""
import os, json, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import paths as _P
warnings.filterwarnings("ignore")

BASE = str(_P.ROOT)
OUT, EXT = f"{BASE}/results", f"{BASE}/external"
WINDOW = 35_000
NPERM = 10000
rng = np.random.default_rng(7)


def load_aao():
    d = pd.read_csv(f"{EXT}/aao_gwas/IPDGC_AAO_GWAS_sumstats_april_2018.txt",
                    sep="\t", usecols=["MarkerName", "Effect", "StdErr", "P-value"])
    cp = d.MarkerName.str.split(":", expand=True)
    d["chr"] = cp[0].str.replace("chr", "", regex=False)
    d["pos"] = pd.to_numeric(cp[1], errors="coerce")
    d = d.rename(columns={"P-value": "p"})
    return d[["chr", "pos", "p", "Effect", "StdErr"]].dropna()


def load_risk():
    d = pd.read_csv(f"{EXT}/risk_gwas/GCST009325.tsv", sep="\t",
                    usecols=["chromosome", "base_pair_location", "p_value", "beta",
                             "standard_error"])
    d = d.rename(columns={"chromosome": "chr", "base_pair_location": "pos",
                          "p_value": "p", "beta": "Effect",
                          "standard_error": "StdErr"})
    d["chr"] = d.chr.astype(str)
    return d.dropna()


def gene_stats(gw, genes, tag):
    rows = []
    for ch, gc in genes.groupby("chr"):
        s = gw[gw.chr == ch]
        if not len(s):
            continue
        s = s.sort_values("pos")
        pos = s.pos.values
        pv = s.p.values.astype(float)
        chi = stats.chi2.isf(np.clip(pv, 1e-300, 1), 1)
        for _, g in gc.iterrows():
            lo = np.searchsorted(pos, g.start - WINDOW, "left")
            hi = np.searchsorted(pos, g.end + WINDOW, "right")
            n = hi - lo
            if n < 5:
                continue
            seg_p = pv[lo:hi]
            seg_c = chi[lo:hi]
            rows.append((g.gene, ch, int(g.start), int(g.end), int(n),
                         float(seg_p.min()), float(seg_c.mean()),
                         float(np.sort(seg_c)[-min(10, n):].mean())))
    d = pd.DataFrame(rows, columns=["gene", "chr", "start", "end", "n_snps",
                                    "min_p", "mean_chi2", "top10_chi2"])
    # empirical calibration against genes of the same variant count
    d["bin"] = pd.qcut(np.log10(d.n_snps), 30, duplicates="drop")
    d["calib_p"] = d.groupby("bin", observed=True).min_p.rank(pct=True)
    d["z"] = stats.norm.isf(np.clip(d.calib_p, 1e-6, 1 - 1e-6))
    d = d.drop(columns=["bin"])
    d.to_csv(f"{OUT}/70_genestats_{tag}.csv", index=False)
    print(f"  {tag}: {len(d)} genes, median {d.n_snps.median():.0f} variants/gene", flush=True)
    return d


def load_genesets(present):
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
            gs = {g.split(",")[0].strip() for g in parts[2:] if g.strip()} & present
            if 10 <= len(gs) <= 500:
                sets[f"{lib}|{parts[0]}"] = gs
    return sets


def competitive(d, gs, adjust_col=None):
    """Is z higher in-set than out-of-set, adjusting for variant count and,
    optionally, for the same gene's statistic in the other GWAS?"""
    m = d.gene.isin(gs).values.astype(float)
    if m.sum() < 10:
        return None
    cols = [np.log10(d.n_snps.values)]
    if adjust_col is not None:
        cols.append(d[adjust_col].values)
    X = sm.add_constant(np.column_stack([m] + cols))
    fit = sm.OLS(d.z.values, X).fit()
    return dict(n_genes=int(m.sum()), beta=float(fit.params[1]),
                se=float(fit.bse[1]), p_onesided=float(fit.pvalues[1] / 2
                                                       if fit.params[1] > 0 else
                                                       1 - fit.pvalues[1] / 2),
                mean_z_in=float(d.z[m.astype(bool)].mean()),
                mean_z_out=float(d.z[~m.astype(bool)].mean()))


def main():
    genes = pd.read_csv(f"{BASE}/data/genes_hg19.csv", dtype={"chr": str})
    genes = genes[genes.chr.isin([str(i) for i in range(1, 23)])]
    print(f"{len(genes)} autosomal genes", flush=True)

    print("loading AAO GWAS ...", flush=True)
    ga = gene_stats(load_aao(), genes, "aao")
    print("loading risk GWAS ...", flush=True)
    gr = gene_stats(load_risk(), genes, "risk")

    j = ga.merge(gr[["gene", "z", "n_snps", "min_p"]], on="gene",
                 suffixes=("", "_risk"))
    j.to_csv(f"{OUT}/71_genestats_joined.csv", index=False)
    r, p = stats.pearsonr(j.z, j.z_risk)
    print(f"\n{len(j)} genes in both. Gene-level onset z vs risk z: r={r:.3f} p={p:.2g}",
          flush=True)

    print("\ntop genes for onset (genome-wide, calibrated):", flush=True)
    for _, x in ga.sort_values("min_p").head(20).iterrows():
        print(f"  {x.gene:14s} chr{x.chr}:{x.start:,} n={int(x.n_snps):5d} "
              f"min_p={x.min_p:.2e} z={x.z:+.2f}")

    present = set(j.gene)
    sets = load_genesets(present)
    print(f"\n{len(sets)} gene sets with 10-500 mapped genes", flush=True)

    rows = []
    for name, gs in sets.items():
        a = competitive(j, gs)                       # onset signal
        b = competitive(j.assign(z=j.z_risk), gs)    # risk signal
        c = competitive(j, gs, adjust_col="z_risk")  # onset over and above risk
        if a and b and c:
            rows.append(dict(pathway=name, n_genes=a["n_genes"],
                             onset_beta=a["beta"], onset_p=a["p_onesided"],
                             risk_beta=b["beta"], risk_p=b["p_onesided"],
                             onset_adj_beta=c["beta"], onset_adj_p=c["p_onesided"],
                             mean_z_onset_in=a["mean_z_in"],
                             mean_z_risk_in=b["mean_z_in"]))
    s = pd.DataFrame(rows)
    for col in ["onset_p", "risk_p", "onset_adj_p"]:
        s[col.replace("_p", "_fdr")] = sm.stats.multipletests(s[col], method="fdr_bh")[1]
    s = s.sort_values("onset_adj_p")
    s.to_csv(f"{OUT}/72_geneset_onset_vs_risk.csv", index=False)

    print("\n=== gene sets carrying onset signal over and above risk signal ===")
    for _, x in s.head(25).iterrows():
        nm = x.pathway.split("|", 1)[1]
        print(f"  {nm[:56]:56s} n={int(x.n_genes):3d} b={x.onset_adj_beta:+.3f} "
              f"p={x.onset_adj_p:.2e} fdr={x.onset_adj_fdr:.3f}")
    print(f"\nsurviving FDR 0.05: {int((s.onset_adj_fdr < 0.05).sum())} of {len(s)}")

    lyso = [k for k in sets if "ysosom" in k or "Lytic Vacuole" in k or "utophag" in k]
    print("\n=== lysosomal and autophagy sets specifically ===")
    for k in lyso[:20]:
        row = s[s.pathway == k]
        if len(row):
            x = row.iloc[0]
            print(f"  {k.split('|')[1][:52]:52s} n={int(x.n_genes):3d} "
                  f"onset p={x.onset_p:.3f} risk p={x.risk_p:.3f} "
                  f"onset|risk p={x.onset_adj_p:.3f}")

    json.dump(dict(n_genes=int(len(j)), gene_z_corr=float(r),
                   n_sets=int(len(s)), n_fdr05=int((s.onset_adj_fdr < 0.05).sum()),
                   top=s.head(30).to_dict("records")),
              open(f"{OUT}/73_geneset_summary.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/72_geneset_onset_vs_risk.csv")


if __name__ == "__main__":
    main()
