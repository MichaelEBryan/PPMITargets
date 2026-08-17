#!/usr/bin/env python
"""
Gene-set analysis of onset age with a null that respects genomic clustering.

Neighbouring genes carry correlated statistics, and gene sets are themselves
spatially clustered (the APOE cluster contributes four genes to any lipoprotein
set). Permuting set membership at random breaks that structure and makes
clustered sets look significant.

Instead the membership vector is circularly shifted along the genome. Genes are
ordered by position and concatenated; a random offset rotates which genes are
labelled as belonging to the set. That preserves both the spatial
autocorrelation of the statistics and the clustering of the set, so a set whose
apparent signal comes only from sitting on top of one strong region is no longer
significant.

Reported for each set: onset signal, risk signal, and onset conditional on risk.
"""
import os, json, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from joblib import Parallel, delayed
import paths as _P
warnings.filterwarnings("ignore")

BASE = str(_P.ROOT)
OUT, EXT = f"{BASE}/results", f"{BASE}/external"
NSHIFT = 2000
NJOBS = 10


def load():
    j = pd.read_csv(f"{OUT}/71_genestats_joined.csv", dtype={"chr": str})
    j["chr_i"] = j.chr.astype(int)
    j = j.sort_values(["chr_i", "start"]).reset_index(drop=True)
    return j


def load_sets(present):
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


def n_clusters(pos_chr, pos_start, mb, gap=1_000_000):
    """Independent genomic clusters occupied by the set."""
    p = sorted(zip(pos_chr[mb], pos_start[mb]))
    n, last = 0, None
    for c, s in p:
        if last is None or c != last[0] or s - last[1] > gap:
            n += 1
        last = (c, s)
    return n


def one(name, gs, genes, Y, X0, chr_i, start, seed):
    mb = genes.isin(gs).values
    n = int(mb.sum())
    if n < 10:
        return None

    def coef(m):
        X = np.column_stack([X0, m.astype(float)])
        return sm.OLS(Y, X).fit().params[-1]

    obs = coef(mb)
    rng = np.random.default_rng(seed)
    N = len(mb)
    null = np.empty(NSHIFT)
    for i in range(NSHIFT):
        null[i] = coef(np.roll(mb, int(rng.integers(1, N))))
    p = (np.sum(null >= obs) + 1) / (NSHIFT + 1)
    return dict(pathway=name, n_genes=n,
                n_clusters=n_clusters(chr_i, start, mb),
                beta=float(obs), shift_p=float(p),
                z=float((obs - null.mean()) / (null.std() + 1e-12)),
                mean_z_in=float(Y[mb].mean()), mean_z_out=float(Y[~mb].mean()))


def main():
    j = load()
    print(f"{len(j)} genes ordered along the genome", flush=True)
    Y = j.z.values
    base = np.column_stack([np.ones(len(j)), np.log10(j.n_snps.values), j.z_risk.values])
    present = set(j.gene)
    sets = load_sets(present)
    print(f"{len(sets)} gene sets with 10-500 mapped genes", flush=True)

    res = Parallel(n_jobs=NJOBS, verbose=1)(
        delayed(one)(k, v, j.gene, Y, base, j.chr_i.values, j.start.values, 900 + i)
        for i, (k, v) in enumerate(sets.items()))
    s = pd.DataFrame([r for r in res if r])
    s["fdr"] = sm.stats.multipletests(s.shift_p, method="fdr_bh")[1]
    s = s.sort_values("shift_p")
    s.to_csv(f"{OUT}/103_circshift_geneset.csv", index=False)

    print(f"\ntested {len(s)} sets with a position-preserving null")
    print("=== onset signal beyond risk ===")
    print(f"{'pathway':56s} {'genes':>5s} {'clus':>5s} {'z':>6s} {'p':>8s} {'fdr':>6s}")
    for _, r in s.head(25).iterrows():
        nm = r.pathway.split("|", 1)[1]
        print(f"  {nm[:54]:54s} {int(r.n_genes):5d} {int(r.n_clusters):5d} "
              f"{r.z:+6.2f} {r.shift_p:8.4f} {r.fdr:6.3f}")
    print(f"\nsurviving FDR 0.05: {int((s.fdr < 0.05).sum())} of {len(s)}")

    for key in ["Lipoprotein", "ysosom", "utophag", "itochondri", "mmune", "ynap"]:
        sub = s[s.pathway.str.contains(key, case=False)]
        if len(sub):
            print(f"\n-- '{key}' ({len(sub)} sets) --")
            for _, r in sub.head(5).iterrows():
                print(f"   {r.pathway.split('|')[1][:50]:50s} n={int(r.n_genes):3d} "
                      f"clusters={int(r.n_clusters):3d} p={r.shift_p:.4f} fdr={r.fdr:.3f}")

    json.dump(dict(n_sets=int(len(s)), n_fdr05=int((s.fdr < 0.05).sum()),
                   top=s.head(30).to_dict("records")),
              open(f"{OUT}/104_circshift_summary.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/103_circshift_geneset.csv")


if __name__ == "__main__":
    main()
