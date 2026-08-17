import sys
from pathlib import Path
import os, json, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

warnings.filterwarnings("ignore")

BASE = str(paths.ROOT)
OUT, EXT = f"{BASE}/results", f"{BASE}/external"
MERGE = 250_000
NPERM = 5000
NJOBS = 10


def collapse(g):
    g = g.sort_values(["chr", "start"]).reset_index(drop=True)
    loci, cur = [], None
    for _, r in g.iterrows():
        if cur is None or r.chr != cur[0] or r.start - cur[1] > MERGE:
            loci.append(dict(chr=r.chr, start=int(r.start), end=int(r.end),
                             genes=[r.gene], min_p=r.min_p, z=r.z,
                             n_snps=r.n_snps, min_p_risk=r.min_p_risk,
                             z_risk=r.z_risk))
        else:
            L = loci[-1]
            L["genes"].append(r.gene)
            L["end"] = max(L["end"], int(r.end))
            if r.min_p < L["min_p"]:
                L["min_p"], L["z"], L["n_snps"] = r.min_p, r.z, r.n_snps
            if r.min_p_risk < L["min_p_risk"]:
                L["min_p_risk"], L["z_risk"] = r.min_p_risk, r.z_risk
        cur = (r.chr, max(int(r.end), cur[1]) if cur and r.chr == cur[0] else int(r.end))
    d = pd.DataFrame(loci)
    d["gene_set"] = d.genes.apply(set)
    return d


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


def one_set(name, gs, d, y, x, nsnp, rng_seed):
    mb = d.gene_set.apply(lambda s: bool(s & gs)).values
    n = int(mb.sum())
    if n < 8:
        return None
    g = mb.astype(float)
    X = sm.add_constant(np.column_stack([g, x, nsnp]))
    obs = sm.OLS(y, X).fit().params[1]
    rng = np.random.default_rng(rng_seed)
    null = np.empty(NPERM)
    Xp = X.copy()
    for i in range(NPERM):
        Xp[:, 1] = g[rng.permutation(len(g))]
        null[i] = sm.OLS(y, Xp).fit().params[1]
    return dict(pathway=name, n_loci=n, beta=float(obs),
                perm_p=float((np.sum(null >= obs) + 1) / (NPERM + 1)),
                z=float((obs - null.mean()) / null.std()),
                mean_z_in=float(d.z[mb].mean()), mean_z_out=float(d.z[~mb].mean()))


def main():
    j = pd.read_csv(f"{OUT}/71_genestats_joined.csv", dtype={"chr": str})
    print(f"{len(j)} genes -> collapsing within {MERGE//1000} kb", flush=True)
    d = collapse(j)
    print(f"{len(d)} independent loci", flush=True)
    d.drop(columns=["gene_set"]).assign(genes=lambda x: x.genes.apply(";".join)) \
        .to_csv(f"{OUT}/100_loci_genomewide.csv", index=False)

    y = d.z.values
    x = d.z_risk.values
    nsnp = np.log10(d.n_snps.values)
    present = set().union(*d.gene_set)
    sets = load_sets(present)
    print(f"{len(sets)} gene sets; testing those with >=8 loci", flush=True)

    res = Parallel(n_jobs=NJOBS, verbose=1)(
        delayed(one_set)(k, v, d, y, x, nsnp, 1000 + i)
        for i, (k, v) in enumerate(sets.items()))
    s = pd.DataFrame([r for r in res if r])
    s["fdr"] = sm.stats.multipletests(s.perm_p, method="fdr_bh")[1]
    s = s.sort_values("perm_p")
    s.to_csv(f"{OUT}/101_locus_geneset.csv", index=False)

    print(f"\ntested {len(s)} sets at locus level")
    print("=== onset signal over and above risk, locus level ===")
    for _, r in s.head(20).iterrows():
        nm = r.pathway.split("|", 1)[1]
        print(f"  {nm[:56]:56s} n={int(r.n_loci):3d} z={r.z:+.2f} "
              f"p={r.perm_p:.4f} fdr={r.fdr:.3f}")
    print(f"\nsurviving FDR 0.05: {int((s.fdr < 0.05).sum())} of {len(s)}")

    for key in ["ysosom", "utophag", "Lipoprotein", "itochondri", "mmune"]:
        sub = s[s.pathway.str.contains(key, case=False)]
        if len(sub):
            print(f"\n-- sets matching '{key}' ({len(sub)}) --")
            for _, r in sub.head(6).iterrows():
                print(f"   {r.pathway.split('|')[1][:52]:52s} n={int(r.n_loci):3d} "
                      f"p={r.perm_p:.4f} fdr={r.fdr:.3f}")

    json.dump(dict(n_loci=int(len(d)), n_sets=int(len(s)),
                   n_fdr05=int((s.fdr < 0.05).sum()),
                   top=s.head(30).to_dict("records")),
              open(f"{OUT}/102_locus_geneset_summary.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/101_locus_geneset.csv")


if __name__ == "__main__":
    main()
