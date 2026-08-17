import sys
from pathlib import Path
import re, json, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

warnings.filterwarnings("ignore")

BASE = str(paths.ROOT)
OUT = f"{BASE}/results"
PCS = [f"Genetic_PRS_PC{i}" for i in range(1, 11)]


def main():
    d = pd.read_parquet(f"{BASE}/data/ppmi_aao.parquet")
    loci = pd.read_csv(f"{OUT}/62_loci_risk_onset.csv")
    snpmap = pd.read_csv(f"{BASE}/data/snp_map.csv")
    import importlib.util
    spec = importlib.util.spec_from_file_location("p17", f"{BASE}/scripts/17_pathway_onset.py")
    p17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p17)

    loci = loci.merge(snpmap[["rsid", "col", "effect_allele"]], on="rsid",
                      how="inner", suffixes=("", "_map"))
    c = d[(d.COHORT_DEFINITION == "Parkinson's Disease") & d.AAO.notna()
          & (d.Genetic_PRS_InfPop == "EUR") & (~d.gen_asc)].copy()
    c = c.dropna(subset=["AAO", "SEX"] + PCS)
    print(f"{len(c)} European sporadic PD cases with onset age, PCs and dosages")

    scores, used = {}, {}
    for name, genes in p17.CURATED.items():
        gs = set(genes)
        sel = loci[loci.gene.fillna("").apply(
            lambda g: any(x in gs for x in re.split(r"[/,]", g)) if g else False)]
        cols = [r.col for _, r in sel.iterrows() if r.col in c.columns]
        if len(cols) < 3:
            continue
        s = np.zeros(len(c))
        n_ok = 0
        for _, r in sel.iterrows():
            if r.col not in c.columns:
                continue
            g = pd.to_numeric(c[r.col], errors="coerce")
            if g.notna().sum() < 100:
                continue
            g = g.fillna(g.mean())
            w = r.risk_beta if r.effect_allele.upper() == str(r.effect_allele_map).upper() else r.risk_beta
            s += w * g.values
            n_ok += 1
        if n_ok >= 3:
            scores[name] = (s - s.mean()) / s.std()
            used[name] = n_ok
    print("pathway scores built from:", used)

    S = pd.DataFrame(scores, index=c.index)
    base_cov = c[["SEX"] + PCS].astype(float)

    rows = []
    for name in S.columns:
        Xm = sm.add_constant(pd.concat([S[[name]], base_cov], axis=1).astype(float))
        mm = sm.OLS(c.AAO.astype(float), Xm).fit()
        Xa = sm.add_constant(pd.concat([S, base_cov], axis=1).astype(float))
        ma = sm.OLS(c.AAO.astype(float), Xa).fit()
        tot = (c.META5_PGS - c.META5_PGS.mean()) / c.META5_PGS.std()
        Xt = sm.add_constant(pd.concat([S[[name]], tot.rename("total_PRS"),
                                        base_cov], axis=1).astype(float))
        mt = sm.OLS(c.AAO.astype(float), Xt).fit()
        rows.append(dict(pathway=name, n_loci=used[name], n=len(c),
                         beta_marginal=mm.params[name], p_marginal=mm.pvalues[name],
                         beta_mutual=ma.params[name], p_mutual=ma.pvalues[name],
                         beta_vs_total=mt.params[name], p_vs_total=mt.pvalues[name]))
    r = pd.DataFrame(rows)
    r["fdr_marginal"] = sm.stats.multipletests(r.p_marginal, method="fdr_bh")[1]
    r["fdr_vs_total"] = sm.stats.multipletests(r.p_vs_total, method="fdr_bh")[1]
    r = r.sort_values("p_vs_total")
    r.to_csv(f"{OUT}/64_pathway_prs_ppmi.csv", index=False)
    print("\npathway polygenic score vs age at onset (years per SD)")
    print(f"{'pathway':30s} {'loci':>4s} {'marginal':>18s} {'vs total load':>20s}")
    for _, x in r.iterrows():
        print(f"{x.pathway:30s} {int(x.n_loci):4d} "
              f"{x.beta_marginal:+7.2f} (p={x.p_marginal:.3f}) "
              f"{x.beta_vs_total:+7.2f} (p={x.p_vs_total:.3f}) fdr={x.fdr_vs_total:.3f}")
    json.dump(dict(n=len(c), rows=r.to_dict("records")),
              open(f"{OUT}/65_pathway_prs_summary.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/64_pathway_prs_ppmi.csv")


if __name__ == "__main__":
    main()
