import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

DATA = str(paths.DATA)
RES = str(paths.RESULTS)


def route(r):
    if r.get("ENRLLRRK2", 0) == 1:
        return "LRRK2 carrier"
    if r.get("ENRLGBA", 0) == 1:
        return "GBA carrier"
    if r.get("ENRLSNCA", 0) == 1:
        return "SNCA carrier"
    if r.get("ENRLPRKN", 0) == 1 or r.get("ENRLPINK1", 0) == 1:
        return "PRKN or PINK1 carrier"
    return "no known variant"


def main():
    d = pd.read_parquet(f"{DATA}/ppmi_aao.parquet")
    cases = d[(d.COHORT_DEFINITION == "Parkinson's Disease") & d.AAO.notna()].copy()
    cases["route"] = cases.apply(route, axis=1)

    rows = []
    for name, g in cases.groupby("route"):
        rows.append(dict(route=name, n=len(g), median=float(g.AAO.median()),
                         q1=float(g.AAO.quantile(0.25)),
                         q3=float(g.AAO.quantile(0.75)),
                         pct_before_50=float((g.AAO < 50).mean() * 100)))
    by_route = pd.DataFrame(rows).sort_values("median")
    by_route.to_csv(f"{RES}/190_onset_by_route.csv", index=False)
    cases[["AAO", "route"]].to_csv(f"{RES}/191_cases_by_route.csv", index=False)

    base = cases.loc[cases.route == "no known variant", "AAO"]
    for name, g in cases.groupby("route"):
        if name != "no known variant" and len(g) >= 20:
            p = stats.mannwhitneyu(g.AAO, base).pvalue
            print(f"{name:24s} n={len(g):5d} median {g.AAO.median():5.1f} "
                  f"vs sporadic P={p:.3g}")

    sub = cases[(cases.Genetic_PRS_InfPop == "EUR") & (~cases.gen_asc)].dropna(
        subset=["AAO", "META5_PGS"]).copy()
    sub["tert"] = pd.qcut(sub.META5_PGS, 3,
                          labels=["lowest third", "middle third",
                                  "highest third"])
    sub.to_csv(f"{RES}/151_cases_for_clinical_fig.csv", index=False)

    ages = np.arange(20, 95.5, 0.5)
    curves = pd.DataFrame(index=ages)
    tert_rows = []
    for name, g in sub.groupby("tert", observed=True):
        curves[name] = [(g.AAO <= a).mean() for a in ages]
        tert_rows.append(dict(tertile=name, n=len(g),
                              median=float(g.AAO.median()),
                              q1=float(g.AAO.quantile(0.25)),
                              q3=float(g.AAO.quantile(0.75)),
                              mean=float(g.AAO.mean()),
                              pct_before_50=float((g.AAO < 50).mean() * 100)))
    curves.to_csv(f"{RES}/150_onset_curves_by_tertile.csv")

    hi = sub.loc[sub.tert == "highest third", "AAO"]
    lo = sub.loc[sub.tert == "lowest third", "AAO"]
    lr = logrank_test(hi, lo, np.ones(len(hi)), np.ones(len(lo)))
    summary = dict(n=len(sub), tertiles=tert_rows,
                   median_diff_years=float(lo.median() - hi.median()),
                   logrank_p=float(lr.p_value))
    json.dump(summary, open(f"{RES}/152_clinical_summary.json", "w"), indent=1)

    print()
    print(pd.DataFrame(tert_rows).to_string(index=False))
    print(f"\nmedian difference {summary['median_diff_years']:.1f} years, "
          f"log-rank P = {summary['logrank_p']:.4f}")


if __name__ == "__main__":
    main()
