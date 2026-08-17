import sys
from pathlib import Path
import os, re, json, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

warnings.filterwarnings("ignore")

BASE = str(paths.ROOT)
OUT, EXT = f"{BASE}/results", f"{BASE}/external"
NPERM = 50000
rng = np.random.default_rng(11)

LYSO_CORE = {"GBA", "TMEM175", "SCARB2", "CTSB", "GALC", "ASAH1", "GRN", "VPS13C",
             "ATP6V0A1", "LAMP3", "TMEM163", "NEU1", "SMPD1", "CTSK", "GUSB",
             "HEXB", "NAGLU", "GPR65", "IDUA", "GAK", "DGKQ", "CTSD", "GLA",
             "NPC1", "NPC2", "TPP1", "PSAP", "SGSH", "HGSNAT", "MCOLN1"}


def load():
    j = pd.read_csv(f"{OUT}/62_loci_risk_onset.csv")
    j["_within"] = j.genes_in_window.fillna("").apply(lambda s: set(s.split(";")) - {""})
    return j


def stat(j, member, x=None):
    y = np.log1p(j.onset_chi2.values)
    if x is None:
        x = np.log1p(j.risk_chi2.values)
    g = member.astype(float)
    return sm.OLS(y, sm.add_constant(np.column_stack([g, x]))).fit().params[1]


def perm_p(j, member, nperm=NPERM):
    obs = stat(j, member)
    x = np.log1p(j.risk_chi2.values)
    y = np.log1p(j.onset_chi2.values)
    g = member.astype(float)
    X = sm.add_constant(np.column_stack([g, x]))
    null = np.empty(nperm)
    for i in range(nperm):
        X[:, 1] = g[rng.permutation(len(g))]
        null[i] = sm.OLS(y, X).fit().params[1]
    return obs, float((np.sum(null >= obs) + 1) / (nperm + 1)), null


def main():
    j = load()
    R = {}
    mb = np.array([bool(w & LYSO_CORE) for w in j._within])
    obs, p, null = perm_p(j, mb)
    R["primary"] = dict(n_loci=int(mb.sum()), beta=float(obs), p=float(p),
                        onset_chi2_in=float(j.onset_chi2[mb].mean()),
                        onset_chi2_out=float(j.onset_chi2[~mb].mean()))
    print(f"[primary] lysosomal loci n={mb.sum()}: beta={obs:+.3f} p={p:.4f}  "
          f"onset chi2 {j.onset_chi2[mb].mean():.2f} vs {j.onset_chi2[~mb].mean():.2f}")
    print(f"  loci: {', '.join(j.rsid[mb])}")

    loo = []
    idx = np.where(mb)[0]
    for i in idx:
        m2 = mb.copy(); m2[i] = False
        o2, p2, _ = perm_p(j, m2, 10000)
        loo.append(dict(dropped=j.rsid.iloc[i],
                        gene=";".join(sorted(j._within.iloc[i] & LYSO_CORE)),
                        beta=float(o2), p=float(p2)))
    loo = pd.DataFrame(loo).sort_values("p", ascending=False)
    loo.to_csv(f"{OUT}/95_lyso_leave_one_out.csv", index=False)
    R["leave_one_out"] = dict(max_p=float(loo.p.max()), min_p=float(loo.p.min()),
                              n_still_p05=int((loo.p < 0.05).sum()), n=len(loo))
    print(f"\n[leave-one-out] p ranges {loo.p.min():.4f}-{loo.p.max():.4f}; "
          f"{int((loo.p<0.05).sum())} of {len(loo)} drops keep p<0.05")
    print(f"  most influential: dropping {loo.iloc[0].dropped} "
          f"({loo.iloc[0].gene}) gives p={loo.iloc[0].p:.4f}")

    n = int(mb.sum())
    rand = []
    for _ in range(2000):
        m2 = np.zeros(len(j), bool)
        m2[rng.choice(len(j), n, replace=False)] = True
        rand.append(stat(j, m2))
    rand = np.array(rand)
    R["size_matched"] = dict(n=n, pct_exceeding=float((rand >= obs).mean()))
    print(f"\n[size-matched] {(rand>=obs).mean()*100:.1f}% of random {n}-locus sets "
          f"reach the lysosomal beta")

    alt = {}
    for lib in ["KEGG_2021_Human", "GO_Cellular_Component_2023", "Reactome_2022",
                "WikiPathway_2023_Human"]:
        p_ = f"{EXT}/genesets/{lib}.txt"
        if not os.path.exists(p_):
            continue
        for line in open(p_):
            parts = line.rstrip("\n").split("\t")
            nm = parts[0]
            if re.search(r"lysosom|lytic vacuole|autophag", nm, re.I):
                gs = {g.split(",")[0].strip() for g in parts[2:] if g.strip()}
                m2 = np.array([bool(w & gs) for w in j._within])
                if m2.sum() >= 4:
                    o2, p2, _ = perm_p(j, m2, 10000)
                    alt[f"{lib}|{nm}"] = dict(n=int(m2.sum()), beta=float(o2), p=float(p2))
    a = pd.DataFrame(alt).T.sort_values("p")
    a.to_csv(f"{OUT}/96_lyso_definitions.csv")
    R["alt_definitions"] = dict(n_tested=len(a), n_p05=int((a.p < 0.05).sum()),
                                median_p=float(a.p.median()))
    print(f"\n[definitions] {len(a)} independent lysosomal/autophagy sets tested; "
          f"{int((a.p<0.05).sum())} reach p<0.05, median p={a.p.median():.3f}")
    for k, r in a.head(6).iterrows():
        print(f"  {k.split('|')[1][:52]:52s} n={int(r.n):2d} p={r.p:.4f}")

    al = pd.read_csv(f"{OUT}/19b_risk_vs_onset_aligned.csv")
    al = al.merge(j[["rsid"]].assign(lyso=mb), on="rsid", how="inner")
    for lab, sel in [("lysosomal", al.lyso), ("other", ~al.lyso)]:
        s = al[sel]
        frac = float((np.sign(s.risk_beta_target) != np.sign(s.aao_beta_aligned)).mean())
        med = float(np.median(np.abs(s.aao_beta_aligned)))
        print(f"\n[direction] {lab:10s} n={len(s):2d}  "
              f"{frac*100:.0f}% of risk alleles bring onset forward, "
              f"median |onset effect| {med:.3f} y/allele")
        R[f"direction_{lab}"] = dict(n=int(len(s)), frac_earlier=frac, median_abs=med)

    json.dump(R, open(f"{OUT}/97_lyso_robustness.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/97_lyso_robustness.json")


if __name__ == "__main__":
    main()
