#!/usr/bin/env python
"""
Null distribution of top-k feature-selection overlap between two strata.

The published claim is: SHAP selected 10 SNPs in the early-onset stratum and 13
in the late-onset stratum, with zero overlap, and this demonstrates distinct
genetic architectures.

That inference requires a null. Here we build three.

  N1  combinatorial   -- two independent uniform draws of size 10 and 13 from the
                         p candidate SNPs. Gives P(overlap = 0) with no data at all.

  N2  split-half      -- take ONE stratum (identical architecture by construction),
                         split it at random into two disjoint subsets whose sizes
                         match n_EOPD and n_LOPD, run the published selection
                         pipeline on each, record the overlap. Repeated B times.
                         This is the null the paper needs and does not report.

  N3  bootstrap stability -- resample the same stratum with replacement, re-run
                         selection, and compare each replicate's top-k with the
                         full-data top-k. Measures whether the procedure is
                         reliable enough for non-overlap to mean anything.

Outputs the observed cross-stratum overlap alongside all three nulls.
"""
import os, re, json, sys, time, warnings, itertools
from math import comb
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import paths as _P

SRC = str(_P.DATA / "Complete DataFrameC.csv")
OUT = str(_P.ROOT / "results")
CLASSES = ['Healthy Control', "Parkinson's Disease", "Prodromal", "SWEDD"]
B = int(os.environ.get("NREP", 2000))
NJOBS = int(os.environ.get("NJOBS", 14))
K_EO, K_LO = 10, 13


def get_model(kind, seed):
    if kind == "XGBoost":
        return xgb.XGBClassifier(objective='multi:softprob', n_estimators=300, max_depth=4,
                                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, random_state=seed, n_jobs=1, verbosity=0)
    if kind == "LightGBM":
        return lgb.LGBMClassifier(objective='multiclass', n_estimators=300, max_depth=4,
                                  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                  reg_lambda=1.0, random_state=seed, n_jobs=1, verbose=-1)
    return CatBoostClassifier(loss_function='MultiClass', iterations=300, depth=4,
                              learning_rate=0.05, random_seed=seed, verbose=0,
                              thread_count=1, allow_writing_files=False)


def shap_rank(X, y, kind, seed, snp_mask):
    """Fit, compute mean|SHAP| across classes, return SNP feature names ranked."""
    m = get_model(kind, seed)
    m.fit(X, y)
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(X)
    if isinstance(sv, list):
        imp = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
    else:
        sv = np.asarray(sv)
        imp = np.abs(sv).mean(0)
        if imp.ndim > 1:
            imp = imp.mean(-1)
    s = pd.Series(imp, index=X.columns)
    return list(s[snp_mask].sort_values(ascending=False).index)


def one_rep(Xv, yv, idx_a, idx_b, kind, seed, snp_mask):
    try:
        ra = shap_rank(Xv.iloc[idx_a], yv[idx_a], kind, seed, snp_mask)
        rb = shap_rank(Xv.iloc[idx_b], yv[idx_b], kind, seed + 100000, snp_mask)
        A, Bs = set(ra[:K_EO]), set(rb[:K_LO])
        return dict(kind=kind, seed=seed, overlap=len(A & Bs),
                    jaccard=len(A & Bs) / len(A | Bs) if (A | Bs) else np.nan,
                    topA=";".join(ra[:K_EO]), topB=";".join(rb[:K_LO]))
    except Exception as e:
        return dict(kind=kind, seed=seed, overlap=np.nan, error=str(e)[:100])


def main():
    t0 = time.time()
    df = pd.read_csv(SRC, low_memory=False)
    d = df[df.COHORT_DEFINITION.isin(CLASSES)].copy()

    snp_pat = re.compile(r'^(chr[0-9XYM]+:|MG_rs)')
    numcols = [c for c in d.select_dtypes(include=['int64', 'float64']).columns
               if c not in ('Unnamed: 0', 'Unnamed: 0.1', 'PATNO')]
    X = d[numcols].copy()
    X.columns = [re.sub(r'[^\w]', '_', c) for c in X.columns]
    snp_cols = [c for c, orig in zip(X.columns, numcols) if snp_pat.match(orig)]
    snp_mask = pd.Index(X.columns).isin(snp_cols)
    X = X.fillna(0)
    y = LabelEncoder().fit_transform(d['COHORT_DEFINITION'])
    age = d['ENROLL_AGE'].values

    p = len(snp_cols)
    eo = np.where(age < 50)[0]
    lo = np.where(age >= 60)[0]
    print(f"p_snp={p}  n_EO={len(eo)}  n_LO={len(lo)}  n_total={len(d)}", flush=True)

    # N1: combinatorial null
    # P(|A n B| = j) for |A|=K_EO, |B|=K_LO drawn uniformly from p features
    n1 = {j: comb(K_EO, j) * comb(p - K_EO, K_LO - j) / comb(p, K_LO)
          for j in range(0, min(K_EO, K_LO) + 1)}
    exp_overlap = K_EO * K_LO / p
    print(f"N1 combinatorial: E[overlap]={exp_overlap:.2f}  P(overlap=0)={n1[0]:.4f}", flush=True)

    # Observed cross-stratum overlap
    obs = []
    for kind in ("XGBoost", "LightGBM", "CatBoost"):
        ra = shap_rank(X.iloc[eo], y[eo], kind, 42, snp_mask)
        rb = shap_rank(X.iloc[lo], y[lo], kind, 42, snp_mask)
        A, Bs = set(ra[:K_EO]), set(rb[:K_LO])
        obs.append(dict(kind=kind, overlap=len(A & Bs),
                        jaccard=len(A & Bs) / len(A | Bs),
                        top_EO=";".join(ra[:K_EO]), top_LO=";".join(rb[:K_LO])))
        print(f"OBSERVED {kind}: overlap={len(A & Bs)}", flush=True)
    pd.DataFrame(obs).to_csv(f"{OUT}/04_observed_overlap.csv", index=False)

    # N2: split-half null inside the LATE-onset stratum
    # Same architecture by construction; sizes matched to n_EO and n_LO.
    Xlo, ylo = X.iloc[lo].reset_index(drop=True), y[lo]
    nA = min(len(eo), len(lo) // 2)
    nB = min(len(lo) - nA, len(lo) // 2)
    print(f"N2 split-half: subset sizes {nA} vs {nB} drawn from LOPD stratum "
          f"({len(lo)}), B={B}", flush=True)

    jobs = []
    rng = np.random.default_rng(7)
    for b in range(B):
        perm = rng.permutation(len(lo))
        ia, ib = perm[:nA], perm[nA:nA + nB]
        jobs.append((ia, ib, ["XGBoost", "LightGBM", "CatBoost"][b % 3], b))
    res = Parallel(n_jobs=NJOBS, verbose=5, batch_size=4)(
        delayed(one_rep)(Xlo, ylo, ia, ib, kind, seed, snp_mask) for ia, ib, kind, seed in jobs)
    n2 = pd.DataFrame(res)
    n2.to_csv(f"{OUT}/05_null_splithalf.csv", index=False)
    ok = n2.overlap.notna()
    print(f"N2: mean overlap={n2.overlap[ok].mean():.2f} "
          f"P(overlap=0)={(n2.overlap[ok] == 0).mean():.4f} "
          f"P(overlap<=obs)={(n2.overlap[ok] <= max(o['overlap'] for o in obs)).mean():.4f}",
          flush=True)

    # N3: bootstrap selection stability within LOPD
    full_rank = {k: shap_rank(Xlo, ylo, k, 1, snp_mask) for k in ("XGBoost", "LightGBM", "CatBoost")}

    def boot(seed):
        r = np.random.default_rng(seed)
        idx = r.integers(0, len(Xlo), len(Xlo))
        kind = ["XGBoost", "LightGBM", "CatBoost"][seed % 3]
        try:
            rb = shap_rank(Xlo.iloc[idx], ylo[idx], kind, seed, snp_mask)
            out = dict(kind=kind, seed=seed)
            for k in (5, 10, 13, 20):
                out[f"jacc_top{k}"] = len(set(rb[:k]) & set(full_rank[kind][:k])) / \
                                      len(set(rb[:k]) | set(full_rank[kind][:k]))
            return out
        except Exception as e:
            return dict(kind=kind, seed=seed, error=str(e)[:100])

    res3 = Parallel(n_jobs=NJOBS, verbose=5, batch_size=4)(
        delayed(boot)(s) for s in range(B))
    n3 = pd.DataFrame(res3)
    n3.to_csv(f"{OUT}/06_bootstrap_stability.csv", index=False)
    print("N3 bootstrap Jaccard vs full-data ranking:", flush=True)
    for k in (5, 10, 13, 20):
        c = f"jacc_top{k}"
        if c in n3:
            print(f"   top-{k}: mean={n3[c].mean():.3f} sd={n3[c].std():.3f}", flush=True)

    json.dump(dict(p_snp=int(p), n_EO=int(len(eo)), n_LO=int(len(lo)),
                   K_EO=K_EO, K_LO=K_LO, B=B,
                   N1_expected_overlap=exp_overlap,
                   N1_pmf={str(k): v for k, v in n1.items()},
                   observed=obs,
                   N2_mean_overlap=float(n2.overlap[ok].mean()),
                   N2_p_zero=float((n2.overlap[ok] == 0).mean()),
                   N2_p_le_observed=float((n2.overlap[ok] <=
                                           max(o['overlap'] for o in obs)).mean()),
                   N3_mean_jaccard={f"top{k}": float(n3[f"jacc_top{k}"].mean())
                                    for k in (5, 10, 13, 20) if f"jacc_top{k}" in n3}),
              open(f"{OUT}/07_overlap_null_summary.json", "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
