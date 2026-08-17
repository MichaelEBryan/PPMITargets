#!/usr/bin/env python
"""
Cloud execution of the two PPMI experiments.

  A. Leakage attribution -- repeated stratified CV over feature-block ablations,
     to work out what actually carries the published classification performance.
  B. Split-half null    -- the distribution of top-k overlap between two subsets
     of ONE stratum, i.e. under identical genetic architecture by construction.

Input is a de-identified numeric matrix: no participant identifiers, no dates.

Run:  modal run 07_modal_ppmi.py
"""
import modal, json, os
import paths as _P

DATA = str(_P.DATA)
app = modal.App("pd-arch-ppmi")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "pyarrow==18.1.0",
                      "scikit-learn==1.5.2", "xgboost==2.1.3", "lightgbm==4.5.0",
                      "catboost==1.2.7", "shap==0.46.0", "scipy==1.13.1")
         .add_local_file(f"{DATA}/ppmi_matrix.parquet", "/data/ppmi_matrix.parquet")
         .add_local_file(f"{DATA}/provenance.json", "/data/provenance.json"))

STRATA = {"EOPD_lt50": ("age", "<", 50), "LOPD_ge60": ("age", ">=", 60), "ALL": None}
NFOLD, NREPEAT = 5, 5
K_A, K_B = 10, 13


def _load():
    import pandas as pd, json
    X = pd.read_parquet("/data/ppmi_matrix.parquet")
    prov = json.load(open("/data/provenance.json"))
    y = X.pop("__label__").values
    age = X.pop("__age__").values
    return X, y, age, prov


def _mask(age, name):
    import numpy as np
    if name == "EOPD_lt50":
        return age < 50
    if name == "LOPD_ge60":
        return age >= 60
    return np.ones(len(age), bool)


def _models(seed):
    import xgboost as xgb, lightgbm as lgb
    from catboost import CatBoostClassifier
    return {
        "XGBoost": xgb.XGBClassifier(objective="multi:softprob", n_estimators=300,
                                     max_depth=4, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, reg_lambda=1.0,
                                     random_state=seed, n_jobs=4, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(objective="multiclass", n_estimators=300,
                                       max_depth=4, learning_rate=0.05, subsample=0.8,
                                       colsample_bytree=0.8, reg_lambda=1.0,
                                       random_state=seed, n_jobs=4, verbose=-1),
        "CatBoost": CatBoostClassifier(loss_function="MultiClass", iterations=300,
                                       depth=4, learning_rate=0.05, random_seed=seed,
                                       verbose=0, thread_count=4,
                                       allow_writing_files=False),
    }


@app.function(image=image, cpu=4.0, memory=8192, timeout=5400,
              max_containers=100, retries=3)
def ablate(job: dict):
    import numpy as np, pandas as pd, warnings
    warnings.filterwarnings("ignore")
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import (f1_score, roc_auc_score, balanced_accuracy_score,
                                 accuracy_score)
    from sklearn.dummy import DummyClassifier

    X, y, age, prov = _load()
    m = _mask(age, job["stratum"])
    Xs, ys = X[m], y[m]
    arm = job["arm"]

    if arm == "full":
        cols = list(Xs.columns)
    elif arm.startswith("minus_"):
        b = arm[6:]
        cols = [c for c in Xs.columns if prov.get(c) != b]
    elif arm.startswith("only_"):
        b = arm[5:]
        cols = [c for c in Xs.columns if prov.get(c) == b]
    elif arm == "genetics_only":
        cols = [c for c in Xs.columns if prov.get(c) in ("geno", "prs", "pcs")]
    elif arm == "genetics_demo_age":
        cols = [c for c in Xs.columns if prov.get(c) in ("geno", "prs", "pcs", "demo", "age")]
    elif arm == "no_leakage":   # drop design + genetics_core + motor
        cols = [c for c in Xs.columns
                if prov.get(c) not in ("design", "genetics_core", "motor")]
    elif arm == "missingness_only":
        M = Xs.isna().astype(np.int8)
        M = M.loc[:, M.nunique() > 1]
        Xs, cols = M, list(M.columns)
    else:
        return []
    if not cols:
        return []

    D = Xs[cols].fillna(0) if arm != "missingness_only" else Xs[cols]
    D = D.replace([np.inf, -np.inf], 0)
    present = np.unique(ys)
    rows = []
    if arm == "majority_baseline":
        pass
    rskf = RepeatedStratifiedKFold(n_splits=NFOLD, n_repeats=NREPEAT, random_state=0)
    for fold, (tr, te) in enumerate(rskf.split(D, ys)):
        if len(np.unique(ys[tr])) < len(present):
            continue
        dm = DummyClassifier(strategy="prior").fit(D.iloc[tr], ys[tr])
        yb = dm.predict(D.iloc[te])
        rows.append(dict(stratum=job["stratum"], arm=arm, model="MajorityBaseline",
                         fold=fold, n_feat=0,
                         f1w=f1_score(ys[te], yb, average="weighted"),
                         f1m=f1_score(ys[te], yb, average="macro"),
                         bacc=balanced_accuracy_score(ys[te], yb),
                         acc=accuracy_score(ys[te], yb), auc=0.5))
        for name, mdl in _models(fold).items():
            try:
                mdl.fit(D.iloc[tr], ys[tr])
                p = mdl.predict_proba(D.iloc[te]); yp = p.argmax(1)
                try:
                    auc = roc_auc_score(ys[te], p, multi_class="ovr", average="macro",
                                        labels=present)
                except Exception:
                    auc = float("nan")
                rows.append(dict(stratum=job["stratum"], arm=arm, model=name, fold=fold,
                                 n_feat=len(cols), n_train=len(tr), n_test=len(te),
                                 f1w=f1_score(ys[te], yp, average="weighted"),
                                 f1m=f1_score(ys[te], yp, average="macro"),
                                 bacc=balanced_accuracy_score(ys[te], yp),
                                 acc=accuracy_score(ys[te], yp), auc=auc))
            except Exception as e:
                rows.append(dict(stratum=job["stratum"], arm=arm, model=name, fold=fold,
                                 error=str(e)[:150]))
    return rows


@app.function(image=image, cpu=4.0, memory=8192, timeout=5400,
              max_containers=120, retries=3)
def null_batch(job: dict):
    """Split-half null inside one stratum: identical architecture by construction."""
    import numpy as np, pandas as pd, warnings, shap
    warnings.filterwarnings("ignore")
    X, y, age, prov = _load()
    m = _mask(age, job["donor"])
    Xs = X[m].fillna(0).replace([np.inf, -np.inf], 0).reset_index(drop=True)
    ys = y[m]
    snp = np.array([prov.get(c) == "geno" for c in Xs.columns])
    nA, nB = job["nA"], job["nB"]

    def rank(idx, kind, seed):
        mdl = _models(seed)[kind]
        mdl.fit(Xs.iloc[idx], ys[idx])
        sv = shap.TreeExplainer(mdl).shap_values(Xs.iloc[idx])
        if isinstance(sv, list):
            imp = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
        else:
            sv = np.asarray(sv); imp = np.abs(sv).mean(0)
            if imp.ndim > 1:
                imp = imp.mean(-1)
        s = pd.Series(imp, index=Xs.columns)[snp]
        return list(s.sort_values(ascending=False).index)

    out = []
    for r in range(job["nrep"]):
        seed = job["seed"] * 1000 + r
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Xs))
        ia, ib = perm[:nA], perm[nA:nA + nB]
        if len(np.unique(ys[ia])) < 2 or len(np.unique(ys[ib])) < 2:
            continue
        for kind in ("XGBoost", "LightGBM", "CatBoost"):
            try:
                ra, rb = rank(ia, kind, seed), rank(ib, kind, seed + 7)
                A, B = set(ra[:K_A]), set(rb[:K_B])
                out.append(dict(donor=job["donor"], model=kind, seed=seed,
                                nA=nA, nB=nB, overlap=len(A & B),
                                jaccard=len(A & B) / len(A | B)))
            except Exception as e:
                out.append(dict(donor=job["donor"], model=kind, seed=seed,
                                error=str(e)[:120]))
    return out


@app.local_entrypoint()
def main():
    import pandas as pd, json, os
    OUT = str(_P.ROOT / "results")
    prov = json.load(open(f"{DATA}/provenance.json"))
    blocks = sorted(set(prov.values()))

    arms = (["full", "no_leakage", "genetics_only", "genetics_demo_age", "missingness_only"]
            + [f"minus_{b}" for b in blocks] + [f"only_{b}" for b in blocks])
    jobs = [dict(stratum=s, arm=a) for s in STRATA for a in arms]
    print(f"[A] {len(jobs)} ablation jobs ({len(arms)} arms x {len(STRATA)} strata)")

    nulls = [dict(donor="LOPD_ge60", nA=230, nB=1693, nrep=12, seed=s) for s in range(100)]
    nulls += [dict(donor="ALL", nA=230, nB=1693, nrep=12, seed=500 + s) for s in range(40)]
    print(f"[B] {len(nulls)} null batches = {sum(n['nrep'] for n in nulls)} replicates "
          f"x 3 model families")

    ra = [r for chunk in ablate.map(jobs, order_outputs=False) if chunk for r in chunk]
    pd.DataFrame(ra).to_csv(f"{OUT}/20_ablation_folds.csv", index=False)
    print(f"wrote 20_ablation_folds.csv  ({len(ra)} rows)")

    rb = [r for chunk in null_batch.map(nulls, order_outputs=False) if chunk for r in chunk]
    pd.DataFrame(rb).to_csv(f"{OUT}/21_null_splithalf.csv", index=False)
    print(f"wrote 21_null_splithalf.csv  ({len(rb)} rows)")

    d = pd.DataFrame(ra)
    d = d[d.f1w.notna()]
    summ = (d.groupby(["stratum", "arm", "model"])
            .agg(n_feat=("n_feat", "first"), folds=("f1w", "size"),
                 f1w=("f1w", "mean"), f1w_sd=("f1w", "std"),
                 bacc=("bacc", "mean"), auc=("auc", "mean"), auc_sd=("auc", "std"))
            .reset_index())
    summ.to_csv(f"{OUT}/22_ablation_summary.csv", index=False)
    for s in ("EOPD_lt50", "LOPD_ge60"):
        print(f"\n--- {s} (XGBoost) ---")
        t = summ[(summ.stratum == s) & (summ.model.isin(["XGBoost", "MajorityBaseline"]))]
        for _, r in t.sort_values("f1w", ascending=False).iterrows():
            print(f"   {r.arm:24s} nfeat={int(r.n_feat):4d} f1w={r.f1w:.3f} auc={r.auc:.3f}")
