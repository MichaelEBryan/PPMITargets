import sys
from pathlib import Path
import json, warnings, time
import numpy as np, pandas as pd
import shap
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

warnings.filterwarnings("ignore")

ROOT = str(paths.ROOT)
RES = str(paths.RESULTS)
N_BOOT = 200
TOPK = 20
SEED = 20260816


def models(seed):
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=8, random_state=seed,
            tree_method="hist", eval_metric="mlogloss", verbosity=0),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.8, n_jobs=8, random_state=seed, verbose=-1),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.08, random_seed=seed,
            verbose=0, allow_writing_files=False, thread_count=8),
    }


def shap_importance(model, X):
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    if isinstance(sv, list):
        a = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        a = np.abs(sv)
        while a.ndim > 2:
            a = a.mean(axis=-1)
        a = a.mean(axis=0)
    return np.asarray(a).ravel()


def main():
    t0 = time.time()
    m = pd.read_parquet(f"{ROOT}/data/ppmi_matrix.parquet")
    prov = json.load(open(f"{ROOT}/data/provenance.json"))
    y = m["__label__"].values
    age = m["__age__"].values
    X = m.drop(columns=["__label__", "__age__"])
    feats = X.columns.tolist()
    block = np.array([prov.get(f, "other") for f in feats])
    print(f"matrix {X.shape}, blocks {pd.Series(block).value_counts().to_dict()}")

    strata = {"EOPD_lt50": age < 50, "LOPD_ge60": age >= 60}
    out_imp, out_boot, out_share, out_dep = [], [], [], []

    for sname, mask in strata.items():
        Xs = X[mask].reset_index(drop=True)
        ys = pd.Series(y[mask]).astype("category").cat.codes.values
        keep = Xs.columns[Xs.notna().sum() > 0]
        Xs = Xs[keep].fillna(-999)
        bl = np.array([prov.get(f, "other") for f in keep])
        print(f"\n{sname}: n={len(Xs)}, p={Xs.shape[1]}, classes={np.bincount(ys)}")

        per_model = {}
        for mn, mdl in models(SEED).items():
            mdl.fit(Xs, ys)
            imp = shap_importance(mdl, Xs)
            per_model[mn] = imp
            for f, b, v in zip(keep, bl, imp):
                out_imp.append(dict(stratum=sname, model=mn, feature=f,
                                    block=b, shap=float(v)))
            tot = imp.sum()
            for b in np.unique(bl):
                out_share.append(dict(stratum=sname, model=mn, block=b,
                                      share=float(imp[bl == b].sum() / tot),
                                      n_features=int((bl == b).sum())))
            print(f"  {mn}: top = {keep[np.argmax(imp)]}, "
                  f"design share = {imp[bl == 'design'].sum() / tot:.3f}")

        rng = np.random.default_rng(SEED)
        ranks = {f: [] for f in keep}
        intop = {f: 0 for f in keep}
        n = len(Xs)
        for b_ in range(N_BOOT):
            idx = rng.choice(n, n, replace=True)
            if len(np.unique(ys[idx])) < 2:
                continue
            mdl = models(SEED + b_)["XGBoost"]
            mdl.fit(Xs.iloc[idx], ys[idx])
            imp = shap_importance(mdl, Xs.iloc[idx])
            order = np.argsort(-imp)
            r = np.empty(len(imp), int)
            r[order] = np.arange(1, len(imp) + 1)
            for f, rr in zip(keep, r):
                ranks[f].append(int(rr))
            for f in keep[order[:TOPK]]:
                intop[f] += 1
            if (b_ + 1) % 25 == 0:
                print(f"    bootstrap {b_ + 1}/{N_BOOT}  "
                      f"({time.time() - t0:.0f}s)")
        nb = max(1, len(ranks[keep[0]]))
        base_order = np.argsort(-per_model["XGBoost"])
        for f, b in zip(keep, bl):
            rr = np.array(ranks[f])
            out_boot.append(dict(
                stratum=sname, feature=f, block=b,
                rank_point=int(np.where(keep[base_order] == f)[0][0] + 1),
                rank_median=float(np.median(rr)),
                rank_q05=float(np.percentile(rr, 5)),
                rank_q95=float(np.percentile(rr, 95)),
                p_in_top20=intop[f] / nb))

        gmask = bl == "genotype"
        if gmask.any():
            gi = np.argmax(np.where(gmask, per_model["XGBoost"], -1))
            gname = keep[gi]
            mdl = models(SEED)["XGBoost"]
            mdl.fit(Xs, ys)
            ex = shap.TreeExplainer(mdl)
            sv = ex.shap_values(Xs)
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv.mean(axis=-1) if sv.shape[-1] < sv.shape[1] else sv[0]
            col = np.asarray(sv)[:, gi] if sv.ndim == 2 else None
            if col is not None:
                for v, s in zip(Xs[gname].values, col):
                    out_dep.append(dict(stratum=sname, feature=gname,
                                        value=float(v), shap=float(s)))

    pd.DataFrame(out_imp).to_csv(f"{RES}/160_shap_importance.csv", index=False)
    pd.DataFrame(out_boot).to_csv(f"{RES}/161_shap_rank_bootstrap.csv", index=False)
    pd.DataFrame(out_share).to_csv(f"{RES}/162_shap_block_share.csv", index=False)
    pd.DataFrame(out_dep).to_csv(f"{RES}/163_shap_dependence.csv", index=False)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
