import sys
from pathlib import Path
import json, itertools, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import statsmodels.api as sm
import xgboost as xgb
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

warnings.filterwarnings("ignore")

BASE = str(paths.ROOT)
OUT = f"{BASE}/results"
PCS = [f"Genetic_PRS_PC{i}" for i in range(1, 11)]
NPERM = 2000
NJOBS = 10


def build():
    d = pd.read_parquet(f"{BASE}/data/ppmi_aao.parquet")
    sm_ = pd.read_csv(f"{BASE}/data/snp_map.csv")
    c = d[(d.COHORT_DEFINITION == "Parkinson's Disease") & d.AAO.notna()
          & (d.Genetic_PRS_InfPop == "EUR") & (~d.gen_asc)].copy()
    c = c.dropna(subset=["AAO", "SEX"] + PCS)
    cols = [x for x in sm_.col if x in c.columns]
    G = c[cols].apply(pd.to_numeric, errors="coerce")
    G = G.loc[:, G.notna().mean() > 0.9]
    G = G.fillna(G.mean())
    G = G.loc[:, G.std() > 0]
    X = pd.concat([G, c[["SEX"] + PCS].astype(float)], axis=1)
    y = c.AAO.astype(float).values
    print(f"{len(X)} European sporadic cases, {G.shape[1]} variants + "
          f"{len(PCS)+1} covariates")
    return X, y, G, c


def nested_cv(X, y, n_out=10, seed=0):
    rows = []
    outer = KFold(n_out, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(outer.split(X)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        preds = {}
        preds["mean only"] = np.full(len(yte), ytr.mean())
        lin = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
        preds["additive linear"] = lin.predict(Xte)
        en = make_pipeline(StandardScaler(),
                           ElasticNetCV(l1_ratio=[.1, .5, .9, 1], cv=5,
                                        n_alphas=40, random_state=seed,
                                        max_iter=5000)).fit(Xtr, ytr)
        preds["elastic net"] = en.predict(Xte)
        gs = GridSearchCV(
            xgb.XGBRegressor(objective="reg:squarederror", n_jobs=2, verbosity=0,
                             random_state=seed),
            {"n_estimators": [200, 400], "max_depth": [2, 3, 4],
             "learning_rate": [0.02, 0.05], "subsample": [0.8],
             "colsample_bytree": [0.8], "reg_lambda": [1.0, 5.0]},
            cv=5, scoring="neg_mean_squared_error", n_jobs=4).fit(Xtr, ytr)
        preds["gradient boosting"] = gs.predict(Xte)
        for name, p in preds.items():
            rows.append(dict(fold=k, model=name, r2=r2_score(yte, p),
                             mae=mean_absolute_error(yte, p),
                             rmse=float(np.sqrt(np.mean((yte - p) ** 2)))))
    return pd.DataFrame(rows)


def interaction_screen(G, y, cov):
    cols = list(G.columns)
    base = sm.add_constant(cov)

    def best_stat(yy):
        best = 0.0
        Y = yy
        for a, b in itertools.combinations(range(len(cols)), 2):
            xa, xb = G.iloc[:, a].values, G.iloc[:, b].values
            X = np.column_stack([base, xa, xb, xa * xb])
            try:
                r = sm.OLS(Y, X).fit()
                t = abs(r.tvalues[-1])
                if t > best:
                    best = t
            except Exception:
                continue
        return best

    obs = best_stat(y)
    rng = np.random.default_rng(3)
    null = Parallel(n_jobs=NJOBS, verbose=0)(
        delayed(best_stat)(y[rng.permutation(len(y))]) for _ in range(200))
    null = np.array(null)
    p = (np.sum(null >= obs) + 1) / (len(null) + 1)
    n_pairs = len(cols) * (len(cols) - 1) // 2
    return dict(n_variants=len(cols), n_pairs=n_pairs, best_abs_t=float(obs),
                perm_p=float(p), null_mean=float(null.mean()),
                null_p95=float(np.percentile(null, 95)))


def main():
    X, y, G, c = build()
    cv = nested_cv(X, y)
    cv.to_csv(f"{OUT}/90_ml_vs_additive_folds.csv", index=False)
    s = cv.groupby("model").agg(r2=("r2", "mean"), r2_sd=("r2", "std"),
                                mae=("mae", "mean"), rmse=("rmse", "mean")).reset_index()
    print("\nheld-out prediction of age at onset, 10-fold nested CV")
    print(f"{'model':20s} {'R2':>8s} {'sd':>7s} {'MAE (y)':>9s} {'RMSE (y)':>9s}")
    for _, r in s.sort_values("r2", ascending=False).iterrows():
        print(f"{r.model:20s} {r.r2:8.4f} {r.r2_sd:7.4f} {r.mae:9.2f} {r.rmse:9.2f}")
    s.to_csv(f"{OUT}/91_ml_vs_additive.csv", index=False)

    piv = cv.pivot(index="fold", columns="model", values="rmse")
    t, p = stats.ttest_rel(piv["gradient boosting"], piv["additive linear"])
    tw, pw = stats.wilcoxon(piv["gradient boosting"], piv["additive linear"])
    print(f"\nboosting vs additive, paired across folds: "
          f"delta RMSE = {(piv['gradient boosting']-piv['additive linear']).mean():+.3f} y "
          f"(t p={p:.3f}, Wilcoxon p={pw:.3f})")

    print("\nscreening every variant pair for interaction ...", flush=True)
    inter = interaction_screen(G, y, c[["SEX"] + PCS].astype(float).values)
    print(f"  {inter['n_pairs']:,} pairs; best |t| = {inter['best_abs_t']:.2f}; "
          f"permutation null 95th pct = {inter['null_p95']:.2f}; p = {inter['perm_p']:.3f}")

    json.dump(dict(n=len(X), n_variants=int(G.shape[1]),
                   cv=s.to_dict("records"),
                   boosting_vs_additive_rmse_delta=float(
                       (piv["gradient boosting"] - piv["additive linear"]).mean()),
                   boosting_vs_additive_p=float(p),
                   interaction=inter),
              open(f"{OUT}/92_ml_value_summary.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/92_ml_value_summary.json")


if __name__ == "__main__":
    main()
