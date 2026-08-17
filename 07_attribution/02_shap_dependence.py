import sys
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd, shap
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

warnings.filterwarnings("ignore")

ROOT = str(paths.ROOT)
RES = str(paths.RESULTS)

m = pd.read_parquet(f"{ROOT}/data/ppmi_matrix.parquet")
prov = json.load(open(f"{ROOT}/data/provenance.json"))
y, age = m["__label__"].values, m["__age__"].values
X = m.drop(columns=["__label__", "__age__"])
imp = pd.read_csv(f"{RES}/160_shap_importance.csv")

rows = []
for sname, mask in [("EOPD_lt50", age < 50), ("LOPD_ge60", age >= 60)]:
    Xs = X[mask].reset_index(drop=True)
    ys = pd.Series(y[mask]).astype("category").cat.codes.values
    keep = Xs.columns[Xs.notna().sum() > 0]
    Xs = Xs[keep].fillna(-999)
    g = imp[(imp.stratum == sname) & (imp.model == "XGBoost")
            & (imp.block == "geno")].sort_values("shap", ascending=False)
    if not len(g):
        continue
    feat = g.feature.iloc[0]
    j = list(keep).index(feat)
    mdl = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=8,
                        random_state=20260816, tree_method="hist",
                        eval_metric="mlogloss", verbosity=0)
    mdl.fit(Xs, ys)
    sv = np.asarray(shap.TreeExplainer(mdl).shap_values(Xs))
    if sv.ndim == 3:
        col = (np.abs(sv[:, j, :]).sum(1) * np.sign(sv[:, j, :].sum(1))
               if sv.shape[0] == len(Xs) else
               np.abs(sv[:, :, j]).sum(0) * np.sign(sv[:, :, j].sum(0)))
    else:
        col = sv[:, j]
    for v, s_ in zip(Xs[feat].values, np.asarray(col).ravel()):
        rows.append(dict(stratum=sname, feature=feat, value=float(v),
                         shap=float(s_)))
    print(f"{sname}: {feat}, {len(Xs)} patients, "
          f"SHAP range {np.min(col):.3f} to {np.max(col):.3f}")

pd.DataFrame(rows).to_csv(f"{RES}/163_shap_dependence.csv", index=False)
print("done")
