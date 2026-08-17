import sys
from pathlib import Path
import modal, itertools, json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

app = modal.App("pd-arch-method")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2",
                      "xgboost==2.1.3", "shap==0.46.0", "scipy==1.13.1",
                      "statsmodels==0.14.4"))

GRID = dict(
    n_small=[100, 230, 500, 1000, 2000],
    n_large=[3386],
    p=[50, 162, 500],
    m_causal=[5, 15],
    beta=[0.05, 0.10, 0.20, 0.40],
    ld=[0.0, 0.6],
    scenario=["null", "dose", "partial", "disjoint"],
    k=[10],
)
KSWEEP = dict(n_small=[230], n_large=[3386], p=[162], m_causal=[15],
              beta=[0.05, 0.10, 0.20], ld=[0.6], scenario=["null", "dose"],
              k=[5, 10, 20, 30])
NREP = 200


@app.function(image=image, cpu=4.0, timeout=5400, max_containers=150, retries=2)
def cell(cfg: dict):
    import numpy as np, warnings
    warnings.filterwarnings("ignore")
    import xgboost as xgb, shap
    from scipy import stats

    n_s, n_l = cfg["n_small"], cfg["n_large"]
    p, m, beta, ld, scen, K = (cfg["p"], cfg["m_causal"], cfg["beta"], cfg["ld"],
                               cfg["scenario"], cfg["k"])
    rng = np.random.default_rng(cfg["seed"])
    lat = np.repeat(np.arange((p + 4) // 5), 5)[:p]

    def geno(n, maf):
        if ld > 0:
            z = rng.standard_normal((n, lat.max() + 1))[:, lat]
            u = np.sqrt(ld) * z + np.sqrt(1 - ld) * rng.standard_normal((n, p))
            q = stats.norm.cdf(u)
            G = np.zeros((n, p), dtype=np.int8)
            for jj in range(p):
                f = maf[jj]
                G[:, jj] = ((q[:, jj] > (1 - f) ** 2).astype(np.int8)
                            + (q[:, jj] > (1 - f) ** 2 + 2 * f * (1 - f)).astype(np.int8))
            return G
        return rng.binomial(2, maf, size=(n, p)).astype(np.int8)

    def pheno(G, causal, betas):
        lin = G[:, causal] @ betas
        return rng.binomial(1, 1 / (1 + np.exp(-(lin - lin.mean()))))

    def shap_top(G, y, k):
        if len(np.unique(y)) < 2:
            return None
        mdl = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                                verbosity=0, eval_metric="logloss")
        mdl.fit(G, y)
        sv = np.abs(np.asarray(shap.TreeExplainer(mdl).shap_values(G))).mean(0)
        if sv.ndim > 1:
            sv = sv.mean(-1)
        return set(np.argsort(-sv)[:k].tolist())

    def formal(GA, yA, GB, yB):
        zs = []
        for jj in range(p):
            out = []
            for G, y in ((GA, yA), (GB, yB)):
                x = G[:, jj].astype(float)
                if x.std() == 0 or len(np.unique(y)) < 2:
                    out = None
                    break
                X = np.column_stack([np.ones(len(x)), x])
                try:
                    import statsmodels.api as sm
                    r = sm.Logit(y, X).fit(disp=0)
                    out.append((r.params[1], r.bse[1]))
                except Exception:
                    out = None
                    break
            if out and len(out) == 2 and all(np.isfinite([o[1] for o in out])) \
                    and all(o[1] > 0 for o in out):
                d = out[0][0] - out[1][0]
                se = np.sqrt(out[0][1] ** 2 + out[1][1] ** 2)
                zs.append(d / se)
        if len(zs) < 5:
            return np.nan
        chi = float(np.sum(np.array(zs) ** 2))
        return float(stats.chi2.sf(chi, len(zs)))

    out = []
    for r in range(cfg["nrep"]):
        maf = rng.uniform(0.05, 0.5, p)
        cA = rng.choice(p, m, replace=False)
        bA = np.full(m, beta) * rng.choice([-1, 1], m)
        if scen == "null":
            cB, bB = cA.copy(), bA.copy()
        elif scen == "dose":
            cB, bB = cA.copy(), bA * 0.5
        elif scen == "partial":
            keep = cA[: m // 2]
            rest = rng.choice(np.setdiff1d(np.arange(p), cA), m - m // 2, replace=False)
            cB = np.concatenate([keep, rest])
            bB = np.full(m, beta) * rng.choice([-1, 1], m)
        else:
            cB = rng.choice(np.setdiff1d(np.arange(p), cA), m, replace=False)
            bB = np.full(m, beta) * rng.choice([-1, 1], m)
        GA, GB = geno(n_s, maf), geno(n_l, maf)
        yA, yB = pheno(GA, cA, bA), pheno(GB, cB, bB)
        sA, sB = shap_top(GA, yA, K), shap_top(GB, yB, K)
        if sA is None or sB is None:
            continue
        ov = len(sA & sB)
        fp = formal(GA, yA, GB, yB)
        out.append(dict(overlap=ov, naive_distinct=int(ov == 0),
                        formal_p=fp, formal_distinct=int(fp < 0.05) if np.isfinite(fp) else None,
                        recall_small=len(sA & set(cA.tolist())) / m,
                        recall_large=len(sB & set(cB.tolist())) / m))
    if not out:
        return dict(**cfg, nvalid=0)
    import numpy as np
    naive = np.mean([o["naive_distinct"] for o in out])
    fd = [o["formal_distinct"] for o in out if o["formal_distinct"] is not None]
    return dict(**cfg, nvalid=len(out),
                naive_calls_distinct=float(naive),
                formal_calls_distinct=float(np.mean(fd)) if fd else None,
                mean_overlap=float(np.mean([o["overlap"] for o in out])),
                mean_recall_small=float(np.mean([o["recall_small"] for o in out])),
                mean_recall_large=float(np.mean([o["recall_large"] for o in out])))


@app.local_entrypoint()
def main():
    cells = []
    i = 0
    for grid in (GRID, KSWEEP):
        keys = list(grid)
        for combo in itertools.product(*[grid[k] for k in keys]):
            c = dict(zip(keys, combo))
            c.update(seed=5000 + i, nrep=NREP)
            cells.append(c)
            i += 1
    seen, uniq = set(), []
    for c in cells:
        key = tuple(sorted((k, v) for k, v in c.items() if k not in ("seed",)))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    print(f"{len(uniq)} cells x {NREP} replicates = {len(uniq)*NREP:,} simulated studies")
    res = [r for r in cell.map(uniq, order_outputs=False) if r]
    out = str(paths.ROOT / "results/80_method_grid.json")
    json.dump(res, open(out, "w"))
    print(f"wrote {out} ({len(res)} cells)")

    import collections
    print("\n=== false-positive rate: architectures are the SAME ===")
    for scen in ("null", "dose"):
        rr = [r for r in res if r.get("scenario") == scen and r.get("n_small") == 230
              and r.get("p") == 162 and r.get("k") == 10 and r.get("beta") in (0.05, 0.1)]
        if rr:
            print(f"  {scen:9s} naive rule {np.mean([r['naive_calls_distinct'] for r in rr]):.3f}"
                  f"   formal test {np.mean([r['formal_calls_distinct'] for r in rr if r['formal_calls_distinct'] is not None]):.3f}")
    print("\n=== power: architectures genuinely DIFFER ===")
    for scen in ("partial", "disjoint"):
        rr = [r for r in res if r.get("scenario") == scen and r.get("n_small") == 230
              and r.get("p") == 162 and r.get("k") == 10 and r.get("beta") in (0.05, 0.1)]
        if rr:
            print(f"  {scen:9s} naive rule {np.mean([r['naive_calls_distinct'] for r in rr]):.3f}"
                  f"   formal test {np.mean([r['formal_calls_distinct'] for r in rr if r['formal_calls_distinct'] is not None]):.3f}")


import numpy as np
