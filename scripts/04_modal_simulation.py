#!/usr/bin/env python
"""
Ground-truth simulation: what does 'no overlap between strata' actually mean?

We generate two strata that share an IDENTICAL genetic architecture by
construction -- same causal SNPs, same effect sizes, same allele frequencies,
same LD blocks -- and differ only in sample size, matching the published
n_EOPD = 230 vs n_LOPD = 3386 asymmetry. We then run the published selection
procedure (gradient-boosted trees + mean|SHAP|, take top-k) separately in each
stratum and record the overlap between the two top-k lists.

Under the published inference rule ("no overlap => distinct genetic
architectures"), every replicate here is a false positive.

We sweep sample size, number of candidate SNPs, number of causal variants,
per-allele effect size, and LD block structure, and also record how often the
procedure recovers the variants that are genuinely causal.

Run:  modal run 04_modal_simulation.py
"""
import modal, json, os
import paths as _P

app = modal.App("pd-arch-selection-null")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==2.1.3", "pandas==2.2.3", "scikit-learn==1.5.2",
                      "xgboost==2.1.3", "shap==0.46.0", "scipy==1.14.1"))
vol = modal.Volume.from_name("pd-arch-sim", create_if_missing=True)

GRID = dict(
    n_small=[100, 230, 500, 1000],
    n_large=[1000, 3386],
    p=[50, 162, 500],
    m_causal=[5, 15],
    beta=[0.05, 0.10, 0.20, 0.40, 0.70],
    ld=[0.0, 0.6],
)
NREP = 200
K_A, K_B = 10, 13


@app.function(image=image, cpu=4.0, timeout=3600, volumes={"/out": vol},
              max_containers=120, retries=2)
def run_cell(cfg: dict):
    import numpy as np, warnings
    warnings.filterwarnings("ignore")
    import xgboost as xgb, shap

    n_small, n_large = cfg["n_small"], cfg["n_large"]
    p, m, beta, ld = cfg["p"], cfg["m_causal"], cfg["beta"], cfg["ld"]
    rng = np.random.default_rng(cfg["seed"])

    def gen(n, maf, causal, betas, latent_map):
        """Genotypes with block LD, then a binary case/control liability."""
        if ld > 0:
            nb = latent_map.max() + 1
            z = rng.standard_normal((n, nb))[:, latent_map]
            e = rng.standard_normal((n, p))
            u = np.sqrt(ld) * z + np.sqrt(1 - ld) * e
            # map latent normal -> genotype dosage under HWE with given MAF
            from scipy.stats import norm
            q = norm.cdf(u)
            G = np.zeros((n, p), dtype=np.int8)
            for j in range(p):
                f = maf[j]
                c0, c1 = (1 - f) ** 2, (1 - f) ** 2 + 2 * f * (1 - f)
                G[:, j] = (q[:, j] > c0).astype(np.int8) + (q[:, j] > c1).astype(np.int8)
        else:
            G = rng.binomial(2, maf, size=(n, p)).astype(np.int8)
        lin = G[:, causal] @ betas
        pr = 1 / (1 + np.exp(-(lin - lin.mean())))
        y = rng.binomial(1, pr)
        return G, y

    def select(G, y, k):
        if len(np.unique(y)) < 2:
            return None
        mdl = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                n_jobs=4, verbosity=0, eval_metric="logloss")
        mdl.fit(G, y)
        sv = shap.TreeExplainer(mdl).shap_values(G)
        imp = np.abs(np.asarray(sv)).mean(0)
        if imp.ndim > 1:
            imp = imp.mean(-1)
        return np.argsort(-imp)[:k]

    out = []
    for r in range(cfg["nrep"]):
        maf = rng.uniform(0.05, 0.5, p)
        causal = rng.choice(p, m, replace=False)
        betas = np.full(m, beta) * rng.choice([-1, 1], m)
        latent_map = np.repeat(np.arange((p + 4) // 5), 5)[:p]
        GA, yA = gen(n_small, maf, causal, betas, latent_map)
        GB, yB = gen(n_large, maf, causal, betas, latent_map)
        sA, sB = select(GA, yA, K_A), select(GB, yB, K_B)
        if sA is None or sB is None:
            continue
        A, B, C = set(sA.tolist()), set(sB.tolist()), set(causal.tolist())
        out.append(dict(overlap=len(A & B), jaccard=len(A & B) / len(A | B),
                        recall_small=len(A & C) / m, recall_large=len(B & C) / m,
                        prec_small=len(A & C) / K_A, prec_large=len(B & C) / K_B,
                        case_frac_small=float(yA.mean()), case_frac_large=float(yB.mean())))
    import numpy as np
    agg = {k: float(np.mean([o[k] for o in out])) for k in out[0]} if out else {}
    agg.update({f"{k}_sd": float(np.std([o[k] for o in out])) for k in ("overlap", "recall_small")} if out else {})
    return dict(**cfg, nvalid=len(out),
                p_zero_overlap=float(np.mean([o["overlap"] == 0 for o in out])) if out else None,
                **{f"mean_{k}": v for k, v in agg.items()},
                raw_overlap=[o["overlap"] for o in out])


@app.local_entrypoint()
def main():
    import itertools, json, os
    keys = list(GRID)
    cells = []
    for i, combo in enumerate(itertools.product(*[GRID[k] for k in keys])):
        cfg = dict(zip(keys, combo))
        cfg.update(seed=1000 + i, nrep=NREP)
        cells.append(cfg)
    print(f"{len(cells)} grid cells x {NREP} replicates = {len(cells)*NREP:,} simulated studies")
    res = list(run_cell.map(cells, order_outputs=True))
    out = str(_P.ROOT / "results/09_simulation_grid.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"))
    print(f"wrote {out}")
    ok = [r for r in res if r and r.get("p_zero_overlap") is not None]
    print(f"\n{len(ok)} cells returned")
    # headline: realistic PD operating point
    for r in ok:
        if r["n_small"] == 230 and r["n_large"] == 3386 and r["p"] == 162 and r["beta"] in (0.10, 0.20):
            print(f"  n=230/3386 p=162 m={r['m_causal']} beta={r['beta']} ld={r['ld']}: "
                  f"P(zero overlap)={r['p_zero_overlap']:.3f} "
                  f"mean overlap={r['mean_overlap']:.2f} "
                  f"recall_small={r['mean_recall_small']:.3f} "
                  f"recall_large={r['mean_recall_large']:.3f}")
