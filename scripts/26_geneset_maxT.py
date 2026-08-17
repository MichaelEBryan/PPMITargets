#!/usr/bin/env python
"""
Genome-wide gene-set test with exact family-wise error control.

The previous run used 2,000 circular shifts, so the smallest attainable p-value
was 5e-4. Benjamini-Hochberg at rank 1 over 6,259 sets requires p <= 8e-6. No set
could have survived regardless of the data, so that negative result carried no
information.

This version fixes two things at once.

Resolution. The test statistic is computed in closed form by Frisch-Waugh-Lovell,
so a shift costs two sparse matrix-vector products instead of 6,259 regressions.
That makes 100,000 shifts cheap.

Multiplicity. Gene sets overlap heavily -- GO is a hierarchy, and Reactome and
KEGG restate much of it -- so Benjamini-Hochberg over 6,259 correlated tests is
the wrong correction. Instead we record the maximum statistic across all sets at
each shift. Comparing a set's observed statistic against that distribution gives
family-wise error control that accounts for the overlap exactly, with no
independence assumption.

Both the onset statistic and the onset-conditional-on-risk statistic are reported.
"""
import os, json, warnings
import numpy as np, pandas as pd
from scipy import stats
import paths as _P
warnings.filterwarnings("ignore")

BASE = str(_P.ROOT)
OUT, EXT = f"{BASE}/results", f"{BASE}/external"
NSHIFT = 20_000
rng = np.random.default_rng(2026)


def load():
    j = pd.read_csv(f"{OUT}/71_genestats_joined.csv", dtype={"chr": str})
    j["chr_i"] = j.chr.astype(int)
    return j.sort_values(["chr_i", "start"]).reset_index(drop=True)


def load_sets(present):
    sets = {}
    for lib in ["KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2023",
                "GO_Cellular_Component_2023", "WikiPathway_2023_Human"]:
        p = f"{EXT}/genesets/{lib}.txt"
        if not os.path.exists(p):
            continue
        for line in open(p):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gs = {g.split(",")[0].strip() for g in parts[2:] if g.strip()} & present
            if 10 <= len(gs) <= 500:
                sets[f"{lib}|{parts[0]}"] = gs
    return sets


def build_membership(j, sets):
    idx = {g: i for i, g in enumerate(j.gene)}
    rows, cols = [], []
    names = []
    for si, (nm, gs) in enumerate(sets.items()):
        names.append(nm)
        for g in gs:
            rows.append(idx[g]); cols.append(si)
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64), names


def run(j, rows, cols, names, y, X, tag):
    """t-statistic on set membership, adjusting for X, for every set at once."""
    n, nset = len(j), len(names)
    XtXi = np.linalg.pinv(X.T @ X)
    P_y = X @ (XtXi @ (X.T @ y))
    yt = y - P_y                      # residualised outcome, fixed across shifts
    sse_y = float(yt @ yt)
    size = np.bincount(cols, minlength=nset).astype(float)

    def stats_for(shift):
        r = rows if shift == 0 else (rows + shift) % n
        # m'y_tilde
        mty = np.bincount(cols, weights=yt[r], minlength=nset)
        # X'm  (3 x nset)
        XtM = np.vstack([size if np.all(X[:, k] == 1) else
                         np.bincount(cols, weights=X[r, k], minlength=nset)
                         for k in range(X.shape[1])])
        # m'P m
        mPm = np.einsum("ij,ij->j", XtM, XtXi @ XtM)
        denom = size - mPm
        denom = np.where(denom > 1e-8, denom, np.nan)
        beta = mty / denom
        # residual variance of the full model, per set
        sse = sse_y - beta * mty
        dof = n - X.shape[1] - 1
        se = np.sqrt(np.clip(sse, 1e-12, None) / dof / denom)
        return beta / se

    obs = stats_for(0)
    maxnull = np.empty(NSHIFT)
    cnt_ge = np.zeros(nset)
    step = max(1, NSHIFT // 10)
    for i in range(NSHIFT):
        s = int(rng.integers(1, n))
        t = stats_for(s)
        maxnull[i] = np.nanmax(t)
        cnt_ge += (t >= obs)
        if (i + 1) % step == 0:
            print(f"    {tag}: {i+1:,}/{NSHIFT:,} shifts", flush=True)
    p_point = (cnt_ge + 1) / (NSHIFT + 1)
    p_fwer = np.array([(np.sum(maxnull >= o) + 1) / (NSHIFT + 1) for o in obs])
    return pd.DataFrame(dict(pathway=names, n_genes=size.astype(int), t=obs,
                             p_pointwise=p_point, p_fwer=p_fwer)), maxnull


def main():
    j = load()
    present = set(j.gene)
    sets = load_sets(present)
    rows, cols, names = build_membership(j, sets)
    print(f"{len(j)} genes, {len(names)} sets, {len(rows):,} gene-set memberships")
    print(f"{NSHIFT:,} circular shifts -> pointwise p floor {1/(NSHIFT+1):.2e}; "
          f"family-wise p is read off the null of the maximum, so this "
          f"resolution is ample")

    y = j.z.values.astype(float)
    ones = np.ones(len(j))
    Xn = np.column_stack([ones, np.log10(j.n_snps.values)])              # size only
    Xc = np.column_stack([ones, np.log10(j.n_snps.values), j.z_risk.values])

    print("\n[1/2] onset signal, adjusting for gene size", flush=True)
    a, mn_a = run(j, rows, cols, names, y, Xn, "onset")
    print("\n[2/2] onset signal, adjusting for gene size and risk signal", flush=True)
    b, mn_b = run(j, rows, cols, names, y, Xc, "onset|risk")

    a = a.rename(columns={"t": "t_onset", "p_pointwise": "p_onset",
                          "p_fwer": "pfwer_onset"})
    b = b.rename(columns={"t": "t_onset_adj", "p_pointwise": "p_onset_adj",
                          "p_fwer": "pfwer_onset_adj"})
    m = a.merge(b.drop(columns=["n_genes"]), on="pathway").sort_values("p_onset_adj")
    m.to_csv(f"{OUT}/110_geneset_maxT.csv", index=False)
    np.save(f"{OUT}/110_maxnull_onset.npy", mn_a)
    np.save(f"{OUT}/110_maxnull_adj.npy", mn_b)

    print("\n=== onset signal beyond risk, family-wise corrected ===")
    print(f"{'pathway':58s} {'n':>4s} {'t':>6s} {'p':>9s} {'p_FWER':>8s}")
    for _, r in m.head(20).iterrows():
        nm = r.pathway.split("|", 1)[1]
        print(f"  {nm[:56]:56s} {int(r.n_genes):4d} {r.t_onset_adj:+6.2f} "
              f"{r.p_onset_adj:9.5f} {r.pfwer_onset_adj:8.4f}")
    n_fwer = int((m.pfwer_onset_adj < 0.05).sum())
    print(f"\nsets significant at family-wise 0.05: {n_fwer} of {len(m)}")
    print(f"null distribution of the maximum t: median {np.median(mn_b):.2f}, "
          f"95th pct {np.percentile(mn_b, 95):.2f}, max {mn_b.max():.2f}")
    print(f"largest observed t: {m.t_onset_adj.max():.2f}")

    for key in ["Lipoprotein", "ysosom", "utophag", "itochondri", "ynap"]:
        sub = m[m.pathway.str.contains(key, case=False)].head(3)
        if len(sub):
            print(f"\n-- '{key}' --")
            for _, r in sub.iterrows():
                print(f"   {r.pathway.split('|')[1][:50]:50s} n={int(r.n_genes):3d} "
                      f"t={r.t_onset_adj:+.2f} p={r.p_onset_adj:.5f} "
                      f"pFWER={r.pfwer_onset_adj:.4f}")

    json.dump(dict(n_sets=int(len(m)), n_shifts=NSHIFT,
                   smallest_attainable_p=1 / (NSHIFT + 1),
                   n_fwer05=n_fwer,
                   max_observed_t=float(m.t_onset_adj.max()),
                   null_max_t_p95=float(np.percentile(mn_b, 95)),
                   top=m.head(25).to_dict("records")),
              open(f"{OUT}/111_maxT_summary.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/110_geneset_maxT.csv")


if __name__ == "__main__":
    main()
