import sys
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

warnings.filterwarnings("ignore")

BASE = str(paths.ROOT)
OUT = f"{BASE}/results"
PCS = [f"Genetic_PRS_PC{i}" for i in range(1, 11)]
R = {}


def core(d):
    c = d[(d.COHORT_DEFINITION == "Parkinson's Disease") & d.AAO.notna()].copy()
    return c


def sec_prs(c):
    rows = []
    defs = [("all cases with PRS", lambda x: x),
            ("sporadic only", lambda x: x[~x.gen_asc]),
            ("EUR only", lambda x: x[x.Genetic_PRS_InfPop == "EUR"]),
            ("EUR + sporadic", lambda x: x[(x.Genetic_PRS_InfPop == "EUR") & (~x.gen_asc)])]
    for lab, f in defs:
        for score in ["META5_PGS", "META5_excl_LRRK2_GBA_PGS", "GP2_PGS"]:
            for adj, cov in [("sex", ["SEX"]), ("sex + 10 PCs", ["SEX"] + PCS)]:
                s = f(c).dropna(subset=[score, "AAO"] + cov)
                if len(s) < 60:
                    continue
                z = (s[score] - s[score].mean()) / s[score].std()
                X = sm.add_constant(pd.concat([z.rename("PRS_z"), s[cov]], axis=1).astype(float))
                m = sm.OLS(s.AAO.astype(float), X).fit()
                ci = m.conf_int().loc["PRS_z"]
                rows.append(dict(subset=lab, score=score, adjustment=adj, n=len(s),
                                 beta_years_per_SD=m.params["PRS_z"], se=m.bse["PRS_z"],
                                 ci_lo=ci[0], ci_hi=ci[1], p=m.pvalues["PRS_z"]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/51_prs_vs_AAO.csv", index=False)
    h = df[(df.subset == "EUR + sporadic") & (df.score == "META5_PGS")
           & (df.adjustment == "sex + 10 PCs")]
    if len(h):
        h = h.iloc[0]
        R["prs_vs_AAO"] = dict(n=int(h.n), beta=float(h.beta_years_per_SD),
                               se=float(h.se), ci=[float(h.ci_lo), float(h.ci_hi)],
                               p=float(h.p))
        print(f"[PRS] META5 vs true onset age (EUR sporadic, +PCs): "
              f"{h.beta_years_per_SD:+.2f} y/SD (95% CI {h.ci_lo:+.2f},{h.ci_hi:+.2f}) "
              f"p={h.p:.3g}  n={int(h.n)}", flush=True)
    s = c.dropna(subset=["META5_PGS", "dx_delay"] + PCS + ["SEX"])
    s = s[(s.dx_delay >= 0) & (s.dx_delay < 20)]
    z = (s.META5_PGS - s.META5_PGS.mean()) / s.META5_PGS.std()
    m = sm.OLS(s.dx_delay.astype(float),
               sm.add_constant(pd.concat([z.rename("PRS_z"), s[["SEX"] + PCS]], axis=1).astype(float))).fit()
    R["prs_vs_dxdelay"] = dict(n=int(len(s)), beta=float(m.params["PRS_z"]),
                               p=float(m.pvalues["PRS_z"]))
    print(f"[PRS] negative control, PRS vs diagnostic delay: "
          f"{m.params['PRS_z']:+.3f} y/SD p={m.pvalues['PRS_z']:.3f} (n={len(s)})", flush=True)
    return df


def sec_survival(c):
    s = c[(c.Genetic_PRS_InfPop == "EUR") & (~c.gen_asc)].dropna(
        subset=["META5_PGS", "AAO", "SEX"] + PCS)
    z = (s.META5_PGS - s.META5_PGS.mean()) / s.META5_PGS.std()
    df = pd.concat([s.AAO.rename("T"), pd.Series(1, index=s.index, name="E"),
                    z.rename("PRS_z"), s[["SEX"] + PCS]], axis=1).astype(float)
    cph = CoxPHFitter().fit(df, "T", "E")
    hr = float(np.exp(cph.params_["PRS_z"]))
    lo, hi = np.exp(cph.confidence_intervals_.loc["PRS_z"].values)
    p = float(cph.summary.loc["PRS_z", "p"])
    R["cox"] = dict(n=int(len(df)), HR_per_SD=hr, ci=[float(lo), float(hi)], p=p)
    print(f"[Cox] hazard of onset per SD of PRS: HR={hr:.3f} "
          f"(95% CI {lo:.3f}-{hi:.3f}) p={p:.3g}  n={len(df)}", flush=True)
    cph.summary.to_csv(f"{OUT}/52_cox_prs.csv")
    return cph


def sec_casecase(c, snpmap):
    s = c[(c.Genetic_PRS_InfPop == "EUR") & (~c.gen_asc)].dropna(subset=["AAO", "SEX"] + PCS)
    s = s[(s.AAO < 50) | (s.AAO >= 60)].copy()
    s["EO"] = (s.AAO < 50).astype(int)
    out = {}
    sp = s.dropna(subset=["META5_PGS"])
    z = (sp.META5_PGS - sp.META5_PGS.mean()) / sp.META5_PGS.std()
    lg = sm.Logit(sp.EO, sm.add_constant(pd.concat(
        [z.rename("PRS_z"), sp[["SEX"] + PCS]], axis=1).astype(float))).fit(disp=0)
    out["prs"] = dict(n_EO=int(sp.EO.sum()), n_LO=int((1 - sp.EO).sum()),
                      OR=float(np.exp(lg.params["PRS_z"])),
                      ci=[float(np.exp(lg.conf_int().loc["PRS_z", 0])),
                          float(np.exp(lg.conf_int().loc["PRS_z", 1]))],
                      p=float(lg.pvalues["PRS_z"]))
    print(f"[case-case] true onset <50 (n={out['prs']['n_EO']}) vs >=60 "
          f"(n={out['prs']['n_LO']}): OR={out['prs']['OR']:.3f} per SD "
          f"(95% CI {out['prs']['ci'][0]:.3f}-{out['prs']['ci'][1]:.3f}) "
          f"p={out['prs']['p']:.4f}", flush=True)

    rows = []
    for _, r in snpmap.iterrows():
        col = r.col
        if col not in s.columns:
            continue
        g = pd.to_numeric(s[col], errors="coerce")
        ok = g.notna()
        if ok.sum() < 80 or g[ok].std() == 0:
            continue
        X = sm.add_constant(pd.concat([g[ok].rename("dose"), s.loc[ok, ["SEX"] + PCS]],
                                      axis=1).astype(float))
        try:
            lg = sm.Logit(s.loc[ok, "EO"], X).fit(disp=0)
            b, p = lg.params["dose"], lg.pvalues["dose"]
        except Exception:
            b, p = np.nan, np.nan
        cs = c[(c.Genetic_PRS_InfPop == "EUR") & (~c.gen_asc)].dropna(subset=["AAO", "SEX"] + PCS)
        gc = pd.to_numeric(cs[col], errors="coerce")
        ok2 = gc.notna()
        m = sm.OLS(cs.loc[ok2, "AAO"].astype(float), sm.add_constant(pd.concat(
            [gc[ok2].rename("dose"), cs.loc[ok2, ["SEX"] + PCS]], axis=1).astype(float))).fit()
        rows.append(dict(rsid=r.rsid, chr37=r.chr37, pos37=r.pos37,
                         effect_allele=r.effect_allele, csq=r.csq,
                         n_cc=int(ok.sum()), cc_logOR=b, cc_p=p,
                         n_aao=int(ok2.sum()), aao_beta=m.params["dose"],
                         aao_se=m.bse["dose"], aao_p=m.pvalues["dose"]))
    pv = pd.DataFrame(rows)
    pv["aao_fdr"] = sm.stats.multipletests(pv.aao_p.fillna(1), method="fdr_bh")[1]
    pv["cc_fdr"] = sm.stats.multipletests(pv.cc_p.fillna(1), method="fdr_bh")[1]
    pv.to_csv(f"{OUT}/53_pervariant_AAO.csv", index=False)
    chi = stats.chi2.isf(pv.cc_p.dropna(), 1).sum()
    ndf = int(pv.cc_p.notna().sum())
    out["global_casecase"] = dict(chi2=float(chi), df=ndf,
                                  p=float(stats.chi2.sf(chi, ndf)))
    print(f"[case-case] global heterogeneity across {ndf} variants: "
          f"chi2={chi:.1f} df={ndf} p={stats.chi2.sf(chi, ndf):.3f}", flush=True)
    print(f"[per-variant] AAO: {int((pv.aao_p<0.05).sum())} at p<0.05 "
          f"(expected {0.05*len(pv):.1f}), {int((pv.aao_fdr<0.05).sum())} survive FDR",
          flush=True)
    R["casecase"] = out
    return pv


def sec_replication(pv):
    aao = pd.read_csv(f"{OUT}/_aao_hits.tsv", sep="\t")
    pv = pv.copy()
    pv["key"] = pv.apply(lambda r: f"chr{int(r.chr37)}:{int(r.pos37)}"
                         if pd.notna(r.pos37) else None, axis=1)
    mg = pv.merge(aao.rename(columns={"MarkerName": "key"}), on="key", how="inner")
    flip = mg.effect_allele.str.upper() != mg.Allele1.str.upper()
    mg["ipdgc_beta"] = np.where(flip, -mg.Effect, mg.Effect)
    mg = mg[mg.aao_beta.notna() & mg.ipdgc_beta.notna()]
    r, p = stats.pearsonr(mg.aao_beta, mg.ipdgc_beta)
    w = 1 / mg.aao_se ** 2
    wls = sm.WLS(mg.aao_beta, sm.add_constant(mg.ipdgc_beta), weights=w).fit()
    sign = float((np.sign(mg.aao_beta) == np.sign(mg.ipdgc_beta)).mean())
    R["replication"] = dict(n=int(len(mg)), pearson_r=float(r), p=float(p),
                            wls_slope=float(wls.params.iloc[1]),
                            wls_se=float(wls.bse.iloc[1]),
                            wls_p=float(wls.pvalues.iloc[1]), sign_concordance=sign)
    print(f"[replication] PPMI true-AAO effects vs IPDGC AAO GWAS ({len(mg)} SNPs): "
          f"r={r:+.3f} p={p:.3g}, WLS slope={wls.params.iloc[1]:+.3f}"
          f"+/-{wls.bse.iloc[1]:.3f}, sign concordance {sign*100:.0f}%", flush=True)
    mg.to_csv(f"{OUT}/54_ppmi_AAO_vs_ipdgc.csv", index=False)
    return mg


def sec_power(pv, mg):
    n = int(pv.n_aao.median())
    sd = 10.7
    rows = []
    for alpha, lab in [(0.05, "nominal"), (0.05 / max(len(pv), 1), "Bonferroni")]:
        zc = stats.norm.ppf(1 - alpha / 2)
        for maf in (0.05, 0.15, 0.30, 0.45):
            se = sd / np.sqrt(n * 2 * maf * (1 - maf))
            rows.append(dict(alpha_label=lab, maf=maf, n=n,
                             mde=(zc + stats.norm.ppf(0.8)) * se))
    pw = pd.DataFrame(rows)
    pw.to_csv(f"{OUT}/55_power_AAO.csv", index=False)
    mde = pw[(pw.maf == 0.30) & (pw.alpha_label == "Bonferroni")].mde.iloc[0]
    R["power"] = dict(n=n, mde_bonferroni_maf030=float(mde),
                      ipdgc_median_abs=float(np.abs(mg.ipdgc_beta).median()),
                      n_above=int((np.abs(mg.ipdgc_beta) > mde).sum()), n_snps=int(len(mg)))
    print(f"[power] n={n}: detectable onset effect (MAF 0.30, Bonferroni) = "
          f"{mde:.2f} y/allele; IPDGC median true effect "
          f"{np.abs(mg.ipdgc_beta).median():.2f}; "
          f"{int((np.abs(mg.ipdgc_beta) > mde).sum())} of {len(mg)} detectable", flush=True)


def main():
    d = pd.read_parquet(f"{BASE}/data/ppmi_aao.parquet")
    snpmap = pd.read_csv(f"{BASE}/data/snp_map.csv")
    c = core(d)
    print(f"PD cases with age at onset: {len(c)}\n")
    sec_prs(c)
    sec_survival(c)
    pv = sec_casecase(c, snpmap)
    mg = sec_replication(pv)
    sec_power(pv, mg)
    json.dump(R, open(f"{OUT}/56_aao_core_summary.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/56_aao_core_summary.json")


if __name__ == "__main__":
    main()
