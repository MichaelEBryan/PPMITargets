#!/usr/bin/env python
"""
The analysis the question actually requires.

1. Ascertainment. Quantify how much of PPMI's genetic subset was recruited
   *because* of carrier status, and how that differs by age stratum.
2. Onset-age proxy. Age at enrolment in the de novo sporadic cohort, with its
   limits stated and tested.
3. Polygenic burden vs onset age among cases, with ancestry PCs, and with the
   genetically-ascertained participants removed.
4. Per-variant onset-age effects in PPMI, and their concordance with the IPDGC
   age-at-onset GWAS (28,568 cases) -- the replication test.
5. Case-case comparison (early-onset cases vs late-onset cases), which is the
   correct design for "distinct architecture", instead of two separate
   case-control models.
6. Power. What effect size PPMI could have detected, against what the AAO GWAS
   says is really there.
"""
import os, re, json, warnings, subprocess
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
import paths as _P
warnings.filterwarnings("ignore")

PPMI = str(_P.DATA / "PPMI" / "_Subject_Characteristics")
BASE = str(_P.ROOT)
OUT, EXT = f"{BASE}/results", f"{BASE}/external"
AAO = f"{EXT}/aao_gwas/IPDGC_AAO_GWAS_sumstats_april_2018.txt"
META5 = f"{EXT}/risk_gwas/GCST009325.tsv"
os.makedirs(OUT, exist_ok=True)
R = {}


def build():
    ps = pd.read_csv(f"{PPMI}/Participant_Status_15Aug2026.csv")
    prs = pd.read_csv(f"{PPMI}/Polygenic_Risk_Scores_15Aug2026.csv")
    p9 = pd.read_csv(f"{PPMI}/PPMI_Project_9001_20250624_15Aug2026.csv")
    dem = pd.read_csv(f"{PPMI}/Demographics_15Aug2026.csv")
    dem = dem.sort_values("PATNO").drop_duplicates("PATNO", keep="first")
    fam = pd.read_csv(f"{PPMI}/Family_History_15Aug2026.csv")
    fam = (fam.groupby("PATNO")["ANYFAMPD"].max().rename("ANYFAMPD").reset_index())

    d = (ps.merge(prs, on="PATNO", how="left")
           .merge(p9.drop(columns=["EVENT_ID"]), on="PATNO", how="left")
           .merge(dem[["PATNO", "SEX", "RAWHITE", "RABLACK", "RAASIAN", "HISPLAT"]],
                  on="PATNO", how="left")
           .merge(fam, on="PATNO", how="left"))
    enrl = [c for c in d.columns if c.startswith("ENRL")]
    d["n_enrl"] = d[enrl].fillna(0).sum(1)
    d["genetically_ascertained"] = d[["ENRLLRRK2", "ENRLGBA", "ENRLSNCA", "ENRLPRKN",
                                      "ENRLPINK1", "ENRLOTHGV"]].fillna(0).sum(1) > 0
    d["route"] = np.select(
        [d.ENRLLRRK2.fillna(0) > 0, d.ENRLGBA.fillna(0) > 0, d.ENRLSNCA.fillna(0) > 0,
         d[["ENRLPRKN", "ENRLPINK1", "ENRLOTHGV"]].fillna(0).sum(1) > 0,
         d.ENRLRBD.fillna(0) > 0, d.ENRLHPSM.fillna(0) > 0,
         d.ENRLNORM.fillna(0) > 0, d.ENRLSRDC.fillna(0) > 0],
        ["LRRK2 carrier", "GBA carrier", "SNCA carrier", "other gene carrier",
         "RBD", "hyposmia", "normal control", "sporadic PD"], default="unspecified")
    return d, enrl


def sec1_ascertainment(d):
    pdc = d[(d.COHORT_DEFINITION == "Parkinson's Disease") & d.ENROLL_AGE.notna()
            & d.META5_PGS.notna()].copy()
    pdc["stratum"] = np.where(pdc.ENROLL_AGE < 50, "EO(<50)",
                       np.where(pdc.ENROLL_AGE >= 60, "LO(>=60)", "mid(50-59)"))
    tab = pd.crosstab(pdc.route, pdc.stratum, margins=True)
    tab.to_csv(f"{OUT}/10_ascertainment_route_by_stratum.csv")

    eo = pdc[pdc.stratum == "EO(<50)"]; lo = pdc[pdc.stratum == "LO(>=60)"]
    a, b = eo.genetically_ascertained.sum(), lo.genetically_ascertained.sum()
    ct = np.array([[a, len(eo) - a], [b, len(lo) - b]])
    orr, pv = stats.fisher_exact(ct)
    R["ascertainment"] = dict(
        n_PD_with_PRS=int(len(pdc)),
        pct_genetically_ascertained_overall=float(pdc.genetically_ascertained.mean() * 100),
        pct_gen_ascertained_EO=float(a / len(eo) * 100), n_EO=int(len(eo)),
        pct_gen_ascertained_LO=float(b / len(lo) * 100), n_LO=int(len(lo)),
        fisher_OR=float(orr), fisher_p=float(pv),
        note=("Fraction of PD cases recruited because they carry a pathogenic variant, "
              "by age stratum. A difference here means the two strata differ in how "
              "they were assembled, before any biology."))
    print(f"[1] genetically-ascertained PD cases: EO {a}/{len(eo)} = {a/len(eo)*100:.1f}%  "
          f"LO {b}/{len(lo)} = {b/len(lo)*100:.1f}%  OR={orr:.2f} p={pv:.3g}", flush=True)
    return pdc


def sec2_prs(pdc):
    """Polygenic burden vs age at enrolment among PD cases."""
    pcs = [f"Genetic_PRS_PC{i}" for i in range(1, 11)]
    rows = []
    for label, sub in [("all PD cases", pdc),
                       ("sporadic only (drop genetic cohort)", pdc[~pdc.genetically_ascertained]),
                       ("EUR-inferred only", pdc[pdc.Genetic_PRS_InfPop == "EUR"]),
                       ("EUR + sporadic only", pdc[(pdc.Genetic_PRS_InfPop == "EUR")
                                                   & (~pdc.genetically_ascertained)])]:
        for score in ["META5_PGS", "META5_excl_LRRK2_GBA_PGS", "GP2_PGS"]:
            for adj in ["sex only", "sex + 10 PCs"]:
                s = sub.dropna(subset=[score, "ENROLL_AGE", "SEX"]).copy()
                cov = ["SEX"] + (pcs if adj == "sex + 10 PCs" else [])
                s = s.dropna(subset=[c for c in cov if c in s.columns])
                if len(s) < 60:
                    continue
                z = (s[score] - s[score].mean()) / s[score].std()
                X = sm.add_constant(pd.concat([z.rename("PRS_z"), s[cov]], axis=1).astype(float))
                m = sm.OLS(s.ENROLL_AGE.astype(float), X).fit()
                rows.append(dict(subset=label, score=score, adjustment=adj, n=len(s),
                                 beta_years_per_SD=m.params["PRS_z"],
                                 se=m.bse["PRS_z"], p=m.pvalues["PRS_z"],
                                 ci_lo=m.conf_int().loc["PRS_z", 0],
                                 ci_hi=m.conf_int().loc["PRS_z", 1],
                                 r2_partial=m.rsquared))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/11_prs_vs_onsetage.csv", index=False)
    head = df[(df.subset == "EUR + sporadic only") & (df.score == "META5_PGS")
              & (df.adjustment == "sex + 10 PCs")]
    if len(head):
        h = head.iloc[0]
        R["prs_vs_age"] = dict(n=int(h.n), beta_years_per_SD=float(h.beta_years_per_SD),
                               se=float(h.se), p=float(h.p),
                               ci=[float(h.ci_lo), float(h.ci_hi)])
        print(f"[2] META5 PRS vs age at enrolment (EUR sporadic, +PCs): "
              f"beta={h.beta_years_per_SD:+.2f} y/SD (95% CI {h.ci_lo:+.2f},{h.ci_hi:+.2f}) "
              f"p={h.p:.3g}  n={int(h.n)}", flush=True)
    return df


def sec3_pervariant(pdc, snpmap):
    """Per-variant onset-age effect in PPMI + case-case EO vs LO test."""
    pcs = [f"Genetic_PRS_PC{i}" for i in range(1, 11)]
    s = pdc[(pdc.Genetic_PRS_InfPop == "EUR")].dropna(subset=["ENROLL_AGE", "SEX"] + pcs).copy()
    rows = []
    for _, r in snpmap.iterrows():
        c = r.col
        if c not in s.columns:
            continue
        g = pd.to_numeric(s[c], errors="coerce")
        if g.notna().sum() < 100 or g.std() == 0:
            continue
        ok = g.notna()
        X = sm.add_constant(pd.concat([g[ok].rename("dose"), s.loc[ok, ["SEX"] + pcs]],
                                      axis=1).astype(float))
        m = sm.OLS(s.loc[ok, "ENROLL_AGE"].astype(float), X).fit()
        # case-case: EO vs LO among cases
        cc = s.loc[ok].copy(); cc["dose"] = g[ok]
        cc = cc[(cc.ENROLL_AGE < 50) | (cc.ENROLL_AGE >= 60)]
        yb = (cc.ENROLL_AGE < 50).astype(int)
        try:
            lg = sm.Logit(yb, sm.add_constant(cc[["dose", "SEX"] + pcs].astype(float))).fit(disp=0)
            cc_beta, cc_p = lg.params["dose"], lg.pvalues["dose"]
        except Exception:
            cc_beta, cc_p = np.nan, np.nan
        rows.append(dict(rsid=r.rsid, chr37=r.chr37, pos37=r.pos37, effect_allele=r.effect_allele,
                         csq=r.csq, n=int(ok.sum()), maf_ppmi=float(g[ok].mean() / 2),
                         beta_years=m.params["dose"], se=m.bse["dose"], p=m.pvalues["dose"],
                         cc_logOR_EOvsLO=cc_beta, cc_p=cc_p, n_cc=int(len(cc))))
    df = pd.DataFrame(rows)
    if len(df):
        df["fdr"] = sm.stats.multipletests(df.p.fillna(1), method="fdr_bh")[1]
        df["cc_fdr"] = sm.stats.multipletests(df.cc_p.fillna(1), method="fdr_bh")[1]
    df.to_csv(f"{OUT}/12_pervariant_onsetage.csv", index=False)
    R["pervariant"] = dict(n_tested=int(len(df)),
                           n_p_lt_0p05=int((df.p < 0.05).sum()),
                           n_fdr_lt_0p05=int((df.fdr < 0.05).sum()),
                           n_bonferroni=int((df.p < 0.05 / max(len(df), 1)).sum()),
                           casecase_n_fdr_lt_0p05=int((df.cc_fdr < 0.05).sum()),
                           expected_by_chance_at_p0p05=float(0.05 * len(df)))
    print(f"[3] per-variant onset-age: {len(df)} SNPs tested, "
          f"{(df.p<0.05).sum()} at p<0.05 (expected {0.05*len(df):.1f} by chance), "
          f"{(df.fdr<0.05).sum()} survive FDR; case-case {(df.cc_fdr<0.05).sum()} survive FDR",
          flush=True)
    return df


def sec4_replication(pv):
    """Concordance of PPMI per-variant onset effects with the IPDGC AAO GWAS."""
    want = set()
    for _, r in pv.iterrows():
        if pd.notna(r.pos37):
            want.add(f"chr{int(r.chr37)}:{int(r.pos37)}")
    q = f"{OUT}/_aao_hits.tsv"
    with open(f"{OUT}/_want.txt", "w") as f:
        f.write("\n".join(want))
    subprocess.run(f"awk 'NR==FNR{{a[$1];next}} FNR==1||($1 in a)' {OUT}/_want.txt {AAO} > {q}",
                   shell=True, check=True)
    aao = pd.read_csv(q, sep="\t")
    print(f"[4] matched {len(aao)-0} of {len(want)} panel SNPs in the IPDGC AAO GWAS", flush=True)
    aao["key"] = aao.MarkerName
    pv = pv.copy()
    pv["key"] = pv.apply(lambda r: f"chr{int(r.chr37)}:{int(r.pos37)}"
                         if pd.notna(r.pos37) else None, axis=1)
    mg = pv.merge(aao, on="key", how="inner")
    # align effect alleles
    flip = mg.effect_allele.str.upper() != mg.Allele1.str.upper()
    mg["aao_beta_aligned"] = np.where(flip, -mg.Effect, mg.Effect)
    mg["aao_se"] = mg.StdErr
    mg = mg[mg.beta_years.notna() & mg.aao_beta_aligned.notna()]
    if len(mg) > 5:
        rp, pp = stats.pearsonr(mg.beta_years, mg.aao_beta_aligned)
        rs, ps = stats.spearmanr(mg.beta_years, mg.aao_beta_aligned)
        sign = float((np.sign(mg.beta_years) == np.sign(mg.aao_beta_aligned)).mean())
        binom = stats.binomtest(int((np.sign(mg.beta_years) ==
                                     np.sign(mg.aao_beta_aligned)).sum()), len(mg), 0.5)
        # inverse-variance weighted regression of PPMI beta on IPDGC beta
        w = 1 / mg.se ** 2
        wls = sm.WLS(mg.beta_years, sm.add_constant(mg.aao_beta_aligned), weights=w).fit()
        R["replication_AAO"] = dict(
            n_snps=int(len(mg)), pearson_r=float(rp), pearson_p=float(pp),
            spearman_r=float(rs), spearman_p=float(ps),
            sign_concordance=sign, sign_binom_p=float(binom.pvalue),
            wls_slope=float(wls.params.iloc[1]), wls_slope_se=float(wls.bse.iloc[1]),
            wls_slope_p=float(wls.pvalues.iloc[1]),
            interpretation=("slope 1 would mean PPMI recovers the IPDGC effect sizes; "
                            "slope indistinguishable from 0 means PPMI per-variant "
                            "onset estimates carry no replicable signal"))
        print(f"[4] PPMI vs IPDGC AAO GWAS ({len(mg)} SNPs): r={rp:+.3f} (p={pp:.3g}), "
              f"sign concordance {sign*100:.0f}% (p={binom.pvalue:.3g}), "
              f"WLS slope={wls.params.iloc[1]:+.3f}+/-{wls.bse.iloc[1]:.3f}", flush=True)
        mg.to_csv(f"{OUT}/13_ppmi_vs_ipdgc_aao.csv", index=False)
    return mg


def sec5_power(pv, mg):
    """What could PPMI have detected, versus what is really there."""
    n = int(pv.n.median())
    sd_age = 9.7
    out = []
    for alpha, lab in [(0.05, "nominal"), (0.05 / max(len(pv), 1), "Bonferroni")]:
        zc = stats.norm.ppf(1 - alpha / 2)
        for maf in (0.05, 0.15, 0.30, 0.45):
            v = 2 * maf * (1 - maf)
            se = sd_age / np.sqrt(n * v)
            mde = (zc + stats.norm.ppf(0.8)) * se   # 80% power
            out.append(dict(alpha_label=lab, alpha=alpha, maf=maf, n=n,
                            mde_years_per_allele=mde))
    pw = pd.DataFrame(out)
    pw.to_csv(f"{OUT}/14_power.csv", index=False)
    if len(mg):
        real = np.abs(mg.aao_beta_aligned)
        R["power"] = dict(
            n_used=n,
            mde_years_per_allele_at_maf0p3_bonferroni=float(
                pw[(pw.maf == 0.30) & (pw.alpha_label == "Bonferroni")].mde_years_per_allele.iloc[0]),
            mde_years_per_allele_at_maf0p3_nominal=float(
                pw[(pw.maf == 0.30) & (pw.alpha_label == "nominal")].mde_years_per_allele.iloc[0]),
            ipdgc_median_abs_effect_years=float(real.median()),
            ipdgc_max_abs_effect_years=float(real.max()),
            n_panel_snps_above_PPMI_bonferroni_MDE=int(
                (real > pw[(pw.maf == 0.30) & (pw.alpha_label == "Bonferroni")]
                 .mde_years_per_allele.iloc[0]).sum()))
        print(f"[5] PPMI n={n}: minimum detectable onset effect (MAF 0.30, 80% power, "
              f"Bonferroni) = {R['power']['mde_years_per_allele_at_maf0p3_bonferroni']:.2f} "
              f"y/allele. IPDGC median true effect = {real.median():.2f} y/allele; "
              f"{R['power']['n_panel_snps_above_PPMI_bonferroni_MDE']} of {len(mg)} panel "
              f"SNPs exceed that threshold.", flush=True)
    return pw


def main():
    d, enrl = build()
    snpmap = pd.read_csv(f"{BASE}/data/snp_map.csv")
    pdc = sec1_ascertainment(d)
    sec2_prs(pdc)
    pv = sec3_pervariant(pdc, snpmap)
    mg = sec4_replication(pv)
    sec5_power(pv, mg)
    json.dump(R, open(f"{OUT}/15_corrected_analysis_summary.json", "w"), indent=2, default=float)
    print("\nwrote", f"{OUT}/15_corrected_analysis_summary.json")


if __name__ == "__main__":
    main()
