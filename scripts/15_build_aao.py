#!/usr/bin/env python
"""
Build the definitive age-at-onset analysis set from PD_Diagnosis_History.

Age at onset is (SXDT - BIRTHDT); age at diagnosis is (PDDXDT - BIRTHDT). Both
are MM/YYYY in PPMI, so ages are accurate to the month. We keep both, plus the
diagnostic delay, and compare each against the enrolment-age proxy the earlier
analysis was forced to use.
"""
import os, json
import numpy as np, pandas as pd
import paths as _P

PPMI = str(_P.DATA / "PPMI")
SC = f"{PPMI}/_Subject_Characteristics"
OUT = str(_P.ROOT)


def mmyyyy(s):
    return pd.to_datetime(s, format="%m/%Y", errors="coerce")


def main():
    dx = pd.read_csv(f"{PPMI}/PD_Diagnosis_History_16Aug2026.csv")
    dx = dx.sort_values(["PATNO", "INFODT"]).drop_duplicates("PATNO", keep="first")
    ps = pd.read_csv(f"{SC}/Participant_Status_15Aug2026.csv")
    prs = pd.read_csv(f"{SC}/Polygenic_Risk_Scores_15Aug2026.csv")
    p9 = pd.read_csv(f"{SC}/PPMI_Project_9001_20250624_15Aug2026.csv").drop(columns=["EVENT_ID"])
    dem = pd.read_csv(f"{SC}/Demographics_15Aug2026.csv").sort_values("PATNO").drop_duplicates("PATNO")
    fam = pd.read_csv(f"{SC}/Family_History_15Aug2026.csv").groupby("PATNO").ANYFAMPD.max().reset_index()
    edu = (pd.read_csv(f"{SC}/Socio-Economics_15Aug2026.csv")
           .sort_values("PATNO").drop_duplicates("PATNO")[["PATNO", "EDUCYRS"]])

    d = (ps.merge(dx[["PATNO", "SXDT", "PDDXDT", "DXTREMOR", "DXRIGID", "DXBRADY",
                      "DXPOSINS", "DOMSIDE"]], on="PATNO", how="left")
           .merge(prs, on="PATNO", how="left")
           .merge(p9, on="PATNO", how="left")
           .merge(dem[["PATNO", "BIRTHDT", "SEX", "RAWHITE", "HISPLAT"]], on="PATNO", how="left")
           .merge(fam, on="PATNO", how="left")
           .merge(edu, on="PATNO", how="left"))

    b = mmyyyy(d.BIRTHDT)
    d["AAO"] = (mmyyyy(d.SXDT) - b).dt.days / 365.25
    d["AAD"] = (mmyyyy(d.PDDXDT) - b).dt.days / 365.25
    d["dx_delay"] = d.AAD - d.AAO
    d["gen_asc"] = d[["ENRLLRRK2", "ENRLGBA", "ENRLSNCA", "ENRLPRKN",
                      "ENRLPINK1", "ENRLOTHGV"]].fillna(0).sum(1) > 0
    d["carrier_route"] = np.select(
        [d.ENRLLRRK2.fillna(0) > 0, d.ENRLGBA.fillna(0) > 0, d.ENRLSNCA.fillna(0) > 0],
        ["LRRK2", "GBA", "SNCA"], default="none")

    pdc = d[d.COHORT_DEFINITION == "Parkinson's Disease"].copy()
    print(f"PD cases: {len(pdc)}")
    print(f"  with age at onset:     {pdc.AAO.notna().sum()}")
    print(f"  with age at diagnosis: {pdc.AAD.notna().sum()}")
    print(f"  with AAO + PRS:        {(pdc.AAO.notna() & pdc.META5_PGS.notna()).sum()}")
    print(f"  with AAO + PRS + PCs:  {(pdc.AAO.notna() & pdc.META5_PGS.notna() & pdc.Genetic_PRS_PC1.notna()).sum()}")
    a = pdc.AAO.dropna()
    print(f"\nage at onset: mean {a.mean():.1f}  sd {a.std():.1f}  "
          f"range {a.min():.1f}-{a.max():.1f}")
    print(f"  <50: {(a<50).sum()}   50-59: {((a>=50)&(a<60)).sum()}   >=60: {(a>=60).sum()}")
    print(f"diagnostic delay: median {pdc.dx_delay.median():.2f} y  "
          f"IQR {pdc.dx_delay.quantile(.25):.2f}-{pdc.dx_delay.quantile(.75):.2f}")

    ok = pdc.dropna(subset=["AAO", "ENROLL_AGE"])
    r = np.corrcoef(ok.AAO, ok.ENROLL_AGE)[0, 1]
    bias = (ok.ENROLL_AGE - ok.AAO)
    print(f"\nenrolment age vs true onset age (n={len(ok)}): r = {r:.3f}")
    print(f"  enrolment age exceeds onset age by median {bias.median():.2f} y "
          f"(IQR {bias.quantile(.25):.2f}-{bias.quantile(.75):.2f}, max {bias.max():.1f})")
    mis = ((ok.AAO < 50) != (ok.ENROLL_AGE < 50)).sum()
    print(f"  misclassified by the <50 cut using enrolment age: {mis} of {len(ok)} "
          f"({100*mis/len(ok):.1f}%)")

    d.to_parquet(f"{OUT}/data/ppmi_aao.parquet", index=False)
    json.dump(dict(n_pd=int(len(pdc)), n_aao=int(pdc.AAO.notna().sum()),
                   n_aao_prs=int((pdc.AAO.notna() & pdc.META5_PGS.notna()).sum()),
                   n_aao_prs_pcs=int((pdc.AAO.notna() & pdc.META5_PGS.notna()
                                      & pdc.Genetic_PRS_PC1.notna()).sum()),
                   aao_mean=float(a.mean()), aao_sd=float(a.std()),
                   n_lt50=int((a < 50).sum()), n_ge60=int((a >= 60).sum()),
                   dx_delay_median=float(pdc.dx_delay.median()),
                   corr_enrolage_aao=float(r),
                   enrolage_minus_aao_median=float(bias.median()),
                   pct_misclassified_by_lt50_cut=float(100 * mis / len(ok))),
              open(f"{OUT}/results/50_aao_construction.json", "w"), indent=2)
    print(f"\nwrote {OUT}/data/ppmi_aao.parquet")


if __name__ == "__main__":
    main()
