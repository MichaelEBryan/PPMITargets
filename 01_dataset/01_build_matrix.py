import sys
from pathlib import Path
import re, json, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

SRC = str(paths.DATA / "Complete DataFrameC.csv")
OUTD = str(paths.ROOT / "data")
CLASSES = ['Healthy Control', "Parkinson's Disease", "Prodromal", "SWEDD"]

BLOCKS = {
    "design": [r"^ENRL", r"STDY$", r"^PPMI_ONLINE_ENROLL$", r"^SCREENEDAM$",
               r"^ENROLL_STATUS$", r"^DATELIG$", r"^APPRDX$", r"^COHORT_y$",
               r"^INEXPAGE$", r"^EVENT_ID", r"^REC_ID", r"^VISNO", r"^PAG_NAME",
               r"^ORIG_ENTRY", r"^LAST_UPDATE", r"^INFODT",
               r"^SE_(REC_ID|EVENT_ID|PAG_NAME|INFODT|ORIG_ENTRY|LAST_UPDATE)",
               r"^AM_(REC_ID|EVENT_ID|SUB_EVENT_ID|PAG_NAME|INFODT)", r"^MG_EVENT_ID$"],
    "genetics_core": [r"^CLIA$", r"^GWAS$", r"^WES$", r"^WGS$", r"^SVs$", r"^SANGER$",
                      r"^IU_Fingerprint$", r"^RNASEQ", r"^APOE$", r"^PATHVAR_COUNT$",
                      r"^VAR_GENE$", r"^LRRK$", r"^GBA$", r"^VPS$", r"^SNCA$",
                      r"^PRKN$", r"^PARK$", r"^PINK$", r"^NOTES$"],
    "motor": [r"^TUG", r"^SW_", r"^STEP_", r"^STR_", r"^SP_", r"^TRA_", r"^T_AMP",
              r"^L_JERK", r"^R_JERK", r"^JERK_T", r"^LA_", r"^RA_[AS]", r"^CAD_",
              r"^ASA_", r"^ASYM_IND", r"^SYM_", r"^GAIT_SUBGROUP",
              r"^(CV|Cadence|Degrees|Velocity|Time|Number|Num|Mean|Sum|Total|Percent|Step|Walk|Wake|Lying|Sitting|Standing|Sedentary|Other|Night|Actual|Activity|Valid|Start|Stop|UpSideDown|NonWear)",
              r"^(amp|rms|wd|str|stp|step|stride|samp|Samp)", r"^x_", r"^x__",
              r"^AM_(AXIVITY|OPAL)"],
    "family": [r"PD\d*$", r"^ANYFAMPD", r"^BIOMOM", r"^BIODAD", r"^FUL", r"^HAFSIB",
               r"^MAGPAR", r"^PAGPAR", r"^MATAU", r"^PATAU", r"^MATCOUS", r"^PATCOUS",
               r"^MAHAFSIB", r"^PAHAFSIB", r"^KIDS", r"^DISFAMPD"],
    "demo": [r"^SEX\d*$", r"^RA(ASIAN|BLACK|HAWOPI|INDALS|NOS|UNKNOWN|WHITE)\d*$",
             r"^HISPLAT\d*$", r"^HANDED\d*$", r"^HOWLIVE\d*$", r"^CHLDBEAR\d*$",
             r"^BIRTHDT", r"^(AFICBERB|ASHKJEW|BASQUE)\d*$", r"^SE_EDUCYRS",
             r"^SE_Education",
             r"^(GAYLES|HETERO|BISEXUAL|PANSEXUAL|ASEXUAL|OTHSEXUALITY)\d*$"],
    "age": [r"^ENROLL_AGE$", r"^AGE_AT_VISIT"],
    "geno": [r"^chr[0-9XYM]+:", r"^MG_rs"],
    "prs": [r"^MG_Genetic_PRS_PRS"],
    "pcs": [r"^MG_Genetic_PRS_PC", r"^MG_Genetic_PRS_InfPop$"],
}


def main():
    df = pd.read_csv(SRC, low_memory=False)
    d = df[df.COHORT_DEFINITION.isin(CLASSES)].copy()
    for c in ("COHORT_x", "COHORT_y"):
        if c in d.columns:
            print(f"dropping label proxy {c}: cross-tab purity vs outcome = "
                  f"{pd.crosstab(d[c], d.COHORT_DEFINITION).max(1).sum() / len(d):.4f}")
    num = [c for c in d.select_dtypes(include=["int64", "float64"]).columns
           if c not in ("Unnamed: 0", "Unnamed: 0.1", "PATNO", "COHORT_x", "COHORT_y")]
    prov = {}
    for c in num:
        prov[c] = next((b for b, pats in BLOCKS.items()
                        if any(re.search(p, c) for p in pats)), "other")
    X = d[num].copy()
    X.columns = [re.sub(r"[^\w]", "_", c) for c in X.columns]
    prov = {re.sub(r"[^\w]", "_", k): v for k, v in prov.items()}
    X["__label__"] = pd.Categorical(d.COHORT_DEFINITION, categories=CLASSES).codes
    X["__age__"] = d.ENROLL_AGE.values
    X = X.reset_index(drop=True)
    X.to_parquet(f"{OUTD}/ppmi_matrix.parquet", index=False)
    json.dump(prov, open(f"{OUTD}/provenance.json", "w"), indent=2)
    import collections
    print("rows", len(X), "cols", X.shape[1] - 2)
    print(collections.Counter(prov.values()))
    print("label counts", np.bincount(X["__label__"]))
    print("EO(<50)", int((X["__age__"] < 50).sum()), "LO(>=60)", int((X["__age__"] >= 60).sum()))
    print(f"wrote {OUTD}/ppmi_matrix.parquet "
          f"({__import__('os').path.getsize(OUTD+'/ppmi_matrix.parquet')/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
