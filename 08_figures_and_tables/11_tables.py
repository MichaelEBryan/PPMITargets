import sys
from pathlib import Path
import os
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

RES = str(paths.RESULTS)
OUT = str(paths.ROOT / "tables")
os.makedirs(OUT, exist_ok=True)

SPEC = [
    ("T01_reproduction", "01_as_published.csv",
     "Reproduction of the published pipeline, single 80:20 split, seed 42"),
    ("T02_enrolment_patterns", "00_enrl_pattern_table.csv",
     "PPMI enrolment-flag pattern against COHORT_DEFINITION"),
    ("T03_ablation", "22_ablation_summary.csv",
     "Feature-block ablation, 5x5-fold repeated stratified CV"),
    ("T04_null_splithalf", "21_null_splithalf.csv",
     "Split-half null: top-k overlap within one stratum"),
    ("T05_observed_overlap", "04_observed_overlap.csv",
     "Observed cross-stratum top-k overlap, by model family"),
    ("T06_simulation", "09_simulation_grid.csv",
     "Simulation grid, 480 cells x 200 replicates, identical architecture in both strata"),
    ("T07_ascertainment", "10_ascertainment_route_by_stratum.csv",
     "Enrolment route of PD cases by age stratum"),
    ("T08_prs_onset", "11_prs_vs_onsetage.csv",
     "Polygenic score against age at enrolment among PD cases"),
    ("T09_prs_control", "16_prs_negative_control.csv",
     "Negative control: same model in participants without PD"),
    ("T10_pervariant", "12_pervariant_onsetage.csv",
     "Per-variant onset-age association and case-case test, 90 panel variants"),
    ("T11_replication", "13_ppmi_vs_ipdgc_aao.csv",
     "PPMI onset effects against the IPDGC age-at-onset GWAS"),
    ("T12_power", "14_power.csv",
     "Minimum detectable onset effect in PPMI"),
    ("T13_panel_annotation", None, "Annotation of the 90-variant PPMI panel"),
    ("T14_ld_17q21_pairwise", "32_panel17_pairwise_LD.csv",
     "Pairwise LD between chromosome 17 panel variants, 1000G EUR"),
    ("T15_ld_17q21_profile", "33_LRRC37A2_variant_LD_profile.csv",
     "LD of rs62053943 with every common variant in chr17:44.8-46.6 Mb"),
    ("T16_galc_structure", "40_galc_structure.csv",
     "ESMFold predictions for GALC wild type against 53 single substitutions"),
]


def main():
    sheets, index = {}, []
    for name, fn, desc in SPEC:
        if name == "T13_panel_annotation":
            df = pd.read_csv(str(paths.ROOT / "data/snp_map.csv"))
        elif fn and os.path.exists(f"{RES}/{fn}"):
            df = pd.read_csv(f"{RES}/{fn}")
        else:
            print(f"  [missing] {name} <- {fn}")
            continue
        if name == "T06_simulation" and "raw_overlap" in df.columns:
            df = df.drop(columns=["raw_overlap"])
        sheets[name] = df
        df.to_csv(f"{OUT}/{name}.csv", index=False)
        index.append(dict(sheet=name, rows=len(df), columns=df.shape[1],
                          source=fn or "snp_map.csv", description=desc))

    idx = pd.DataFrame(index)
    with pd.ExcelWriter(f"{OUT}/Supplementary_Tables.xlsx", engine="openpyxl") as w:
        idx.to_excel(w, sheet_name="Index", index=False)
        for k, v in sheets.items():
            v.head(50000).to_excel(w, sheet_name=k[:31], index=False)
    print(idx.to_string(index=False))
    print(f"\nwrote {OUT}/Supplementary_Tables.xlsx ({len(sheets)} sheets)")


if __name__ == "__main__":
    main()
