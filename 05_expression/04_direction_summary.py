import sys
from pathlib import Path
import warnings
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

warnings.filterwarnings("ignore")

ROOT = str(paths.ROOT)
RES = str(paths.RESULTS)
WINDOW = 250_000
P_MULT = 100.0

LEAD = {
    "SNCA": (90626111, 3.007e-41), "GBA1": (155135036, 5.022e-30),
    "TMEM175": (951947, 7.974e-23), "GAK": (951947, 7.974e-23),
    "DGKQ": (951947, 7.974e-23), "IDUA": (951947, 7.974e-23),
    "LRRC37A2": (44095467, 1.363e-21), "MAPT": (44095467, 1.363e-21),
    "LRRK2": (40773684, 1.108e-12), "SCARB2": (77183300, 1.528e-09),
    "GPNMB": (23245569, 3.831e-09), "VPS13C": (61993702, 4.626e-08),
    "CTSB": (11328236, 1.244e-06), "ASAH1": (17580690, 5.433e-05),
    "APOE": (45851331, 7.264e-05), "CTSD": (2266487, 1.172e-04),
    "GALC": (88193968, 1.437e-04),
}


def main():
    e = pd.read_csv(f"{RES}/180_risk_allele_eqtl_direction.csv")
    rows, per_tissue = [], []
    for gene, (pos, lead_p) in LEAD.items():
        s = e[e.gene == gene]
        w = s[((s.pos37 - pos).abs() <= WINDOW)
              & (s.risk_p <= lead_p * P_MULT)]
        if not len(w):
            rows.append(dict(gene=gene, lead_p=lead_p, n_variants=0,
                             n_tissues=0, n_up=0, n_down=0,
                             median_slope=np.nan,
                             risk_allele_effect="no expression signal",
                             drug_would_need_to="unknown", confidence="none"))
            continue
        tm = w.groupby("tissue").slope_on_risk_allele.median()
        for t, v in tm.items():
            per_tissue.append(dict(gene=gene, tissue=t, median_slope=float(v),
                                   n=int((w.tissue == t).sum())))
        up, dn = int((tm > 0).sum()), int((tm < 0).sum())
        n = up + dn
        agree = max(up, dn) / n if n else 0
        direction = ("raises" if up > dn and agree >= 0.75 else
                     "lowers" if dn > up and agree >= 0.75 else "mixed")
        conf = ("strong" if n >= 5 and agree >= 0.9 else
                "moderate" if n >= 3 and agree >= 0.75 else "weak")
        rows.append(dict(
            gene=gene, lead_p=lead_p, n_variants=len(w), n_tissues=n,
            n_up=up, n_down=dn, agreement=float(agree),
            median_slope=float(tm.median()),
            risk_allele_effect=direction,
            drug_would_need_to=("lower" if direction == "raises" else
                                "raise" if direction == "lowers" else
                                "unknown"),
            confidence=conf))
    d = pd.DataFrame(rows).sort_values("lead_p")
    d.to_csv(f"{RES}/182_direction_final.csv", index=False)
    pd.DataFrame(per_tissue).to_csv(f"{RES}/183_direction_per_tissue.csv",
                                    index=False)
    print(d[["gene", "n_variants", "n_tissues", "n_up", "n_down",
             "median_slope", "risk_allele_effect", "drug_would_need_to",
             "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
