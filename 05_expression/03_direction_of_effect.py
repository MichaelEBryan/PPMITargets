import sys
from pathlib import Path
import warnings
import numpy as np, pandas as pd
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

warnings.filterwarnings("ignore")

ROOT = str(paths.ROOT)
RES = str(paths.RESULTS)
GTEX = str(paths.EXTERNAL / "gtex" / "GTEx_Analysis_v8_eQTL")

GENES = {
    "SNCA": "ENSG00000145335", "TMEM175": "ENSG00000127419",
    "GAK": "ENSG00000178950", "DGKQ": "ENSG00000145214",
    "IDUA": "ENSG00000127415", "APOE": "ENSG00000130203",
    "CTSB": "ENSG00000164733", "ASAH1": "ENSG00000104763",
    "GALC": "ENSG00000054983", "SCARB2": "ENSG00000138760",
    "MAPT": "ENSG00000186868", "GPNMB": "ENSG00000136235",
    "LRRC37A2": "ENSG00000238083", "CTSD": "ENSG00000117984",
    "VPS13C": "ENSG00000129003", "GBA1": "ENSG00000177628",
    "LRRK2": "ENSG00000188906",
}


def load_risk():
    d = pd.read_csv(f"{ROOT}/external/risk_gwas/GCST009325.tsv", sep="\t",
                    low_memory=False)
    ren = {}
    for c in d.columns:
        cl = c.lower()
        if cl in ("chromosome", "chr"):
            ren[c] = "chr"
        elif cl in ("base_pair_location", "bp", "position"):
            ren[c] = "pos"
        elif cl in ("effect_allele", "a1"):
            ren[c] = "ea"
        elif cl in ("other_allele", "a2"):
            ren[c] = "oa"
        elif cl in ("beta",):
            ren[c] = "beta"
        elif cl in ("p_value", "p"):
            ren[c] = "p"
    d = d.rename(columns=ren)
    need = {"chr", "pos", "ea", "oa", "beta", "p"}
    if not need <= set(d.columns):
        raise SystemExit(f"risk sumstats missing {need - set(d.columns)}")
    d["chr"] = pd.to_numeric(d["chr"], errors="coerce")
    return d.dropna(subset=["chr", "pos", "beta", "p"])


def gene_windows():
    g = pd.read_csv(f"{ROOT}/data/genes_hg19.csv")
    cols = {c.lower(): c for c in g.columns}
    name = cols.get("gene") or cols.get("symbol") or cols.get("name")
    return g, name, cols


def main():
    risk = load_risk()
    g, name, cols = gene_windows()
    chrom_c = cols.get("chr") or cols.get("chrom")
    start_c = cols.get("start"); end_c = cols.get("end")
    alias = {"GBA1": "GBA"}

    lead = {}
    for gene in GENES:
        sym = alias.get(gene, gene)
        row = g[g[name].astype(str) == sym]
        if not len(row):
            print(f"  [no coordinates] {gene}")
            continue
        r = row.iloc[0]
        ch = int(str(r[chrom_c]).replace("chr", ""))
        lo, hi = int(r[start_c]) - 500_000, int(r[end_c]) + 500_000
        w = risk[(risk.chr == ch) & (risk.pos.between(lo, hi))]
        if not len(w):
            print(f"  [no risk variants] {gene}")
            continue
        b = w.loc[w.p.idxmin()]
        ra, oa_, bt = (b.ea, b.oa, b.beta) if b.beta > 0 else (b.oa, b.ea, -b.beta)
        lead[gene] = dict(chr=ch, pos=int(b.pos), risk_allele=str(ra).upper(),
                          other_allele=str(oa_).upper(), risk_beta=float(bt),
                          risk_p=float(b.p))
        print(f"{gene:9s} lead chr{ch}:{int(b.pos)} risk allele {ra} "
              f"beta {bt:.3f} P {b.p:.2e}")

    from pyliftover import LiftOver
    lo = LiftOver("hg38", "hg19")
    want = {v: k for k, v in GENES.items()}
    risk_idx = {}
    for ch, sub in risk.groupby("chr"):
        risk_idx[int(ch)] = sub.set_index("pos")[["ea", "oa", "beta", "p"]]

    rows = []
    for f in sorted(os.listdir(GTEX)):
        if not f.endswith(".gz"):
            continue
        tis = f.split(".")[0].replace("Brain_", "").replace("_", " ")
        d = pd.read_csv(f"{GTEX}/{f}", sep="\t", compression="gzip",
                        usecols=["variant_id", "gene_id", "slope",
                                 "pval_nominal"])
        d["gene_stub"] = d.gene_id.str.split(".").str[0]
        d = d[d.gene_stub.isin(want)]
        if not len(d):
            continue
        parts = d.variant_id.str.split("_", expand=True)
        d = d.assign(chr38=parts[0], pos38=pd.to_numeric(parts[1],
                                                         errors="coerce"),
                     ref=parts[2].str.upper(), alt=parts[3].str.upper())
        for _, h in d.iterrows():
            m = lo.convert_coordinate(h.chr38, int(h.pos38) - 1)
            if not m:
                continue
            ch37 = int(m[0][0].replace("chr", "")) if m[0][0][3:].isdigit() \
                else None
            p37 = int(m[0][1]) + 1
            if ch37 is None or ch37 not in risk_idx:
                continue
            R = risk_idx[ch37]
            if p37 not in R.index:
                continue
            rr = R.loc[p37]
            if isinstance(rr, pd.DataFrame):
                rr = rr.iloc[0]
            ea, oa_, bt = str(rr.ea).upper(), str(rr.oa).upper(), float(rr.beta)
            ra = ea if bt > 0 else oa_
            if {ea, oa_} != {h.ref, h.alt}:
                continue
            slope = h.slope if h.alt == ra else -h.slope
            rows.append(dict(gene=want[h.gene_stub], tissue=tis,
                             variant=h.variant_id, pos37=p37,
                             risk_allele=ra, risk_beta=abs(bt),
                             risk_p=float(rr.p),
                             slope_on_risk_allele=float(slope),
                             eqtl_p=float(h.pval_nominal)))
        print(f"  {tis}: {len(rows)} aligned records so far")

    e = pd.DataFrame(rows)
    e.to_csv(f"{RES}/180_risk_allele_eqtl_direction.csv", index=False)

    summ = []
    for gene in GENES:
        s = e[e.gene == gene] if len(e) else e
        L = lead.get(gene, {})
        strong = s[s.risk_p < 1e-4] if len(s) else s
        use = strong if len(strong) >= 3 else s
        up = int((use.slope_on_risk_allele > 0).sum()) if len(use) else 0
        dn = int((use.slope_on_risk_allele < 0).sum()) if len(use) else 0
        n = up + dn
        direction = ("raises" if up > 0.75 * n and n else
                     "lowers" if dn > 0.75 * n and n else
                     "mixed" if n else "no eQTL")
        summ.append(dict(
            gene=gene, lead_p=L.get("risk_p", np.nan),
            n_eqtl=len(s), n_strong=len(use), n_up=up, n_down=dn,
            median_slope=float(use.slope_on_risk_allele.median())
            if len(use) else np.nan,
            n_tissues=int(use.tissue.nunique()) if len(use) else 0,
            risk_allele_effect=direction,
            drug_would_need_to=("lower" if direction == "raises" else
                                "raise" if direction == "lowers" else
                                "unknown")))
    S = pd.DataFrame(summ).sort_values("lead_p")
    S.to_csv(f"{RES}/181_direction_summary.csv", index=False)
    print("\n" + S[["gene", "n_eqtl", "n_strong", "n_up", "n_down",
                     "n_tissues", "median_slope", "risk_allele_effect",
                     "drug_would_need_to"]].to_string(index=False))


if __name__ == "__main__":
    main()
