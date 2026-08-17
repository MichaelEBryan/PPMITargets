#!/usr/bin/env python
"""Per-tissue eQTL effect of the risk allele on each candidate gene.

Significant brain eQTLs at each gene's risk locus are aligned onto the
risk-increasing allele and combined within tissue by inverse-variance
weighting, giving one effect and standard error per gene per brain region.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd, requests
import paths as _P
warnings.filterwarnings("ignore")

ROOT = str(_P.ROOT)
RES = str(_P.RESULTS)
GTEX = str(_P.EXTERNAL / "gtex" / "GTEx_Analysis_v8_eQTL")
STR = str(_P.STRUCTURES)

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
LEAD = {                       # locus lead variant, GRCh37, and its risk P
    "SNCA": (4, 90626111, 3.007e-41), "GBA1": (1, 155135036, 5.022e-30),
    "TMEM175": (4, 951947, 7.974e-23), "GAK": (4, 951947, 7.974e-23),
    "DGKQ": (4, 951947, 7.974e-23), "IDUA": (4, 951947, 7.974e-23),
    "LRRC37A2": (17, 44095467, 1.363e-21), "MAPT": (17, 44095467, 1.363e-21),
    "LRRK2": (12, 40773684, 1.108e-12), "SCARB2": (4, 77183300, 1.528e-09),
    "GPNMB": (7, 23245569, 3.831e-09), "VPS13C": (15, 61993702, 4.626e-08),
    "CTSB": (8, 11328236, 1.244e-06), "ASAH1": (8, 17580690, 5.433e-05),
    "APOE": (19, 45851331, 7.264e-05), "CTSD": (11, 2266487, 1.172e-04),
    "GALC": (14, 88193968, 1.437e-04),
}
WINDOW = 250_000
P_MULT = 100.0
GTEX_ALIAS = {"GBA1": "GBA"}


def load_risk():
    d = pd.read_csv(f"{ROOT}/external/risk_gwas/GCST009325.tsv", sep="\t",
                    low_memory=False)
    d = d.rename(columns={"chromosome": "chr", "base_pair_location": "pos",
                          "effect_allele": "ea", "other_allele": "oa",
                          "p_value": "p"})
    d["chr"] = pd.to_numeric(d["chr"], errors="coerce")
    return d.dropna(subset=["chr", "pos", "beta", "p"])


def gtex_expression(gene):
    sym = GTEX_ALIAS.get(gene, gene)
    path = f"{STR}/gtex_{sym}.json"
    if os.path.exists(path):
        return json.load(open(path))
    try:
        g = requests.get("https://gtexportal.org/api/v2/reference/gene",
                         params={"geneId": sym, "page": 0, "itemsPerPage": 5},
                         timeout=90).json()
        hits = [h for h in g.get("data", [])
                if h.get("geneSymbolUpper") == sym.upper()]
        if not hits:
            return {}
        r = requests.get(
            "https://gtexportal.org/api/v2/expression/medianGeneExpression",
            params={"gencodeId": hits[0]["gencodeId"], "datasetId": "gtex_v8",
                    "page": 0, "itemsPerPage": 100}, timeout=120)
        out = {x["tissueSiteDetailId"]: x["median"]
               for x in (r.json().get("data", []) if r.ok else [])
               if x.get("tissueSiteDetailId", "").startswith("Brain")}
        json.dump(out, open(path, "w"))
        return out
    except Exception:
        return {}


def main():
    from pyliftover import LiftOver
    lo = LiftOver("hg38", "hg19")
    risk = load_risk()
    risk_idx = {int(c): sub.set_index("pos")[["ea", "oa", "beta", "p"]]
                for c, sub in risk.groupby("chr")}
    want = {v: k for k, v in GENES.items()}

    rows = []
    for f in sorted(os.listdir(GTEX)):
        if not f.endswith(".gz"):
            continue
        tissue = f.split(".")[0].replace("Brain_", "").replace("_", " ")
        d = pd.read_csv(f"{GTEX}/{f}", sep="\t", compression="gzip",
                        usecols=["variant_id", "gene_id", "slope", "slope_se",
                                 "pval_nominal", "maf"])
        d["stub"] = d.gene_id.str.split(".").str[0]
        d = d[d.stub.isin(want)]
        if not len(d):
            continue
        parts = d.variant_id.str.split("_", expand=True)
        d = d.assign(chr38=parts[0], pos38=pd.to_numeric(parts[1],
                                                         errors="coerce"),
                     ref=parts[2].str.upper(), alt=parts[3].str.upper())
        for _, h in d.iterrows():
            gene = want[h.stub]
            ch_lead, pos_lead, p_lead = LEAD[gene]
            m = lo.convert_coordinate(h.chr38, int(h.pos38) - 1)
            if not m or not m[0][0][3:].isdigit():
                continue
            ch37, p37 = int(m[0][0].replace("chr", "")), int(m[0][1]) + 1
            if ch37 != ch_lead or abs(p37 - pos_lead) > WINDOW:
                continue
            R = risk_idx.get(ch37)
            if R is None or p37 not in R.index:
                continue
            rr = R.loc[p37]
            if isinstance(rr, pd.DataFrame):
                rr = rr.iloc[0]
            if float(rr.p) > p_lead * P_MULT:
                continue
            ea, oa_, bt = str(rr.ea).upper(), str(rr.oa).upper(), float(rr.beta)
            ra = ea if bt > 0 else oa_
            if {ea, oa_} != {h.ref, h.alt}:
                continue
            rows.append(dict(
                gene=gene, tissue=tissue, variant=h.variant_id, pos37=p37,
                slope=float(h.slope if h.alt == ra else -h.slope),
                slope_se=float(h.slope_se), maf=float(h.maf),
                eqtl_p=float(h.pval_nominal), risk_p=float(rr.p),
                risk_beta=abs(bt)))
        print(f"{tissue}: {len(rows)} records so far")

    v = pd.DataFrame(rows)
    v.to_csv(f"{RES}/184_eqtl_variants.csv", index=False)

    # one inverse-variance weighted effect per gene per tissue
    out = []
    for (gene, tissue), s in v.groupby(["gene", "tissue"]):
        w = 1.0 / s.slope_se ** 2
        beta = float((s.slope * w).sum() / w.sum())
        se = float(np.sqrt(1.0 / w.sum()))
        out.append(dict(gene=gene, tissue=tissue, n_variants=len(s),
                        beta=beta, se=se, lo=beta - 1.96 * se,
                        hi=beta + 1.96 * se,
                        min_eqtl_p=float(s.eqtl_p.min())))
    t = pd.DataFrame(out)
    t.to_csv(f"{RES}/185_eqtl_by_tissue.csv", index=False)

    # expression level for context
    ex = []
    for gene in GENES:
        for tis, tpm in gtex_expression(gene).items():
            ex.append(dict(gene=gene,
                           tissue=tis.replace("Brain_", "").replace("_", " "),
                           tpm=float(tpm)))
    e = pd.DataFrame(ex)
    e.to_csv(f"{RES}/186_brain_expression.csv", index=False)

    print(f"\n{len(v)} variant records, {len(t)} gene-tissue effects, "
          f"{t.gene.nunique()} genes")
    print(t.groupby("gene").agg(tissues=("tissue", "nunique"),
                                median_beta=("beta", "median")).round(3)
          .sort_values("median_beta").to_string())


if __name__ == "__main__":
    main()
