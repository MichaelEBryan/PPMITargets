import sys
from pathlib import Path
import os, glob, gzip, json
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

BASE = str(paths.ROOT)
GT = f"{BASE}/external/gtex/GTEx_Analysis_v8_eQTL"
OUT = f"{BASE}/results"

WIN = (45_300_000, 46_600_000)
IDX = {"rs62053943": 45_666_837, "rs17649553": 45_917_282}
LDFILE = f"{OUT}/33_LRRC37A2_variant_LD_profile.csv"


def main():
    ld = pd.read_csv(LDFILE)
    tag = set(ld.loc[ld.Dprime >= 0.9, "pos38"].astype(int))
    print(f"{len(tag)} variants at |D'| >= 0.9 with rs62053943", flush=True)

    rows = []
    for f in sorted(glob.glob(f"{GT}/Brain_*signif_variant_gene_pairs.txt.gz")):
        tis = os.path.basename(f).split(".v8")[0]
        n = 0
        with gzip.open(f, "rt") as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            i_v, i_g = hdr.index("variant_id"), hdr.index("gene_id")
            i_p, i_s = hdr.index("pval_nominal"), hdr.index("slope")
            for line in fh:
                p = line.split("\t")
                v = p[i_v]
                if not v.startswith("chr17_"):
                    continue
                pos = int(v.split("_")[1])
                if not (WIN[0] <= pos <= WIN[1]):
                    continue
                rows.append((tis, v, pos, p[i_g].split(".")[0], float(p[i_p]),
                             float(p[i_s]), pos in tag))
                n += 1
        print(f"  {tis:42s} {n:6d} significant pairs in window", flush=True)

    d = pd.DataFrame(rows, columns=["tissue", "variant_id", "pos38", "gene_id",
                                    "pval", "slope", "on_haplotype"])
    SYM = {"ENSG00000186868": "MAPT", "ENSG00000120071": "KANSL1",
           "ENSG00000185829": "ARL17A", "ENSG00000228696": "ARL17B",
           "ENSG00000238083": "LRRC37A2", "ENSG00000176681": "LRRC37A",
           "ENSG00000073969": "NSF", "ENSG00000120088": "CRHR1",
           "ENSG00000185294": "SPPL2C", "ENSG00000225190": "PLEKHM1",
           "ENSG00000214425": "LRRC37A4P", "ENSG00000262633": "MAPT-AS1",
           "ENSG00000108379": "WNT3", "ENSG00000228775": "WNT9B-AS",
           "ENSG00000264589": "LRRC37A17P", "ENSG00000238083.10": "LRRC37A2"}
    d["gene"] = d.gene_id.map(SYM).fillna(d.gene_id)
    d.to_csv(f"{OUT}/35_gtex_17q21_eqtls.csv", index=False)

    hap = d[d.on_haplotype]
    print(f"\n{len(d):,} significant brain eQTL pairs in the window; "
          f"{len(hap):,} at variants on the rs62053943 haplotype", flush=True)

    named = hap[hap.gene.isin(SYM.values())]
    tab = (named.groupby("gene")
           .agg(tissues=("tissue", "nunique"), variants=("variant_id", "nunique"),
                min_p=("pval", "min"), median_abs_slope=("slope", lambda s: np.median(np.abs(s))))
           .sort_values("variants", ascending=False))
    print("\n=== genes regulated by variants on the rs62053943 haplotype ===")
    print(tab.to_string())
    tab.to_csv(f"{OUT}/36_gtex_haplotype_genes.csv")

    print("\n=== the two index variants, directly ===")
    idx_rows = []
    for rs, pos in IDX.items():
        sub = d[d.pos38 == pos]
        genes = sorted(set(sub.gene))
        idx_rows.append(dict(rsid=rs, pos38=pos, n_pairs=len(sub),
                             n_tissues=sub.tissue.nunique(), n_genes=len(genes),
                             genes=";".join(genes)))
        print(f"  {rs} (chr17:{pos:,}): significant eQTL for {len(genes)} genes "
              f"across {sub.tissue.nunique()} brain tissues")
        print(f"     {', '.join(genes)}")
    pd.DataFrame(idx_rows).to_csv(f"{OUT}/37_index_variant_eqtls.csv", index=False)

    json.dump(dict(n_haplotype_variants=len(tag),
                   n_signif_pairs_in_window=int(len(d)),
                   n_signif_pairs_on_haplotype=int(len(hap)),
                   n_distinct_genes_on_haplotype=int(hap.gene.nunique()),
                   named_block_genes=list(tab.index),
                   index_variants=idx_rows),
              open(f"{OUT}/38_gtex_17q21_summary.json", "w"), indent=2)
    print(f"\nwrote {OUT}/38_gtex_17q21_summary.json")


if __name__ == "__main__":
    main()
