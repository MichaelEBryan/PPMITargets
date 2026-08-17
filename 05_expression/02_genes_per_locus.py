import sys
from pathlib import Path
import os
import pandas as pd
from pyliftover import LiftOver

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

RES = str(paths.RESULTS)
GTEX = str(paths.EXTERNAL / "gtex" / "GTEx_Analysis_v8_eQTL")

LEAD = {
    "SNCA": (4, 90626111), "GBA1": (1, 155135036), "TMEM175": (4, 951947),
    "GAK": (4, 951947), "DGKQ": (4, 951947), "IDUA": (4, 951947),
    "LRRC37A2": (17, 44095467), "MAPT": (17, 44095467),
    "LRRK2": (12, 40773684), "SCARB2": (4, 77183300),
    "GPNMB": (7, 23245569), "VPS13C": (15, 61993702),
    "CTSB": (8, 11328236), "ASAH1": (8, 17580690), "APOE": (19, 45851331),
    "CTSD": (11, 2266487), "GALC": (14, 88193968),
}


def main():
    lo = LiftOver("hg38", "hg19")
    loci = {}
    for gene, pos in LEAD.items():
        loci.setdefault(pos, []).append(gene)
    hits = {k: set() for k in loci}

    for f in sorted(os.listdir(GTEX)):
        if not f.endswith(".gz"):
            continue
        d = pd.read_csv(f"{GTEX}/{f}", sep="\t", compression="gzip",
                        usecols=["variant_id", "gene_id"])
        parts = d.variant_id.str.split("_", expand=True)
        d = d.assign(chr38=parts[0],
                     pos38=pd.to_numeric(parts[1], errors="coerce"))
        for (chrom, pos) in loci:
            near = d[(d.chr38 == f"chr{chrom}")
                     & ((d.pos38 - pos).abs() < 2_000_000)]
            for _, r in near.iterrows():
                m = lo.convert_coordinate(r.chr38, int(r.pos38) - 1)
                if m and m[0][0] == f"chr{chrom}" and int(m[0][1]) + 1 == pos:
                    hits[(chrom, pos)].add(r.gene_id.split(".")[0])
        print(f"{f.split('.')[0]}: "
              f"{ {loci[k][0]: len(v) for k, v in hits.items() if v} }")

    rows = [dict(gene=g, chr=k[0], pos=k[1], n_genes_controlled=len(v))
            for k, v in hits.items() for g in loci[k]]
    out = pd.DataFrame(rows).sort_values("n_genes_controlled", ascending=False)
    out.to_csv(f"{RES}/187_locus_gene_count.csv", index=False)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
