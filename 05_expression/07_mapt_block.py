import sys
from pathlib import Path
import json, gzip
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import paths

BASE = str(paths.ROOT)
VCF = f"{BASE}/external/1kg/chr17_MAPT.vcf.gz"
OUT = f"{BASE}/results"
GENES38 = {
    "PLEKHM1": (45_460_000, 45_524_000), "LRRC37A2": (45_619_000, 45_674_000),
    "CRHR1": (45_784_000, 45_895_000), "SPPL2C": (45_878_000, 45_884_000),
    "MAPT": (45_894_382, 46_028_334), "KANSL1": (46_029_916, 46_225_403),
    "ARL17B": (46_197_000, 46_275_000), "ARL17A": (46_274_000, 46_357_000),
    "LRRC37A": (46_360_000, 46_420_000), "NSF": (46_596_000, 46_762_000),
}
PANEL17 = {"rs62053943": 45_666_837, "rs117615688": 45_720_942,
           "rs17649553": 45_917_282, "rs11658976": 46_789_439}


def gene_at(p):
    hits = [g for g, (s, e) in GENES38.items() if s <= p <= e]
    if hits:
        return hits[0]
    near = min(GENES38.items(), key=lambda kv: min(abs(p - kv[1][0]), abs(p - kv[1][1])))
    return f"~{near[0]}"


def read_vcf(path):
    ids, pos, gts = [], [], []
    samples = None
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            p = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                samples = p[9:]; continue
            k = p[8].split(":").index("GT")
            hap = np.empty((len(p) - 9, 2), dtype=np.int8)
            for i, s in enumerate(p[9:]):
                g = s.split(":")[k].replace("|", "/").split("/")
                try:
                    hap[i] = (int(g[0]), int(g[1]))
                except Exception:
                    hap[i] = (-1, -1)
            ids.append(p[2]); pos.append(int(p[1])); gts.append(hap)
    return np.array(ids), np.array(pos), np.stack(gts), samples


def ld_pair(hA, hB):
    a, b = hA.ravel(), hB.ravel()
    m = (a >= 0) & (b >= 0)
    a, b = a[m], b[m]
    if len(a) < 60:
        return np.nan, np.nan
    pA, pB = a.mean(), b.mean()
    if min(pA, pB) in (0, 1) or max(pA, pB) in (0, 1):
        return np.nan, np.nan
    pAB = (a * b).mean()
    D = pAB - pA * pB
    denom = pA * (1 - pA) * pB * (1 - pB)
    r2 = D * D / denom if denom > 0 else np.nan
    Dmax = min(pA * (1 - pB), (1 - pA) * pB) if D > 0 else min(pA * pB, (1 - pA) * (1 - pB))
    Dp = abs(D) / Dmax if Dmax > 0 else np.nan
    return float(r2), float(Dp)


def main():
    ids, pos, H, samples = read_vcf(VCF)
    ped = pd.read_csv(f"{BASE}/external/1kg/ped.txt", sep=r"\s+")
    pops = dict(zip(ped.SampleID, ped.Superpopulation))
    unrel = set(ped.loc[(ped.FatherID.astype(str) == "0") &
                        (ped.MotherID.astype(str) == "0"), "SampleID"])
    sp = np.array([pops.get(s, "NA") for s in samples])
    ok = np.array([s in unrel for s in samples])
    eur = (sp == "EUR") & ok
    print(f"{H.shape[0]} SNPs, EUR unrelated n={eur.sum()}", flush=True)

    He = H[:, eur, :]
    af = np.array([np.nanmean(np.where(He[i] >= 0, He[i], np.nan)) for i in range(len(ids))])
    common = np.where((af > 0.05) & (af < 0.95))[0]
    print(f"{len(common)} common SNPs (MAF>5% EUR) in chr17:{pos.min():,}-{pos.max():,}", flush=True)

    idx = {}
    for rs, p in PANEL17.items():
        w = np.where(ids == rs)[0]
        idx[rs] = int(w[0]) if len(w) else int(np.argmin(np.abs(pos - p)))

    rows = []
    for a in PANEL17:
        for b in PANEL17:
            if a >= b:
                continue
            r2, dp = ld_pair(He[idx[a]], He[idx[b]])
            rows.append(dict(snp_a=a, gene_a=gene_at(pos[idx[a]]),
                             snp_b=b, gene_b=gene_at(pos[idx[b]]),
                             dist_kb=abs(pos[idx[a]] - pos[idx[b]]) / 1e3, r2=r2, Dprime=dp))
    pair = pd.DataFrame(rows)
    pair.to_csv(f"{OUT}/32_panel17_pairwise_LD.csv", index=False)
    print("\n=== pairwise LD between the chr17 panel variants (1000G EUR) ===")
    print(pair.round(3).to_string(index=False), flush=True)

    iA = idx["rs62053943"]
    r2s, dps = [], []
    for j in common:
        r2, dp = ld_pair(He[iA], He[j])
        r2s.append(r2); dps.append(dp)
    r2s, dps = np.array(r2s), np.array(dps)
    prof = pd.DataFrame(dict(rsid=ids[common], pos38=pos[common],
                             gene=[gene_at(p) for p in pos[common]], r2=r2s, Dprime=dps))
    prof.to_csv(f"{OUT}/33_LRRC37A2_variant_LD_profile.csv", index=False)

    res = {}
    for thr in (0.2, 0.5, 0.8):
        sel = prof[prof.r2 >= thr]
        res[f"r2_ge_{thr}"] = dict(n=int(len(sel)),
                                   span_kb=float((sel.pos38.max() - sel.pos38.min()) / 1e3) if len(sel) else 0,
                                   genes=sorted(set(g for g in sel.gene if not g.startswith("~"))))
    for thr in (0.8, 0.9):
        sel = prof[prof.Dprime >= thr]
        res[f"Dprime_ge_{thr}"] = dict(n=int(len(sel)),
                                       span_kb=float((sel.pos38.max() - sel.pos38.min()) / 1e3) if len(sel) else 0,
                                       genes=sorted(set(g for g in sel.gene if not g.startswith("~"))))
    res["pairwise_panel"] = pair.to_dict("records")
    res["max_r2_partner"] = prof.loc[prof[prof.rsid != ids[iA]].r2.idxmax()].to_dict() \
        if len(prof) > 1 else None
    json.dump(res, open(f"{OUT}/34_mapt_block_summary.json", "w"), indent=2, default=float)

    print("\n=== extent of the haplotype the LRRC37A2 variant sits on (EUR) ===")
    for k, v in res.items():
        if k.startswith(("r2_", "Dprime_")):
            print(f"  {k:14s} n={v['n']:5d}  span={v['span_kb']:8.1f} kb  "
                  f"genes: {', '.join(v['genes'])}")
    print(f"\nwrote {OUT}/34_mapt_block_summary.json", flush=True)


if __name__ == "__main__":
    main()
