#!/usr/bin/env python
"""
Can a predicted structure support the GALC mechanism claim?

The paper states that the GALC variant produces a gain of function by
stabilising a flexible loop at the edge of the binding pocket, and cites
AlphaFold model confidence as the evidence. Two things are being asked of
structure prediction that it cannot deliver:

  1. rs8005172 is intronic (PPMI's own annotation is GALC/GPR65; Ensembl calls
     it an intron variant). There is no substituted residue to model.

  2. Even for real missense variants, a folding model's predicted structure of a
     point mutant is not a mechanism assay. We test that directly: fold GALC
     wild type plus ClinVar pathogenic missense variants, gnomAD common
     (tolerated) missense variants, and random substitutions, then ask whether
     backbone deviation from wild type or change in predicted confidence
     separates pathogenic from tolerated.

If they do not separate, no claim about loop stabilisation can rest on this.

Run:  modal run 10_modal_galc_structure.py
"""
import modal, json, os
import paths as _P

BASE = str(_P.ROOT)
app = modal.App("pd-arch-galc-fold")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "transformers==4.46.3", "accelerate==1.1.1",
                      "numpy==1.26.4", "scipy==1.13.1", "biopython==1.84")
         .env({"HF_HOME": "/cache"}))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
       "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
       "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
       "Tyr": "Y", "Val": "V"}


@app.function(image=image, gpu="A100-40GB", timeout=5400, volumes={"/cache": cache},
              max_containers=10, retries=2)
def fold_batch(job: dict):
    import torch, numpy as np
    from transformers import AutoTokenizer, EsmForProteinFolding

    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",
                                                 low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True
    model.trunk.set_chunk_size(64)

    out = []
    for item in job["items"]:
        seq = item["seq"]
        with torch.no_grad():
            enc = tok([seq], return_tensors="pt", add_special_tokens=False)
            enc = {k: v.cuda() for k, v in enc.items()}
            o = model(**enc)
        pos = o["positions"][-1, 0].cpu().numpy()      # (L, 14, 3) atom14
        plddt = o["plddt"][0, :, 1].cpu().numpy()      # per-residue CA pLDDT
        out.append(dict(name=item["name"], group=item["group"], mut=item.get("mut"),
                        ca=pos[:, 1, :].astype(np.float32).tolist(),
                        plddt=plddt.astype(np.float32).tolist(),
                        mean_plddt=float(plddt.mean())))
        print(f"folded {item['name']} pLDDT={plddt.mean():.1f}", flush=True)
    return out


@app.local_entrypoint()
def main():
    import json, random, numpy as np
    seq = open(f"{BASE}/data/galc/GALC_P54803.fasta").read().split("\n")[1].strip()
    L = len(seq)
    cv = json.load(open(f"{BASE}/data/galc/clinvar_missense.json"))
    gn = json.load(open(f"{BASE}/data/galc/gnomad_common_missense.json"))

    def mk(pos, aa_to):
        if not (1 <= pos <= L):
            return None
        return seq[:pos - 1] + aa_to + seq[pos:]

    items = [dict(name="WT", group="wildtype", seq=seq)]
    seen = set()

    path = [v for v in cv if v["cls"] in ("Pathogenic", "Likely pathogenic",
                                          "Pathogenic/Likely pathogenic")]
    for v in path[:22]:
        a1, a2, p = AA3.get(v["aa3_from"]), AA3.get(v["aa3_to"]), v["pos"]
        if not a1 or not a2 or p > L or seq[p - 1] != a1 or (p, a2) in seen:
            continue
        seen.add((p, a2))
        items.append(dict(name=f"{a1}{p}{a2}", group="ClinVar pathogenic",
                          mut=f"{a1}{p}{a2}", seq=mk(p, a2)))

    import re
    for v in gn:
        m = re.match(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})", v["hgvsp"])
        if not m:
            continue
        a1, p, a2 = AA3.get(m.group(1)), int(m.group(2)), AA3.get(m.group(3))
        if not a1 or not a2 or p > L or seq[p - 1] != a1 or (p, a2) in seen:
            continue
        seen.add((p, a2))
        items.append(dict(name=f"{a1}{p}{a2}", group="gnomAD common (tolerated)",
                          mut=f"{a1}{p}{a2}", seq=mk(p, a2)))

    rng = random.Random(11)
    while sum(1 for i in items if i["group"] == "random substitution") < 18:
        p = rng.randint(1, L); a2 = rng.choice(list(AA3.values()))
        if seq[p - 1] == a2 or (p, a2) in seen:
            continue
        seen.add((p, a2))
        items.append(dict(name=f"{seq[p-1]}{p}{a2}", group="random substitution",
                          mut=f"{seq[p-1]}{p}{a2}", seq=mk(p, a2)))

    print(f"GALC {L} aa; folding {len(items)} sequences")
    import collections
    print(collections.Counter(i["group"] for i in items))

    nb = 8
    batches = [dict(items=items[i::nb]) for i in range(nb)]
    res = [r for chunk in fold_batch.map(batches, order_outputs=False) if chunk for r in chunk]

    by = {r["name"]: r for r in res}
    wt = np.array(by["WT"]["ca"], dtype=float)
    wt_plddt = np.array(by["WT"]["plddt"], dtype=float)

    def kabsch_rmsd(P, Q):
        Pc, Qc = P - P.mean(0), Q - Q.mean(0)
        V, S, Wt = np.linalg.svd(Pc.T @ Qc)
        d = np.sign(np.linalg.det(V @ Wt))
        R = V @ np.diag([1, 1, d]) @ Wt
        return float(np.sqrt((((Pc @ R) - Qc) ** 2).sum(1).mean()))

    rows = []
    for r in res:
        if r["name"] == "WT":
            continue
        ca = np.array(r["ca"], dtype=float)
        pl = np.array(r["plddt"], dtype=float)
        pos = int(r["mut"][1:-1])
        rows.append(dict(variant=r["mut"], group=r["group"], pos=pos,
                         rmsd_ca=kabsch_rmsd(ca, wt),
                         mean_plddt=r["mean_plddt"],
                         delta_mean_plddt=r["mean_plddt"] - float(wt_plddt.mean()),
                         local_plddt=float(pl[pos - 1]),
                         delta_local_plddt=float(pl[pos - 1] - wt_plddt[pos - 1])))
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(f"{BASE}/results/40_galc_structure.csv", index=False)
    json.dump(dict(wt_mean_plddt=float(wt_plddt.mean()), n=len(d)),
              open(f"{BASE}/results/40_galc_wt.json", "w"), indent=2)
    print("\n=== does predicted structure separate pathogenic from tolerated? ===")
    print(d.groupby("group")[["rmsd_ca", "delta_mean_plddt", "delta_local_plddt"]]
          .agg(["mean", "std", "count"]).round(3).to_string())
    from scipy import stats
    a = d[d.group == "ClinVar pathogenic"]
    b = d[d.group == "gnomAD common (tolerated)"]
    if len(a) > 2 and len(b) > 2:
        for m in ("rmsd_ca", "delta_mean_plddt", "delta_local_plddt"):
            u = stats.mannwhitneyu(a[m], b[m])
            auc = u.statistic / (len(a) * len(b))
            print(f"  {m:20s} pathogenic vs tolerated: AUC={auc:.3f} p={u.pvalue:.3f}")
