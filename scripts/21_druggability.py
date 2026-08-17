#!/usr/bin/env python
"""
Druggability of the genes carrying onset signal.

Takes the genes ranked by the genome-wide onset statistic, plus the members of
any pathway that survives the competitive test, and annotates each with Open
Targets Platform data: small-molecule and antibody tractability, known drugs and
their maximum clinical phase in any indication, and existing association with
Parkinson's disease. The output is a ranked candidate table with the evidence for
each candidate stated, not a nomination.
"""
import json, time, urllib.request, os
import numpy as np, pandas as pd
import paths as _P

BASE = str(_P.ROOT)
OUT = f"{BASE}/results"
OT = "https://api.platform.opentargets.org/api/v4/graphql"
PD_EFO = "MONDO_0005180"

Q = """
query T($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id approvedSymbol biotype
    tractability { modality label value }
    knownDrugs(size: 60) {
      count
      rows { drug { name maximumClinicalTrialPhase isApproved drugType }
             phase status disease { name id } }
    }
    associatedDiseases(efoIds: ["MONDO_0005180"]) {
      rows { score disease { id name } }
    }
  }
}
"""


def gql(q, v, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(
                OT, data=json.dumps({"query": q, "variables": v}).encode(),
                headers={"Content-Type": "application/json", "User-Agent": "research/1.0"})
            return json.load(urllib.request.urlopen(req, timeout=90))
        except Exception:
            time.sleep(3 * (i + 1))
    return None


def resolve(symbols):
    """Symbol -> Ensembl id via the Open Targets search endpoint."""
    q = """query S($q: String!) { search(queryString: $q, entityNames: ["target"], page:{index:0,size:5})
           { hits { id name entity object { ... on Target { approvedSymbol } } } } }"""
    out = {}
    for s in symbols:
        r = gql(q, {"q": s})
        if not r:
            continue
        for h in (r.get("data", {}).get("search", {}) or {}).get("hits", []):
            sym = (h.get("object") or {}).get("approvedSymbol")
            if sym == s:
                out[s] = h["id"]
                break
        time.sleep(0.12)
    return out


def main():
    ga = pd.read_csv(f"{OUT}/70_genestats_aao.csv")
    gj = pd.read_csv(f"{OUT}/71_genestats_joined.csv")
    lyso = set()
    for lib in ["KEGG_2021_Human", "GO_Cellular_Component_2023", "Reactome_2022"]:
        p = f"{BASE}/external/genesets/{lib}.txt"
        if not os.path.exists(p):
            continue
        for line in open(p):
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            nm = parts[0].lower()
            if "lysosom" in nm or "lytic vacuole" in nm or "autophag" in nm:
                lyso |= {g.split(",")[0].strip() for g in parts[2:] if g.strip()}

    # candidates: strongest onset genes overall, and lysosomal genes with onset signal
    top = ga.sort_values("min_p").head(60)
    lyso_hits = gj[gj.gene.isin(lyso)].sort_values("min_p").head(40)
    cand = sorted(set(top.gene) | set(lyso_hits.gene))
    cand = [c for c in cand if not c.startswith(("LINC", "LOC", "MIR", "SNOR"))]
    print(f"{len(cand)} candidate genes ({len(lyso)} lysosomal genes in the libraries)")

    ids = resolve(cand)
    print(f"resolved {len(ids)} to Ensembl ids", flush=True)

    rows = []
    for sym, eid in ids.items():
        r = gql(Q, {"ensemblId": eid})
        t = ((r or {}).get("data") or {}).get("target")
        if not t:
            continue
        tr = t.get("tractability") or []
        sm_ok = any(x["modality"] == "SM" and x["value"] and
                    x["label"] in ("Approved Drug", "Advanced Clinical",
                                   "Phase 1 Clinical", "Structure with Ligand",
                                   "High-Quality Ligand", "High-Quality Pocket")
                    for x in tr)
        ab_ok = any(x["modality"] == "AB" and x["value"] and
                    x["label"] in ("Approved Drug", "Advanced Clinical",
                                   "Phase 1 Clinical", "UniProt loc high conf",
                                   "GO CC high conf") for x in tr)
        best_sm = [x["label"] for x in tr if x["modality"] == "SM" and x["value"]]
        kd = t.get("knownDrugs") or {}
        drows = kd.get("rows") or []
        maxph = max([d["drug"]["maximumClinicalTrialPhase"] or 0 for d in drows], default=0)
        approved = sorted({d["drug"]["name"] for d in drows if d["drug"]["isApproved"]})
        pd_drugs = sorted({d["drug"]["name"] for d in drows
                           if "parkinson" in (d.get("disease") or {}).get("name", "").lower()})
        assoc = (t.get("associatedDiseases") or {}).get("rows") or []
        pd_score = max([a["score"] for a in assoc], default=0.0)
        g = ga[ga.gene == sym]
        rows.append(dict(gene=sym, ensembl=eid,
                         onset_min_p=float(g.min_p.iloc[0]) if len(g) else np.nan,
                         onset_z=float(g.z.iloc[0]) if len(g) else np.nan,
                         lysosomal=sym in lyso,
                         sm_tractable=sm_ok, ab_tractable=ab_ok,
                         sm_labels=";".join(best_sm[:4]),
                         n_known_drugs=kd.get("count", 0), max_phase=maxph,
                         approved_drugs=";".join(approved[:5]),
                         pd_drugs=";".join(pd_drugs[:5]),
                         opentargets_pd_score=pd_score))
        time.sleep(0.12)

    d = pd.DataFrame(rows).sort_values("onset_min_p")
    d.to_csv(f"{OUT}/85_druggability.csv", index=False)
    print(f"\nannotated {len(d)} targets")
    print("\n=== onset-signal genes that are small-molecule tractable ===")
    sel = d[d.sm_tractable].sort_values("onset_min_p").head(25)
    print(f"{'gene':12s} {'onset p':>9s} {'lyso':>5s} {'phase':>5s} {'drugs':>6s}  approved / PD")
    for _, x in sel.iterrows():
        print(f"{x.gene:12s} {x.onset_min_p:9.2e} {str(x.lysosomal):>5s} "
              f"{int(x.max_phase):5d} {int(x.n_known_drugs):6d}  "
              f"{(x.approved_drugs or '-')[:44]}")
    json.dump(dict(n=len(d), n_sm=int(d.sm_tractable.sum()),
                   n_lyso=int(d.lysosomal.sum()),
                   n_with_approved=int((d.approved_drugs.fillna('') != '').sum())),
              open(f"{OUT}/86_druggability_summary.json", "w"), indent=2)
    print(f"\nwrote {OUT}/85_druggability.csv")


if __name__ == "__main__":
    main()
