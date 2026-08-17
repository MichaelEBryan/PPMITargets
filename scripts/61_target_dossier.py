#!/usr/bin/env python
"""Structural and annotation dossier for each candidate gene.

Pulls the AlphaFold model, finds buried cavities, checks whether the largest
cavity contains the UniProt-annotated catalytic residues, and adds experimental
structure counts, tractability categories, loss-of-function constraint and
brain expression. The same pipeline is run on twenty control proteins, ten with
an approved small-molecule drug and ten without, so the cavity descriptors can
be scored rather than only described.
"""
import os, sys, json, time, gzip, io, warnings
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pocket import read_pdb, find_pockets
import paths as _P
warnings.filterwarnings("ignore")

ROOT = str(_P.ROOT)
RES = str(_P.RESULTS)
STR = str(_P.STRUCTURES)
os.makedirs(STR, exist_ok=True)

CAND = {
    "SNCA": ("P37840", "ENSG00000145335"),
    "TMEM175": ("Q9BSE2", "ENSG00000127419"),
    "GAK": ("O14976", "ENSG00000178950"),
    "DGKQ": ("P52824", "ENSG00000145214"),
    "IDUA": ("P35475", "ENSG00000127415"),
    "APOE": ("P02649", "ENSG00000130203"),
    "CTSB": ("P07858", "ENSG00000164733"),
    "ASAH1": ("Q13510", "ENSG00000104763"),
    "GALC": ("P54803", "ENSG00000054983"),
    "SCARB2": ("Q14108", "ENSG00000138760"),
    "MAPT": ("P10636", "ENSG00000186868"),
    "GPNMB": ("Q14956", "ENSG00000136235"),
    "LRRC37A2": ("A6NM11", "ENSG00000238083"),
    "CTSD": ("P07339", "ENSG00000117984"),
    "VPS13C": ("Q709C8", "ENSG00000129003"),
    "GBA1": ("P04062", "ENSG00000177628"),
    "LRRK2": ("Q5S007", "ENSG00000188906"),
}

# Twenty controls. The first ten each carry at least one approved
# small-molecule drug; the second ten are long-standing hard targets with none.
CONTROL = {
    "EGFR": ("P00533", 1), "HMGCR": ("P04035", 1), "PTGS2": ("P35354", 1),
    "ACE": ("P12821", 1), "DHFR": ("P00374", 1), "CA2": ("P00918", 1),
    "ADRB2": ("P07550", 1), "BRAF": ("P15056", 1), "PARP1": ("P09874", 1),
    "HDAC2": ("Q92769", 1),
    "MYC": ("P01106", 0), "TP53": ("P04637", 0), "CTNNB1": ("P35222", 0),
    "STAT3": ("P40763", 0), "NPM1": ("P06748", 0), "MAX": ("P61244", 0),
    "JUN": ("P05412", 0), "FOXO3": ("O43524", 0), "RUNX1": ("Q01196", 0),
    "MECP2": ("P51608", 0),
}

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OT_QUERY = """
query T($id: String!) {
  target(ensemblId: $id) {
    id approvedSymbol
    tractability { modality label value }
  }
}
"""


def fetch_alphafold(gene, acc):
    path = f"{STR}/AF-{acc}-F1.pdb"
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    try:
        meta = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}",
                            timeout=120).json()
        if not meta:
            return None
        r = requests.get(meta[0]["pdbUrl"], timeout=300)
        if r.ok and len(r.content) > 1000:
            open(path, "wb").write(r.content)
            return path
    except Exception as e:
        print(f"    [alphafold] {gene}: {e}")
    return None


def uniprot_sites(acc):
    """Annotated catalytic and ligand-binding residues."""
    path = f"{STR}/uniprot_{acc}.json"
    if os.path.exists(path):
        d = json.load(open(path))
    else:
        try:
            d = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json",
                             timeout=90).json()
            json.dump(d, open(path, "w"))
        except Exception:
            return set(), set(), None
    act, bind = set(), set()
    for f in d.get("features", []):
        t = f.get("type")
        try:
            s = f["location"]["start"]["value"]
            e = f["location"]["end"]["value"]
        except Exception:
            continue
        if t == "Active site":
            act.update(range(s, e + 1))
        elif t == "Binding site":
            bind.update(range(s, e + 1))
    length = (d.get("sequence") or {}).get("length")
    return act, bind, length


def open_targets(gene, eid):
    try:
        r = requests.post(OT_URL, json={"query": OT_QUERY,
                                        "variables": {"id": eid}}, timeout=90)
        d = (r.json().get("data") or {}).get("target")
        if not d:
            return {}
        tr = {}
        for t in d.get("tractability", []):
            if t["value"]:
                tr.setdefault(t["modality"], []).append(t["label"])
        return tr
    except Exception as e:
        print(f"    [open targets] {gene}: {e}")
        return {}


def rcsb(acc):
    base = [{"type": "terminal", "service": "text", "parameters": {
        "attribute": "rcsb_polymer_entity_container_identifiers."
                     "reference_sequence_identifiers.database_accession",
        "operator": "exact_match", "value": acc}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity_container_identifiers."
                         "reference_sequence_identifiers.database_name",
            "operator": "exact_match", "value": "UniProt"}}]

    def run(nodes):
        q = {"query": {"type": "group", "logical_operator": "and",
                       "nodes": nodes},
             "return_type": "entry",
             "request_options": {"paginate": {"start": 0, "rows": 2000}}}
        try:
            r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query",
                              json=q, timeout=120)
            return len(r.json().get("result_set", [])) if r.ok else 0
        except Exception:
            return 0

    n_all = run(base)
    n_lig = run(base + [{"type": "terminal", "service": "text", "parameters": {
        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
        "operator": "greater", "value": 0}}]) if n_all else 0
    return dict(n_pdb=n_all, n_pdb_ligand=n_lig)


def gnomad_constraint():
    path = f"{ROOT}/external/gnomad_constraint.tsv"
    if not os.path.exists(path):
        url = ("https://storage.googleapis.com/gcp-public-data--gnomad/release/"
               "2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz")
        r = requests.get(url, timeout=900); r.raise_for_status()
        with gzip.open(io.BytesIO(r.content), "rt") as fh:
            open(path, "w").write(fh.read())
    d = pd.read_csv(path, sep="\t", low_memory=False)
    keep = [c for c in ["gene", "oe_lof_upper", "pLI", "mis_z", "oe_mis"]
            if c in d.columns]
    return d[keep].dropna(subset=["gene"]).groupby("gene").first()


GTEX_ALIAS = {"GBA1": "GBA"}


def gtex_brain(gene):
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
        rows = r.json().get("data", []) if r.ok else []
        out = {x["tissueSiteDetailId"]: x["median"] for x in rows
               if x.get("tissueSiteDetailId", "").startswith("Brain")}
        json.dump(out, open(path, "w"))
        return out
    except Exception as e:
        print(f"    [gtex] {gene}: {e}")
        return {}


def profile(gene, acc, want_external=True, eid=None, cons=None):
    rec = dict(gene=gene, uniprot=acc)
    act, bind, length = uniprot_sites(acc)
    rec["n_annotated_site_residues"] = len(act | bind)
    path = fetch_alphafold(gene, acc)
    pocket_pts = None
    if path:
        xyz, plddt, resid, resn, elem = read_pdb(path)
        rec["n_residues"] = int(len(set(resid)))
        rec["mean_plddt"] = float(np.nanmean(plddt))
        rec["frac_confident"] = float(np.nanmean(plddt >= 70))
        rec["frac_disordered"] = float(np.nanmean(plddt < 50))
        pk = find_pockets(xyz, resn, resid, elem, min_psp=4, min_volume=50,
                          n_report=5)
        rec["n_pockets"] = len(pk)
        if pk:
            top = pk[0]
            pocket_pts = top["points"]
            for k in ("volume", "enclosure", "n_lining",
                      "hydrophobic_fraction", "depth_below_surface"):
                rec[f"pocket_{k}"] = top[k]
            rec["pocket_lining"] = ";".join(top["lining"])
            sites = act | bind
            hit = None
            for i, q in enumerate(pk):
                nums = {int("".join(ch for ch in x if ch.isdigit()))
                        for x in q["lining"]}
                if sites and nums & sites:
                    hit = i + 1
                    rec["site_pocket_rank"] = hit
                    rec["site_pocket_volume"] = q["volume"]
                    rec["n_site_residues_in_pocket"] = len(nums & sites)
                    break
            rec["top_pocket_is_catalytic"] = (hit == 1) if sites else np.nan
    if want_external:
        tr = open_targets(gene, eid) if eid else {}
        for mod, labs in tr.items():
            rec[f"tract_{mod}"] = ";".join(labs)
        rec.update(rcsb(acc))
        key = GTEX_ALIAS.get(gene, gene)
        if cons is not None and key in cons.index:
            c = cons.loc[key]
            rec["loeuf"] = float(c.get("oe_lof_upper", np.nan))
            rec["pLI"] = float(c.get("pLI", np.nan))
            rec["mis_z"] = float(c.get("mis_z", np.nan))
        ex = gtex_brain(gene)
        if ex:
            rec["tpm_substantia_nigra"] = ex.get("Brain_Substantia_nigra")
            rec["tpm_brain_median"] = float(np.median(list(ex.values())))
            rec["tpm_brain_max"] = float(np.max(list(ex.values())))
    return rec, pocket_pts


def main():
    t0 = time.time()
    cons = gnomad_constraint()

    rows, pts = [], {}
    print("=== candidate genes ===")
    for g, (acc, eid) in CAND.items():
        r, p = profile(g, acc, True, eid, cons)
        rows.append(r)
        if p is not None:
            pts[g] = p
        print(f"{g:9s} pLDDT {r.get('mean_plddt', float('nan')):5.1f}  "
              f"pocket {r.get('pocket_volume', float('nan')):7.0f} A^3  "
              f"catalytic rank {r.get('site_pocket_rank', '-')}  "
              f"PDB+ligand {r.get('n_pdb_ligand', 0):4d}  "
              f"LOEUF {r.get('loeuf', float('nan')):.2f}")
    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/170_target_dossier.csv", index=False)
    np.savez_compressed(f"{RES}/171_pocket_points.npz", **pts)

    print("\n=== benchmark controls ===")
    brows = []
    for g, (acc, lab) in CONTROL.items():
        r, _ = profile(g, acc, want_external=False)
        r["has_approved_sm_drug"] = lab
        brows.append(r)
        print(f"{g:8s} drug={lab}  pLDDT {r.get('mean_plddt', float('nan')):5.1f}  "
              f"pocket {r.get('pocket_volume', float('nan')):7.0f} A^3  "
              f"enclosure {r.get('pocket_enclosure', float('nan')):.2f}")
    b = pd.DataFrame(brows)
    b.to_csv(f"{RES}/172_pocket_benchmark.csv", index=False)

    from sklearn.metrics import roc_auc_score
    auc = {}
    for c in ["pocket_volume", "pocket_enclosure", "pocket_hydrophobic_fraction",
              "pocket_n_lining", "mean_plddt", "frac_disordered"]:
        s = b[["has_approved_sm_drug", c]].dropna()
        if s.has_approved_sm_drug.nunique() == 2 and len(s) >= 8:
            a = roc_auc_score(s.has_approved_sm_drug, s[c])
            auc[c] = float(max(a, 1 - a))
            auc[c + "_direction"] = "higher" if a >= 0.5 else "lower"
    json.dump(auc, open(f"{RES}/173_pocket_benchmark_auc.json", "w"), indent=1)
    print("\nbenchmark AUC:",
          json.dumps({k: round(v, 3) for k, v in auc.items()
                      if not k.endswith("_direction")}, indent=1))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
