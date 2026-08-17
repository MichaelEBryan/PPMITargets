# PPMITargets

Genetic analysis of age at onset in Parkinson's disease, using the Parkinson's
Progression Markers Initiative (PPMI) together with public genome-wide,
expression and structural resources.

The question is whether Parkinson's disease beginning before 50 differs from
disease beginning after 60 in the mechanisms involved, or only in how much
inherited risk the patient carries. The analysis also asks what the genetics
implies for drug targets: which genes carry evidence for an effect on onset,
whether their proteins have a site a drug could bind, and in which direction a
drug would have to act.

## Data

PPMI is controlled-access. Nothing derived from individual participants is in
this repository. To reproduce the PPMI parts you need your own approved
download from https://www.ppmi-info.org, placed in `data/`.

Everything else is public and is downloaded by the scripts:

| Source | Used for |
|---|---|
| IPDGC age-at-onset GWAS (GCST007780) | per-variant effects on onset, 28,568 patients |
| Nalls risk meta-analysis (GCST009325) | per-variant effects on risk |
| GTEx v8 brain eQTL and median expression | effect of the risk allele on gene expression |
| gnomAD v2.1.1 constraint | tolerance of loss of function |
| AlphaFold DB, UniProt, RCSB PDB | predicted structures, catalytic sites, ligand-bound structures |
| Open Targets Platform | tractability categories |
| 1000 Genomes phase 3 | linkage disequilibrium |

## Running the analysis

Scripts are numbered in execution order and are independent of each other
except through the files they write to `results/`.

```bash
python scripts/06_prep_matrix.py        # build the analysis matrix from PPMI
python scripts/15_build_aao.py          # age at first motor symptom
python scripts/16_aao_core.py           # polygenic burden against onset
python scripts/60_shap_importance.py    # classifier attributions, 200 bootstraps
python scripts/61_target_dossier.py     # structures, pockets, tractability
python scripts/65_eqtl_evidence.py      # risk-allele effect on brain expression
```

`60_shap_importance.py` takes roughly two hours on eight cores.
`65_eqtl_evidence.py` needs `pyliftover` and downloads the UCSC hg38-to-hg19
chain on first use.

## Figures

Each figure script reads only from `results/` and writes a PDF and a PNG to
`figures/`. The shared style lives in `scripts/clinical_style.py`.

```bash
python scripts/74_fig_summary.py        # Figure 1, summary
python scripts/50_clinical_fig1.py      # onset by genetic cause
python scripts/75_fig_eqtl.py           # risk-allele effect on expression
python scripts/70_fig_targets.py        # structures and target criteria
python scripts/71_fig_model.py          # what the classifier attends to
python scripts/76_fig_effectsize.py     # effect sizes against method sensitivity
```

## How the risk-allele expression effects are assessed

This is the part most easily got wrong, so the steps are set out in full.

1. For each candidate gene, take the strongest risk-associated variant within
   500 kb as the locus lead, and orient it onto its risk-increasing allele.
2. Take every significant brain eQTL for that gene in all 13 GTEx v8 brain
   tissues.
3. Lift each eQTL variant from GRCh38 to GRCh37 with the UCSC chain, and match
   it to the risk statistics by position.
4. Keep the variant only if the two allele pairs agree. Flip the sign of the
   expression slope where GTEx's alternate allele is not the risk allele. This
   step is the one that silently reverses the answer if skipped.
5. Keep variants within 250 kb of the lead whose risk P-value is within two
   orders of magnitude of the lead's own, so a weak locus is judged on its own
   scale rather than against a fixed threshold.
6. Within each tissue, combine the surviving variants by inverse-variance
   weighting, giving one effect and one standard error per gene per region.
7. Each tissue then votes once. A direction is called where at least three
   quarters of voting tissues agree.

A strong, consistent eQTL does not on its own identify the gene. The same
figure therefore reports how many genes each locus lead variant controls in
brain: one or none at most loci, but 28 at 17q21.31, which is why the
*LRRC37A2* signal cannot be attributed to *LRRC37A2* rather than to any of its
neighbours on the same inversion haplotype.

## Main outputs

| File | Contents |
|---|---|
| `results/51_prs_vs_AAO.csv` | polygenic burden against onset, 24 specifications |
| `results/19b_risk_vs_onset_aligned.csv` | risk and onset effects for 86 loci, allele-aligned |
| `results/53_pervariant_AAO.csv` | per-variant onset and case-case statistics |
| `results/110_geneset_maxT.csv` | 6,259 gene sets, family-wise corrected |
| `results/160_shap_importance.csv` | Shapley attribution per feature per model |
| `results/161_shap_rank_bootstrap.csv` | bootstrap rank interval per feature |
| `results/170_target_dossier.csv` | structure, pocket, constraint and tractability per gene |
| `results/172_pocket_benchmark.csv` | the same descriptors on 20 control proteins |
| `results/185_eqtl_by_tissue.csv` | risk-allele effect on expression, per gene per region |
| `results/187_locus_gene_count.csv` | genes controlled by each locus lead variant |

`results/` is not tracked here because several files contain participant-level
values derived from PPMI.

## Manuscript

`manuscript/` holds the LaTeX source and the compiled PDF. Build with
`xelatex manuscript.tex`, run twice.

## Requirements

Python 3.12 with numpy, pandas, scipy, scikit-learn, matplotlib, xgboost,
lightgbm, catboost, shap, requests and pyliftover.
