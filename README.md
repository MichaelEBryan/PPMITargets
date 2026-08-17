# PPMITargets

Genetic analysis of age at onset in Parkinson's disease, using the Parkinson's
Progression Markers Initiative together with public genome-wide, expression and
structural resources.

The question is whether disease beginning before 50 differs from disease
beginning after 60 in the mechanisms involved, or only in how much inherited
risk the patient carries. A second question follows from it: what the genetics
implies for drug targets, meaning which genes carry evidence for an effect on
onset, whether their proteins have a site a drug could bind, and in which
direction a drug would have to act.

## Layout

Directories run in the order the work was done. Each contains numbered scripts
that run in order within it. Scripts communicate only through files in
`results/`, so any stage can be rerun on its own once its inputs exist.

| | |
|---|---|
| `01_dataset` | build the analysis matrix and the onset variable from PPMI |
| `02_earlier_pipeline` | audit of the earlier feature-ranking approach that motivated this work |
| `03_onset` | onset by genetic cause, by polygenic burden, and whether it is predictable |
| `04_genomewide` | per-gene, per-pathway and gene-set tests of onset against risk |
| `05_expression` | effect of the risk allele on gene expression in brain |
| `06_targets` | structures, binding pockets, constraint and tractability |
| `07_attribution` | Shapley attributions for the earlier classifier |
| `08_figures_and_tables` | every figure and table |
| `lib` | paths, plotting style, pocket detection, protein drawing |

## Data

PPMI is controlled-access. Nothing derived from individual participants is in
this repository. Reproducing the PPMI parts needs your own approved download
from https://www.ppmi-info.org placed in `data/`. The scripts expect the study
files under their download names, including `Participant_Status`,
`Demographics`, `PD_Diagnosis_History`, `Family_History`,
`Polygenic_Risk_Scores` and `PPMI_Project_9001`.

Everything else is public and is fetched by the scripts that need it.

| Source | Used for |
|---|---|
| IPDGC age-at-onset GWAS (GCST007780) | per-variant effects on onset, 28,568 patients |
| Nalls risk meta-analysis (GCST009325) | per-variant effects on risk |
| GTEx v8 brain eQTL and median expression | effect of the risk allele on expression |
| gnomAD v2.1.1 constraint | tolerance of loss of function |
| AlphaFold DB, UniProt, RCSB PDB | predicted structures, catalytic sites, ligand-bound structures |
| Open Targets Platform | tractability categories |
| 1000 Genomes phase 3 | linkage disequilibrium |

Two small reference tables are also expected in `data/`: `genes_hg19.csv`
(gene coordinates) and `snp_map.csv` (variant identifiers to positions).

## What each stage does

**01_dataset.** Builds a numeric feature matrix from the PPMI tables, dropping
identifiers, dates and free text. Derives age at onset as the age at the first
recorded motor symptom. `02_variant_calls.py` runs on Modal and is needed only
to regenerate the cloud results.

**02_earlier_pipeline.** Reproduces the earlier analysis as published, then
measures how much of its classifier performance comes from study-design
variables rather than biology, and how often its feature-ranking comparison
would report a difference between two groups that in truth share an
architecture. `03_simulation_grid.py` and `04_method_grid.py` run on Modal.

**03_onset.** Onset by genetic cause and by third of polygenic burden, with
Kaplan-Meier curves and a log-rank test, then whether a model can predict one
patient's onset age rather than describe the group.

**04_genomewide.** Per-variant and per-gene effects on onset set against the
same variants' effects on risk, then the same comparison across 6,259 gene
sets, and a leave-one-out check on the lysosomal set.

**05_expression.** For each candidate gene, the effect of the risk allele on
its expression in each of 13 GTEx brain regions, and how many genes each locus
lead variant controls. This stage is set out in full below because it is the
one most easily got wrong.

**06_targets.** Predicted structures, cavity detection, catalytic-site
recovery, constraint and tractability for the candidate genes and for 20
control proteins with known outcomes.

**07_attribution.** Shapley attributions for the earlier classifier, with the
rank of each feature bootstrapped over 200 resamples.
`01_shap_importance.py` takes roughly two hours on eight cores.

**08_figures_and_tables.** Reads only from `results/` and writes a PDF and a
PNG for each figure into `figures/`. Five are main figures, `fig01` to
`fig05`, and twenty are supplementary, `supp01` to `supp20`.

## How the risk-allele expression effects are assessed

1. For each candidate gene, take the strongest risk-associated variant within
   500 kb as the locus lead, and orient it onto its risk-increasing allele.
2. Take every significant brain eQTL for that gene in all 13 GTEx v8 brain
   tissues.
3. Lift each eQTL variant from GRCh38 to GRCh37 with the UCSC chain and match
   it to the risk statistics by position.
4. Keep the variant only if the two allele pairs agree. Flip the sign of the
   expression slope where GTEx's alternate allele is not the risk allele. This
   step reverses the answer if skipped.
5. Keep variants within 250 kb of the lead whose risk P-value is within two
   orders of magnitude of the lead's own, so a weak locus is judged on its own
   scale rather than against a fixed threshold.
6. Within each tissue, combine the surviving variants by inverse-variance
   weighting, giving one effect and one standard error per gene per region.
7. Each tissue votes once. A direction is called where at least three quarters
   of voting tissues agree.

A strong, consistent eQTL does not on its own identify the gene, so the same
stage counts how many genes each locus lead variant controls in brain: one or
none at most loci, but 28 at 17q21.31, which is why the *LRRC37A2* signal
cannot be attributed to *LRRC37A2* rather than to its neighbours on the same
inversion haplotype.

## Assumptions

These are choices that change the answer, listed so they can be argued with.

- Onset is the age at the first recorded motor symptom. Age at enrolment is
  used only in `08_figures_and_tables/09_supplementary_b.py`, to show what the
  earlier analysis obtained from that proxy.
- Enrolment and study-design variables are excluded from every onset model.
  They carry the ascertainment that put a patient into one PPMI arm rather
  than another, and including them recovers the design instead of the biology.
- Polygenic analyses are restricted to inferred-European patients recruited
  without a known pathogenic variant, and burden is split into thirds within
  that group.
- Gene-set tests use circular rotation of the genome rather than gene
  permutation, so that clustering of related genes is preserved under the null,
  and are corrected family-wise by the maximum statistic.
- Expression effects are aligned onto the risk-increasing allele, as above.
  Unaligned effects point the wrong way for roughly half the loci.
- Cavities are found by LIGSITE on a 0.8 Å grid with a 1.4 Å probe, requiring
  at least four protein-solvent-protein events. The enclosure threshold of
  0.705 is the best cut on a benchmark of 20 control proteins, 10 with an
  approved small-molecule drug and 10 without after long effort.
- Structures are AlphaFold DB models where one exists and ESMFold predictions
  for variants. Neither is experimental evidence of a druggable site.

## Reproducing

```bash
pip install -r requirements.txt
python 01_dataset/01_build_matrix.py
python 01_dataset/03_age_at_onset.py
python 03_onset/01_onset_by_group.py
python 03_onset/02_polygenic_burden.py
python 05_expression/01_eqtl_by_tissue.py
python 06_targets/01_target_dossier.py
python 07_attribution/01_shap_importance.py
```

Then the figures:

```bash
python 08_figures_and_tables/01_summary.py
python 08_figures_and_tables/02_onset.py
python 08_figures_and_tables/03_expression.py
```

`05_expression/01_eqtl_by_tissue.py` needs `pyliftover` and downloads the UCSC
hg38-to-hg19 chain on first use. Scripts named for Modal run remotely and are
not needed unless those results are being regenerated.

## Inspecting results without rerunning anything

`results/` is not tracked here, because several files hold participant-level
values derived from PPMI. The figures in `figures/` were generated from those
files and can be read directly. The main result files are:

| File | Contents |
|---|---|
| `51_prs_vs_AAO.csv` | polygenic burden against onset, 24 specifications |
| `19b_risk_vs_onset_aligned.csv` | risk and onset effects for 86 loci, allele-aligned |
| `53_pervariant_AAO.csv` | per-variant onset and case-case statistics |
| `110_geneset_maxT.csv` | 6,259 gene sets, family-wise corrected |
| `160_shap_importance.csv` | Shapley attribution per feature per model |
| `161_shap_rank_bootstrap.csv` | bootstrap rank interval per feature |
| `170_target_dossier.csv` | structure, pocket, constraint and tractability per gene |
| `172_pocket_benchmark.csv` | the same descriptors on 20 control proteins |
| `185_eqtl_by_tissue.csv` | risk-allele effect on expression, per gene per region |
| `187_locus_gene_count.csv` | genes controlled by each locus lead variant |

## Steps not yet scripted

The work grew over several sessions and some intermediate steps were run
interactively. The files below are read by scripts here but have no producing
script, so a rebuild from raw data stops at them. None affects the figures as
published, which were made from these files as they stand. Each is a short
step rather than a large computation.

| File | What it holds | Where it came from |
|---|---|---|
| `09_simulation_grid.csv` | simulation grid, flattened | the JSON written by `02_earlier_pipeline/03_simulation_grid.py` |
| `81_method_grid.csv` | method comparison grid | the JSON written by `02_earlier_pipeline/04_method_grid.py` |
| `19b_risk_vs_onset_aligned.csv` | 86 loci, risk and onset effects on a common allele | the allele alignment step |
| `70_genestats_aao.csv` | gene-level onset statistics | `04_genomewide/01_gene_level.py` writes `70_genestats_{tag}.csv`; the tag is set at the call site |
| `85_target_annotation.csv` | per-gene onset statistic and tractability | a join of `70_genestats_*` and `85_druggability.csv` |
| `120_galc_region_risk.csv`, `121_galc_eqtl_brain.csv` | GALC regional and eQTL extracts | extraction preceding `06_targets/03_galc_activity.py` |
| `131_galc_variants_geometry.csv`, `132_galc_method_comparison.csv` | ESMFold and AlphaFold variant geometry | the Modal output of `06_targets/04_galc_structure.py` |
| `140_learning_curve.csv` | learning curves | run alongside `03_onset/03_predictability.py` |
| `94b_interaction_nulls.json` | interaction permutation nulls | a rerun of the interaction screen with the carrier filter |
| `16_prs_negative_control.csv` | polygenic negative control | a variant of `02_earlier_pipeline/05_corrected_analysis.py` |

## Requirements

Python 3.12. `pip install -r requirements.txt`.
`06_targets/04_galc_structure.py` needs `torch` and `transformers` and is
intended to run on Modal rather than locally.
