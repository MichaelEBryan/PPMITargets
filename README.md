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

## What is not yet scripted

The analysis grew over several sessions and some intermediate steps were run
interactively rather than saved. Two of those have since been written up as
scripts (`66_onset_groups.py` and `67_locus_gene_count.py`, both verified to
reproduce the stored values exactly). The following files are still read by
figure scripts but have no producing script here, so a clean rebuild from raw
data will stop at them:

| File | What it holds | Where it came from |
|---|---|---|
| `09_simulation_grid.csv` | simulation grid, flattened | derived from `09_simulation_grid.json` written by `04_modal_simulation.py` |
| `81_method_grid.csv` | method comparison grid | derived from the JSON written by `20_modal_method.py` |
| `19b_risk_vs_onset_aligned.csv` | 86 loci, risk and onset effects on a common allele | allele alignment step |
| `85_target_annotation.csv` | per-gene onset statistic and tractability | join of `70_genestats_*` and `85_druggability.csv` |
| `70_genestats_aao.csv` | gene-level onset statistics | `19_genewide_onset.py` writes `70_genestats_{tag}.csv`; the tag is set at the call site |
| `120_galc_region_risk.csv`, `121_galc_eqtl_brain.csv` | GALC regional and eQTL extracts | extraction steps preceding `30_galc_activity.py` |
| `131_galc_variants_geometry.csv`, `132_galc_method_comparison.csv` | ESMFold and AlphaFold variant geometry | assembled from the Modal output of `10_modal_galc_structure.py` |
| `140_learning_curve.csv` | learning curves | run alongside `22_ml_value.py` |
| `94b_interaction_nulls.json` | interaction permutation nulls | rerun of the interaction screen with the carrier filter |
| `16_prs_negative_control.csv` | polygenic negative control | `05_corrected_analysis.py` variant |

None of these affect the figures as published, which were generated from the
files as they stand. They matter only for rebuilding from scratch, and each is
a short step rather than a large computation.

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
`xelatex manuscript_v2.tex`, run twice.

The Word version required for submission is generated from the same source, so
the two cannot drift apart:

```bash
python tools/make_reference.py   # formatting taken from the journal template
python tools/tex_to_jei.py       # LaTeX to Markdown, in the required order
pandoc tools/manuscript_jei.md --reference-doc=tools/jei_reference.docx \
  --from=markdown+raw_attribute+pipe_tables+superscript+subscript \
  --to=docx --resource-path=.:figures -o manuscript/Khare_Bryan_JEI.docx
python tools/check_docx.py       # verify against the journal's requirements
python tools/page_count.py       # printed length, measured in Arial 11
```

`make_reference.py` reads the journal's own author template and keeps its page
setup, one-inch margins and continuous line numbering, then sets every style to
Arial 11 at 1.5 line spacing with bold headings. Headings are written as bold
paragraphs rather than Word heading styles, as the template asks.

`check_docx.py` tests the built file against each requirement in turn: page
size, margins, line numbering, font, size, spacing, section order, figure
sizing, and that every embedded image has a declared content type. It reports
each check separately so a failure says which requirement is unmet.

`page_count.py` lays out the text with the installed Arial metrics to estimate
the printed length from the Introduction to the end of the Methods, which the
journal caps at ten pages.

## Requirements

Python 3.12 with numpy, pandas, scipy, scikit-learn, matplotlib, xgboost,
lightgbm, catboost, shap, requests and pyliftover.
