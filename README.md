# How much can embryo selection gain?

**Headroom of embryo selection, ceiling for AI, and reproducibility of the choice, estimated
from sibling blastocyst cohorts.** Course project MMD6020 (Université de Montréal), 2026.

When an IVF patient has several blastocysts, only one is transferred first. AI tools promise a
better choice, but the largest randomised trial (Illingworth et al., *Nat Med* 2024) did not
show non-inferiority. This project measures, on an open dataset, **how many live births a
better choice can add at the first transfer**, how much of that gain embryologists already
capture, and how much is left for an AI.

| Estimand | Contrast | Role |
|---|---|---|
| E1 | best Gardner-graded embryo − random embryo | primary |
| E2 | embryologist's choice − random | secondary |
| E3 | best embryo according to the model − embryologist's choice | secondary |
| E4 | best − worst embryo | descriptive |
| E5 | gain over morphology of a predictor with higher AUROC (simulation) + trial size | secondary |
| E6 | share of choices changed, and births lost, when another embryologist grades | secondary |
| E7 | live birth with k vs 1 blastocyst available (model-free check) | secondary |

The full pre-registered analysis plan is in [`docs/analysis_plan.md`](docs/analysis_plan.md);
deviations are logged in [`docs/deviations.md`](docs/deviations.md).

## Data

Kromp F. et al. *An annotated human blastocyst dataset to benchmark deep learning architectures
for in vitro fertilization.* Sci Data 10, 271 (2023).
[doi:10.1038/s41597-023-02182-3](https://doi.org/10.1038/s41597-023-02182-3) ·
[Figshare](https://doi.org/10.6084/m9.figshare.20123153.v3) ·
[GitHub](https://github.com/software-competence-center-hagenberg/Blastocyst-Dataset)
(pinned commit `d2acd67`).

- 2,344 blastocyst images of 837 patients (Kepler University Linz), expert Gardner grades.
- Clinical data of 752 fresh transfers: age, AMH, endometrium, oocytes, hCG, heart activity,
  live birth.
- 300 test images graded by 11 embryologists from 11 clinics + consensus.

The data are **not** redistributed here: `make data` downloads them and verifies MD5
checksums (`data/raw/MANIFEST.csv`). If a download is blocked, place the files manually:

```
data/raw/Clincial_annotations.csv        (Figshare; sic, original file name)
data/raw/complete.csv                    (GitHub)
data/raw/annotations/*.csv               (GitHub: Annotator_0..10, Gardner_Expert, test_rev)
data/raw/Images/                         (optional, step G only)
```

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and the exact package
versions from `uv.lock`) and `make`.

```bash
git clone <this repository> && cd selection-headroom
uv sync --group dev          # exact environment from uv.lock
make test                    # unit tests (no data needed)
make data                    # download + checksum verification
make quick                   # whole pipeline with small replicate counts (~3 min)
make clean && make all       # full analysis (~1 h on a laptop), writes results/
```

`make quick` and `make all` write to the same folders: always run `make clean` before the
final `make all`. `results/summary.md` states which mode produced it.

Optional step G (images; GPU or Apple Silicon recommended, about 1-2 h): quality control and
contact sheet; pre-processing `pad` (whole embryo, no crop) or `clahe`; three feature extractors
(ResNet-50 ImageNet, DINOv2 ViT-S/14, ResNet-50 fine-tuned on Gardner grades of the
**non-transferred** embryos only); 10/20/50 principal components; repeated leak-free
cross-fitting; AUROC with CI, calibration and Brier, overall and with / without a choice,
paired difference vs the Gardner model; shortcut test. The primary configuration is fixed in
`config.yaml` and every configuration of the grid is reported (`images_metrics.csv`).

```bash
uv sync --group dev --extra images
make data-images             # or copy the Figshare "Images" folder into data/raw/
make images
make explain                 # step H: explainability (after `make images`)
```

Optional step H (explainability, descriptive): grade probes evaluated on held-out embryos
(confusion matrices, weighted kappa); exact signed class activation maps for each Gardner
criterion and for the image part of the hCG and live-birth models; share of the maps that
falls on the embryo; model-randomisation sanity check; what the image components encode
(correlations with grades, photo quality, day, age, outcomes; t-SNE); tabular coefficients by
outcome; Figure 3 (outcome cascade).

## Pipeline

| Step | Script | Module | Output |
|---|---|---|---|
| 0 | `00_download.py` | `download` | `data/raw/`, `MANIFEST.csv` |
| A | `01_audit.py` | `audit` | `results/tables/audit.csv`, `repaired_values.csv` |
| A | `02_clean.py` | `cleaning` | `data/processed/transfers.csv`, `embryos.csv`, `flow.csv` |
| B | `03_model.py` | `model` | cross-fitted predictions, `model_performance.csv`, calibration plot |
| C | `04_strategies.py` | `strategies` | `strategies.csv` (E1-E4), subgroups, crude check |
| F | `05_choice_value.py` | `choice_value` | `choice_value.csv` (E7), `hidden_selection_test.csv` |
| D | `06_simulation.py` | `simulation` | `simulation_grid.csv` (E5), trial sizes |
| E | `07_observers.py` | `observers` | `observers.csv` (E6), agreement |
| S | `08_sensitivity.py` | `sensitivity` | `sensitivity.csv` (S1-S10, S8b), `cascade.csv` |
| - | `09_figures.py` | `figures` | Table 1, Figures 1-2, Markdown tables, `summary.md` |
| G | `10_images.py` | `images` | `images_metrics.csv`, QC, shortcut test (optional) |
| H | `11_explain.py` | `explain` | `explain_*.csv`, `supp_explain_*.png`, Figure 3 (optional) |

Every random step draws from `seed` in `config.yaml` plus a fixed offset, so re-running gives
identical numbers. Software versions of each run are written to
`results/logs/environment.json`.

## Methods in one paragraph

A ridge-penalised logistic model predicts live birth for each embryo from its Gardner grades
and the patient's age (restricted cubic spline), AMH, mature oocytes, endometrial thickness
and day of transfer, fitted on 709 single-embryo fresh transfers. Predictions for sibling
embryos are cross-fitted by patient and **leak-free**: missing covariates are imputed inside
each fold (models fitted on training patients only, without the outcome for test patients)
and the ridge penalty is chosen by nested cross-validation; the whole procedure is repeated
over several stochastic imputations and averaged. For the 396 day-5
patients with at least two blastocysts, g-computation gives the expected live birth under each
selection strategy; 95% CIs come from a patient-level bootstrap that refits the fold-wise
imputation and the model (penalty fixed at the median of the nested search, for speed). A simulation keeps the real cohort structure, is anchored on the observed Gardner gain,
and asks what a predictor with a higher AUROC would add, with the trial size needed to detect
it. Embryologists' confusion matrices from the 300-image test set are used to re-grade
siblings and measure how often the choice changes.

## Data-quality issues found (and handled)

- AMH and endometrium values converted to dates by a German-locale spreadsheet
  (`"Jän.26"` = 1.26, `"08.Mai"` = 8.5): reversed by explicit rules, flagged.
- Image names not zero-padded consistently; one image duplicated with conflicting grades.
- Grades differ between the clinical file and the expert file for 3-4% of transferred embryos.
- On day-4 transfers, siblings look more expanded than the transferred embryo (likely
  photographed later at vitrification): day 4 excluded from the strategy comparison.
- Variable `Fond` undocumented: not used.

## Repository layout

```
config.yaml            seeds, replicate counts, grids (single source of truth)
Makefile               pipeline (make all / quick / test / lint / images)
pyproject.toml, uv.lock
src/headroom/          analysis package (one module per step)
scripts/               thin command-line entry points
tests/                 unit tests of repair rules and estimators
docs/                  analysis plan, deviations
data/raw/              downloaded data (git-ignored)
data/processed/        derived data (git-ignored)
results/               tables, figures, logs, summary.md
```

## Citation and licence

Code: MIT licence (see `LICENSE`). Data: see the licence on the Figshare record and cite
Kromp et al. 2023. Please cite this repository with `CITATION.cff`.
