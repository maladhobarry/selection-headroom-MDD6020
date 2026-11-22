# Deviations from the pre-registered analysis plan

Every change made after the plan was frozen is listed here, with its reason and date.

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-11-22 | Step D (simulation) | Predictors defined by an AUROC *increment* over morphology, and the morphology predictor anchored on the observed Gardner gain (E1), instead of absolute AUROC targets | In a pipeline check, absolute AUROC targets mapped to selection gains about twice the observed one at the model's own AUROC: the AUROC of a pooled population depends on patient-level spread, which the simulation cannot reproduce exactly. Anchoring on E1 makes the comparison internally consistent. |
| 2026-11-22 | Step F | Step F used as a model-free consistency check of the value of choice, not to narrow the embryo-level SD | With the morphology predictor fixed, the implied slope barely changes with the embryo-level SD; the SD is bounded only by its ability to reproduce E1. |
| 2026-11-22 | Step E | 6 retained embryologists (not 7) | Kromp et al. retain Annotator 0, 1, 2, 5, 7, 8 (+ the Gardner expert) after consensus filtering. |
| 2026-11-22 | Step B | 709 transfers (not 710) | One transferred image (846_01) has no expert grade. |
| 2026-11-22 | Step B | Imputation moved inside each cross-fitting fold (fitted on training patients; outcome used for training patients only), and ridge penalty chosen by nested CV in each training fold | Review of possible data leakage: imputing once on all patients with the outcome, and tuning the penalty on all data, let test-fold information reach test predictions (small effect, removed). Bootstrap replicates reuse the median nested penalty for speed. |
| 2026-11-22 | Step B | Fold-wise imputation by proper stochastic regression (one posterior draw of coefficients and residual SD) instead of chained equations | 10x faster, needed to run the imputation inside every fold of every bootstrap replicate; same variables and order. Chained equations kept for step F (inference). |
| 2026-11-22 | Step F | Hidden-selection test: leak-free cross-fitting re-run in each bootstrap replicate (was: stored predictions resampled); failed step-F replicates now counted | CI must include model uncertainty; transparency of the bootstrap. |
| 2026-11-22 | Step C | Negative control added: estimands recomputed with outcomes permuted across transfers (noise floor, incl. winner's curse) | A synthetic check showed that choosing the maximum of noisy predictions creates an apparent gain; E3 must be read against this floor. |
| 2026-11-22 | Steps B, C | Clinical pregnancy with heart activity (HA) added as key secondary outcome (discrimination by group, model-free check), primary outcome unchanged (live birth) | Comparability with Illingworth et al. (primary endpoint HA); more events. |
| 2026-11-22 | Step B | Discrimination reported separately for patients with and without a choice | The overall AUROC hides the absence of discrimination among chosen embryos. |
| 2026-11-22 | Step G | Strategies valued by the image model's own probabilities not reported; within-choice-set AUROC is the decisive quantity | Self-valuation is inflated by winner's curse (a pure-noise image model showed +5.8 points). |
| 2026-11-22 | Step G | Image analysis extended before looking at image results of the extended grid: QC, `pad`/`clahe` pre-processing (no centre crop), DINOv2 and grade-fine-tuned ResNet-50 extractors, 10/20/50 components, 5 repeated cross-fittings, calibration and Brier by group. Primary configuration fixed in advance (resnet50_gardner, pad, 20); all configurations reported | First pass (ImageNet ResNet-50, centre crop, single split) used generic features, possibly cropped embryo edges, and reported AUROC only. |

## 2026-11-22 — Duplicated image files (data integrity)

- **Finding.** The image QC (step 10) found 31 byte-identical files in 15 groups, 12 of
  them shared across patients: 422/503 (same transferred image, same age and AMH),
  426/427 (same transferred image, different women), 835/858 (all images identical; 858
  has no clinical row), 629_06/701_03; within-patient duplicates in 105, 175 and 846.
- **Change.** Patients 105, 175, 422, 426, 427, 503 and 835 are flagged (`dup_image`) and
  kept in the pre-specified primary analysis. New sensitivity analysis **S10** excludes them
  (strategies) and a `choice_S10` group is reported for the image models.
- **Change.** The fine-tuning set of `resnet50_gardner` now drops non-transferred images that
  are byte-identical to a transferred image (e.g. 858_01 = 835_01) and repeated copies.
  This removes grade-label (not outcome) overlap between fine-tuning and evaluation.
  Cached weights and features carry a fingerprint of the training set.
- **Figures.** Figure 1 states the permutation noise floor of E3; figure 2B legend moved.

## 2026-11-22 — Pregnancy outcomes added as secondary outcomes

- **Change.** Positive hCG (`SS`) is added next to heart activity (`HA`) as a secondary
  outcome: tabular model performance (step B, nested penalty per outcome), crude check
  (step C), sensitivity **S8b** (estimands with SS), and every image configuration
  (`images.outcomes: [LB, HA, SS]`). Live birth remains the only primary outcome.
- **Fix.** The fold-wise imputation now uses the outcome being modelled (it used LB for every
  outcome). Training rows only; no leakage either way. Affects HA results slightly.
- **Rationale.** Tests where along the cascade (implantation -> heart activity -> birth)
  the morphological signal is lost.

## 2026-11-22 — Explainability step added (step H, descriptive)

- New optional step `make explain` (`src/headroom/explain.py`), run after `make images`.
  It re-estimates nothing from steps A–G and adds no test of the primary hypothesis.
- Contents: grade probes on held-out (transferred) embryos with confusion matrices and
  weighted kappa; exact signed class activation maps (global-average-pooling read-out) for
  the three Gardner criteria and for the image part of the hCG and live-birth models;
  localisation of the maps on the embryo (Hough circle); model-randomisation sanity check
  (Adebayo et al., 2018); Spearman correlations between image components and grades,
  photo quality, day, age and outcomes; t-SNE map; standardised ridge coefficients of the
  tabular model for each outcome; figure 3 (value of selection and discrimination along the
  outcome cascade).
- Outcome maps are labelled exploratory: the image model does not beat Gardner (step G).
- Article figures added: `figure0_flow.png` (participant flow), `figure4_images_spec_curve.png`
  (specification curve of the 18 image configurations vs Gardner, by outcome) and
  `supp_sensitivity_forest.png` (E1 across all sensitivity analyses).
- Figures 3 and the sensitivity forest show, for the primary analysis, the step-C estimate
  of E1 (pooled over 20 imputations, 2,000 bootstrap replicates), as in Figure 1 and the
  text; the S0 row of `sensitivity.csv` (one imputation, sensitivity bootstrap) differs
  slightly (1.8 vs 1.7) and is kept in the table for transparency.
