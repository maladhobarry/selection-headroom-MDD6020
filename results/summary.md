# Key results

Run mode: full; seed 20261122.

- Model: cross-validated AUROC 0.552, calibration slope 0.89; within choice set 0.489 [0.429, 0.549].
- E1 best Gardner vs random (primary): 1.7 [-0.4, 3.8] points.
- E2 embryologist vs random: 1.7 [0.2, 3.1] points.
- E3 best by model vs embryologist (upper bound, model-dependent): 2.1 [1.3, 5.2] points; noise floor (permuted outcomes) 2.9 (95th pct 5.1).
- Noise floor of E1: 0.0 (95th pct 1.8).
- E4 best vs worst: 8.0 [6.7, 13.0] points.
- E7 live birth 4+ vs 1 blastocyst (adjusted): 4.4 [-7.4, 13.6] points.
- Model-free check: best-Gardner embryo transferred vs not (age-adjusted): -3.8 [-15.8, 6.3] points (n = 307 vs 89).
- E6 pick changed with another embryologist: 39.3% of patients [37.9, 40.7] (live-birth change not reported: circular, the model is built on expert grades).
- E5, predictor AUROC +0.02 over morphology: gain 0.6 to 1.4 points over Gardner (across sigma_e); trial size 17395 to 91180 per arm.
- E5, predictor AUROC +0.05 over morphology: gain 2.7 to 3.3 points over Gardner (across sigma_e); trial size 3148 to 4773 per arm.
- E5, predictor AUROC +0.1 over morphology: gain 5.8 to 6.7 points over Gardner (across sigma_e); trial size 810 to 1065 per arm.
- sigma_e able to reproduce E1: [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
