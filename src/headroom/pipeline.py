"""Orchestration of the analysis steps. Each ``step_*`` reads the outputs of the previous
steps from disk and writes its own, so any step can be re-run alone (see Makefile)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import audit, choice_value, cleaning, download, observers, sensitivity, simulation
from . import figures as F
from . import model as M
from . import strategies as S
from .config import get_logger, get_paths, load_config, write_environment


def _ctx(quick: bool | None = None):
    cfg = load_config(quick=quick)
    return cfg, get_paths(cfg)


# --------------------------------------------------------------------------- step 0 / A
def step_download(images: bool = False, quick=None):
    cfg, paths = _ctx(quick)
    download.download_all(cfg, paths, images=images)


def step_audit(quick=None):
    cfg, paths = _ctx(quick)
    write_environment(paths, cfg)
    if not download.verify(paths):
        raise SystemExit("Raw files missing: run `make data` first.")
    audit.run(paths)


def step_clean(quick=None):
    cfg, paths = _ctx(quick)
    data = cleaning.build(paths, cfg)
    cleaning.save(data, paths)


# --------------------------------------------------------------------------- step B
def step_model(quick=None):
    cfg, paths = _ctx(quick)
    log = get_logger("03_model", paths)
    t, e = cleaning.load_processed(paths)
    m_imp = cfg["model"]["n_imputations"]
    kn = cfg["model"]["age_spline_knots"]

    def make(c):
        return M.EmbryoModel(C=c, n_knots=kn)

    # Leak-free cross-fitting: imputation and penalty search inside each fold, repeated over
    # m stochastic imputations and averaged.
    p_tr = np.zeros(len(t))
    p_emb = np.zeros(len(e))
    fold_Cs = []
    for i in range(m_imp):
        a, b, cs = M.crossfit(t, e, make, cfg["model"]["cv_folds"], cfg["seed"] + i,
                              C=None, n_knots=kn)
        p_tr += a / m_imp
        p_emb += b / m_imp
        fold_Cs += cs
    C = float(np.median(fold_Cs))
    log.info("Nested ridge penalty: median C = %.4g (range %.4g-%.4g over %d folds)", C,
             min(fold_Cs), max(fold_Cs), len(fold_Cs))
    t.assign(p_cv=p_tr).to_csv(paths.processed / "predictions_transfers.csv", index=False)
    e.assign(p=p_emb).to_csv(paths.processed / "predictions_embryos.csv", index=False)

    y = t["LB"].to_numpy()
    perf = M.performance(y, p_tr)
    t0 = M.impute_patients(t, cfg["seed"])
    opt = M.optimism_corrected(t0, lambda: make(C), cfg["bootstrap"]["n_secondary"],
                               cfg["seed"] + 7)
    rows = [{"metric": "Cross-validated AUROC", "value": perf["auroc"]},
            {"metric": "Cross-validated calibration intercept", "value": perf["cal_intercept"]},
            {"metric": "Cross-validated calibration slope", "value": perf["cal_slope"]},
            {"metric": "Brier score", "value": perf["brier"]},
            {"metric": "Apparent AUROC", "value": opt["apparent_auroc"]},
            {"metric": "Optimism-corrected AUROC", "value": opt["corrected_auroc"]},
            {"metric": "Optimism-corrected calibration slope", "value": opt["corrected_slope"]},
            {"metric": "Transfers / live births", "value": f"{len(t)} / {int(y.sum())}"},
            {"metric": "Ridge C (median of nested folds)", "value": C}]
    # Discrimination within the groups that matter (day 5): with vs without a choice.
    rng_g = np.random.default_rng(cfg["seed"] + 8)
    for name, m in (("day 5, choice (>=2 blastocysts)", (t["day"] == 5) & (t["n_blast"] >= 2)),
                    ("day 5, no choice (1 blastocyst)", (t["day"] == 5) & (t["n_blast"] == 1))):
        yy, pp = y[m.to_numpy()], p_tr[m.to_numpy()]
        bs = []
        for _ in range(cfg["bootstrap"]["n_secondary"]):
            i = rng_g.integers(0, len(yy), len(yy))
            if len(np.unique(yy[i])) == 2:
                bs.append(M.roc_auc_score(yy[i], pp[i]))
        rows.append({"metric": f"Cross-validated AUROC, {name}",
                     "value": f"{M.roc_auc_score(yy, pp):.3f} "
                              f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"})
    # Secondary outcomes (pregnancy): positive hCG (SS) and clinical pregnancy with fetal
    # heart activity (HA, Illingworth's primary endpoint). Same leak-free cross-fitting with
    # nested penalty, the imputation model using the outcome being predicted.
    for oc, label in (("SS", "positive hCG"), ("HA", "heart activity")):
        p_oc, _, _ = M.crossfit(t, e.iloc[:0], make, cfg["model"]["cv_folds"], cfg["seed"],
                                outcome=oc, C=None, n_knots=kn)
        t.assign(**{f"p_cv_{oc}": p_oc}).to_csv(
            paths.processed / f"predictions_transfers_{oc}.csv", index=False)
        yo = t[oc].to_numpy()
        po = M.performance(yo, p_oc)
        rows += [{"metric": f"{oc} ({label}): cross-validated AUROC", "value": po["auroc"]},
                 {"metric": f"{oc} ({label}): calibration slope", "value": po["cal_slope"]},
                 {"metric": f"{oc} ({label}): transfers / events",
                  "value": f"{len(t)} / {int(yo.sum())}"}]
        for name, m in (("day 5, choice (>=2 blastocysts)",
                         (t["day"] == 5) & (t["n_blast"] >= 2)),
                        ("day 5, no choice (1 blastocyst)",
                         (t["day"] == 5) & (t["n_blast"] == 1))):
            mm = m.to_numpy()
            yy, pp = yo[mm], p_oc[mm]
            bs = []
            for _ in range(cfg["bootstrap"]["n_secondary"]):
                i = rng_g.integers(0, len(yy), len(yy))
                if len(np.unique(yy[i])) == 2:
                    bs.append(M.roc_auc_score(yy[i], pp[i]))
            rows.append({"metric": f"{oc} ({label}): cross-validated AUROC, {name}",
                         "value": f"{M.roc_auc_score(yy, pp):.3f} "
                                  f"[{np.percentile(bs, 2.5):.3f}, "
                                  f"{np.percentile(bs, 97.5):.3f}]"})
    pd.DataFrame(rows).to_csv(paths.tables / "model_performance.csv", index=False)
    full = make(C).fit(t0, y)
    coef = pd.DataFrame({"term": M.design(t0, "full", full.knots_).columns,
                         "coef_standardised": full.est_.coef_[0]})
    coef.to_csv(paths.tables / "model_coefficients.csv", index=False)
    (paths.tables / "model_C.json").write_text(json.dumps({"C": C, "fold_Cs": fold_Cs}),
                                               encoding="utf-8")
    grp = np.where(t["n_blast"] >= 2, "choice", "no_choice")
    F.calibration_plot(y, p_tr, grp, paths.figures / "supp_calibration.png")
    log.info("Cross-validated AUROC %.3f, calibration slope %.2f", perf["auroc"],
             perf["cal_slope"])


def _C(paths) -> float:
    return json.loads((paths.tables / "model_C.json").read_text())["C"]


# --------------------------------------------------------------------------- step C
def step_strategies(quick=None):
    cfg, paths = _ctx(quick)
    log = get_logger("04_strategies", paths)
    t, e = cleaning.load_processed(paths)
    C, opt = _C(paths), S.Options()
    m_imp = cfg["model"]["n_imputations"]
    pts, vals_list = [], []
    for i in range(m_imp):
        p, v = S.run_once(t, e, cfg, None, opt, cfg["seed"] + i)   # nested penalty
        pts.append(p)
        vals_list.append(v)
    point = pd.DataFrame(pts).mean().to_dict()
    vals = sum(vals_list) / m_imp
    log.info("Bootstrap of strategies (%d replicates)...", cfg["bootstrap"]["n_main"])
    boot = S.bootstrap(t, e, cfg, C, opt, cfg["bootstrap"]["n_main"], cfg["seed"] + 100)
    summ = S.summarise(point, boot)
    n_perm = max(5, cfg["bootstrap"]["n_secondary"] // 5)
    null = S.permutation_null(t, e, cfg, C, opt, n_perm, cfg["seed"] + 150)
    summ["null_mean"] = summ["quantity"].map(null.mean())
    summ["null_95th"] = summ["quantity"].map(null.quantile(0.95))
    summ["above_noise"] = summ["estimate"] - summ["null_mean"]
    summ.to_csv(paths.tables / "strategies.csv", index=False)
    boot.to_csv(paths.processed / "strategies_boot.csv", index=False)
    vals.to_csv(paths.processed / "strategy_values_per_patient.csv")

    het = S.heterogeneity(vals, t)
    hb = S.heterogeneity_boot(t, e, cfg, C, opt, cfg["bootstrap"]["n_secondary"],
                              cfg["seed"] + 200)
    ci = hb.groupby(["subgroup", "level"])[["E1", "E2", "E3"]].quantile([0.025, 0.975])
    for k in ("E1", "E2", "E3"):
        lo = ci[k].xs(0.025, level=2)
        hi = ci[k].xs(0.975, level=2)
        het[f"{k}_ci"] = [f"[{lo.loc[(s, lv)]:.1f}, {hi.loc[(s, lv)]:.1f}]"
                          for s, lv in zip(het["subgroup"], het["level"], strict=True)]
    het.to_csv(paths.tables / "strategies_subgroups.csv", index=False)
    pd.concat([S.crude_check(t, cfg["bootstrap"]["n_secondary"], cfg["seed"] + 250, "LB"),
               S.crude_check(t, cfg["bootstrap"]["n_secondary"], cfg["seed"] + 251, "HA"),
               S.crude_check(t, cfg["bootstrap"]["n_secondary"], cfg["seed"] + 252, "SS")]
              ).to_csv(paths.tables / "crude_check.csv", index=False)
    s = summ.set_index("quantity")
    for k in ("E1", "E2", "E3", "E4"):
        log.info("%s = %.2f points [%.2f, %.2f]", k, s.loc[k, "estimate"], s.loc[k, "ci_low"],
                 s.loc[k, "ci_high"])


# --------------------------------------------------------------------------- step F
def step_choice_value(quick=None):
    cfg, paths = _ctx(quick)
    log = get_logger("05_choice_value", paths)
    t, _ = cleaning.load_processed(paths)
    summ, boot, crude = choice_value.run(t, cfg, cfg["bootstrap"]["n_secondary"],
                                         cfg["seed"] + 300)
    summ.to_csv(paths.tables / "choice_value.csv", index=False)
    crude.to_csv(paths.tables / "choice_value_crude.csv", index=False)
    pt = pd.read_csv(paths.processed / "predictions_transfers.csv")
    C, kn = _C(paths), cfg["model"]["age_spline_knots"]

    def refit(tb, s):
        # Leak-free cross-fitting re-run on the bootstrap sample (penalty fixed).
        p, _, _ = M.crossfit(tb, tb.iloc[:0], lambda c: M.EmbryoModel(C=c, n_knots=kn),
                             cfg["model"]["cv_folds"], s, C=C, n_knots=kn)
        return p

    hs = choice_value.hidden_selection_test(t, pt["p_cv"].to_numpy(),
                                            cfg["bootstrap"]["n_secondary"], cfg["seed"] + 301,
                                            refit=refit)
    hs.to_csv(paths.tables / "hidden_selection_test.csv", index=False)
    s = summ.set_index("quantity")
    log.info("Adjusted RD 4+ vs 1 = %.2f [%.2f, %.2f]", *s.loc["RD_4+_vs_1",
                                                               ["estimate", "ci_low", "ci_high"]])


# --------------------------------------------------------------------------- step D
def step_simulation(quick=None):
    cfg, paths = _ctx(quick)
    log = get_logger("06_simulation", paths)
    pe = pd.read_csv(paths.processed / "predictions_embryos.csv")
    pe = pe[pe["day"] == cfg["population"]["main_day"]]
    lp = np.log(pe["p"] / (1 - pe["p"]))
    patients = (pe.assign(lp=lp).groupby("patient")
                .agg(b=("lp", "mean"), n_blast=("n_blast", "first"), target=("p", "mean"))
                .reset_index())
    strat = pd.read_csv(paths.tables / "strategies.csv").set_index("quantity")
    e1 = strat.loc["E1", "estimate"] / 100
    obs_level = strat.loc["observed", "estimate"] / 100
    f_summ = pd.read_csv(paths.tables / "choice_value.csv")
    grid, e7 = simulation.run(patients, cfg, e1, obs_level, f_summ, cfg["seed"] + 400)
    grid.to_csv(paths.tables / "simulation_grid.csv", index=False)
    e7.to_csv(paths.tables / "simulation_value_of_choice.csv", index=False)
    log.info("sigma_e able to reproduce E1: %s",
             e7.loc[e7["reproduces_E1"], "sigma_e"].tolist())


# --------------------------------------------------------------------------- step E
def step_observers(quick=None):
    cfg, paths = _ctx(quick)
    log = get_logger("07_observers", paths)
    pe = pd.read_csv(paths.processed / "predictions_embryos.csv")
    pe = pe[pe["choice_set"] == 1]
    ann, gold = cleaning.load_annotators(paths)
    names = cfg["observers"]["retained"]
    observers.agreement(ann, gold, names + ["Gardner_Expert"]).to_csv(
        paths.tables / "observers_agreement.csv", index=False)
    summ, sims = observers.run(pe, ann, gold, names, cfg["observers"]["n_reps"],
                               cfg["seed"] + 500)
    summ.to_csv(paths.tables / "observers.csv", index=False)
    a = summ.set_index("annotator").loc["ALL (mean of embryologists)"]
    log.info("Pick changed: %.1f%%; delta LB %.2f points", a["pick_changed_pct_mean"],
             a["delta_LB_points_mean"])


# --------------------------------------------------------------------------- sensitivity
def step_sensitivity(quick=None):
    cfg, paths = _ctx(quick)
    log = get_logger("08_sensitivity", paths)
    t, e = cleaning.load_processed(paths)
    res = sensitivity.run(t, e, cfg, _C(paths), cfg["bootstrap"]["n_secondary"],
                          cfg["seed"] + 600, log)
    res.to_csv(paths.tables / "sensitivity.csv", index=False)
    sensitivity.cascade(t).to_csv(paths.tables / "cascade.csv", index=False)


# --------------------------------------------------------------------------- figures
def step_figures(quick=None):
    cfg, paths = _ctx(quick)
    t, _ = cleaning.load_processed(paths)
    tab1 = F.table1(t)
    tab1.to_csv(paths.tables / "table1.csv", index=False)
    F.to_markdown(tab1, paths.tables / "table1.md")
    strat = pd.read_csv(paths.tables / "strategies.csv")
    F.figure1(strat, paths.figures / "figure1_strategies.png")
    F.figure2(pd.read_csv(paths.tables / "choice_value.csv"),
              pd.read_csv(paths.tables / "simulation_value_of_choice.csv"),
              pd.read_csv(paths.tables / "simulation_grid.csv"),
              paths.figures / "figure2_value_and_ceiling.png")
    flow_p, sens_p = paths.tables / "flow.csv", paths.tables / "sensitivity.csv"
    if flow_p.exists():
        F.flow_diagram(pd.read_csv(flow_p), paths.figures / "figure0_flow.png")
    if sens_p.exists():
        F.sensitivity_forest(F.with_primary(pd.read_csv(sens_p), paths.tables),
                             paths.figures / "supp_sensitivity_forest.png")
    img_p = paths.tables / "images_metrics.csv"
    if img_p.exists() and "outcome" in pd.read_csv(img_p, nrows=1).columns:
        F.specification_curve(pd.read_csv(img_p), paths.figures / "figure4_images_spec_curve.png")
    for name in ("strategies", "strategies_subgroups", "crude_check", "choice_value",
                 "hidden_selection_test",
                 "simulation_value_of_choice", "simulation_grid", "observers", "sensitivity",
                 "model_performance", "cascade", "audit", "flow"):
        p = paths.tables / f"{name}.csv"
        if p.exists():
            F.to_markdown(pd.read_csv(p), paths.tables / f"{name}.md")
    _summary(cfg, paths)


def _summary(cfg, paths):
    """One-page Markdown summary of the key numbers (results/summary.md)."""
    s = pd.read_csv(paths.tables / "strategies.csv").set_index("quantity")
    f = pd.read_csv(paths.tables / "choice_value.csv").set_index("quantity")
    o = pd.read_csv(paths.tables / "observers.csv").set_index("annotator")
    perf = pd.read_csv(paths.tables / "model_performance.csv").set_index("metric")["value"]
    grid = pd.read_csv(paths.tables / "simulation_grid.csv")

    def ci(df, k):
        r = df.loc[k]
        return f"{r['estimate']:.1f} [{r['ci_low']:.1f}, {r['ci_high']:.1f}]"

    ok = grid[grid["reachable"].astype(bool)]
    cc = pd.read_csv(paths.tables / "crude_check.csv").set_index("outcome").loc["LB"]
    lines = [
        "# Key results", "",
        f"Run mode: {'QUICK (not for reporting)' if cfg['_quick'] else 'full'}; "
        f"seed {cfg['seed']}.", "",
        f"- Model: cross-validated AUROC {float(perf['Cross-validated AUROC']):.3f}, "
        f"calibration slope {float(perf['Cross-validated calibration slope']):.2f}; within "
        f"choice set {perf.get('Cross-validated AUROC, day 5, choice (>=2 blastocysts)', 'n/a')}.",
        f"- E1 best Gardner vs random (primary): {ci(s, 'E1')} points.",
        f"- E2 embryologist vs random: {ci(s, 'E2')} points.",
        f"- E3 best by model vs embryologist (upper bound, model-dependent): {ci(s, 'E3')}"
        f" points; noise floor (permuted outcomes) {s.loc['E3', 'null_mean']:.1f} "
        f"(95th pct {s.loc['E3', 'null_95th']:.1f}).",
        f"- Noise floor of E1: {s.loc['E1', 'null_mean']:.1f} "
        f"(95th pct {s.loc['E1', 'null_95th']:.1f}).",
        f"- E4 best vs worst: {ci(s, 'E4')} points.",
        f"- E7 live birth 4+ vs 1 blastocyst (adjusted): {ci(f, 'RD_4+_vs_1')} points.",
        f"- Model-free check: best-Gardner embryo transferred vs not (age-adjusted): "
        f"{cc['adjusted_RD_points']:.1f} [{cc['ci_low']:.1f}, {cc['ci_high']:.1f}] points "
        f"(n = {int(cc['n_best_transferred'])} vs {int(cc['n_other_transferred'])}).",
        f"- E6 pick changed with another embryologist: "
        f"{o.loc['ALL (mean of embryologists)', 'pick_changed_pct_mean']:.1f}% of patients "
        f"[{o.loc['ALL (mean of embryologists)', 'pick_changed_pct_lo']:.1f}, "
        f"{o.loc['ALL (mean of embryologists)', 'pick_changed_pct_hi']:.1f}] "
        "(live-birth change not reported: circular, the model is built on expert grades).",
    ]
    for d in ("0.02", "0.05", "0.1"):
        top = ok[ok["delta_auroc"].astype(str) == d]
        if len(top):
            lines.append(f"- E5, predictor AUROC +{d} over morphology: gain "
                         f"{top['gain_vs_gardner_choice_set'].min():.1f} to "
                         f"{top['gain_vs_gardner_choice_set'].max():.1f} points over Gardner "
                         f"(across sigma_e); trial size "
                         f"{int(top['n_per_arm_vs_morphology'].min())} to "
                         f"{int(top['n_per_arm_vs_morphology'].max())} per arm.")
    e7 = pd.read_csv(paths.tables / "simulation_value_of_choice.csv")
    lines.append(f"- sigma_e able to reproduce E1: "
                 f"{e7.loc[e7['reproduces_E1'], 'sigma_e'].tolist()}")
    (paths.root / "results" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- step G (optional)
def step_images(quick=None, pretrained: bool = True):
    from . import images as G

    cfg, paths = _ctx(quick)
    log = get_logger("10_images", paths)
    ic = cfg["images"]
    found = G.find_images(paths.raw)
    if not found:
        raise SystemExit("No images in data/raw/: run `make data-images` first.")
    log.info("Images found: %d", len(found))

    # 1) Quality control
    qc = G.image_qc(found, paths.tables / "images_qc.csv",
                    paths.figures / "supp_image_contact_sheet.png", cfg["seed"])
    G.qc_summary(qc).to_csv(paths.tables / "images_qc_summary.csv", index=False)
    log.info("QC: %d images flagged (see images_qc.csv)", int(qc["flag_any"].sum()))

    t, e = cleaning.load_processed(paths)
    t["dup_image"] = t["patient"].isin(cleaning.DUPLICATE_IMAGE_PATIENTS).astype(int)
    grades, _ = cleaning.load_expert_grades(paths)
    transferred = set(zip(t["patient"], t["embryo"], strict=True))
    not_transferred, ft_tag = G.finetune_pool(found, transferred, qc)
    log.info("Fine-tuning pool: %d images (duplicates of transferred images removed)",
             len(not_transferred))
    kn = cfg["model"]["age_spline_knots"]
    k_folds = cfg["model"]["cv_folds"]
    feat_dir = paths.processed / "image_features"
    feat_dir.mkdir(exist_ok=True)

    # Gardner reference model, same repeated folds (grades + clinical, no image).
    def make_gar(c):
        return G.ImageModel([], with_grades=True, C=c, n_knots=kn)

    tt = t.reset_index(drop=True)
    outcomes = ic.get("outcomes", ["LB"])
    ys = {oc: tt[oc].to_numpy() for oc in outcomes}
    p_gar = {oc: G.crossfit_repeated(tt, make_gar, ic["cv_repeats"], k_folds, cfg["seed"], oc)
             for oc in outcomes}
    groups = {"all": np.ones(len(tt), bool),
              "choice": ((tt["day"] == 5) & (tt["n_blast"] >= 2)).to_numpy(),
              "no_choice": ((tt["day"] == 5) & (tt["n_blast"] == 1)).to_numpy()}
    # Sensitivity S10: choice set without patients whose image file is duplicated.
    groups["choice_S10"] = groups["choice"] & (tt["dup_image"] == 0).to_numpy()
    n_boot = cfg["bootstrap"]["n_secondary"]
    rows, ft_rows = [], []
    r0 = dict(config="gardner_reference", backbone="-", preprocess="-", n_pcs=0)
    for oc in outcomes:
        for gname, m in groups.items():
            rows.append({**r0, "outcome": oc, "group": gname, "n": int(m.sum()),
                         "events": int(ys[oc][m].sum()),
                         **G.metrics(ys[oc][m], p_gar[oc][m], n_boot, cfg["seed"] + 1)})

    primary = ic["primary"]
    primary_pcs = None
    for bb in ic["backbones"]:
        for pre in ic["preprocess"]:
            # The fine-tuned backbone's files carry a fingerprint of its training set, so a
            # stale cache from an earlier training set is never reused.
            tag = f"_{ft_tag}" if bb == "resnet50_gardner" else ""
            cache = feat_dir / f"{bb}_{pre}{tag}.npz"
            weights = None
            try:
                if bb == "resnet50_gardner":
                    weights = feat_dir / f"{bb}_{pre}{tag}.pt"
                    if not weights.exists():
                        acc = G.finetune_on_grades(found, grades, not_transferred, pre, cfg,
                                                   weights, pretrained, log)
                        ft_rows.append({"preprocess": pre, **acc})
                feats = G.extract_features(found, bb, pre, cache, pretrained, weights)
            except Exception as exc:          # e.g. DINOv2 download blocked: log and go on
                log.warning("Skipping %s/%s: %s", bb, pre, exc)
                continue
            for n in ic["n_pcs"]:
                pcs = G.principal_components(feats, n, cfg["seed"])
                cols = list(pcs.columns)
                tp = tt.merge(pcs, left_on=["patient", "embryo"], right_index=True, how="left")
                if tp[cols].isna().any().any():
                    log.warning("Missing image features for %d transfers",
                                int(tp[cols[0]].isna().sum()))
                    continue

                def make(c, cols=cols):
                    return G.ImageModel(cols, C=c, n_knots=kn)

                name = f"{bb}|{pre}|{n}"
                is_primary = (bb == primary["backbone"] and pre == primary["preprocess"]
                              and n == primary["n_pcs"])
                if is_primary:
                    primary_pcs = pcs
                for oc in outcomes:
                    p_img = G.crossfit_repeated(tp, make, ic["cv_repeats"], k_folds,
                                                cfg["seed"], oc)
                    for gname, m in groups.items():
                        rows.append({"config": name + (" (PRIMARY)" if is_primary else ""),
                                     "backbone": bb, "preprocess": pre, "n_pcs": n,
                                     "outcome": oc, "group": gname, "n": int(m.sum()),
                                     "events": int(ys[oc][m].sum()),
                                     **G.metrics(ys[oc][m], p_img[m], n_boot, cfg["seed"] + 1,
                                                 ref=p_gar[oc][m])})
                    ch = next(r for r in rows[-len(groups):] if r["group"] == "choice")
                    log.info("%s [%s]: AUROC choice set %.3f (Gardner %.3f)", name, oc,
                             ch["auroc"], ch["auroc"] - ch["delta_auroc_vs_gardner"])
    res = pd.DataFrame(rows)
    res.to_csv(paths.tables / "images_metrics.csv", index=False)
    F.specification_curve(res, paths.figures / "figure4_images_spec_curve.png")
    F.to_markdown(res, paths.tables / "images_metrics.md", ".3f")
    if ft_rows:
        pd.DataFrame(ft_rows).to_csv(paths.tables / "images_finetune.csv", index=False)
    if primary_pcs is not None:
        cs = e[e["choice_set"] == 1].merge(primary_pcs, left_on=["patient", "embryo"],
                                           right_index=True, how="inner")
        sc = G.shortcut_test(cs, list(primary_pcs.columns), cfg["seed"])
        pd.Series(sc).to_csv(paths.tables / "images_shortcut_test.csv", header=["value"])
        log.info("Shortcut test (primary): excess AUROC from images %.3f",
                 sc["excess_auroc_from_images"])


# --------------------------------------------------------------------------- step H
def step_explain(quick=None, pretrained: bool = True):
    """Explainability (optional): tabular coefficients, cascade figure, and - when the
    images and the step-G backbone are available - probes, CAMs, localisation, sanity check
    and representation maps."""
    from scipy.stats import spearmanr
    from sklearn.manifold import TSNE

    from . import explain as X
    from . import images as G

    cfg, paths = _ctx(quick)
    log = get_logger("11_explain", paths)
    ic, ex, seed = cfg["images"], cfg.get("explain", {}), cfg["seed"]
    kn = cfg["model"]["age_spline_knots"]
    t, e = cleaning.load_processed(paths)
    outcomes = ic.get("outcomes", ["LB"])

    # --- tabular model and cascade (no images needed)
    coefs = X.tabular_coefficients(t, outcomes, kn, seed)
    coefs.to_csv(paths.tables / "explain_tabular_coefficients.csv")
    F.to_markdown(coefs.reset_index(names="term"),
                  paths.tables / "explain_tabular_coefficients.md", ".3f")
    X.figure_coefficients(coefs, paths.figures / "supp_explain_tabular.png")
    sens_p, img_p = paths.tables / "sensitivity.csv", paths.tables / "images_metrics.csv"
    if sens_p.exists():
        img = pd.read_csv(img_p) if img_p.exists() else None
        if img is not None and "outcome" not in img.columns:
            img = None
        X.figure_cascade(F.with_primary(pd.read_csv(sens_p), paths.tables), img,
                         paths.figures / "figure3_cascade.png")
    log.info("Tabular coefficients and cascade figure written")

    found = G.find_images(paths.raw)
    if not found:
        log.info("No images: image explainability skipped")
        return
    pr = ic["primary"]
    bb, pre = pr["backbone"], pr["preprocess"]
    if not bb.startswith("resnet50"):
        raise SystemExit("CAMs are implemented for the ResNet-50 backbones only.")
    qc_p = paths.tables / "images_qc.csv"
    if not qc_p.exists():
        raise SystemExit("Run `make images` first (step G).")
    qc = pd.read_csv(qc_p)
    grades, _ = cleaning.load_expert_grades(paths)
    transferred = set(zip(t["patient"], t["embryo"], strict=True))
    pool, tag = G.finetune_pool(found, transferred, qc)
    feat_dir = paths.processed / "image_features"
    suffix = f"_{tag}" if bb == "resnet50_gardner" else ""
    weights = feat_dir / f"{bb}_{pre}{suffix}.pt" if bb == "resnet50_gardner" else None
    if weights is not None and not weights.exists():
        raise SystemExit(f"{weights.name} not found: run `make images` first (step G).")
    feats = G.extract_features(found, bb, pre, feat_dir / f"{bb}_{pre}{suffix}.npz",
                               pretrained, weights)
    held = sorted(k for k in transferred if k in feats.index)

    # --- 1. grade probes (fitted on the fine-tuning pool, evaluated on transferred embryos)
    pm, conf, W = X.grade_probes(feats, grades, pool, held, seed)
    pm.to_csv(paths.tables / "explain_grade_probes.csv", index=False)
    X.figure_probes(pm, conf, paths.figures / "supp_explain_grade_probes.png")
    log.info("Grade probes, held-out kappa: %s",
             {r.criterion: round(r.weighted_kappa, 2) for r in pm.itertuples()})

    # --- 2. read-outs: grades + image part of the outcome model
    t_imp = M.impute_patients(t, seed)
    w_out, contrib = {}, {}
    for oc in ex.get("outcomes", ["SS", "LB"]):
        w_out[oc], contrib[oc] = X.outcome_readout(feats, t_imp, pr["n_pcs"], oc, kn, seed)
    names = [f"{c} probe" for c in X.CRITERIA] + [f"{X.OUTCOME_LABEL[o]} model" for o in w_out]
    Wall = np.vstack([W[c] for c in X.CRITERIA] + list(w_out.values()))

    # --- 3. CAMs on all transferred (held-out) images + localisation on the embryo
    net = X.load_resnet(weights, pretrained)
    imgs = [G.load_image(found[k], pre) for k in held]
    maps = X.cams(net, imgs, Wall)
    masks = [X.embryo_mask(im) for im in imgs]
    rows = []
    for h, nm in enumerate(names):
        r = np.array([X.enrichment(X.upsample(maps[i, h]), masks[i][0])
                      for i in range(len(imgs))])
        q = np.nanpercentile(r, [25, 50, 75])
        rows.append({"read_out": nm, "images": len(imgs), "enrichment_median": q[1],
                     "enrichment_q1": q[0], "enrichment_q3": q[2],
                     "share_enrichment_above_1": float(np.nanmean(r > 1))})
    loc = pd.DataFrame(rows)
    loc["embryo_circle_found_pct"] = 100 * np.mean([m[1] for m in masks])

    # --- 4. sanity check: randomise the last block, same read-outs
    n_s = min(ex.get("n_sanity", 100), len(imgs))
    rnet = X.randomise_last_block(X.load_resnet(weights, pretrained), seed)
    rmaps = X.cams(rnet, imgs[:n_s], Wall)
    loc["sanity_rank_corr_trained_vs_random"] = [
        float(np.nanmedian([spearmanr(maps[i, h].ravel(), rmaps[i, h].ravel())[0]
                            for i in range(n_s)])) for h in range(len(names))]
    loc.to_csv(paths.tables / "explain_localisation.csv", index=False)
    F.to_markdown(loc, paths.tables / "explain_localisation.md", ".2f")
    log.info("Localisation (median enrichment): %s; sanity rank corr: %s",
             dict(zip(loc.read_out, loc.enrichment_median.round(2), strict=True)),
             dict(zip(loc.read_out, loc.sanity_rank_corr_trained_vs_random.round(2),
                      strict=True)))

    # --- galleries
    rng = np.random.default_rng(seed)
    g = grades.set_index(["patient", "embryo"])
    idx = {k: i for i, k in enumerate(held)}
    n_gal = ex.get("n_gallery", 8)
    choice = t[(t["day"] == 5) & (t["n_blast"] >= 2)]
    ck = [k for k in zip(choice["patient"], choice["embryo"], strict=True) if k in idx]
    by_exp = sorted(ck, key=lambda k: (g.loc[k, "EXP"], rng.random()))
    pick = [by_exp[int(j)] for j in np.linspace(0, len(by_exp) - 1, n_gal)] if ck else []

    def label(k):
        r = g.loc[k]
        icm = "ABCN"[int(r["ICM"])]
        te = "ABCN"[int(r["TE"])]
        return f"{k[0]}_{k[1]:02d}  {int(r['EXP']) + 1}{icm}{te}"

    if pick:
        X.figure_cams([imgs[idx[k]] for k in pick],
                      maps[[idx[k] for k in pick]][:, :3], [f"{c} probe" for c in X.CRITERIA],
                      [label(k) for k in pick], paths.figures / "supp_explain_cam_grades.png",
                      "Where the network sees each Gardner criterion (held-out embryos, "
                      "sorted by expansion)")
    oc0 = next(iter(w_out))
    cs = contrib[oc0].reindex(ck).dropna().sort_values()
    half = n_gal // 2
    sel = list(cs.index[:half]) + list(cs.index[-half:])
    if sel:
        ys = t.set_index(["patient", "embryo"])[oc0]
        X.figure_cams([imgs[idx[k]] for k in sel], maps[[idx[k] for k in sel]][:, 3:],
                      names[3:],
                      [f"{label(k)}  {'+' if ys.loc[k] == 1 else '-'}" for k in sel],
                      paths.figures / "supp_explain_cam_outcomes.png",
                      f"Image evidence for {X.OUTCOME_LABEL[oc0]} (exploratory): lowest "
                      f"{half} and highest {half} image contributions, patients with a choice")

    # --- 5. what the representation encodes
    pcs10 = G.principal_components(feats, 10, seed)
    corr = X.pc_correlations(pcs10, grades, qc, t)
    corr.to_csv(paths.tables / "explain_pc_correlations.csv")
    pcs50 = G.principal_components(feats, 50, seed)
    xy = TSNE(2, init="pca", perplexity=30, random_state=seed).fit_transform(pcs50.to_numpy())
    emb2d = pd.DataFrame(xy, index=pcs50.index, columns=["x", "y"])
    emb2d["EXP"] = g["EXP"].reindex(emb2d.index)
    emb2d["SS"] = t.set_index(["patient", "embryo"])["SS"].reindex(emb2d.index)
    emb2d.to_csv(paths.tables / "explain_tsne.csv")
    X.figure_representation(corr, emb2d, paths.figures / "supp_explain_representation.png")
    log.info("Explainability done: tables explain_*.csv, figures supp_explain_*.png")
