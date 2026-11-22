"""Sensitivity analyses S1-S10 for the strategy estimands, and the exploratory cascade.

Each analysis reruns step C with one switch changed (see ``strategies.Options``) and its own
patient-level bootstrap. The conclusion is judged robust if E1 stays in the same order of
magnitude (below 5 points or not).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from . import strategies as S
from .cleaning import DUPLICATE_IMAGE_PATIENTS, add_gardner_rank


def _variants(cfg: dict) -> list[tuple[str, str, S.Options, str]]:
    v = [
        ("S0", "Primary analysis", S.Options(), "primary"),
        ("S1a", "Day 4 included", S.Options(days=(4, 5)), "photo timing"),
        ("S1b", "Gardner rank ignores expansion", S.Options(rank_col="rank_no_exp"),
         "photo timing"),
        ("S2a", "Model: embryo grades + age only", S.Options(feature_set="embryo_age"),
         "model form"),
        ("S2b", "Model: ordinal Gardner scores", S.Options(feature_set="ordinal"),
         "model form"),
        ("S2c", "Model: gradient boosting", S.Options(kind="gbm"), "model form"),
        ("S3", "Complete cases (AMH observed)", S.Options(impute=False), "missing data"),
        ("S4", "Grades of the clinical file for transferred embryos", S.Options(),
         "grade inconsistency"),
        ("S5", "Positivity: sibling grade combinations seen among transfers",
         S.Options(positivity=True), "positivity"),
    ]
    for o in cfg["sensitivity"]["hidden_advantage_or"]:
        v.append((f"S6 (OR {o})", f"Hidden advantage of transferred embryo, OR {o}",
                  S.Options(hidden_or=o), "unmeasured confounding"))
    v += [
        ("S7", "Gardner rank: TE before ICM", S.Options(rank_col="rank_te_first"),
         "ranking rule"),
        ("S8", "Outcome: heart activity (HA)", S.Options(outcome="HA"), "outcome"),
        ("S8b", "Outcome: positive hCG (SS)", S.Options(outcome="SS"), "outcome"),
        ("S9", "Model trained on no-choice patients only", S.Options(train_mask="no_choice"),
         "selected training data"),
        ("S10", "Patients with duplicated image files excluded", S.Options(),
         "data integrity"),
    ]
    return v


def _prepare(code: str, transfers: pd.DataFrame, embryos: pd.DataFrame
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    t, e = transfers.copy(), embryos.copy()
    e = add_gardner_rank(e)
    e["rank_no_exp"] = add_gardner_rank(e, order=("ICM", "TE"))["gardner_rank"]
    e["rank_te_first"] = add_gardner_rank(e, order=("TE", "ICM", "EXP"))["gardner_rank"]
    if code == "S3":
        keep = t["AMH"].notna() & t["Endo"].notna() & t["COC"].notna()
        t = t[keep]
        e = e[e["patient"].isin(t["patient"])]
    if code == "S4":
        tr = t.set_index("patient")[["embryo", "EXP_silver", "ICM_silver", "TE_silver"]]
        m = e["transferred"] == 1
        for c in ("EXP", "ICM", "TE"):
            e.loc[m, c] = e.loc[m, "patient"].map(tr[f"{c}_silver"]).astype(int).to_numpy()
            t[c] = t["patient"].map(tr[f"{c}_silver"]).astype(int)
        e = add_gardner_rank(e)
    if code == "S10":
        bad = set(DUPLICATE_IMAGE_PATIENTS)
        t = t[~t["patient"].isin(bad)]
        e = e[~e["patient"].isin(bad)]
    if code == "S8":
        e = e.drop(columns=["HA"], errors="ignore")
    if code == "S8b":
        e = e.drop(columns=["SS"], errors="ignore")
    return t, e


def run(transfers: pd.DataFrame, embryos: pd.DataFrame, cfg: dict, C: float, n_boot: int,
        seed: int, log=None) -> pd.DataFrame:
    rows = []
    for i, (code, label, opt, threat) in enumerate(_variants(cfg)):
        t, e = _prepare(code, transfers, embryos)
        point, _ = S.run_once(t, e, cfg, C, opt, seed)
        boot = S.bootstrap(t, e, cfg, C, opt, n_boot, seed + 10_000 * (i + 1))
        summ = S.summarise(point, boot, ["E1", "E2", "E3", "E4"]).set_index("quantity")
        row = {"code": code, "analysis": label, "threat": threat}
        for k in ("E1", "E2", "E3", "E4"):
            row[k] = summ.loc[k, "estimate"]
            row[f"{k}_ci"] = f"[{summ.loc[k, 'ci_low']:.1f}, {summ.loc[k, 'ci_high']:.1f}]"
        rows.append(row)
        if log:
            log.info("%s: E1 = %.2f %s", code, row["E1"], row["E1_ci"])
    return pd.DataFrame(rows)


def cascade(transfers: pd.DataFrame) -> pd.DataFrame:
    """Exploratory: at which stage does morphology act? (hCG -> heart activity -> birth)."""
    t = transfers[transfers["day"] == 5].copy()
    t["ICM_cat"] = t["ICM"].map({0: "A", 1: "B", 2: "C", 3: "ND"})
    t["TE_cat"] = t["TE"].map({0: "A", 1: "B", 2: "C", 3: "ND"})
    steps = [("hCG positive (all)", t, "SS"),
             ("Heart activity | hCG positive", t[t["SS"] == 1], "HA"),
             ("Live birth | heart activity", t[t["HA"] == 1], "LB")]
    rows = []
    for name, d, y in steps:
        fit = smf.glm(f"{y} ~ C(ICM_cat, Treatment('A')) + C(TE_cat, Treatment('A')) + EXP"
                      " + age", d, family=sm.families.Binomial()).fit()
        ci = fit.conf_int()
        for term in fit.params.index:
            if term == "Intercept":
                continue
            rows.append({"stage": name, "n": len(d), "events": int(d[y].sum()), "term": term,
                         "OR": float(np.exp(np.clip(fit.params[term], -30, 30))),
                         "ci_low": float(np.exp(np.clip(ci.loc[term, 0], -30, 30))),
                         "ci_high": float(np.exp(np.clip(ci.loc[term, 1], -30, 30))),
                         "p": float(fit.pvalues[term])})
    return pd.DataFrame(rows)
