"""Step F: observed value of choice (E7) and test of hidden selection.

If choosing among siblings is worth a lot, live birth must rise with the number of
blastocysts available (1 = no choice). We estimate, on day-5 single-embryo transfers,
standardised live-birth risks by number of blastocysts (1, 2, 3, 4+), adjusted for age,
AMH and mature oocytes (g-computation over the observed covariate distribution).

The number of blastocysts also reflects oocyte quality, which pushes the slope *up*:
the adjusted contrast is therefore read as an upper bound of the value of choice.

Hidden-selection test: calibration-in-the-large of the step-B model (cross-fitted
predictions for the transferred embryo) in patients without vs with a choice. Observed
births above predictions only when a choice existed would suggest that embryologists see
signs not captured by the grades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from . import model as M

LEVELS = ["1", "2", "3", "4+"]


def _prep(t: pd.DataFrame, knots: np.ndarray) -> pd.DataFrame:
    out = t.copy()
    out["nb"] = pd.Categorical(pd.cut(out["n_blast"], [0, 1, 2, 3, 99], labels=LEVELS),
                               categories=LEVELS)
    basis = M.rcs_basis(out["age"].to_numpy(), knots)
    out["age0"], out["age1"] = basis[:, 0], basis[:, 1]
    out["nb_num"] = out["n_blast"].clip(upper=6)
    return out


FORMULA = "LB ~ C(nb, Treatment('1')) + age0 + age1 + logAMH + MII"
TREND = "LB ~ nb_num + age0 + age1 + logAMH + MII"


def standardised_risks(t_imp: pd.DataFrame, knots: np.ndarray) -> dict:
    d = _prep(t_imp, knots)
    fit = smf.glm(FORMULA, d, family=sm.families.Binomial()).fit()
    res = {}
    for lvl in LEVELS:
        cf = d.assign(nb=pd.Categorical([lvl] * len(d), categories=LEVELS))
        res[f"risk_{lvl}"] = float(fit.predict(cf).mean()) * 100
    for lvl in LEVELS[1:]:
        res[f"RD_{lvl}_vs_1"] = res[f"risk_{lvl}"] - res["risk_1"]
    trend = smf.glm(TREND, d, family=sm.families.Binomial()).fit()
    res["OR_per_blastocyst"] = float(np.exp(trend.params["nb_num"]))
    return res


def run(transfers: pd.DataFrame, cfg: dict, n_boot: int, seed: int
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = transfers[transfers["day"] == cfg["population"]["main_day"]].reset_index(drop=True)
    knots = M.rcs_knots(t["age"], cfg["model"]["age_spline_knots"])
    m = cfg["model"]["n_imputations"]
    # Point estimate: average over multiple imputations.
    pts = [standardised_risks(M.impute_patients(t, seed + i), knots) for i in range(m)]
    point = pd.DataFrame(pts).mean().to_dict()
    # Bootstrap: one stochastic imputation per replicate.
    rng = np.random.default_rng(seed)
    boots, failed = [], 0
    for b in range(n_boot):
        tb = t.iloc[rng.integers(0, len(t), len(t))].reset_index(drop=True)
        try:
            boots.append(standardised_risks(M.impute_patients(tb, seed + 1000 + b), knots))
        except Exception:  # rare non-convergence (e.g. an empty category) in a replicate
            failed += 1
    boot = pd.DataFrame(boots)
    point["bootstrap_replicates_failed"] = failed
    summary = pd.DataFrame([{"quantity": k, "estimate": point[k],
                             "ci_low": np.percentile(boot[k], 2.5) if k in boot else np.nan,
                             "ci_high": np.percentile(boot[k], 97.5) if k in boot else np.nan}
                            for k in point])
    crude = (t.assign(nb=pd.cut(t["n_blast"], [0, 1, 2, 3, 99], labels=LEVELS))
             .groupby("nb", observed=True)
             .agg(n=("LB", "size"), live_births=("LB", "sum"), age=("age", "mean"),
                  MII=("MII", "mean"))
             .assign(LB_pct=lambda d: d.live_births / d.n * 100).reset_index())
    return summary, boot, crude


def hidden_selection_test(transfers: pd.DataFrame, p_transfer: np.ndarray, n_boot: int,
                          seed: int, refit=None) -> pd.DataFrame:
    """Observed/expected live births by choice group (day 5), cross-fitted predictions.

    ``refit(transfers_boot, seed) -> predictions`` re-runs the leak-free cross-fitting in each
    bootstrap replicate, so the CI includes model uncertainty. Without it, the stored
    predictions are resampled (narrower CI).
    """
    def oe(d):
        d = d[d["day"] == 5]
        grp = np.where(d["n_blast"] >= 2, "choice", "no_choice")
        g = d.groupby(grp)
        r = (g["LB"].sum() / g["p"].sum()).to_dict()
        r["ratio_choice_vs_no_choice"] = r["choice"] / r["no_choice"]
        return r

    t = transfers.assign(p=p_transfer)
    point = oe(t)
    rng = np.random.default_rng(seed)
    rows_b = []
    for b in range(n_boot):
        idx = rng.integers(0, len(t), len(t))
        if refit is None:
            rows_b.append(oe(t.iloc[idx]))
        else:
            tb = transfers.iloc[idx].reset_index(drop=True)
            rows_b.append(oe(tb.assign(p=refit(tb, seed + b))))
    boot = pd.DataFrame(rows_b)
    rows = []
    for k, v in point.items():
        rows.append({"quantity": f"O/E {k}" if "ratio" not in k else k, "estimate": v,
                     "ci_low": np.percentile(boot[k], 2.5),
                     "ci_high": np.percentile(boot[k], 97.5)})
    d5 = t[t["day"] == 5]
    counts = d5.groupby(np.where(d5["n_blast"] >= 2, "choice", "no_choice")).agg(
        n=("LB", "size"), observed=("LB", "sum"), expected=("p", "sum"))
    for g, r in counts.iterrows():
        rows.append({"quantity": f"n / observed / expected ({g})",
                     "estimate": r["n"], "ci_low": r["observed"], "ci_high": r["expected"]})
    return pd.DataFrame(rows)
