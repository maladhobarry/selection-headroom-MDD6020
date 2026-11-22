"""Step C: g-computation of embryo-selection strategies (estimands E1-E4).

For each patient of the choice set (day 5, >= 2 blastocysts) and each of her blastocysts j,
the cross-fitted model gives p_ij, the probability of live birth if embryo j were transferred.
The value of a strategy for the patient is the p of the embryo it would pick:

- random      mean_j p_ij
- observed    p of the embryo actually transferred
- gardner     p of the best Gardner-ranked embryo (ties: mean over tied embryos)
- model_best  max_j p_ij   (best according to the model)
- worst       min_j p_ij

Estimands (percentage points, averaged over patients):
E1 = gardner - random (primary), E2 = observed - random, E3 = model_best - observed,
E4 = model_best - worst.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import model as M

STRATEGIES = ["worst", "random", "observed", "gardner", "model_best"]
ESTIMANDS = {
    "E1": ("gardner", "random"),
    "E2": ("observed", "random"),
    "E3": ("model_best", "observed"),
    "E4": ("model_best", "worst"),
    "gardner_vs_observed": ("gardner", "observed"),
}


@dataclass
class Options:
    """Switches used by the sensitivity analyses (defaults = primary analysis)."""
    feature_set: str = "full"
    kind: str = "ridge"
    days: tuple = (5,)
    rank_col: str = "gardner_rank"
    outcome: str = "LB"
    impute: bool = True
    hidden_or: float = 1.0          # S6: hidden advantage of the transferred embryo
    train_mask: str = "all"         # "all" or "no_choice" (S9)
    positivity: bool = False        # S5


def strategy_values(emb: pd.DataFrame, p_col: str = "p", rank_col: str = "gardner_rank"
                    ) -> pd.DataFrame:
    """Per-patient value of each strategy."""
    g = emb.groupby("patient")
    out = pd.DataFrame({
        "random": g[p_col].mean(),
        "worst": g[p_col].min(),
        "model_best": g[p_col].max(),
        "observed": emb[emb["transferred"] == 1].set_index("patient")[p_col],
        "gardner": emb[emb[rank_col] == 1].groupby("patient")[p_col].mean(),
        "n_blast": g.size(),
    })
    return out


def estimands(values: pd.DataFrame, weights: np.ndarray | None = None) -> dict:
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    res = {s: float(np.average(values[s], weights=w)) * 100 for s in STRATEGIES}
    for e, (a, b) in ESTIMANDS.items():
        res[e] = res[a] - res[b]
    return res


def choice_mask(transfers: pd.DataFrame, opt: Options) -> pd.Series:
    return transfers["day"].isin(opt.days) & (transfers["n_blast"] >= 2)


def fit_predict(transfers: pd.DataFrame, embryos: pd.DataFrame, cfg: dict, C: float | None,
                opt: Options, seed: int) -> pd.DataFrame:
    """Return the choice-set embryos with leak-free cross-fitted predictions ``p``.

    Imputation and (if ``C`` is None) the penalty search happen inside each fold
    (see ``model.crossfit``). Bootstrap replicates pass the fixed C of the main analysis.
    """
    t = transfers.reset_index(drop=True)
    chosen = set(t.loc[choice_mask(t, opt), "patient"])
    e = embryos[embryos["patient"].isin(chosen)]
    knots_n = cfg["model"]["age_spline_knots"]

    def make(c):
        return M.EmbryoModel(feature_set=opt.feature_set, kind=opt.kind, C=c,
                             n_knots=knots_n, seed=seed)

    if opt.train_mask == "no_choice":
        train = t[t["n_blast"] == 1]
        test = t[t["patient"].isin(chosen)].drop_duplicates("patient")
        if opt.impute:
            tr, te = M.impute_fold(train, test, seed, opt.outcome)
        else:
            tr, te = M.add_log_amh(train), M.add_log_amh(test)
        c = C if C is not None else M.choose_C_inner(tr, opt.feature_set, knots_n, seed,
                                                     opt.outcome)
        m = make(c).fit(tr, tr[opt.outcome].to_numpy())
        e = M.attach_patient_covariates(e, te)
        e = e.assign(p=m.predict(e))
    else:
        _, p_emb, _ = M.crossfit(t, e, make, cfg["model"]["cv_folds"], seed,
                                 outcome=opt.outcome, impute=opt.impute, C=C,
                                 feature_set=opt.feature_set, n_knots=knots_n)
        e = e.reset_index(drop=True).assign(p=p_emb)

    if opt.hidden_or != 1.0:
        # The model learned from selected embryos; if the transferred embryo had a hidden
        # advantage (odds ratio OR), siblings' true probabilities are lower than predicted.
        lo = np.log(e["p"] / (1 - e["p"])) - np.log(opt.hidden_or)
        sib = e["transferred"] == 0
        e.loc[sib, "p"] = (1 / (1 + np.exp(-lo)))[sib]

    if opt.positivity:
        seen = set(map(tuple, transfers[["EXP", "ICM", "TE"]].to_numpy()))
        keep = e[["EXP", "ICM", "TE"]].apply(tuple, axis=1).isin(seen) | (e["transferred"] == 1)
        e = e[keep]
        n = e.groupby("patient").size()
        e = e[e["patient"].isin(n[n >= 2].index)]
        from .cleaning import add_gardner_rank
        e = add_gardner_rank(e)
    return e


def run_once(transfers: pd.DataFrame, embryos: pd.DataFrame, cfg: dict, C: float,
             opt: Options, seed: int) -> tuple[dict, pd.DataFrame]:
    e = fit_predict(transfers, embryos, cfg, C, opt, seed)
    vals = strategy_values(e, rank_col=opt.rank_col)
    return estimands(vals), vals


def bootstrap(transfers: pd.DataFrame, embryos: pd.DataFrame, cfg: dict, C: float,
              opt: Options, n_boot: int, seed: int) -> pd.DataFrame:
    """Patient-level bootstrap refitting imputation, model and cross-fitted predictions."""
    rng = np.random.default_rng(seed)
    n = len(transfers)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        tb = transfers.iloc[idx].reset_index(drop=True)
        counts = tb["patient"].value_counts()
        try:
            e = fit_predict(tb, embryos, cfg, C, opt, seed + 1 + b)
        except ValueError:
            continue
        vals = strategy_values(e, rank_col=opt.rank_col)
        w = counts.reindex(vals.index).fillna(0).to_numpy()
        if w.sum() == 0:
            continue
        rows.append(estimands(vals, w))
    return pd.DataFrame(rows)


def permutation_null(transfers: pd.DataFrame, embryos: pd.DataFrame, cfg: dict, C: float,
                     opt: Options, n_perm: int, seed: int) -> pd.DataFrame:
    """Negative control: estimands when outcomes are permuted across transfers.

    Any link between grades and live birth is destroyed, so every non-zero estimand is
    produced by noise and by choosing the maximum of noisy predictions ("winner's curse").
    E3 above this noise floor is the part that reflects real signal.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_perm):
        tp = transfers.copy()
        tp[opt.outcome] = rng.permutation(tp[opt.outcome].to_numpy())
        rows.append(run_once(tp, embryos, cfg, C, opt, seed + 1 + i)[0])
    return pd.DataFrame(rows)


def summarise(point: dict, boot: pd.DataFrame, keys=None) -> pd.DataFrame:
    keys = keys or (STRATEGIES + list(ESTIMANDS))
    out = []
    for k in keys:
        lo, hi = np.percentile(boot[k], [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
        out.append({"quantity": k, "estimate": point[k], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)


def heterogeneity(vals: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """E1 and E3 by number of blastocysts and age group (point estimates)."""
    v = vals.join(transfers.set_index("patient")[["age"]])
    v["n_group"] = pd.cut(v["n_blast"], [1, 2, 3, 99], labels=["2", "3", "4+"])
    v["age_group"] = pd.cut(v["age"], [0, 34.99, 39.99, 99], labels=["<35", "35-39", ">=40"])
    rows = []
    for col in ("n_group", "age_group"):
        for lvl, d in v.groupby(col, observed=True):
            r = estimands(d)
            rows.append({"subgroup": col, "level": str(lvl), "n": len(d), "E1": r["E1"],
                         "E2": r["E2"], "E3": r["E3"]})
    return pd.DataFrame(rows)


def heterogeneity_boot(transfers, embryos, cfg, C, opt, n_boot, seed) -> pd.DataFrame:
    """Bootstrap CIs for subgroup estimands (refits as in ``bootstrap``)."""
    rng = np.random.default_rng(seed)
    n = len(transfers)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        tb = transfers.iloc[idx].reset_index(drop=True)
        counts = tb["patient"].value_counts()
        e = fit_predict(tb, embryos, cfg, C, opt, seed + 1 + b)
        vals = strategy_values(e, rank_col=opt.rank_col)
        vals = vals.loc[vals.index.repeat(counts.reindex(vals.index).fillna(0).astype(int))]
        h = heterogeneity(vals, transfers.drop_duplicates("patient"))
        h["b"] = b
        rows.append(h)
    return pd.concat(rows, ignore_index=True)


def crude_check(transfers: pd.DataFrame, n_boot: int = 0, seed: int = 0, outcome: str = "LB"
                ) -> pd.DataFrame:
    """Model-free check: live birth when the best-Gardner embryo was transferred or not,
    age-adjusted (logistic model, standardised risk difference), with bootstrap CI."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    t = transfers[(transfers["day"] == 5) & (transfers["n_blast"] >= 2)].copy()
    t["best_transferred"] = (t["gardner_rank_transferred"] == 1).astype(int)

    def rd(d):
        fit = smf.glm(f"{outcome} ~ best_transferred + age", d,
                      family=sm.families.Binomial()).fit()
        return (fit.predict(d.assign(best_transferred=1)).mean()
                - fit.predict(d.assign(best_transferred=0)).mean()) * 100

    point = rd(t)
    rng = np.random.default_rng(seed)
    boot = [rd(t.iloc[rng.integers(0, len(t), len(t))]) for _ in range(n_boot)]
    return pd.DataFrame([{
        "outcome": outcome,
        "n_best_transferred": int(t["best_transferred"].sum()),
        "n_other_transferred": int((1 - t["best_transferred"]).sum()),
        "crude_rate_best": t.loc[t.best_transferred == 1, outcome].mean() * 100,
        "crude_rate_other": t.loc[t.best_transferred == 0, outcome].mean() * 100,
        "adjusted_RD_points": point,
        "ci_low": np.percentile(boot, 2.5) if boot else np.nan,
        "ci_high": np.percentile(boot, 97.5) if boot else np.nan,
    }])
