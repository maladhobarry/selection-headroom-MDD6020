"""Step E: reproducibility of the choice between embryologists (E6).

1. On the 300-image test set, estimate for each retained embryologist the probability of
   each grade given the consensus grade (one confusion matrix per criterion, add-0.5
   smoothing).
2. For each patient of the choice set, regrade every sibling as that embryologist would
   (draws from her confusion matrices, starting from the expert's grade), re-apply the
   Gardner ranking, and compare the embryo picked with the expert's pick.
3. Report the share of patients whose pick changes and the change in expected live birth
   (probabilities from the step-B model, i.e. the expert-graded "truth").

Consistency rule (Gardner): expansion 1-2 (codes 0-1) implies ICM/TE "not defined" (3);
expansion >= 3 implies a defined ICM/TE grade (0-2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import add_gardner_rank

K = {"EXP": 5, "ICM": 4, "TE": 4}


def confusion(ann: pd.DataFrame, gold: pd.DataFrame, annotator: str) -> dict[str, np.ndarray]:
    a = ann[ann["annotator"] == annotator].merge(
        gold, on=["patient", "embryo"], suffixes=("", "_gold"))
    out = {}
    for c, k in K.items():
        m = np.full((k, k), 0.5)                      # rows: consensus grade, cols: annotator
        d = a.dropna(subset=[c, f"{c}_gold"])
        for g, x in zip(d[f"{c}_gold"].astype(int), d[c].astype(int), strict=True):
            if 0 <= g < k and 0 <= x < k:
                m[g, x] += 1
        out[c] = m / m.sum(1, keepdims=True)
    return out


def agreement(ann: pd.DataFrame, gold: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for n in names:
        a = ann[ann["annotator"] == n].merge(gold, on=["patient", "embryo"],
                                             suffixes=("", "_gold"))
        r = {"annotator": n, "images": len(a)}
        for c in K:
            d = a.dropna(subset=[c, f"{c}_gold"])
            r[f"{c}_agreement"] = float((d[c] == d[f"{c}_gold"]).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def _draw(true: np.ndarray, cm: np.ndarray, rng: np.random.Generator,
          allowed: np.ndarray | None = None) -> np.ndarray:
    probs = cm[true.astype(int)]
    if allowed is not None:
        probs = probs * allowed
        probs = probs / probs.sum(1, keepdims=True)
    cum = probs.cumsum(1)
    return (rng.random(len(true))[:, None] > cum).sum(1)


def regrade(emb: pd.DataFrame, cms: dict, rng: np.random.Generator) -> pd.DataFrame:
    out = emb.copy()
    exp = _draw(out["EXP"].to_numpy(), cms["EXP"], rng)
    defined = exp >= 2
    allow_def = np.array([1.0, 1.0, 1.0, 0.0])
    allow_nd = np.array([0.0, 0.0, 0.0, 1.0])
    for c in ("ICM", "TE"):
        allow = np.where(defined[:, None], allow_def[None, :], allow_nd[None, :])
        out[c] = _draw(out[c].to_numpy(), cms[c], rng, allow)
    out["EXP"] = exp
    return out


def pick_distribution(emb: pd.DataFrame) -> pd.Series:
    """Probability that each embryo is picked (uniform among tied best Gardner ranks)."""
    best = (emb["gardner_rank"] == 1).astype(float)
    return best / best.groupby(emb["patient"]).transform("sum")


def run(embryos_p: pd.DataFrame, ann: pd.DataFrame, gold: pd.DataFrame, names: list[str],
        n_reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """embryos_p: choice-set embryos with expert grades and model probability ``p``."""
    rng = np.random.default_rng(seed)
    base = add_gardner_rank(embryos_p)
    w_exp = pick_distribution(base).to_numpy()
    value_exp = pd.Series(w_exp * base["p"].to_numpy()).groupby(base["patient"].to_numpy()).sum()
    rows = []
    for name in names:
        cms = confusion(ann, gold, name)
        for r in range(n_reps):
            sim = add_gardner_rank(regrade(base, cms, rng))
            w = pick_distribution(sim).to_numpy()
            same = pd.Series(w * w_exp).groupby(base["patient"].to_numpy()).sum()
            value = pd.Series(w * base["p"].to_numpy()).groupby(base["patient"].to_numpy()).sum()
            rows.append({"annotator": name, "rep": r,
                         "pick_changed_pct": float((1 - same).mean()) * 100,
                         "delta_LB_points": float((value - value_exp).mean()) * 100,
                         "abs_delta_LB_points": float((value - value_exp).abs().mean()) * 100})
    sims = pd.DataFrame(rows)
    summ = (sims.groupby("annotator")[["pick_changed_pct", "delta_LB_points",
                                       "abs_delta_LB_points"]]
            .agg(["mean", lambda x: np.percentile(x, 2.5), lambda x: np.percentile(x, 97.5)]))
    summ.columns = [f"{a}_{b}".replace("<lambda_0>", "lo").replace("<lambda_1>", "hi")
                    for a, b in summ.columns]
    overall = sims.groupby("rep")[["pick_changed_pct", "delta_LB_points",
                                   "abs_delta_LB_points"]].mean()
    tot = {f"{c}_mean": overall[c].mean() for c in overall}
    tot.update({f"{c}_lo": np.percentile(overall[c], 2.5) for c in overall})
    tot.update({f"{c}_hi": np.percentile(overall[c], 97.5) for c in overall})
    summ.loc["ALL (mean of embryologists)"] = pd.Series(tot)
    return summ.reset_index(), sims
