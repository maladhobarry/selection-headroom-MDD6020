"""Step D: ceiling of the gain reachable by a better predictor (E5), and trial size.

A small observed headroom has two possible explanations: a weak predictor (Gardner grades),
or little difference in viability between sibling embryos. The simulation separates them.

Model (logit scale), keeping the real cohort structure of day-5 patients:
    viability  l_ij = b_i + c + u_ij,   u_ij ~ N(0, sigma_e^2)
    predictor  s_ij = l_ij + eta_ij,    eta_ij ~ N(0, tau^2)
- b_i: patient baseline from the step-B model (mean predicted logit of her embryos),
- c:   calibrates the mean live-birth rate of a randomly chosen embryo to the observed level,
- sigma_e: embryo-level spread of viability, not identifiable with one transfer per patient,
  explored on a grid and narrowed by the slope of step F,
- tau: noise of the predictor.

Anchoring. The "current morphology" predictor is the one whose simulated selection gain
equals the observed gain of the Gardner strategy (E1, step C); its simulated AUROC is A_G.
Better predictors are defined by an AUROC *increment* over morphology (A_G + delta), the
scale on which AI tools are usually compared (e.g. iDAScore vs KIDScore: about +0.01).
A sigma_e too small to reproduce E1 even with a perfect predictor is ruled out: this gives
a lower bound on the embryo-level spread of viability.

Gain of a predictor = E[p of the embryo it ranks first] - E[p of a random embryo].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(60)   # E[f(Z)], Z ~ N(0,1)
GH_W = GH_W / GH_W.sum()


def expit(x):
    return 1 / (1 + np.exp(-x))


def calibrate_c(b: np.ndarray, sigma: float, target: float) -> float:
    """Intercept c such that mean_i E_u[expit(b_i + c + u)] = target (Gauss-Hermite)."""
    def mean_rate(c):
        return float(np.mean(expit(b[:, None] + c + sigma * GH_X[None, :]) @ GH_W))
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_rate(mid) < target else (lo, mid)
    return (lo + hi) / 2


def auroc_for_tau(b, c, sigma, tau, rng, m=60_000) -> float:
    i = rng.integers(0, len(b), m)
    lat = b[i] + c + rng.normal(0, sigma, m)
    y = rng.random(m) < expit(lat)
    s = lat + rng.normal(0, tau, m)
    return roc_auc_score(y, s)


def tau_for_auroc(b, c, sigma, target, seed) -> tuple[float, bool]:
    """Noise SD giving the target AUROC; (nan, False) if unreachable even with tau = 0."""
    best = auroc_for_tau(b, c, sigma, 0.0, np.random.default_rng(seed))
    if target >= best:
        return np.nan, False
    lo, hi = 0.0, 50.0
    for _ in range(30):
        mid = (lo + hi) / 2
        a = auroc_for_tau(b, c, sigma, mid, np.random.default_rng(seed))
        lo, hi = (mid, hi) if a > target else (lo, mid)
    return (lo + hi) / 2, True


def expected_gain(b, n_blast, c, sigma, tau, reps, rng) -> np.ndarray:
    """Per-patient expected gain (chosen - random), averaged over repetitions."""
    n = len(b)
    k = int(n_blast.max())
    valid = np.arange(k)[None, :] < n_blast[:, None]
    gain = np.zeros(n)
    for _ in range(reps):
        u = rng.normal(0, sigma, (n, k))
        p = expit(b[:, None] + c + u)
        s = u + (rng.normal(0, tau, (n, k)) if tau > 0 else 0.0)
        s = np.where(valid, s, -np.inf)
        pick = p[np.arange(n), s.argmax(1)]
        rand = np.where(valid, p, 0).sum(1) / n_blast
        gain += pick - rand
    return gain / reps


def n_per_arm(p0: float, delta: float, alpha: float, power: float) -> float:
    if delta <= 0:
        return np.inf
    p1 = p0 + delta
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(np.ceil(z**2 * (p0 * (1 - p0) + p1 * (1 - p1)) / delta**2))


def tau_for_gain(b, nb, c, sigma, target_gain, reps, seed) -> tuple[float, bool]:
    """Noise SD whose selection gain on the choice set equals ``target_gain`` (proportion)."""
    multi = nb >= 2

    def gain(tau):
        return expected_gain(b, nb, c, sigma, tau, reps, np.random.default_rng(seed))[
            multi].mean()

    if gain(0.0) < target_gain:
        return np.nan, False
    lo, hi = 0.0, 60.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gain(mid) > target_gain else (lo, mid)
    return (lo + hi) / 2, True


def run(patients: pd.DataFrame, cfg: dict, e1_gain: float, observed_level: float,
        f_summary: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    patients: day-5 patients with columns b (baseline logit), n_blast, target (random level).
    e1_gain: observed gain of the Gardner strategy over random (step C, proportion).
    observed_level: live-birth level under the embryologist's choice (step C), proportion.
    f_summary: step-F summary (RD_k_vs_1 estimates and CIs), model-free consistency check.
    """
    sim = cfg["simulation"]
    b = patients["b"].to_numpy()
    nb = patients["n_blast"].to_numpy().astype(int)
    target = float(patients["target"].mean())
    multi = nb >= 2
    k4 = int(round(patients.loc[nb >= 4, "n_blast"].mean()))
    f = f_summary.set_index("quantity")
    rows, e7_rows = [], []
    for si, sigma in enumerate(sim["embryo_sd_grid"]):
        c = calibrate_c(b, sigma, target)
        tau_g, ok_g = tau_for_gain(b, nb, c, sigma, e1_gain, max(50, sim["n_reps"] // 5),
                                   seed + si)
        if not ok_g:
            # Even a perfect predictor cannot reproduce the observed Gardner gain.
            e7_rows.append({"sigma_e": sigma, "reproduces_E1": False})
            rows.append({"sigma_e": sigma, "delta_auroc": "all", "reachable": False,
                         "reproduces_E1": False})
            continue
        a_g = auroc_for_tau(b, c, sigma, tau_g, np.random.default_rng(seed + si))
        g_gard = expected_gain(b, nb, c, sigma, tau_g, sim["n_reps"],
                               np.random.default_rng(seed + 100 + si))

        # Model-free check (step F): give every patient k blastocysts, choose by morphology.
        e7 = {}
        for lvl, k in (("2", 2), ("3", 3), ("4+", k4)):
            g = expected_gain(b, np.full(len(b), k), c, sigma, tau_g,
                              max(20, sim["n_reps"] // 10), np.random.default_rng(seed + k))
            e7[lvl] = float(g.mean()) * 100
        obs = f.loc["RD_4+_vs_1"]
        consistent = bool(obs["ci_low"] <= e7["4+"] <= obs["ci_high"])
        e7_rows.append({"sigma_e": sigma, "reproduces_E1": True, "gardner_tau": tau_g,
                        "gardner_sim_auroc": a_g,
                        **{f"implied_RD_{k}_vs_1": v for k, v in e7.items()},
                        "observed_RD_4+_vs_1": obs["estimate"],
                        "observed_ci": f"[{obs['ci_low']:.1f}, {obs['ci_high']:.1f}]",
                        "consistent_with_step_F": consistent})

        for ai, d in enumerate([*sim["auroc_delta_grid"], "perfect"]):
            if d == "perfect":
                tau, ok = 0.0, True
            else:
                tau, ok = tau_for_auroc(b, c, sigma, a_g + d, seed + 1000 + 10 * si + ai)
            if not ok:
                rows.append({"sigma_e": sigma, "delta_auroc": d, "reachable": False,
                             "reproduces_E1": True})
                continue
            g = expected_gain(b, nb, c, sigma, tau, sim["n_reps"],
                              np.random.default_rng(seed + 2000 + 10 * si + ai))
            over_gardner = float(g[multi].mean() - g_gard[multi].mean())
            row = {"sigma_e": sigma, "delta_auroc": d, "reachable": True,
                   "reproduces_E1": True, "tau": tau,
                   "sim_auroc": a_g + d if d != "perfect" else np.nan,
                   "gain_vs_random_choice_set": float(g[multi].mean()) * 100,
                   "gain_vs_gardner_choice_set": over_gardner * 100,
                   "n_per_arm_vs_morphology": n_per_arm(observed_level, over_gardner,
                                                        sim["alpha"], sim["power"])}
            for lvl, lo, hi in (("2", 2, 2), ("3", 3, 3), ("4+", 4, 999)):
                sel = (nb >= lo) & (nb <= hi)
                row[f"gain_vs_gardner_n{lvl}"] = float(g[sel].mean() - g_gard[sel].mean()) * 100
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(e7_rows)
