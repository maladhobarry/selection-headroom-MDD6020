"""Tests of the estimators on small synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest

from headroom import model as M
from headroom import simulation as D
from headroom.strategies import estimands, strategy_values


def test_strategy_values_known_answer():
    emb = pd.DataFrame({
        "patient": [1, 1, 1, 2, 2],
        "p": [0.2, 0.4, 0.6, 0.3, 0.5],
        "transferred": [0, 1, 0, 1, 0],
        "gardner_rank": [2, 2, 1, 1, 2],
    })
    v = strategy_values(emb)
    assert v.loc[1, "random"] == pytest.approx(0.4)
    assert v.loc[1, "gardner"] == pytest.approx(0.6)
    assert v.loc[2, "observed"] == pytest.approx(0.3)
    e = estimands(v)
    assert e["E1"] == pytest.approx(((0.6 - 0.4) + (0.3 - 0.4)) / 2 * 100)
    assert e["E4"] == pytest.approx(((0.6 - 0.2) + (0.5 - 0.3)) / 2 * 100)


def test_estimand_weights_equal_duplication():
    v = pd.DataFrame({k: [0.3, 0.5] for k in ["worst", "random", "observed", "gardner",
                                              "model_best"]})
    v["gardner"] = [0.4, 0.5]
    w = estimands(v, np.array([2, 1]))
    dup = estimands(pd.concat([v.iloc[[0]], v]).reset_index(drop=True))
    assert w["E1"] == pytest.approx(dup["E1"])


def test_rcs_is_linear_beyond_outer_knots():
    k = np.array([25.0, 33.0, 40.0])
    x = np.array([41.0, 42.0, 43.0])
    b = M.rcs_basis(x, k)[:, 1]
    assert np.diff(b, 2) == pytest.approx(np.zeros(1), abs=1e-9)


def test_calibrate_c_hits_target():
    b = np.random.default_rng(0).normal(-1, 0.3, 500)
    c = D.calibrate_c(b, 1.0, 0.3)
    rate = np.mean(D.expit(b[:, None] + c + 1.0 * D.GH_X[None, :]) @ D.GH_W)
    assert rate == pytest.approx(0.3, abs=1e-6)


def test_perfect_predictor_beats_noisy_one():
    rng = np.random.default_rng(1)
    b = rng.normal(-1, 0.2, 300)
    nb = rng.integers(1, 6, 300)
    c = D.calibrate_c(b, 1.0, 0.3)
    g0 = D.expected_gain(b, nb, c, 1.0, 0.0, 50, np.random.default_rng(2)).mean()
    g5 = D.expected_gain(b, nb, c, 1.0, 5.0, 50, np.random.default_rng(2)).mean()
    assert g0 > g5 > -0.005


def test_singletons_have_no_gain():
    b = np.zeros(10)
    g = D.expected_gain(b, np.ones(10, dtype=int), -1.0, 1.0, 0.0, 20,
                        np.random.default_rng(3))
    assert np.allclose(g, 0)


def test_n_per_arm_matches_textbook():
    # 35% vs 40%, alpha 5%, power 80%: about 1,470 per arm.
    assert D.n_per_arm(0.35, 0.05, 0.05, 0.80) == pytest.approx(1471, abs=5)


def test_impute_fold_is_leak_free():
    """Changing a test patient's outcome must not change its imputed values."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "patient": np.arange(n), "age": rng.normal(33, 4, n), "MII": rng.integers(1, 15, n),
        "day": 5, "COC": rng.integers(2, 20, n).astype(float),
        "Endo": rng.normal(10, 2, n), "AMH": np.exp(rng.normal(1, 0.6, n)),
        "LB": rng.integers(0, 2, n)})
    df.loc[rng.choice(n, 50, replace=False), "AMH"] = np.nan
    train, test = df.iloc[:150], df.iloc[150:].copy()
    _, te1 = M.impute_fold(train, test, seed=1)
    test["LB"] = 1 - test["LB"]
    _, te2 = M.impute_fold(train, test, seed=1)
    assert np.allclose(te1["logAMH"], te2["logAMH"])
    assert te1["logAMH"].notna().all()


def test_s10_excludes_duplicate_image_patients():
    """S10 removes every patient listed as having a byte-identical image file."""
    import pandas as pd

    from headroom.cleaning import DUPLICATE_IMAGE_PATIENTS
    from headroom.sensitivity import _prepare

    pts = sorted(DUPLICATE_IMAGE_PATIENTS)[:2] + [9001, 9002]
    t = pd.DataFrame({"patient": pts, "embryo": 1, "AMH": 1.0, "Endo": 1.0, "COC": 1.0})
    e = pd.DataFrame({"patient": pts, "embryo": 1, "EXP": 3, "ICM": 0, "TE": 0,
                      "transferred": 1})
    t2, e2 = _prepare("S10", t, e)
    assert set(t2["patient"]) == {9001, 9002}
    assert set(e2["patient"]) == {9001, 9002}
