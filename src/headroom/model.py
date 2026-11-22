"""Step B: embryo-level prognostic model of live birth (reported following TRIPOD+AI).

Unit of analysis: one single-embryo fresh transfer (one embryo per patient).
Outcome: live birth (LB). Predictors fixed in advance:

- embryo: expansion (categories), inner cell mass (A/B/C/not defined), trophectoderm (idem)
- patient: age (restricted cubic spline, 3 knots), log AMH, mature oocytes (MII),
  endometrial thickness, day of transfer (4 vs 5)

Estimation: ridge-penalised logistic regression (penalty chosen once by cross-validation).
Missing AMH / endometrium / oocyte counts: multiple imputation by chained equations at the
patient level, outcome included in the imputation model.

Sibling predictions are *cross-fitted*: the model that predicts a patient's embryos never saw
that patient's transfer, which prevents an over-fitted model from inflating the
"best according to the model" strategy.

No information from a test fold reaches its predictions:
- missing covariates are imputed inside each fold; the imputation model is fitted on the
  training patients only, with the outcome for the training set and *without* the outcome
  for the test patients;
- the ridge penalty is chosen inside each training fold (nested cross-validation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PATIENT_NUMERIC = ["age", "AMH", "Endo", "COC", "MII"]
IMPUTE_X = ["age", "logAMH", "Endo", "COC", "MII", "day"]
IMPUTE_COLS = [*IMPUTE_X, "LB"]
INNER_CS = np.logspace(-3, 0, 8)     # penalty grid for the nested (inner) search


# --------------------------------------------------------------------------- splines
def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted (natural) cubic spline basis, Harrell's parameterisation.

    Returns k-1 columns for k knots: x itself and k-2 non-linear terms, linear beyond the
    outer knots.
    """
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    cols = [x]
    norm = (k[-1] - k[0]) ** 2
    for j in range(len(k) - 2):
        term = (np.clip(x - k[j], 0, None) ** 3
                - np.clip(x - k[-2], 0, None) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
                + np.clip(x - k[-1], 0, None) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2]))
        cols.append(term / norm)
    return np.column_stack(cols)


def rcs_knots(x: np.ndarray, n_knots: int = 3) -> np.ndarray:
    q = {3: [10, 50, 90], 4: [5, 35, 65, 95], 5: [5, 27.5, 50, 72.5, 95]}[n_knots]
    return np.percentile(np.asarray(x, dtype=float), q)


# --------------------------------------------------------------------------- design matrix
FEATURE_SETS = {
    # Full pre-specified model.
    "full": {"embryo": "categorical", "patient": ["age", "logAMH", "MII", "Endo", "day5"]},
    # S2a: embryo grades + age only.
    "embryo_age": {"embryo": "categorical", "patient": ["age"]},
    # S2b: Gardner grades as ordinal scores instead of categories.
    "ordinal": {"embryo": "ordinal", "patient": ["age", "logAMH", "MII", "Endo", "day5"]},
}


def design(df: pd.DataFrame, feature_set: str, knots: np.ndarray) -> pd.DataFrame:
    spec = FEATURE_SETS[feature_set]
    X = {}
    if spec["embryo"] == "categorical":
        for lvl in (0, 1, 2, 4):                      # reference: EXP=3 (Gardner 4), the mode
            X[f"EXP_{lvl}"] = (df["EXP"] == lvl).astype(float)
        for c in ("ICM", "TE"):
            for lvl, lab in ((1, "B"), (2, "C"), (3, "ND")):   # reference: A
                X[f"{c}_{lab}"] = (df[c] == lvl).astype(float)
    else:
        X["EXP"] = df["EXP"].astype(float)
        nd = df["ICM"] == 3
        X["ND"] = nd.astype(float)
        X["ICM"] = np.where(nd, 0.0, df["ICM"])
        X["TE"] = np.where(df["TE"] == 3, 0.0, df["TE"])
    for v in spec["patient"]:
        if v == "age":
            basis = rcs_basis(df["age"].to_numpy(), knots)
            for j in range(basis.shape[1]):
                X[f"age_{j}"] = basis[:, j]
        elif v == "day5":
            X["day5"] = (df["day"] == 5).astype(float)
        else:
            X[v] = df[v].astype(float)
    return pd.DataFrame(X, index=df.index)


# --------------------------------------------------------------------------- imputation
def add_log_amh(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["logAMH"] = np.log(out["AMH"])
    return out


def impute_patients(transfers: pd.DataFrame, seed: int, sample_posterior: bool = True
                    ) -> pd.DataFrame:
    """One stochastic imputation of patient-level covariates (MICE, outcome included)."""
    t = add_log_amh(transfers)
    imp = IterativeImputer(sample_posterior=sample_posterior, max_iter=15, random_state=seed)
    filled = imp.fit_transform(t[IMPUTE_COLS])
    out = t.copy()
    out[IMPUTE_COLS] = filled
    out["AMH"] = np.exp(out["logAMH"])
    return out


IMPUTE_ORDER = ["COC", "Endo", "logAMH"]      # columns with missing values (MII, age complete)
IMPUTE_BASE = ["age", "MII", "day"]


def _fit_draw(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """OLS fit with one posterior draw of the coefficients and residual SD (proper MI)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = max(len(y) - X1.shape[1], 1)
    sigma = np.sqrt(resid @ resid / rng.chisquare(dof))
    cov = sigma**2 * np.linalg.pinv(X1.T @ X1)
    beta_draw = rng.multivariate_normal(beta, cov)
    return beta_draw, sigma


def impute_fold(train: pd.DataFrame, test: pd.DataFrame, seed: int, outcome: str = "LB"
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leak-free stochastic regression imputation for one fold (proper, one draw).

    Each incomplete variable (oocytes, endometrium, log AMH, in this order) is regressed on
    age, mature oocytes, day and the variables already completed. Models are fitted on the
    *training* patients only:
    - training patients: the outcome is an extra predictor (recommended when the completed
      data are used to fit the outcome model; ``outcome`` is the one being modelled);
    - test patients: separate models *without* the outcome, so a test outcome never
      influences its own prediction.
    """
    rng = np.random.default_rng(seed)
    tr, te = add_log_amh(train).reset_index(drop=True), add_log_amh(test).reset_index(drop=True)
    raw = tr.copy()                                  # observed training values only
    for with_y, target in ((True, tr), (False, te)):
        done = list(IMPUTE_BASE)
        src = tr if with_y else raw.copy()           # training rows provide the fits
        for col in IMPUTE_ORDER:
            preds = done + ([outcome] if with_y else [])
            obs = src[col].notna().to_numpy()
            if obs.sum() < 10:
                done.append(col)
                continue
            beta, sigma = _fit_draw(src.loc[obs, preds].to_numpy(float),
                                    src.loc[obs, col].to_numpy(float), rng)
            miss = target[col].isna().to_numpy()
            if miss.any():
                X1 = np.column_stack([np.ones(miss.sum()),
                                      target.loc[miss, preds].to_numpy(float)])
                target.loc[miss, col] = X1 @ beta + rng.normal(0, sigma, miss.sum())
            if not with_y:
                # keep the training copy complete for the next column's no-outcome fit
                m2 = src[col].isna().to_numpy()
                if m2.any():
                    X2 = np.column_stack([np.ones(m2.sum()), src.loc[m2, preds].to_numpy(float)])
                    src.loc[m2, col] = X2 @ beta + rng.normal(0, sigma, m2.sum())
            done.append(col)
    for d in (tr, te):
        d["AMH"] = np.exp(d["logAMH"])
    return tr, te


def attach_patient_covariates(embryos: pd.DataFrame, transfers_imp: pd.DataFrame
                              ) -> pd.DataFrame:
    cols = ["patient", "age", "logAMH", "AMH", "Endo", "COC", "MII", "day"]
    e = embryos.drop(columns=[c for c in cols if c != "patient" and c in embryos.columns])
    return e.merge(transfers_imp[cols].drop_duplicates("patient"), on="patient", how="left")


# --------------------------------------------------------------------------- model
@dataclass
class EmbryoModel:
    feature_set: str = "full"
    kind: str = "ridge"          # "ridge" or "gbm"
    C: float = 1.0
    n_knots: int = 3
    seed: int = 0
    knots_: np.ndarray = field(default=None, repr=False)
    mean_: pd.Series = field(default=None, repr=False)
    sd_: pd.Series = field(default=None, repr=False)
    est_: object = field(default=None, repr=False)

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        X = design(df, self.feature_set, self.knots_)
        return ((X - self.mean_) / self.sd_).to_numpy()

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> EmbryoModel:
        self.knots_ = rcs_knots(df["age"], self.n_knots)
        X = design(df, self.feature_set, self.knots_)
        self.mean_ = X.mean()
        self.sd_ = X.std(ddof=0).replace(0, 1.0)
        Xs = ((X - self.mean_) / self.sd_).to_numpy()
        if self.kind == "ridge":
            self.est_ = LogisticRegression(C=self.C, max_iter=5000).fit(Xs, y)
        elif self.kind == "gbm":
            self.est_ = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=30,
                l2_regularization=1.0, random_state=self.seed).fit(Xs, y)
        else:
            raise ValueError(self.kind)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.est_.predict_proba(self._X(df))[:, 1]

    def linear_predictor(self, df: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.predict(df), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))


def choose_C(transfers_imp: pd.DataFrame, cfg: dict, feature_set: str = "full") -> float:
    """Ridge penalty chosen once by 10-fold CV on the log-loss (kept fixed afterwards)."""
    knots = rcs_knots(transfers_imp["age"], cfg["model"]["age_spline_knots"])
    X = design(transfers_imp, feature_set, knots)
    Xs = ((X - X.mean()) / X.std(ddof=0).replace(0, 1)).to_numpy()
    cv = StratifiedKFold(cfg["model"]["cv_folds"], shuffle=True, random_state=cfg["seed"])
    m = LogisticRegressionCV(Cs=np.logspace(-3, 2, cfg["model"]["ridge_Cs"]), cv=cv,
                             scoring="neg_log_loss", max_iter=5000)
    m.fit(Xs, transfers_imp["LB"].to_numpy())
    return float(m.C_[0])


def choose_C_inner(train_imp: pd.DataFrame, feature_set: str, n_knots: int, seed: int,
                   outcome: str = "LB", folds: int = 3) -> float:
    """Nested search of the ridge penalty on a training fold only (log-loss, 3-fold)."""
    knots = rcs_knots(train_imp["age"], n_knots)
    X = design(train_imp, feature_set, knots)
    Xs = ((X - X.mean()) / X.std(ddof=0).replace(0, 1)).to_numpy()
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    m = LogisticRegressionCV(Cs=INNER_CS, cv=cv, scoring="neg_log_loss", max_iter=5000)
    m.fit(Xs, train_imp[outcome].to_numpy())
    return float(m.C_[0])


def patient_folds(patients: np.ndarray, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold id per *unique original patient* (stratified on outcome)."""
    uniq = pd.DataFrame({"patient": patients, "y": y}).drop_duplicates("patient")
    folds = np.empty(len(uniq), dtype=int)
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for f, (_, idx) in enumerate(skf.split(uniq, uniq["y"])):
        folds[idx] = f
    fmap = dict(zip(uniq["patient"], folds, strict=True))
    return np.array([fmap[p] for p in patients])


def crossfit(transfers: pd.DataFrame, embryos: pd.DataFrame, make_model, k: int, seed: int,
             outcome: str = "LB", impute: bool = True, C: float | None = None,
             feature_set: str = "full", n_knots: int = 3
             ) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Cross-fitted predictions for transfers and for all embryos of the same patients.

    ``transfers`` and ``embryos`` are *not* imputed: imputation is done inside each fold.
    ``make_model(C)`` returns an unfitted model. ``C=None`` -> nested choice of the penalty
    in each training fold; otherwise the given C is used (bootstrap replicates).
    ``transfers`` may contain repeated patients (bootstrap); folds are defined on the
    original patient id, so copies of a patient always fall in the same fold.
    Returns (p_transfers, p_embryos, C used in each fold).
    """
    t = transfers.reset_index(drop=True)
    y = t[outcome].to_numpy()
    folds = patient_folds(t["patient"].to_numpy(), y, k, seed)
    fold_of = dict(zip(t["patient"], folds, strict=False))
    e = embryos.reset_index(drop=True)
    emb_fold = e["patient"].map(fold_of).to_numpy()
    p_tr = np.full(len(t), np.nan)
    p_emb = np.full(len(e), np.nan)
    Cs = []
    for f in range(k):
        train, test = folds != f, folds == f
        if impute:
            tr, te = impute_fold(t[train], t[test], seed + 17 * f, outcome)
        else:
            tr, te = add_log_amh(t[train]), add_log_amh(t[test])
        c_f = C if C is not None else choose_C_inner(tr, feature_set, n_knots, seed + f,
                                                     outcome)
        Cs.append(c_f)
        m = make_model(c_f).fit(tr, y[train])
        if test.any():
            p_tr[test] = m.predict(te)
        sel = emb_fold == f
        if sel.any():
            ef = attach_patient_covariates(e[sel], te.drop_duplicates("patient"))
            p_emb[sel] = m.predict(ef)
    return p_tr, p_emb, Cs


# --------------------------------------------------------------------------- performance
def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration intercept (calibration-in-the-large, slope fixed at 1) and slope."""
    import statsmodels.api as sm

    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    intercept = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                       offset=lp).fit().params[0]
    return float(intercept), float(slope)


def performance(y: np.ndarray, p: np.ndarray) -> dict:
    a, b = calibration(y, p)
    return {"auroc": float(roc_auc_score(y, p)), "cal_intercept": a, "cal_slope": b,
            "brier": float(np.mean((p - y) ** 2)), "mean_pred": float(p.mean()),
            "obs_rate": float(y.mean())}


def optimism_corrected(transfers_imp: pd.DataFrame, make_model, n_boot: int, seed: int
                       ) -> dict:
    """Harrell's bootstrap optimism correction of AUROC and calibration slope."""
    rng = np.random.default_rng(seed)
    y = transfers_imp["LB"].to_numpy()
    full = make_model().fit(transfers_imp, y)
    app = performance(y, full.predict(transfers_imp))
    opt = {"auroc": [], "cal_slope": []}
    n = len(transfers_imp)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = transfers_imp.iloc[idx]
        if b["LB"].nunique() < 2:
            continue
        m = make_model().fit(b, b["LB"].to_numpy())
        pb, po = performance(b["LB"].to_numpy(), m.predict(b)), performance(y, m.predict(
            transfers_imp))
        for key in opt:
            opt[key].append(pb[key] - po[key])
    return {"apparent_auroc": app["auroc"],
            "corrected_auroc": app["auroc"] - float(np.mean(opt["auroc"])),
            "apparent_slope": app["cal_slope"],
            "corrected_slope": app["cal_slope"] - float(np.mean(opt["cal_slope"]))}
