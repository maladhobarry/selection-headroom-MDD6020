"""Step A: load, repair, link and filter the raw data.

Known problems in the raw files (documented in the audit report):

- Decimal values typed in a German-locale spreadsheet were converted to dates, e.g.
  AMH "1.26" -> "Jän.26" (month.year) and endometrium "8.5" -> "08.Mai" (day.month).
  Both patterns are reversed exactly; values are flagged with the rule used.
- A few oocyte counts are typed as sums ("7+1"): the first number is kept (flagged).
- Image names are not zero-padded consistently ("0001_01", "198_01", "187_1"):
  a numeric (patient, embryo) key is used for every join.
- One image appears twice with different grades in the expert file (838_02): first kept.
- The variable "Fond" is undocumented and is not used.
- Byte-identical image files (MD5, found by the image QC of step 10) are shared by
  different patients: 422/503 (same transferred image, same age and AMH: probably one
  woman counted twice), 426/427 (same transferred image, different women: mislabelled
  file), 835/858 (all 10 images identical; 858 has no clinical row). Within patients 105
  and 175 one photo is used for two sibling embryos. These patients are kept in the
  pre-specified primary analysis, flagged (``dup_image``) and removed in sensitivity S10.

Grade coding (Kromp 2023, Table 1):
  EXP 0-4 = Gardner expansion 1-5; ICM/TE 0 = A, 1 = B, 2 = C, 3 = not defined (EXP 1-2).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Paths, get_logger

MONTHS = {
    "jan": 1, "jän": 1, "jaen": 1, "feb": 2, "mär": 3, "mar": 3, "mrz": 3, "maer": 3,
    "apr": 4, "mai": 5, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "okt": 10, "oct": 10, "nov": 11, "dez": 12, "dec": 12,
}
GRADE_LABELS = {0: "A", 1: "B", 2: "C", 3: "ND"}
ANNOTATOR_NAMES = [f"Annotator_{i}" for i in range(11)]
# Patients with an analysed image that is byte-identical to another image (see docstring;
# reproduced from data by images.image_qc -> results/tables/images_qc.csv).
DUPLICATE_IMAGE_PATIENTS = frozenset({105, 175, 422, 426, 427, 503, 835})


# --------------------------------------------------------------------------- parsing helpers
def parse_image_key(name: str) -> tuple[int, int]:
    """'0001_02.png' -> (1, 2); '187_1.png' -> (187, 1)."""
    m = re.fullmatch(r"\s*(\d+)_(\d+)\.png\s*", str(name))
    if not m:
        raise ValueError(f"Unexpected image name: {name!r}")
    return int(m.group(1)), int(m.group(2))


def _month(token: str) -> int | None:
    return MONTHS.get(token.strip().lower().rstrip("."))


def repair_excel_decimal(value) -> tuple[float, str]:
    """Undo a spreadsheet's decimal-to-date conversion.

    Returns (value, rule) with rule in {"missing", "numeric", "month_year", "day_month",
    "unparsed"}.

    - "Jän.26"  (month.YY)  <- typed "1.26"  -> 1 + 26/100
    - "10.Sep"  (DD.month)  <- typed "10.9"  -> 10 + 9/10  (months 10-12 -> .10/.11/.12)
    The day.month rule cannot tell "10.9" from "10.09" (difference <= 0.09); we keep the
    one-decimal reading, which is the usual entry format, and flag the rule.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan, "missing"
    s = str(value).strip().replace(",", ".")
    if s == "":
        return np.nan, "missing"
    try:
        return float(s), "numeric"
    except ValueError:
        pass
    m = re.fullmatch(r"([^\W\d_]+)\.(\d{1,2})", s)          # month.YY
    if m and _month(m.group(1)):
        return _month(m.group(1)) + int(m.group(2)) / 100, "month_year"
    m = re.fullmatch(r"(\d{1,2})\.([^\W\d_]+)", s)          # DD.month
    if m and _month(m.group(2)):
        mo = _month(m.group(2))
        return int(m.group(1)) + mo / (100 if mo >= 10 else 10), "day_month"
    return np.nan, "unparsed"


def parse_count(value) -> tuple[float, str]:
    """'7+1' -> (7, 'sum_first'); '12' -> (12, 'numeric')."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan, "missing"
    s = str(value).strip()
    if s == "":
        return np.nan, "missing"
    if re.fullmatch(r"\d+", s):
        return float(s), "numeric"
    m = re.fullmatch(r"(\d+)\s*\+\s*(\d+)", s)
    if m:
        return float(m.group(1)), "sum_first"
    return np.nan, "unparsed"


# --------------------------------------------------------------------------- loading
def load_clinical(paths: Paths) -> pd.DataFrame:
    df = pd.read_csv(paths.raw / "Clincial_annotations.csv", sep=";", encoding="latin-1",
                     dtype=str)
    df.columns = [c.strip() for c in df.columns]
    keys = df["Image"].map(parse_image_key)
    df["patient"] = [k[0] for k in keys]
    df["embryo"] = [k[1] for k in keys]
    return df


def load_expert_grades(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expert ('silver') grades of all images. Returns (grades, conflicting_duplicates)."""
    g = pd.read_csv(paths.raw / "complete.csv", header=None, skipinitialspace=True,
                    names=["Image", "EXP", "ICM", "TE"])
    keys = g["Image"].map(parse_image_key)
    g["patient"] = [k[0] for k in keys]
    g["embryo"] = [k[1] for k in keys]
    dup = g[g.duplicated(["patient", "embryo"], keep=False)]
    g = g.drop_duplicates(["patient", "embryo"], keep="first").reset_index(drop=True)
    return g, dup


def load_annotators(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long table of individual annotations on the test set + consensus ('gold')."""
    rows = []
    for name in ANNOTATOR_NAMES + ["Gardner_Expert"]:
        a = pd.read_csv(paths.raw / "annotations" / f"{name}.csv", header=None,
                        names=["Image", "EXP", "ICM", "TE"])
        a["annotator"] = name
        rows.append(a)
    ann = pd.concat(rows, ignore_index=True)
    gold = pd.read_csv(paths.raw / "annotations" / "test_rev.csv", header=None,
                       names=["Image", "EXP", "ICM", "TE"])
    for d in (ann, gold):
        keys = d["Image"].map(parse_image_key)
        d["patient"] = [k[0] for k in keys]
        d["embryo"] = [k[1] for k in keys]
        for c in ("EXP", "ICM", "TE"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return ann, gold


# --------------------------------------------------------------------------- building
@dataclass
class CleanData:
    transfers: pd.DataFrame   # one row per single-embryo fresh transfer (patient level)
    embryos: pd.DataFrame     # all imaged blastocysts of those patients
    flow: pd.DataFrame        # participant flow
    flags: pd.DataFrame       # repaired / flagged values


def clean_clinical(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    flags = []
    for col, fn in (("AMH", repair_excel_decimal), ("Endo", repair_excel_decimal),
                    ("COC", parse_count), ("MII", parse_count)):
        parsed = out[col].map(fn)
        out[col + "_raw"] = out[col]
        out[col] = [p[0] for p in parsed]
        out[col + "_rule"] = [p[1] for p in parsed]
        for img, raw, val, rule in zip(out["Image"], out[col + "_raw"], out[col],
                                       out[col + "_rule"], strict=True):
            if rule not in ("numeric", "missing"):
                flags.append({"Image": img, "variable": col, "raw": raw, "value": val,
                              "rule": rule})
    for col in ("d", "Age", "EXP_silver", "ICM_silver", "TE_silver", "SS", "HA", "LB"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.rename(columns={"d": "day", "Age": "age"})
    return out, pd.DataFrame(flags)


def build(paths: Paths, cfg: dict) -> CleanData:
    log = get_logger("02_cleaning", paths)
    clin_raw = load_clinical(paths)
    grades, dups = load_expert_grades(paths)
    clin, flags = clean_clinical(clin_raw)

    flow = [("Fresh transfers in clinical file (rows)", len(clin), clin["patient"].nunique())]

    # Single-embryo transfers only (a double transfer gives one outcome for two embryos).
    n_tr = clin.groupby("patient").size()
    det = n_tr[n_tr > 1].index
    clin = clin[~clin["patient"].isin(det)].copy()
    flow.append(("Excluded: double-embryo transfers", -len(det) * 2, -len(det)))

    # Link to expert grades.
    clin = clin.merge(grades[["patient", "embryo", "EXP", "ICM", "TE"]],
                      on=["patient", "embryo"], how="left", validate="one_to_one")
    no_grade = clin["EXP"].isna()
    flow.append(("Excluded: transferred image without expert grade", -int(no_grade.sum()),
                 -int(no_grade.sum())))
    clin = clin[~no_grade].copy()
    for c in ("EXP", "ICM", "TE"):
        clin[c] = clin[c].astype(int)
    flow.append(("Single-embryo transfers analysed (model, step B)", len(clin),
                 clin["patient"].nunique()))

    # Sibling blastocysts of these patients (photographed for transfer or vitrification).
    emb = grades[grades["patient"].isin(clin["patient"])].copy()
    emb = emb.merge(clin[["patient", "embryo"]].assign(transferred=1),
                    on=["patient", "embryo"], how="left")
    emb["transferred"] = emb["transferred"].fillna(0).astype(int)
    n_blast = emb.groupby("patient").size().rename("n_blast")
    clin = clin.merge(n_blast, left_on="patient", right_index=True, how="left")
    patient_cols = ["patient", "day", "age", "AMH", "Endo", "COC", "MII", "SS", "HA", "LB",
                    "n_blast"]
    emb = emb.merge(clin[patient_cols], on="patient", how="left")

    d5 = clin["day"] == cfg["population"]["main_day"]
    flow.append(("Day-5 single-embryo transfers (steps C, F)", int(d5.sum()),
                 int(clin.loc[d5, "patient"].nunique())))
    choice = d5 & (clin["n_blast"] >= cfg["population"]["min_blastocysts"])
    flow.append(("  ... with >= 2 blastocysts (strategies, E1-E4)", int(choice.sum()),
                 int(choice.sum())))
    flow.append(("  ... with 1 blastocyst (no-choice group, step F)",
                 int((d5 & (clin["n_blast"] == 1)).sum()),
                 int((d5 & (clin["n_blast"] == 1)).sum())))

    clin["choice_set"] = choice.astype(int)
    emb = emb.merge(clin[["patient", "choice_set"]], on="patient", how="left")

    # Gardner rank used by the "best by Gardner" strategy (lower = better).
    emb = add_gardner_rank(emb)
    rk = emb.loc[emb["transferred"] == 1, ["patient", "gardner_rank"]]
    clin = clin.merge(rk.rename(columns={"gardner_rank": "gardner_rank_transferred"}),
                      on="patient", how="left")

    clin["dup_image"] = clin["patient"].isin(DUPLICATE_IMAGE_PATIENTS).astype(int)
    emb["dup_image"] = emb["patient"].isin(DUPLICATE_IMAGE_PATIENTS).astype(int)
    flags = pd.concat([flags, clin.loc[clin["dup_image"] == 1, ["Image"]].assign(
        variable="image", value="byte-identical to another image", rule="flag_S10")],
        ignore_index=True)

    if len(dups):
        flags = pd.concat([flags, dups.assign(variable="grade_duplicate", rule="kept_first")
                           .rename(columns={"EXP": "value"})[["Image", "variable", "value",
                                                               "rule"]]], ignore_index=True)
    flow_df = pd.DataFrame(flow, columns=["step", "transfers_or_rows", "patients"])
    log.info("Transfers for model: %d (%d live births)", len(clin), int(clin["LB"].sum()))
    log.info("Choice set (day 5, >=2 blastocysts): %d patients", int(choice.sum()))
    return CleanData(transfers=clin.reset_index(drop=True), embryos=emb.reset_index(drop=True),
                     flow=flow_df, flags=flags)


def add_gardner_rank(emb: pd.DataFrame, order: tuple[str, ...] = ("ICM", "TE", "EXP")
                     ) -> pd.DataFrame:
    """Lexicographic Gardner rank within patient; 1 = best. Ties share the same rank.

    ICM and TE: A(0) > B(1) > C(2) > not defined(3). EXP: higher expansion is better.
    """
    out = emb.copy()
    # Each criterion takes values 0-4, so a base-10 composite preserves lexicographic order.
    key = np.zeros(len(out))
    for c in order:
        part = (4 - out[c]) if c == "EXP" else out[c]
        key = key * 10 + part.to_numpy()
    out["_k"] = key
    out["gardner_rank"] = out.groupby("patient")["_k"].rank(method="min").astype(int)
    return out.drop(columns="_k")


def save(data: CleanData, paths: Paths) -> None:
    data.transfers.to_csv(paths.processed / "transfers.csv", index=False)
    data.embryos.to_csv(paths.processed / "embryos.csv", index=False)
    data.flow.to_csv(paths.tables / "flow.csv", index=False)
    data.flags.to_csv(paths.tables / "repaired_values.csv", index=False)


def load_processed(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_csv(paths.processed / "transfers.csv"),
            pd.read_csv(paths.processed / "embryos.csv"))
