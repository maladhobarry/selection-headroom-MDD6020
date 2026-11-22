"""Step A: data-quality audit of the raw files (reported as supplementary material).

Every check returns a row of a table: what was checked, what was found, what is done.
"""

from __future__ import annotations

import pandas as pd

from .cleaning import (
    clean_clinical,
    load_annotators,
    load_clinical,
    load_expert_grades,
)
from .config import Paths, get_logger


def run(paths: Paths) -> pd.DataFrame:
    log = get_logger("01_audit", paths)
    clin_raw = load_clinical(paths)
    grades, dups = load_expert_grades(paths)
    clin, flags = clean_clinical(clin_raw)
    ann, gold = load_annotators(paths)

    rows = []

    def add(check, found, action):
        rows.append({"check": check, "finding": found, "action": action})
        log.info("%s: %s", check, found)

    add("Clinical rows / patients", f"{len(clin)} / {clin['patient'].nunique()}",
        "Single-embryo transfers kept")
    n_tr = clin.groupby("patient").size()
    add("Double-embryo transfers", f"{int((n_tr > 1).sum())} patients",
        "Excluded (one outcome for two embryos)")
    names = clin_raw["Image"]
    irregular = (~names.str.fullmatch(r"\d{4}_\d{2}\.png")).sum()
    add("Image names not in NNNN_NN.png form", f"{int(irregular)} of {len(names)}",
        "Numeric (patient, embryo) key for all joins")
    add("Expert-grade file", f"{len(grades) + len(dups) // 2} rows, "
        f"{len(dups)} rows in conflicting duplicates",
        "First occurrence kept")
    merged = clin.merge(grades.drop(columns="Image"), on=["patient", "embryo"], how="left")
    add("Transferred images without expert grade",
        ", ".join(merged.loc[merged["EXP"].isna(), "Image"]) or "none", "Excluded")
    m = merged.dropna(subset=["EXP"])
    for c in ("EXP", "ICM", "TE"):
        agree = (m[f"{c}_silver"].astype(float) == m[c]).mean()
        add(f"{c}: clinical file vs expert-grade file", f"{agree:.1%} identical",
            "Expert-grade file used; clinical grades in sensitivity S4")
    for var in ("AMH", "Endo", "COC", "MII"):
        rules = clin[f"{var}_rule"].value_counts().to_dict()
        add(f"{var} parsing rules", "; ".join(f"{k}={v}" for k, v in rules.items()),
            "Reversed date conversion / first number of sums; flagged")
    add("Variable 'Fond'", "Not documented in Kromp et al. 2023 (Table 1)", "Not used")
    # Photo timing: expansion of siblings vs transferred embryo, by transfer day.
    e = grades.drop(columns="Image").merge(clin[["patient", "embryo", "day"]].assign(tr=1),
                     on=["patient", "embryo"], how="left")
    e["day"] = e.groupby("patient")["day"].transform("max")
    e = e.dropna(subset=["day"])
    e["tr"] = e["tr"].fillna(0)
    t = e.groupby(["day", "tr"])["EXP"].mean().unstack()
    for d in t.index:
        add(f"Mean expansion, day {int(d)}: siblings vs transferred",
            f"{t.loc[d, 0.0]:.2f} vs {t.loc[d, 1.0]:.2f}",
            "Day 4 excluded from strategy comparison (siblings likely photographed later)")
    add("Annotator files", f"{ann['annotator'].nunique()} files, "
        f"{ann[['EXP', 'ICM', 'TE']].isna().any(axis=1).sum()} incomplete annotations; "
        f"consensus on {len(gold)} images",
        "Incomplete criteria ignored per criterion")
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "audit.csv", index=False)
    flags.to_csv(paths.tables / "repaired_values.csv", index=False)
    return out
