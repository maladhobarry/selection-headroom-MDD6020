"""Smoke tests of the article figures and of the fine-tuning pool (data integrity)."""

import pandas as pd
import pytest

from headroom import figures as F


def _sens():
    return pd.DataFrame({"code": ["S0", "S8", "S8b", "S10"],
                         "analysis": ["Primary", "HA", "SS", "No duplicates"],
                         "E1": [1.7, 3.0, 3.6, 1.7],
                         "E1_ci": ["[-0.4, 3.8]", "[0.9, 5.0]", "[1.1, 5.7]", "[-0.5, 3.6]"]})


def _img():
    rows = []
    for oc in ("SS", "HA", "LB"):
        rows.append({"config": "gardner_reference", "outcome": oc, "group": "choice",
                     "auroc": 0.55, "auroc_ci": "[0.50, 0.60]", "delta_auroc_vs_gardner": None,
                     "delta_ci": None})
        for i, name in enumerate(["a|pad|10", "b|pad|20 (PRIMARY)", "c|clahe|50"]):
            rows.append({"config": name, "outcome": oc, "group": "choice", "auroc": 0.56,
                         "auroc_ci": "[0.50, 0.62]", "delta_auroc_vs_gardner": 0.01 * i,
                         "delta_ci": "[-0.05, 0.06]"})
    return pd.DataFrame(rows)


def test_article_figures_render(tmp_path):
    flow = pd.DataFrame({
        "step": ["Fresh", "Excluded: double", "Excluded: no grade", "Single", "Day-5",
                 "choice", "no choice"],
        "transfers_or_rows": [754, -44, -1, 709, 605, 396, 209],
        "patients": [732, -22, -1, 709, 605, 396, 209]})
    F.flow_diagram(flow, tmp_path / "flow.png")
    F.sensitivity_forest(_sens(), tmp_path / "forest.png")
    F.specification_curve(_img(), tmp_path / "spec.png")
    for f in ("flow.png", "forest.png", "spec.png"):
        assert (tmp_path / f).stat().st_size > 5_000


def test_cascade_figure_renders(tmp_path):
    X = pytest.importorskip("headroom.explain")
    X.figure_cascade(_sens(), _img(), tmp_path / "cascade.png")
    assert (tmp_path / "cascade.png").stat().st_size > 5_000


def test_finetune_pool_drops_copies_of_transferred_images():
    from headroom.images import finetune_pool

    found = {(1, 1): "a", (1, 2): "b", (2, 1): "c", (2, 2): "d", (3, 1): "e"}
    qc = pd.DataFrame({"patient": [1, 1, 2, 2, 3], "embryo": [1, 2, 1, 2, 1],
                       "md5": ["x", "y", "x", "z", "z"]})
    pool, tag = finetune_pool(found, transferred={(1, 1)}, qc=qc)
    # (2,1) is a copy of the transferred image; (3,1) repeats (2,2)
    assert pool == [(1, 2), (2, 2)]
    assert len(tag) == 8
