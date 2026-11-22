"""Unit tests of the data-repair rules (no data files needed)."""

import math

import numpy as np
import pandas as pd
import pytest

from headroom.cleaning import add_gardner_rank, parse_count, parse_image_key, repair_excel_decimal


@pytest.mark.parametrize("raw, value, rule", [
    ("2.6", 2.6, "numeric"),
    ("16.73", 16.73, "numeric"),
    ("Jän.26", 1.26, "month_year"),      # typed 1.26 -> January 1926
    ("Mär.79", 3.79, "month_year"),
    ("Dez.54", 12.54, "month_year"),
    ("Okt.64", 10.64, "month_year"),
    ("10.Sep", 10.9, "day_month"),       # typed 10.9 -> 10 September
    ("08.Mai", 8.5, "day_month"),
    ("04.Nov", 4.11, "day_month"),
    ("02.Dez", 2.12, "day_month"),
])
def test_repair_excel_decimal(raw, value, rule):
    v, r = repair_excel_decimal(raw)
    assert r == rule
    assert v == pytest.approx(value)


@pytest.mark.parametrize("raw", [None, float("nan"), "", "  "])
def test_repair_missing(raw):
    v, r = repair_excel_decimal(raw)
    assert math.isnan(v) and r == "missing"


def test_repair_unparsed():
    v, r = repair_excel_decimal("abc")
    assert math.isnan(v) and r == "unparsed"


def test_parse_count():
    assert parse_count("12") == (12.0, "numeric")
    assert parse_count("7+1") == (7.0, "sum_first")
    assert math.isnan(parse_count(None)[0])


@pytest.mark.parametrize("name, key", [("0001_02.png", (1, 2)), ("187_1.png", (187, 1)),
                                       ("198_01.png", (198, 1))])
def test_parse_image_key(name, key):
    assert parse_image_key(name) == key


def test_parse_image_key_rejects_garbage():
    with pytest.raises(ValueError):
        parse_image_key("image.png")


def test_gardner_rank_order_and_ties():
    # patient 1: AA exp 4 beats AA exp 3 beats AB; ND is worst.
    emb = pd.DataFrame({
        "patient": [1, 1, 1, 1, 2, 2],
        "EXP": [3, 2, 3, 1, 3, 3],
        "ICM": [0, 0, 0, 3, 1, 1],
        "TE": [0, 0, 1, 3, 0, 0],
    })
    r = add_gardner_rank(emb)["gardner_rank"].tolist()
    assert r[:4] == [1, 2, 3, 4]
    assert r[4:] == [1, 1]                  # exact tie shares rank 1


def test_gardner_rank_te_first():
    emb = pd.DataFrame({"patient": [1, 1], "EXP": [3, 3], "ICM": [0, 1], "TE": [1, 0]})
    assert add_gardner_rank(emb)["gardner_rank"].tolist() == [1, 2]
    assert add_gardner_rank(emb, order=("TE", "ICM", "EXP"))["gardner_rank"].tolist() == [2, 1]
    assert np.all(emb.columns == ["patient", "EXP", "ICM", "TE"])   # input not modified
