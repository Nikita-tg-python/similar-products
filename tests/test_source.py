from pathlib import Path

import pytest

from app.ingest.source import SOURCE_COLUMNS, iter_local_rows, normalize, pick_text
from scripts.build_fixture import CASES, FIXTURE

ROOT = Path(__file__).resolve().parent.parent


def raw_row(**overrides):
    """A row shaped like the live HF parquet (not like the dataset card)."""
    row = {
        "code": "0000101209159",
        "product_name": [
            {"lang": "main", "text": "Pâte à tartiner"},
            {"lang": "en", "text": "Hazelnut spread"},
        ],
        "brands": "Bovetti",
        "lang": "fr",
        "categories_tags": ["en:spreads", "en:hazelnut-spreads"],
        "ingredients_text": [
            {"lang": "main", "text": "NOISETTES,\r\n sucre"},
            {"lang": "en", "text": '<span class="allergen">HAZELNUTS</span>, sugar &amp; cocoa'},
        ],
        "allergens_tags": ["en:nuts"],
        "traces_tags": ["en:milk"],
        "ingredients_analysis_tags": ["en:palm-oil-free", "en:maybe-vegan", "en:vegetarian"],
        "nutriments": [
            {"name": "energy-kj", "100g": 2524.0},
            {"name": "sugars", "100g": 32.0},
        ],
        "nutriscore_grade": "e",
        "nova_group": 3,
        "environmental_score_grade": "d",
        "completeness": 0.79,
        "countries_tags": ["en:france"],
        "obsolete": False,
    }
    return row | overrides


def test_normalize_flattens_the_live_schema():
    p = normalize(raw_row())
    assert p is not None
    assert p.product_name == "Hazelnut spread"
    assert p.ingredients_text == "HAZELNUTS, sugar & cocoa"  # markup and entities stripped
    assert p.vegan == "maybe"
    assert p.sugars_100g == 32.0
    assert p.nutriscore_grade == "e"
    assert p.allergens_tags == ["en:nuts"]
    assert p.traces_tags == ["en:milk"]


def test_pick_text_prefers_english_then_own_language_then_main():
    entries = [{"lang": "main", "text": "M"}, {"lang": "fr", "text": "F"}]
    assert pick_text(entries + [{"lang": "en", "text": "E"}], "fr") == "E"
    assert pick_text(entries, "fr") == "F"
    assert pick_text(entries, "de") == "M"
    assert pick_text([{"lang": "main", "text": "  "}], "fr") is None
    assert pick_text(None, "fr") is None


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["en:vegan", "en:vegetarian"], "yes"),
        (["en:non-vegan"], "no"),
        (["en:maybe-vegan"], "maybe"),
        (["en:vegan-status-unknown"], "unknown"),
        (None, "unknown"),
    ],
)
def test_vegan_status(tags, expected):
    assert normalize(raw_row(ingredients_analysis_tags=tags)).vegan == expected


@pytest.mark.parametrize(
    ("overrides", "field", "expected"),
    [
        ({"nutriscore_grade": "unknown"}, "nutriscore_grade", None),
        ({"nutriscore_grade": "not-applicable"}, "nutriscore_grade", None),
        ({"nutriments": [{"name": "sugars", "100g": 320.0}]}, "sugars_100g", None),
        ({"nutriments": [{"name": "sugars", "100g": None}]}, "sugars_100g", None),
        ({"nutriments": None}, "sugars_100g", None),
        ({"environmental_score_grade": "unknown"}, "environmental_score_grade", None),
        ({"ingredients_text": []}, "ingredients_text", None),
    ],
)
def test_unknown_values_become_none(overrides, field, expected):
    assert getattr(normalize(raw_row(**overrides)), field) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"code": ""},
        {"code": None},
        {"product_name": []},
        {"product_name": [{"lang": "main", "text": ""}]},
        {"obsolete": True},
        {"categories_tags": [], "ingredients_text": []},
    ],
)
def test_useless_rows_are_dropped(overrides):
    assert normalize(raw_row(**overrides)) is None


def test_fixture_is_raw_source_rows():
    rows = list(iter_local_rows(ROOT / FIXTURE))
    assert 200 <= len(rows) <= 500
    assert set(rows[0]) == set(SOURCE_COLUMNS)
    assert len({r["code"] for r in rows}) == len(rows)


def test_fixture_covers_every_interesting_case():
    products = [normalize(row) for row in iter_local_rows(ROOT / FIXTURE)]
    assert all(products), "the fixture holds only rows that normalize accepts"
    missing = [name for name, case in CASES.items() if not any(case(p) for p in products)]
    assert not missing


@pytest.mark.parametrize("placeholder", ["undefined", "Undefined ", "null"])
def test_placeholder_ingredients_are_unknown(placeholder):
    row = raw_row(ingredients_text=[{"lang": "main", "text": placeholder}])
    assert normalize(row).ingredients_text is None
