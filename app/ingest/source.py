"""Open Food Facts rows, streamed from the Hugging Face parquet or read from a local sample, and
normalized.

`food.parquet` (7.86 GB, ~4.77M rows, 145 columns) is ordered by barcode and split into ~4.7k row
groups of ~1k rows. We never download it: pyarrow reads the footer once (~100 MB, the one fixed
cost) and then, per row group, only the byte ranges of the ~16 columns we need (~0.2 MB instead
of ~2 MB).
Why not `datasets.load_dataset(..., streaming=True)`: measured on 2026-09-30, it did not yield even
5k rows in 20 minutes, while this reader does ~50k useful rows in a few minutes.

Row groups are picked in a seeded random order across the whole file, not the first N: the file is
sorted by barcode, so the first rows are one narrow barcode range (mostly legacy/odd codes).

The live schema differs from the dataset card: `product_name` and `ingredients_text` are lists of
`{lang, text}`, `nutriments` is a list of `{name, 100g, ...}`, and the vegan status lives in
`ingredients_analysis_tags` (`en:vegan` / `en:non-vegan` / `en:maybe-vegan` /
`en:vegan-status-unknown`). `normalize` turns a raw row into a flat `Product`.
"""

import html
import random
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

HF_PATH = "datasets/openfoodfacts/product-database/food.parquet"

SOURCE_COLUMNS = [
    "code",
    "product_name",
    "brands",
    "lang",
    "categories_tags",
    "ingredients_text",
    "allergens_tags",
    "traces_tags",
    "ingredients_analysis_tags",
    "nutriments",
    "nutriscore_grade",
    "nova_group",
    "environmental_score_grade",
    "completeness",
    "countries_tags",
    "obsolete",
]

Vegan = Literal["yes", "no", "maybe", "unknown"]

VEGAN_TAGS: dict[str, Vegan] = {
    "en:vegan": "yes",
    "en:non-vegan": "no",
    "en:maybe-vegan": "maybe",
}
NUTRISCORE_GRADES = {"a", "b", "c", "d", "e"}
ENV_SCORE_GRADES = {"a-plus", "a", "b", "c", "d", "e", "f"}

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


class Product(BaseModel):
    code: str
    product_name: str
    brands: str | None = None
    lang: str | None = None
    categories_tags: list[str] = []
    ingredients_text: str | None = None
    allergens_tags: list[str] = []
    traces_tags: list[str] = []
    vegan: Vegan = "unknown"
    nutriscore_grade: str | None = None
    sugars_100g: float | None = None
    nova_group: int | None = None
    environmental_score_grade: str | None = None
    completeness: float | None = None
    countries_tags: list[str] = []


def _clean(text: str | None) -> str | None:
    """OFF texts carry allergen markup (`<span class="allergen">`), entities and \\r\\n."""
    if not text:
        return None
    text = _SPACE_RE.sub(" ", html.unescape(_TAG_RE.sub("", text))).strip()
    return text or None


def pick_text(entries: list[dict[str, Any]] | None, lang: str | None) -> str | None:
    """English first (the embedding text and the demo queries are English), then the product's own
    language, then `main`, then whatever is there."""
    texts = {e.get("lang"): _clean(e.get("text")) for e in entries or []}
    texts = {k: v for k, v in texts.items() if v}
    for key in ("en", lang, "main"):
        if key in texts:
            return texts[key]
    return next(iter(texts.values()), None)


def vegan_status(analysis_tags: list[str] | None) -> Vegan:
    for tag in analysis_tags or []:
        if tag in VEGAN_TAGS:
            return VEGAN_TAGS[tag]
    return "unknown"


def sugars_per_100g(nutriments: list[dict[str, Any]] | None) -> float | None:
    for n in nutriments or []:
        if n.get("name") == "sugars":
            value = n.get("100g")
            # a handful of rows have typos like 320 g/100 g or negatives: unknown, not a fact
            if value is None or not 0 <= value <= 100:
                return None
            return float(value)
    return None


def normalize(row: dict[str, Any]) -> Product | None:
    """A flat `Product`, or None when the row is useless for similarity search: no barcode, no
    name, obsolete, or nothing to describe it by (neither categories nor ingredients)."""
    code = (row.get("code") or "").strip()
    lang = row.get("lang") or None
    name = pick_text(row.get("product_name"), lang)
    ingredients = pick_text(row.get("ingredients_text"), lang)
    categories = list(row.get("categories_tags") or [])
    if not code or not name or row.get("obsolete") or not (categories or ingredients):
        return None
    nutriscore = row.get("nutriscore_grade")
    env_grade = row.get("environmental_score_grade")
    return Product(
        code=code,
        product_name=name,
        brands=_clean(row.get("brands")),
        lang=lang,
        categories_tags=categories,
        ingredients_text=ingredients,
        allergens_tags=list(row.get("allergens_tags") or []),
        traces_tags=list(row.get("traces_tags") or []),
        vegan=vegan_status(row.get("ingredients_analysis_tags")),
        nutriscore_grade=nutriscore if nutriscore in NUTRISCORE_GRADES else None,
        sugars_100g=sugars_per_100g(row.get("nutriments")),
        nova_group=row.get("nova_group"),
        environmental_score_grade=env_grade if env_grade in ENV_SCORE_GRADES else None,
        completeness=row.get("completeness"),
        countries_tags=list(row.get("countries_tags") or []),
    )


class _CountingFile:
    """Read-only file wrapper that counts bytes actually pulled over the network."""

    def __init__(self, raw: Any, counter: list[int]) -> None:
        self._raw, self._counter = raw, counter

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        self._counter[0] += len(data)
        return data

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def iter_hf_tables(
    seed: int = 0, workers: int = 8, fetched: list[int] | None = None
) -> Iterator[pa.Table]:
    """Row groups (SOURCE_COLUMNS only) in a seeded random order, one table each, until the file is
    exhausted — the caller decides when to stop. `fetched[0]` accumulates bytes downloaded.
    Needs network, never used in tests."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    counter = fetched if fetched is not None else [0]

    def open_file() -> _CountingFile:
        # cache_type="none": fetch exactly the ranges pyarrow asks for; the default read-ahead
        # cache turned ~0.2 MB of needed columns into ~8 MB per row group
        return _CountingFile(fs.open(HF_PATH, cache_type="none"), counter)

    metadata = pq.read_metadata(open_file())
    groups = list(range(metadata.num_row_groups))
    random.Random(seed).shuffle(groups)

    def read_group(group: int) -> pa.Table:
        parquet = pq.ParquetFile(open_file(), metadata=metadata, pre_buffer=True)
        return parquet.read_row_group(group, columns=SOURCE_COLUMNS)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(groups), workers):
            # map keeps order, so the output is the same for the same seed
            yield from pool.map(read_group, groups[start : start + workers])


def iter_hf_rows(seed: int = 0, workers: int = 8) -> Iterator[dict[str, Any]]:
    for table in iter_hf_tables(seed, workers):
        yield from table.to_pylist()


def iter_local_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Raw rows from a local parquet written by `scripts/build_sample.py` (or the test fixture)."""
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1000):
        yield from batch.to_pylist()
