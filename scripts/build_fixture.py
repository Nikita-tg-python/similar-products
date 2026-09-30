"""Cut the committed test fixture (~300 raw rows) out of the local sample, covering every case the
filters and search must handle.

    python -m scripts.build_fixture [--sample data/sample_50k.parquet] [--size 300] [--seed 0]

Each case in CASES gets up to PER_CASE rows first, the rest is filled at random; tests assert that
every case is present, so the fixture cannot silently lose one when it is rebuilt.
"""

import argparse
import random
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from app.ingest.source import Product, normalize

FIXTURE = Path("tests/fixtures/sample_products.parquet")
PER_CASE = 12

# EU claim "low sugar" is <= 5 g/100 g (Reg. (EC) 1924/2006); "high" follows the UK FSA
# front-of-pack threshold (> 22.5 g/100 g). The search thresholds themselves are fixed later
# (KAN-59/KAN-60).
LOW_SUGAR, HIGH_SUGAR = 5.0, 22.5
NUTS = {"en:nuts", "en:peanuts"}


def _is_biscuit(p: Product) -> bool:
    return bool({"en:biscuits", "en:cookies"} & set(p.categories_tags))


CASES: dict[str, Callable[[Product], bool]] = {
    "vegan_yes": lambda p: p.vegan == "yes",
    "vegan_no": lambda p: p.vegan == "no",
    "vegan_maybe": lambda p: p.vegan == "maybe",
    "vegan_unknown": lambda p: p.vegan == "unknown",
    **{f"nutriscore_{g}": (lambda p, g=g: p.nutriscore_grade == g) for g in "abcde"},
    "nutriscore_none": lambda p: p.nutriscore_grade is None,
    "sugar_low": lambda p: p.sugars_100g is not None and p.sugars_100g <= LOW_SUGAR,
    "sugar_high": lambda p: p.sugars_100g is not None and p.sugars_100g > HIGH_SUGAR,
    "sugar_unknown": lambda p: p.sugars_100g is None,
    **{
        f"allergen_{a.removeprefix('en:')}": (lambda p, a=a: a in p.allergens_tags)
        for a in ("en:nuts", "en:peanuts", "en:milk", "en:gluten", "en:eggs", "en:soybeans")
    },
    "no_allergens": lambda p: not p.allergens_tags,
    "nut_traces_only": lambda p: (
        bool(NUTS & set(p.traces_tags)) and not NUTS & set(p.allergens_tags)
    ),
    "biscuits": _is_biscuit,
    # the task's example: vegan biscuits without nuts, low in sugar — true matches, and near misses
    "demo_match": lambda p: (
        _is_biscuit(p)
        and p.vegan == "yes"
        and not NUTS & set(p.allergens_tags)
        and p.sugars_100g is not None
        and p.sugars_100g <= LOW_SUGAR
    ),
    "biscuits_vegan": lambda p: _is_biscuit(p) and p.vegan == "yes",
    "biscuits_with_nuts": lambda p: _is_biscuit(p) and bool(NUTS & set(p.allergens_tags)),
    "not_english": lambda p: p.lang != "en",
    "no_ingredients": lambda p: p.ingredients_text is None,
    "no_categories": lambda p: not p.categories_tags,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=Path, default=Path("data/sample_50k.parquet"))
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIXTURE)
    args = parser.parse_args()

    table = pq.read_table(args.sample)
    products = [normalize(row) for row in table.to_pylist()]
    order = list(range(len(products)))
    random.Random(args.seed).shuffle(order)

    chosen: list[int] = []
    taken: set[int] = set()
    for name, case in CASES.items():
        hits = [i for i in order if i not in taken and products[i] and case(products[i])]
        chosen += hits[:PER_CASE]
        taken.update(hits[:PER_CASE])
        total = sum(bool(p and case(p)) for p in products)
        print(f"{name:<22} {len(hits[:PER_CASE]):>3} (of {total})")
    for i in order:
        if len(chosen) >= args.size:
            break
        if i not in taken and products[i]:
            chosen.append(i)
            taken.add(i)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(pa.array(sorted(chosen))), args.out, compression="zstd")
    print(f"saved {len(chosen)} rows to {args.out} ({args.out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
