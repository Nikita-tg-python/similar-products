"""Qdrant vs pgvector on the same vectors: latency (p50/p95) and recall@10 against exact search.

    python -m eval.compare_backends [--runs 20] [--seed 0] [--no-readme]

- Queries: 10 random products (seed) searched by their own vector, themselves excluded, and 10 text
  queries embedded with the same model. No LLM: the run must be reproducible.
- Two scenarios: no filters, and a selective filter (vegan, no nuts in allergens/traces,
  sugars <= 5 g/100 g) - where HNSW engines differ most.
- Ground truth: exact cosine over every vector (numpy), restricted to the rows that pass the
  filter - computed here independently of both stores' filter code.
- Latency: `store.search()` as the API calls it, measured from this client (so it includes the
  network round trip: HTTP or gRPC for Qdrant, the Postgres protocol for pgvector); warm-up runs
  first, then the backends are interleaved run by run so drift hits all of them alike.
- Writes eval/comparison_report.md and the table between the markers in README.md.
"""

import argparse
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

from app import db
from app.config import settings
from app.embeddings import get_embedder
from app.schemas import SearchFilters
from app.vectorstores.base import SEARCH_EF, VectorStore
from app.vectorstores.pgvector_store import PgvectorStore
from app.vectorstores.qdrant_store import HNSW, QdrantStore

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "eval" / "comparison_report.md"
README = ROOT / "README.md"
MARK_START, MARK_END = "<!-- comparison:start -->", "<!-- comparison:end -->"
K = 10
WARMUP = 3

TEXT_QUERIES = [
    "chocolate biscuits",
    "sparkling mineral water",
    "greek yogurt",
    "breakfast cereals",
    "hazelnut chocolate spread",
    "smoked salmon",
    "tomato pasta sauce",
    "dark chocolate",
    "orange juice",
    "potato chips crisps",
]
SCENARIOS = {
    "no filters": SearchFilters(),
    "vegan, no nuts, sugar <= 5 g": SearchFilters(
        exclude_allergens=["nuts"], vegan_only=True, sugars_max=5
    ),
}


@dataclass
class Query:
    label: str
    vector: np.ndarray
    exclude_code: str | None = None


@dataclass
class Result:
    latencies_ms: list[float] = field(default_factory=list)
    recalls: list[float] = field(default_factory=list)


def load_catalog(conn) -> tuple[list[str], np.ndarray, list[dict]]:
    """Every vector with the filter fields, straight from Postgres (identical to Qdrant's copy:
    scripts/check_stores.py)."""
    rows = conn.execute(
        "SELECT code, embedding, allergens_tags, traces_tags, vegan, nutriscore_grade, "
        "sugars_100g, ingredients_text IS NOT NULL "
        "FROM products WHERE embedding IS NOT NULL ORDER BY code"
    ).fetchall()
    codes = [r[0] for r in rows]
    vectors = np.stack([r[1].to_numpy() for r in rows]).astype(np.float32)
    fields = [
        {
            "allergens": set(r[2]) | set(r[3]),
            "vegan": r[4],
            "nutriscore": r[5],
            "sugars": r[6],
            "ingredients_known": r[7],
        }
        for r in rows
    ]
    return codes, vectors, fields


def mask_for(f: SearchFilters, fields: list[dict]) -> np.ndarray:
    """The filter's meaning, re-implemented here as the independent reference."""
    banned = set(f.exclude_allergens)
    grades = f.nutriscore_grades
    return np.array(
        [
            not banned & row["allergens"]
            and (not banned or row["ingredients_known"])
            and (not f.vegan_only or row["vegan"] == "yes")
            and (grades is None or row["nutriscore"] in grades)
            and (
                f.sugars_max is None
                or (row["sugars"] is not None and row["sugars"] <= f.sugars_max)
            )
            for row in fields
        ]
    )


def exact_top_k(query: Query, vectors: np.ndarray, codes: list[str], mask: np.ndarray) -> set[str]:
    scores = vectors @ query.vector  # all vectors are L2-normalized: dot product == cosine
    scores = np.where(mask, scores, -np.inf)
    if query.exclude_code is not None:
        scores[codes.index(query.exclude_code)] = -np.inf
    top = np.argpartition(-scores, K)[:K]
    return {codes[i] for i in top if np.isfinite(scores[i])}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def run(runs: int, seed: int) -> tuple[str, str]:
    """(full report, README table)."""
    embedder = get_embedder()
    with db.connect(settings.database_url) as conn:
        pgvector = PgvectorStore(conn)  # also registers the vector type for load_catalog
        codes, vectors, fields = load_catalog(conn)
        rng = random.Random(seed)
        product_codes = rng.sample(codes, 10)
        by_code = dict(zip(codes, vectors, strict=True))
        queries = [Query(f"product {c}", by_code[c], exclude_code=c) for c in product_codes]
        text_vectors = embedder.encode(TEXT_QUERIES)
        queries += [
            Query(f"text {t!r}", v) for t, v in zip(TEXT_QUERIES, text_vectors, strict=True)
        ]

        # Both Qdrant transports: gRPC is what the API uses, HTTP shows what the transport costs.
        stores: dict[str, VectorStore] = {
            name: QdrantStore(
                QdrantClient(
                    url=settings.qdrant_url,
                    prefer_grpc=grpc,
                    grpc_port=settings.qdrant_grpc_port,
                    timeout=30,
                ),
                settings.qdrant_collection,
            )
            for name, grpc in (("qdrant (gRPC)", True), ("qdrant (HTTP)", False))
        }
        stores["pgvector"] = pgvector
        info = stores["qdrant (gRPC)"].client.get_collection(settings.qdrant_collection)

        results: dict[tuple[str, str], Result] = {}
        selectivity: dict[str, float] = {}
        for scenario, filters in SCENARIOS.items():
            mask = mask_for(filters, fields)
            selectivity[scenario] = float(mask.mean())
            truth = [exact_top_k(q, vectors, codes, mask) for q in queries]
            for name, store in stores.items():
                result = results.setdefault((scenario, name), Result())
                for q, expected in zip(queries, truth, strict=True):
                    hits = {h.code for h in store.search(q.vector, K, q.exclude_code, filters)}
                    result.recalls.append(len(hits & expected) / len(expected))
                    for _ in range(WARMUP):
                        store.search(q.vector, K, q.exclude_code, filters)
            for _ in range(runs):
                for q in queries:
                    for name, store in stores.items():
                        started = time.perf_counter()
                        store.search(q.vector, K, q.exclude_code, filters)
                        ms = (time.perf_counter() - started) * 1000
                        results[(scenario, name)].latencies_ms.append(ms)

    detail_rows = [
        f"| {scenario} | {q.label} | "
        + " | ".join(f"{results[(scenario, n)].recalls[i]:.1f}" for n in stores)
        + " |"
        for scenario in SCENARIOS
        for i, q in enumerate(queries)
    ]

    table = [
        "| Сценарій | Backend | p50, мс | p95, мс | recall@10 (середній) | recall@10 (мін.) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for (scenario, name), r in results.items():
        table.append(
            f"| {scenario} | {name} | {percentile(r.latencies_ms, 50):.2f} | "
            f"{percentile(r.latencies_ms, 95):.2f} | "
            f"{np.mean(r.recalls):.3f} | {min(r.recalls):.1f} |"
        )

    env = (
        f"{len(codes)} векторів × {vectors.shape[1]}, cosine; HNSW в обох `m={HNSW.m}`, "
        f"`ef_construct={HNSW.ef_construct}`, пошук `ef={SEARCH_EF}`; Qdrant: "
        f"{info.points_count} точок, indexed_vectors_count={info.indexed_vectors_count} "
        "(лічильник Qdrant: менше за точки — малий сегмент шукається перебором, більше — після "
        "перезапису ще не прибрані старі версії); "
        f"{len(queries)} запитів × {runs} прогонів після {WARMUP} прогрівних; "
        f"{platform.machine()}, {platform.system()} (Docker), seed {seed}"
    )
    sel = "; ".join(f"«{s}» проходить {v:.1%} продуктів" for s, v in selectivity.items())
    report = "\n".join(
        [
            "# Qdrant vs pgvector",
            "",
            "Згенеровано `make compare` (`eval/compare_backends.py`). Не редагувати вручну.",
            "",
            f"Умови: {env}. Селективність фільтрів: {sel}.",
            "",
            *table,
            "",
            "recall@10 — частка точного top-10 (brute-force косинус у numpy по всіх векторах,",
            "що проходять фільтр), яку повернула база. Латентність — виклик `store.search()`",
            "з клієнта, разом з мережею.",
            "",
            "**qdrant (HTTP)**: сталі ~40 мс поверх — це транспорт, а не пошук. Той самий",
            "REST-запит через `httpx` (на ньому REST-клієнт Qdrant) займає ~46 мс, через",
            "`http.client` — ~3 мс; найімовірніше, Nagle + delayed ACK на невеликих POST.",
            "Тому API ходить у Qdrant по gRPC.",
            "",
            "## Recall@10 по запитах",
            "",
            "| Сценарій | Запит | " + " | ".join(stores) + " |",
            "| --- | --- | " + " | ".join("---:" for _ in stores) + " |",
            *detail_rows,
            "",
        ]
    )
    return report, "\n".join([f"Умови: {env}. Селективність: {sel}.", "", *table])


def update_readme(table: str) -> None:
    text = README.read_text(encoding="utf-8")
    start, end = text.index(MARK_START) + len(MARK_START), text.index(MARK_END)
    README.write_text(f"{text[:start]}\n{table}\n{text[end:]}", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-readme", action="store_true")
    args = parser.parse_args()
    report, table = run(args.runs, args.seed)
    REPORT.write_text(report, encoding="utf-8")
    if not args.no_readme:
        update_readme(table)
    print(report)


if __name__ == "__main__":
    main()
