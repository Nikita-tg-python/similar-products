"""Check that Qdrant and pgvector hold the same vectors and agree on neighbours.

    python -m scripts.check_stores [--samples 200] [--show 3] [--seed 0]

Prints both counts, the mean top-10 overlap for random products (by the product's own vector,
itself excluded), and the two top-5 lists side by side for a few of them.
"""

import argparse
import random

import numpy as np

from app import db
from app.config import settings
from app.vectorstores import get_qdrant_store
from app.vectorstores.pgvector_store import PgvectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    qdrant = get_qdrant_store()
    with db.connect(settings.database_url) as conn:
        pgvector = PgvectorStore(conn)
        counts = {"qdrant": qdrant.count(), "pgvector": pgvector.count()}
        print(f"vectors: qdrant {counts['qdrant']}, pgvector {counts['pgvector']}")

        rows = conn.execute("SELECT code FROM products WHERE embedding IS NOT NULL")
        codes = [r[0] for r in rows]
        codes = random.Random(args.seed).sample(codes, min(args.samples, len(codes)))
        names = dict(conn.execute("SELECT code, product_name FROM products").fetchall())

        overlaps, max_diffs = [], []
        for i, code in enumerate(codes):
            vector = pgvector.get_vector(code)
            max_diffs.append(float(np.abs(vector - qdrant.get_vector(code)).max()))
            q = qdrant.search(vector, limit=10, exclude_code=code)
            p = pgvector.search(vector, limit=10, exclude_code=code)
            overlaps.append(len({h.code for h in q} & {h.code for h in p}) / 10)
            if i < args.show:
                print(f"\n{code} {names[code]!r}")
                print(f"  {'qdrant':<46} pgvector")
                for hq, hp in zip(q[:5], p[:5], strict=False):
                    left = f"{hq.score:.3f} {names[hq.code][:38]}"
                    print(f"  {left:<46} {hp.score:.3f} {names[hp.code][:38]}")

    print(
        f"\n{len(codes)} products: top-10 overlap mean {np.mean(overlaps):.2f}, "
        f"min {min(overlaps):.1f}; max |qdrant - pgvector| vector diff {max(max_diffs):.1e}"
    )


if __name__ == "__main__":
    main()
