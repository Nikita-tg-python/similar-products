"""Save >= N useful Open Food Facts rows locally without downloading the 7.86 GB parquet.

    python -m scripts.build_sample [--limit 50000] [--seed 0] [--workers 8] [--out PATH]

Rows are kept raw (SOURCE_COLUMNS, original nested schema) so ingest workers normalize them exactly
as they would a live stream; only rows that `normalize` accepts are written. Prints how many bytes
were actually downloaded — the proof that the file was not fetched whole.
"""

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from app.ingest.source import iter_hf_tables, normalize

DEFAULT_OUT = Path("data/sample_50k.parquet")
FULL_FILE_BYTES = 7_864_721_494


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=50_000, help="useful rows to keep")
    parser.add_argument("--seed", type=int, default=0, help="row group order")
    parser.add_argument("--workers", type=int, default=8, help="parallel row group reads")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".parquet.part")
    fetched = [0]
    started = time.monotonic()
    groups = read = kept = 0
    seen: set[str] = set()
    writer: pq.ParquetWriter | None = None
    try:
        for table in iter_hf_tables(args.seed, args.workers, fetched):
            groups += 1
            read += table.num_rows
            mask = []
            for row in table.to_pylist():
                product = normalize(row)
                useful = product is not None and product.code not in seen and kept < args.limit
                if useful:
                    seen.add(product.code)
                    kept += 1
                mask.append(useful)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table.filter(pa.array(mask)))
            if groups % args.workers == 0 or kept >= args.limit:
                print(
                    f"row groups {groups:>4}  read {read:>7}  kept {kept:>6}  "
                    f"downloaded {fetched[0] / 1e6:6.1f} MB  {time.monotonic() - started:6.1f}s",
                    flush=True,
                )
            if kept >= args.limit:
                break
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(args.out)
    print(
        f"saved {kept} rows to {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) "
        f"from {read} read; "
        f"downloaded {fetched[0] / 1e6:.1f} MB = {fetched[0] / FULL_FILE_BYTES:.1%} of the "
        f"{FULL_FILE_BYTES / 1e9:.2f} GB file in {time.monotonic() - started:.0f}s"
    )


if __name__ == "__main__":
    main()
