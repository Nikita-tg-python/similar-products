"""Block until every host:port accepts TCP connections (k8s initContainers: the ordering docker
compose gets from depends_on).

    python -m scripts.wait_for postgres:5432 qdrant:6333 redis:6379 [--timeout 300]
"""

import argparse
import socket
import sys
import time


def wait_for(targets: list[str], timeout_s: float, poll_s: float = 2.0) -> list[str]:
    """The targets still closed when time ran out (empty list: all open)."""
    pending = list(targets)
    deadline = time.monotonic() + timeout_s
    while pending:
        for target in list(pending):
            host, port = target.rsplit(":", 1)
            try:
                with socket.create_connection((host, int(port)), timeout=poll_s):
                    print(f"{target} is up", flush=True)
                    pending.remove(target)
            except OSError:
                pass
        if pending and time.monotonic() >= deadline:
            return pending
        if pending:
            time.sleep(poll_s)
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="+", help="host:port")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if missing := wait_for(args.targets, args.timeout):
        sys.exit(f"still not reachable after {args.timeout:g}s: {', '.join(missing)}")


if __name__ == "__main__":
    main()
