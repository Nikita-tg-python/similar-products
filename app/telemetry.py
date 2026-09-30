"""OpenTelemetry tracing for the API and the ingest worker.

TRACING_EXPORTER picks where spans go:
- tree (default): when a request or job finishes, its whole span tree is printed to stdout, one
  indented line per span with its duration and key attributes (the plain ConsoleSpanExporter
  prints one JSON blob per span, which hides the tree);
- console: the SDK's ConsoleSpanExporter as is (JSON per span);
- otlp: OTLP/HTTP to OTEL_EXPORTER_OTLP_ENDPOINT (Jaeger in the `observability` compose profile);
- memory: kept in `memory_exporter`, for tests;
- none: tracing off.
"""

import sys
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Any, TextIO

from opentelemetry import context, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.config import settings

tracer = trace.get_tracer("similar_products")
memory_exporter = InMemorySpanExporter()
_configured = False
_lock = threading.Lock()


class TreeConsoleExporter(SpanExporter):
    """Buffers finished spans per trace and prints the tree once its local root ends."""

    def __init__(self, out: TextIO = sys.stdout) -> None:
        self.out = out
        self._pending: dict[int, list[ReadableSpan]] = defaultdict(list)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            trace_id = span.context.trace_id
            with self._lock:
                self._pending[trace_id].append(span)
                if span.parent is not None and not span.parent.is_remote:
                    continue
                done = self._pending.pop(trace_id)
            self.out.write(render_tree(done))
            self.out.flush()
        return SpanExportResult.SUCCESS


def render_tree(spans: list[ReadableSpan]) -> str:
    children: dict[int | None, list[ReadableSpan]] = defaultdict(list)
    ids = {s.context.span_id for s in spans}
    for s in spans:
        parent = s.parent.span_id if s.parent is not None and s.parent.span_id in ids else None
        children[parent].append(s)
    lines: list[str] = []

    def walk(span: ReadableSpan, depth: int) -> None:
        ms = (span.end_time - span.start_time) / 1e6
        attrs = " ".join(
            f"{k}={v}"
            for k, v in (span.attributes or {}).items()
            if not k.startswith("http.") and not k.startswith("url.") and not k.startswith("net.")
        )
        status = " ERROR" if span.status.status_code == trace.StatusCode.ERROR else ""
        lines.append(
            f"{'  ' * depth}{span.name:<{max(1, 40 - 2 * depth)}} {ms:9.1f} ms{status}"
            f"  {attrs}".rstrip()
        )
        for child in sorted(children[span.context.span_id], key=lambda c: c.start_time):
            walk(child, depth + 1)

    trace_id = f"{spans[0].context.trace_id:032x}"
    lines.append(f"── trace {trace_id} " + "─" * 30)
    for root in sorted(children[None], key=lambda s: s.start_time):
        walk(root, 0)
    return "\n".join(lines) + "\n"


def configure(service_name: str) -> None:
    """Install the global tracer provider once per process (the API or a worker)."""
    global _configured
    with _lock:
        if _configured:
            return
        _configured = True
        exporter = settings.tracing_exporter
        if exporter == "none":
            return
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if exporter == "tree":
            provider.add_span_processor(SimpleSpanProcessor(TreeConsoleExporter()))
        elif exporter == "console":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif exporter == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        elif exporter == "memory":
            provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
        else:
            raise ValueError(f"unknown TRACING_EXPORTER: {exporter}")
        trace.set_tracer_provider(provider)


def span(name: str, **attributes: Any) -> AbstractContextManager[trace.Span]:
    """A child span of whatever is current; None-valued attributes are skipped."""
    return tracer.start_as_current_span(
        name, attributes={k: v for k, v in attributes.items() if v is not None}
    )


def in_current_context[T](fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap `fn` so a plain thread pool runs it under the caller's trace context (thread pools
    do not copy contextvars, so the spans it opens would otherwise start new traces)."""
    ctx = context.get_current()

    def run(*args: Any, **kwargs: Any) -> T:
        token = context.attach(ctx)
        try:
            return fn(*args, **kwargs)
        finally:
            context.detach(token)

    return run
