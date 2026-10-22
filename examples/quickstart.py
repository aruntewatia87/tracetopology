#!/usr/bin/env python3
"""End-to-end demo: a multi-agent trace through the whole toolkit.

The trace below is a realistic support-agent run: a planner fans out to three
tools, a retriever pulls context, and a final LLM call writes the answer. The
planner also retries the same flaky search three times, which is the classic
agent bug this toolkit is built to surface.
"""

from __future__ import annotations

from tracetopology import (
    CardinalityController,
    IngestionAudit,
    IngestionRules,
    LabelPolicy,
    LLMAgentTopology,
    MetricBuilder,
    PriceBook,
    Span,
    SpanKind,
    attribute_trace_cost,
    build_trace_tree,
    span_cost_resolver,
)

MS = 1_000_000
TRACE_ID = "trace-8f3a"


def span(
    span_id: str,
    parent: str | None,
    name: str,
    kind: SpanKind,
    start_ms: float,
    end_ms: float,
    *,
    model: str | None = None,
    prompt: int = 0,
    completion: int = 0,
    cached: int = 0,
    cost: float | None = None,
    attributes: dict | None = None,
    tenant: str = "acme",
) -> Span:
    """Build one span of the demo trace."""
    attrs: dict = dict(attributes or {})
    if model is not None:
        attrs["gen_ai.request.model"] = model
    if prompt:
        attrs["llm.token_count.prompt"] = prompt
    if completion:
        attrs["llm.token_count.completion"] = completion
    if cached:
        attrs["llm.token_count.cached"] = cached
    if cost is not None:
        attrs["cost.usd"] = cost
    return Span(
        span_id=span_id,
        parent_span_id=parent,
        trace_id=TRACE_ID,
        name=name,
        kind=kind,
        start_ns=int(start_ms * MS),
        end_ns=int(end_ms * MS),
        tenant_id=tenant,
        attributes=attrs,
    )


def build_demo_trace() -> list[Span]:
    """Build the demo trace: planner -> 3 tools -> retriever -> LLM -> final."""
    return [
        span(
            "planner",
            None,
            "support_agent.plan",
            SpanKind.AGENT,
            0.0,
            1420.0,
            model="gpt-4o",
            prompt=1200,
            completion=180,
            cached=800,
        ),
        span("tool_search", "planner", "web_search", SpanKind.TOOL, 10.0, 310.0, cost=0.012),
        span("tool_kb", "planner", "kb_lookup", SpanKind.TOOL, 20.0, 140.0, cost=0.004),
        span("tool_ticket", "planner", "ticket_lookup", SpanKind.TOOL, 30.0, 95.0, cost=0.002),
        # The same flaky search, retried with identical arguments.
        span(
            "tool_search_retry1",
            "planner",
            "web_search",
            SpanKind.TOOL,
            320.0,
            640.0,
            cost=0.012,
            attributes={"tool.arguments": "q=acme refund status"},
        ),
        span(
            "tool_search_retry2",
            "planner",
            "web_search",
            SpanKind.TOOL,
            650.0,
            980.0,
            cost=0.012,
            attributes={"tool.arguments": "q=acme refund status"},
        ),
        span(
            "retriever",
            "planner",
            "vector_store.query",
            SpanKind.RETRIEVAL,
            990.0,
            1180.0,
            cost=0.003,
        ),
        span(
            "composer",
            "planner",
            "llm.compose",
            SpanKind.LLM,
            1190.0,
            1420.0,
            model="claude-3-5-sonnet",
            prompt=2400,
            completion=320,
        ),
    ]


def build_fleet(count: int) -> list[Span]:
    """Build a batch of request spans, each with its own high-cardinality ids.

    Every request carries a distinct prompt hash, session id and request id.
    Turning those into metric labels is exactly how a metrics backend is
    brought down.
    """
    fleet: list[Span] = []
    for index in range(count):
        fleet.append(
            Span(
                span_id=f"req_{index:04d}",
                parent_span_id=None,
                trace_id=f"trace-{index:04d}",
                name="llm.call",
                kind=SpanKind.LLM,
                start_ns=index * MS,
                end_ns=index * MS + 250_000,
                tenant_id="acme" if index % 4 else "globex",
                attributes={
                    "gen_ai.request.model": "gpt-4o" if index % 3 else "gpt-4o-mini",
                    "llm.token_count.prompt": 500 + index,
                    "llm.token_count.completion": 120,
                    "prompt.hash": f"sha256-{index:04d}",
                    "session.id": f"session-{index // 10:03d}",
                    "request.id": f"req-{index:04d}",
                    "user.email": f"user{index:04d}@example.com",
                    # One pod per request: the classic real-world label bomb.
                    "deployment.instance": f"pod-{index:04d}",
                },
            )
        )
    return fleet


def banner(title: str) -> None:
    """Print a section banner."""
    print()
    print(title)
    print("=" * len(title))


def main() -> None:
    """Run the full demo."""
    spans = build_demo_trace()
    prices = PriceBook.default()
    # The resolver makes the tree's cost rollups use the same price book as
    # the attribution breakdown, so the two can never disagree.
    tree = build_trace_tree(spans, cost_resolver=span_cost_resolver(prices))

    banner("1. TRACE TOPOLOGY")
    print(tree.render_tree())
    print()
    print(f"spans={len(tree)}  depth={tree.depth()}  leaves={tree.leaf_count()}")
    print(f"total tokens={tree.total_tokens()}")

    banner("2. AGENT TOPOLOGY")
    topology = LLMAgentTopology(tree)
    print(f"tool fan-out per agent : {topology.fan_out()}")
    print(f"tool calls per agent    : {topology.tool_calls_by_agent()}")
    top = topology.cost_concentration()
    if top is not None:
        print(
            f"cost concentration      : {top.name} "
            f"(${top.span.cost_usd:.4f}, {topology.cost_concentration_share() * 100:.1f}% of trace)"
        )
    print("retry loops detected    :")
    for loop in topology.retry_loops():
        print(
            f"  - {loop.tool_name} called {loop.occurrences}x with args {loop.arguments!r} "
            f"(wasted ${loop.wasted_cost:.4f})"
        )

    banner("3. COST ATTRIBUTION")
    breakdown = attribute_trace_cost(spans, prices)
    print(f"total attributed cost   : ${breakdown.total_cost:.6f}")
    print("by kind:")
    for kind, value in breakdown.cost_by_kind.items():
        share = value / breakdown.total_cost * 100 if breakdown.total_cost else 0.0
        print(f"  {kind:<10} ${value:.6f}  ({share:5.1f}%)")
    print("by model:")
    for model, value in breakdown.cost_by_model.items():
        share = value / breakdown.total_cost * 100 if breakdown.total_cost else 0.0
        print(f"  {model:<20} ${value:.6f}  ({share:5.1f}%)")
    print()
    print(f"critical path           : {' -> '.join(breakdown.critical_path_ids)}")
    print(f"critical path cost      : ${breakdown.critical_path_cost:.6f}")
    print(f"off-path (wasted) cost  : ${breakdown.off_path_cost:.6f}")
    print(f"critical path share     : {breakdown.critical_path_share * 100:.2f}%")
    per_span_total = sum(entry.total_cost for entry in breakdown.per_span)
    print(f"invariant (sum==total)  : {per_span_total:.6f} == {breakdown.total_cost:.6f}")

    banner("4. CARDINALITY CONTROL")
    # A realistic batch: 200 requests, each with its own prompt hash, session
    # id, request id and pod. These are the attributes that destroy a metrics
    # backend when they are turned into labels.
    fleet = build_fleet(200)
    naive = CardinalityController(LabelPolicy())
    naive_report = naive.report(fleet)
    print("naive - every span attribute becomes a metric label:")
    print(f"  distinct label values   : {naive_report.before_labels}")
    print(f"  estimated series        : {naive_report.estimated_series_before:,}")
    print("  label values per label  :")
    for label, count in naive_report.value_counts_before.items():
        print(f"    {label:<28} {count}")
    print()

    # The safe approach: a fixed, bounded label space. The metric families
    # only ever use these dimensions, and each has a handful of values.
    controller = CardinalityController(
        LabelPolicy(
            allowed_labels=("tenant_id", "model", "kind", "status", "direction"),
            max_values_per_label=50,
            hash_remaining=True,
        )
    )
    fleet_bundle = MetricBuilder(controller).build(fleet)
    fleet_report = fleet_bundle.cardinality
    assert fleet_report is not None
    print("bounded - only the fixed metric label space is used:")
    print(f"  label values per label  : {fleet_report.value_counts_before}")
    print(
        f"  estimated series        : {fleet_report.estimated_series_before:,} -> "
        f"{fleet_report.estimated_series_after:,}"
    )
    print(f"  reduction               : {fleet_report.reduction_pct():.2f}%")
    print()

    # The cap does the rest: an unbounded label is bucketed into 50 hash
    # buckets instead of being allowed to create one series per pod.
    capped = CardinalityController(
        LabelPolicy(
            allowed_labels=(
                "tenant_id",
                "model",
                "kind",
                "status",
                "direction",
                "deployment.instance",
            ),
            max_values_per_label=50,
            hash_remaining=True,
        )
    )
    capped_report = capped.report(fleet)
    print("bounded - and anything above the cap is hashed into 50 buckets:")
    print(
        f"  estimated series        : {capped_report.estimated_series_before:,} -> "
        f"{capped_report.estimated_series_after:,}"
    )
    print(f"  reduction               : {capped_report.reduction_pct():.2f}%")
    print(f"  dropped attributes      : {capped_report.dropped_attributes or '(none)'}")
    print(f"  high cardinality labels : {capped_report.high_cardinality_labels or '(none)'}")
    print(f"  hashed labels           : {capped_report.hashed_labels or '(none)'}")

    banner("5. PROMETHEUS METRICS")
    print(MetricBuilder(controller).build(spans).render())

    banner("6. INGESTION AUDIT")
    # A healthy run: a configured sampling rate drops some observations, but
    # that is deliberate, so the run is still complete.
    audit = IngestionAudit(
        IngestionRules(
            sampling_rate=0.5,
            allowed_tenants=("acme",),
            tenant_quotas={"acme": 100},
        )
    )
    for index in range(20):
        audit.submit(span(f"ingest_{index}", None, "llm.call", SpanKind.LLM, 0.0, 50.0))
    healthy = audit.report()
    print("healthy run (sampling is deliberate):")
    print(f"  submitted               : {healthy.submitted}")
    print(f"  accepted                : {healthy.accepted}")
    print(f"  dropped                 : {healthy.dropped}")
    print(f"  drop reasons            : {healthy.drop_reasons}")
    print(f"  dropped fraction        : {healthy.dropped_fraction:.3f}")
    print(f"  complete                : {healthy.complete}")
    print()

    # A broken run: a malformed observation and a tenant-isolation rejection.
    # These are not deliberate, so the run is NOT complete. This is the
    # difference between "we sampled" and "we lost your data".
    broken_audit = IngestionAudit(IngestionRules(allowed_tenants=("acme",)))
    for index in range(4):
        broken_audit.submit(span(f"ok_{index}", None, "llm.call", SpanKind.LLM, 0.0, 50.0))
    broken_audit.submit({"span_id": "malformed"})
    broken_audit.submit(
        span("intruder", None, "llm.call", SpanKind.LLM, 0.0, 50.0, tenant="evilcorp")
    )
    report = broken_audit.report()
    print("broken run (data was lost for a reason nobody asked for):")
    print(f"  submitted               : {report.submitted}")
    print(f"  accepted                : {report.accepted}")
    print(f"  dropped                 : {report.dropped}")
    print(f"  drop reasons            : {report.drop_reasons}")
    print(f"  dropped fraction        : {report.dropped_fraction:.3f}")
    print(f"  complete                : {report.complete}")
    print("  drop details:")
    for reason, detail in broken_audit.drop_details():
        print(f"    {reason:<18} {detail}")


if __name__ == "__main__":
    main()
