"""Tests for cardinality control and Prometheus metric building."""

from __future__ import annotations

import pytest

from tracetopology import (
    CardinalityController,
    LabelPolicy,
    LabelSet,
    MetricBuilder,
    Span,
    SpanKind,
    stable_hash_bucket,
)

MS = 1_000_000


def llm_span(
    span_id: str,
    *,
    model: str = "gpt-4o",
    prompt: str = "p",
    tenant: str = "acme",
    start_ms: float = 0.0,
    end_ms: float = 100.0,
    kind: SpanKind = SpanKind.LLM,
) -> Span:
    """Build an LLM span with high-cardinality attributes."""
    return Span(
        span_id=span_id,
        parent_span_id=None,
        trace_id="t1",
        name="llm.call",
        kind=kind,
        start_ns=int(start_ms * MS),
        end_ns=int(end_ms * MS),
        tenant_id=tenant,
        attributes={
            "tenant_id": tenant,
            "gen_ai.request.model": model,
            "llm.token_count.prompt": 100,
            "llm.token_count.completion": 50,
            "prompt.hash": prompt,
            "session.id": f"session-{prompt}",
            "user.email": f"user-{prompt}@example.com",
            "request.id": f"req-{prompt}",
        },
    )


def test_label_set_cardinality_is_the_product_of_value_counts() -> None:
    label_set = LabelSet()
    label_set.update({"model": "gpt-4o"})
    label_set.update({"model": "claude"})
    label_set.update({"tenant": "acme"})
    label_set.update({"tenant": "globex"})
    assert label_set.value_count("model") == 2
    assert label_set.value_count("tenant") == 2
    assert label_set.cardinality_estimate() == 4
    assert label_set.label_count() == 4


def test_empty_label_set_is_one_series() -> None:
    assert LabelSet().cardinality_estimate() == 1
    assert LabelSet().label_count() == 0


def test_non_allow_listed_attributes_are_dropped() -> None:
    policy = LabelPolicy(allowed_labels=("gen_ai.request.model", "tenant_id"))
    controller = CardinalityController(policy)
    spans = [llm_span("s1", prompt="abc")]
    report = controller.report(spans)
    assert "prompt.hash" in report.dropped_attributes
    assert "user.email" in report.dropped_attributes
    assert "gen_ai.request.model" not in report.dropped_attributes


def test_denied_labels_win_over_the_allow_list() -> None:
    policy = LabelPolicy(allowed_labels=("model",), denied_labels=("model",))
    controller = CardinalityController(policy)
    report = controller.report([llm_span("s1")])
    # "model" is denied, so it never survives; the span attribute carrying
    # the model is a different label name and is not allow-listed either.
    assert "model" not in report.dropped_attributes
    assert "gen_ai.request.model" in report.dropped_attributes
    assert report.after_labels == 0


def test_denied_span_attribute_is_dropped() -> None:
    policy = LabelPolicy(denied_labels=("prompt.hash",))
    controller = CardinalityController(policy)
    report = controller.report([llm_span("s1")])
    assert "prompt.hash" in report.dropped_attributes


def test_drop_patterns_remove_matching_labels() -> None:
    policy = LabelPolicy(drop_patterns=(r"^user\.", r"\.id$"))
    controller = CardinalityController(policy)
    report = controller.report([llm_span("s1")])
    assert "user.email" in report.dropped_attributes
    assert "session.id" in report.dropped_attributes


def test_high_cardinality_labels_are_hashed_to_a_bounded_bucket_count() -> None:
    policy = LabelPolicy(
        allowed_labels=("prompt.hash",), max_values_per_label=10, hash_remaining=True
    )
    controller = CardinalityController(policy)
    spans = [llm_span(f"s{i}", prompt=f"prompt-{i}") for i in range(100)]
    report = controller.report(spans)
    assert report.value_counts_before["prompt.hash"] == 100
    assert report.value_counts_after["prompt.hash"] <= 10
    assert "prompt.hash" in report.high_cardinality_labels
    assert "prompt.hash" in report.hashed_labels


def test_high_cardinality_labels_can_be_dropped_instead_of_hashed() -> None:
    policy = LabelPolicy(
        allowed_labels=("prompt.hash",), max_values_per_label=10, hash_remaining=False
    )
    controller = CardinalityController(policy)
    spans = [llm_span(f"s{i}", prompt=f"prompt-{i}") for i in range(100)]
    report = controller.report(spans)
    assert "prompt.hash" in report.dropped_attributes
    assert report.value_counts_after.get("prompt.hash", 0) == 0


def test_hash_buckets_are_stable_and_bounded() -> None:
    first = stable_hash_bucket("model", "gpt-4o", 16)
    second = stable_hash_bucket("model", "gpt-4o", 16)
    assert first == second
    assert first.startswith("bucket_")
    buckets = {stable_hash_bucket("model", f"v{i}", 16) for i in range(500)}
    assert len(buckets) <= 16


def test_estimated_series_drops_dramatically() -> None:
    # Four labels (prompt.hash, session.id, request.id, user.email) each take
    # 50 distinct values, so the label space describes 50**4 == 6_250_000
    # time series. This is the multiplicative explosion: a handful of
    # unbounded labels turns into millions of series.
    controller = CardinalityController(LabelPolicy(allowed_labels=("tenant_id",)))
    spans = [llm_span(f"s{i}", prompt=f"prompt-{i}") for i in range(50)]
    report = controller.report(spans)
    exploding = dict.fromkeys(("prompt.hash", "session.id", "request.id", "user.email"), 50)
    assert {k: v for k, v in report.value_counts_before.items() if v == 50} == exploding
    assert report.estimated_series_before == 50**4 == 6_250_000
    assert report.estimated_series_after == 1
    assert report.reduction_pct() > 99.0
    assert report.label_reduction_pct() > 99.0


def test_reduction_pct_is_mathematically_correct() -> None:
    controller = CardinalityController(LabelPolicy(allowed_labels=("model", "tenant")))
    spans = [
        llm_span(
            f"s{i}",
            model="gpt-4o" if i % 2 == 0 else "claude",
            tenant="acme" if i % 3 else "globex",
        )
        for i in range(12)
    ]
    report = controller.report(spans)
    expected = (1 - report.estimated_series_after / report.estimated_series_before) * 100.0
    assert report.reduction_pct() == pytest.approx(expected)


def test_reduction_pct_is_zero_when_there_is_nothing_to_reduce() -> None:
    controller = CardinalityController(LabelPolicy())
    report = controller.report([])
    assert report.reduction_pct() == 0.0


def test_sanitise_labels_drops_disallowed_labels() -> None:
    controller = CardinalityController(LabelPolicy(allowed_labels=("model",)))
    result = controller.sanitise_labels({"model": "gpt-4o", "secret": "x"})
    assert result == {"model": "gpt-4o"}


def test_sanitise_labels_hashes_a_label_over_the_cap() -> None:
    controller = CardinalityController(
        LabelPolicy(allowed_labels=("model",), max_values_per_label=4, hash_remaining=True)
    )
    result = controller.sanitise_labels({"model": "gpt-4o"}, value_counts={"model": 100})
    assert result["model"].startswith("bucket_")


def test_metric_builder_emits_the_expected_families() -> None:
    builder = MetricBuilder(CardinalityController(LabelPolicy()))
    bundle = builder.build([llm_span("s1")])
    names = [family.name for family in bundle.families]
    assert names == [
        "llm_requests_total",
        "llm_tokens_total",
        "llm_request_duration_seconds",
        "llm_cost_usd_total",
    ]


def test_requests_counter_counts_one_per_span() -> None:
    builder = MetricBuilder()
    bundle = builder.build([llm_span(f"s{i}") for i in range(4)])
    family = bundle.family("llm_requests_total")
    assert family is not None
    assert sum(entry.value for entry in family.series) == 4.0


def test_tokens_counter_is_split_by_direction() -> None:
    builder = MetricBuilder()
    bundle = builder.build([llm_span("s1")])
    family = bundle.family("llm_tokens_total")
    assert family is not None
    by_direction = {entry.labels["direction"]: entry.value for entry in family.series}
    assert by_direction == {"input": 100.0, "output": 50.0}


def test_histogram_bucket_counts_are_cumulative_and_exact() -> None:
    builder = MetricBuilder(buckets=(0.1, 0.5, 1.0, 5.0))
    spans = [
        llm_span("s1", end_ms=50.0),  # 0.05s -> bucket 0.1
        llm_span("s2", end_ms=200.0),  # 0.2s  -> bucket 0.5
        llm_span("s3", end_ms=400.0),  # 0.4s  -> bucket 0.5
        llm_span("s4", end_ms=900.0),  # 0.9s  -> bucket 1.0
        llm_span("s5", end_ms=6000.0),  # 6.0s  -> +Inf only
    ]
    bundle = builder.build(spans)
    family = bundle.family("llm_request_duration_seconds")
    assert family is not None
    series = family.series[0]
    assert series.bucket_counts[0.1] == 1
    assert series.bucket_counts[0.5] == 3
    assert series.bucket_counts[1.0] == 4
    assert series.bucket_counts[5.0] == 4
    assert series.total_count == 5
    assert series.total_sum == pytest.approx(0.05 + 0.2 + 0.4 + 0.9 + 6.0)


def test_histogram_renders_cumulative_buckets_in_prometheus_format() -> None:
    builder = MetricBuilder(buckets=(0.1, 1.0))
    bundle = builder.build([llm_span("s1", end_ms=50.0), llm_span("s2", end_ms=500.0)])
    rendered = bundle.family("llm_request_duration_seconds").render()
    assert 'le="0.1"} 1' in rendered
    assert 'le="1.0"} 2' in rendered
    assert 'le="+Inf"} 2' in rendered
    assert "llm_request_duration_seconds_count{" in rendered
    assert "# TYPE llm_request_duration_seconds histogram" in rendered


def test_histogram_bucket_labels_put_le_last() -> None:
    builder = MetricBuilder(buckets=(0.1,))
    bundle = builder.build([llm_span("s1")])
    rendered = bundle.family("llm_request_duration_seconds").render()
    line = next(
        line
        for line in rendered.splitlines()
        if line.startswith("llm_request_duration_seconds_bucket{")
    )
    assert line.endswith('le="0.1"} 1')
    assert line.index("le=") > line.index("kind=")


def test_counter_renders_help_type_and_labels() -> None:
    builder = MetricBuilder()
    bundle = builder.build([llm_span("s1", tenant="acme", model="gpt-4o")])
    rendered = bundle.family("llm_requests_total").render()
    assert "# HELP llm_requests_total" in rendered
    assert "# TYPE llm_requests_total counter" in rendered
    assert 'model="gpt-4o"' in rendered
    assert 'tenant_id="acme"' in rendered


def test_metric_bundle_renders_every_family() -> None:
    builder = MetricBuilder()
    bundle = builder.build([llm_span("s1")])
    rendered = bundle.render()
    assert "llm_requests_total" in rendered
    assert "llm_tokens_total" in rendered
    assert "llm_request_duration_seconds" in rendered
    assert "llm_cost_usd_total" in rendered


def test_metric_builder_report_describes_the_metric_label_space() -> None:
    builder = MetricBuilder(CardinalityController(LabelPolicy(allowed_labels=("tenant_id",))))
    bundle = builder.build([llm_span("s1", prompt="x")])
    assert bundle.cardinality is not None
    # The metric families only ever use tenant_id, model, kind, status and
    # direction, so the report describes that space, not the raw attributes.
    assert "tenant_id" in bundle.cardinality.value_counts_before
    assert "prompt.hash" not in bundle.cardinality.value_counts_before


def test_builder_is_deterministic() -> None:
    spans = [llm_span(f"s{i}", prompt=f"p{i}") for i in range(5)]
    first = MetricBuilder().build(spans).render()
    second = MetricBuilder().build(spans).render()
    assert first == second


def test_label_policy_rejects_a_negative_cap() -> None:
    with pytest.raises(ValueError, match="max_values_per_label"):
        LabelPolicy(max_values_per_label=-1)


def test_report_for_labels_assesses_an_explicit_label_space() -> None:
    controller = CardinalityController(LabelPolicy(allowed_labels=("tenant",)))
    first = LabelSet()
    first.update({"tenant": "acme", "model": "gpt-4o"})
    second = LabelSet()
    second.update({"tenant": "globex", "model": "claude"})
    report = controller.report_for_labels([first, second])
    assert report.value_counts_before == {"model": 2, "tenant": 2}
    assert report.estimated_series_before == 4
    # "model" is dropped, leaving two tenants and therefore two series.
    assert report.estimated_series_after == 2
    assert report.dropped_attributes == ["model"]
    assert report.reduction_pct() == pytest.approx(50.0)


def test_report_for_labels_merges_the_label_spaces() -> None:
    controller = CardinalityController(LabelPolicy())
    first = LabelSet()
    first.update({"a": "1"})
    second = LabelSet()
    second.update({"a": "2"})
    report = controller.report_for_labels([first, second])
    assert report.value_counts_before == {"a": 2}
    assert report.estimated_series_before == 2


def test_the_metric_report_describes_the_real_series_space() -> None:
    spans = [llm_span(f"s{i}", prompt=f"p{i}") for i in range(10)]
    bundle = MetricBuilder(CardinalityController(LabelPolicy())).build(spans)
    assert bundle.cardinality is not None
    # The metric families never expose prompt.hash as a label.
    assert "prompt.hash" not in bundle.cardinality.value_counts_before
    assert "model" in bundle.cardinality.value_counts_before
    assert "kind" in bundle.cardinality.value_counts_before


def test_a_family_can_be_looked_up_by_name() -> None:
    bundle = MetricBuilder().build([llm_span("s1")])
    assert bundle.family("llm_requests_total") is not None
    assert bundle.family("does_not_exist") is None


def test_a_counter_family_reports_the_value_of_a_label_set() -> None:
    bundle = MetricBuilder().build([llm_span("s1", tenant="acme", model="gpt-4o")])
    family = bundle.family("llm_requests_total")
    assert family is not None
    assert (
        family.value({"tenant_id": "acme", "model": "gpt-4o", "kind": "llm", "status": "ok"}) == 1.0
    )
    assert family.value({"tenant_id": "nobody"}) == 0.0


def test_a_bounded_label_is_dropped_rather_than_hashed_when_asked() -> None:
    controller = CardinalityController(
        LabelPolicy(allowed_labels=("model",), max_values_per_label=2, hash_remaining=False)
    )
    result = controller.sanitise_labels({"model": "gpt-4o"}, value_counts={"model": 10})
    assert result == {}


def test_the_policy_permits_a_label_when_nothing_denies_it() -> None:
    policy = LabelPolicy()
    assert policy.permits("anything") is True
    assert policy.drop_reason("anything") is None


def test_an_empty_builder_produces_empty_families() -> None:
    bundle = MetricBuilder().build([])
    assert bundle.family("llm_requests_total").series == []
    assert bundle.family("llm_request_duration_seconds").series == []
    assert bundle.render() != ""
