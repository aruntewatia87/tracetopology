"""Tests for the no-silent-drop ingestion audit."""

from __future__ import annotations

import pytest

from tracetopology import (
    DropReason,
    IngestionAudit,
    IngestionReport,
    IngestionRules,
    Span,
    audit_observations,
)


def span(span_id: str, *, tenant: str | None = "acme", attrs: dict | None = None) -> Span:
    """Build a simple span for ingestion."""
    return Span(
        span_id=span_id,
        parent_span_id=None,
        trace_id="t1",
        name="llm.call",
        tenant_id=tenant,
        attributes=dict(attrs or {}),
    )


def test_an_empty_submission_is_handled() -> None:
    audit = IngestionAudit()
    report = audit.report()
    assert report.submitted == 0
    assert report.accepted == 0
    assert report.dropped == 0
    assert report.drop_reasons == {}
    assert report.dropped_fraction == 0.0
    assert report.complete is True


def test_everything_is_accepted_by_default() -> None:
    audit = IngestionAudit()
    accepted = audit.submit_many([span("a"), span("b"), span("c")])
    report = audit.report()
    assert accepted == 3
    assert report.submitted == 3
    assert report.accepted == 3
    assert report.dropped == 0
    assert report.complete is True


def test_dropped_by_sampling_is_reported_as_expected() -> None:
    # A configured sampling rate is deliberate, so the run stays complete.
    rules = IngestionRules(sampling_rate=0.5)
    audit = IngestionAudit(rules)
    for index in range(200):
        audit.submit(span(f"s{index}"))
    report = audit.report()
    assert report.dropped > 0
    assert report.reason_count(DropReason.SAMPLED) == report.dropped
    assert report.complete is True


def test_sampling_is_deterministic() -> None:
    spans = [span(f"s{index}") for index in range(100)]
    first = IngestionAudit(IngestionRules(sampling_rate=0.5)).submit_many(spans)
    second = IngestionAudit(IngestionRules(sampling_rate=0.5)).submit_many(spans)
    assert first == second
    assert 0 < first < 100


def test_a_sampling_rate_of_one_keeps_everything() -> None:
    audit = IngestionAudit(IngestionRules(sampling_rate=1.0))
    for index in range(50):
        audit.submit(span(f"s{index}"))
    assert audit.report().dropped == 0


def test_a_sampling_rate_of_zero_keeps_nothing() -> None:
    audit = IngestionAudit(IngestionRules(sampling_rate=0.0))
    for index in range(10):
        audit.submit(span(f"s{index}"))
    report = audit.report()
    assert report.accepted == 0
    assert report.dropped == 10
    assert report.complete is True


def test_dropped_by_malformed_sets_complete_false_and_records_the_reason() -> None:
    audit = IngestionAudit()
    audit.submit({"span_id": "a", "trace_id": "t1", "name": "ok"})
    audit.submit({"span_id": "b"})  # missing trace_id
    audit.submit(42)  # not coercible at all
    report = audit.report()
    assert report.submitted == 3
    assert report.accepted == 1
    assert report.dropped == 2
    assert report.reason_count(DropReason.MALFORMED) == 2
    assert report.complete is False


def test_dropped_by_a_closed_window_sets_complete_false() -> None:
    audit = IngestionAudit(IngestionRules(window_open=False))
    audit.submit(span("a"))
    report = audit.report()
    assert report.accepted == 0
    assert report.dropped == 1
    assert report.reason_count(DropReason.WINDOW_CLOSED) == 1
    assert report.complete is False


def test_dropped_by_tenant_isolation_sets_complete_false() -> None:
    audit = IngestionAudit(IngestionRules(allowed_tenants=("acme",)))
    assert audit.submit(span("a", tenant="acme")) is True
    assert audit.submit(span("b", tenant="evilcorp")) is False
    report = audit.report()
    assert report.accepted == 1
    assert report.dropped == 1
    assert report.reason_count(DropReason.TENANT_ISOLATION) == 1
    assert report.complete is False


def test_dropped_by_a_tenant_quota_sets_complete_false() -> None:
    audit = IngestionAudit(IngestionRules(tenant_quotas={"acme": 2}))
    assert audit.submit(span("a")) is True
    assert audit.submit(span("b")) is True
    assert audit.submit(span("c")) is False
    report = audit.report()
    assert report.accepted == 2
    assert report.dropped == 1
    assert report.reason_count(DropReason.QUOTA_EXCEEDED) == 1
    assert report.complete is False


def test_quotas_are_tracked_per_tenant() -> None:
    audit = IngestionAudit(IngestionRules(tenant_quotas={"acme": 1, "globex": 1}))
    assert audit.submit(span("a", tenant="acme")) is True
    assert audit.submit(span("b", tenant="globex")) is True
    assert audit.submit(span("c", tenant="acme")) is False
    assert audit.submit(span("d", tenant="globex")) is False
    assert audit.report().accepted == 2


def test_the_drop_reasons_sum_exactly_to_the_number_dropped() -> None:
    rules = IngestionRules(
        sampling_rate=0.5,
        allowed_tenants=("acme",),
        tenant_quotas={"acme": 3},
    )
    audit = IngestionAudit(rules)
    for index in range(200):
        audit.submit(span(f"s{index}", tenant="acme" if index % 2 else "evilcorp"))
    for index in range(5):
        audit.submit({"span_id": f"bad{index}"})
    report = audit.report()
    assert sum(report.drop_reasons.values()) == report.dropped
    assert report.accepted + report.dropped == report.submitted
    assert report.dropped_fraction == pytest.approx(report.dropped / report.submitted)


def test_the_report_enforces_its_accounting_invariant() -> None:
    with pytest.raises(ValueError, match="invariant violated"):
        IngestionReport(submitted=10, accepted=5, dropped=4, drop_reasons={"sampled": 4})


def test_the_report_rejects_drop_reasons_that_do_not_sum() -> None:
    with pytest.raises(ValueError, match="invariant violated"):
        IngestionReport(submitted=10, accepted=8, dropped=2, drop_reasons={"sampled": 1})


def test_mixed_drop_reasons_make_the_run_incomplete() -> None:
    rules = IngestionRules(sampling_rate=0.0, allowed_tenants=("acme",))
    audit = IngestionAudit(rules)
    audit.submit(span("a", tenant="acme"))
    audit.submit(span("b", tenant="evilcorp"))
    report = audit.report()
    assert report.drop_reasons["sampled"] == 1
    assert report.drop_reasons["tenant_isolation"] == 1
    assert report.complete is False


def test_drop_details_are_recorded() -> None:
    audit = IngestionAudit(IngestionRules(allowed_tenants=("acme",)))
    audit.submit(span("a", tenant="evilcorp"))
    details = audit.drop_details()
    assert len(details) == 1
    assert details[0][0] == "tenant_isolation"
    assert "evilcorp" in details[0][1]


def test_accepted_spans_are_returned_in_order() -> None:
    audit = IngestionAudit()
    audit.submit_many([span("a"), span("b"), span("c")])
    assert [s.span_id for s in audit.accepted_spans] == ["a", "b", "c"]


def test_a_mapping_observation_is_coerced_into_a_span() -> None:
    report, accepted = audit_observations(
        [
            {
                "span_id": "a",
                "parent_span_id": None,
                "trace_id": "t1",
                "name": "llm.call",
                "kind": "llm",
                "start_ns": 0,
                "end_ns": 1000,
                "attributes": {"gen_ai.request.model": "gpt-4o"},
                "tenant_id": "acme",
            }
        ]
    )
    assert report.accepted == 1
    assert len(accepted) == 1
    assert accepted[0].kind.value == "llm"
    assert accepted[0].model == "gpt-4o"
    assert accepted[0].duration_ns == 1000


def test_attribute_count_limit_marks_a_span_as_malformed() -> None:
    rules = IngestionRules(max_attributes=2)
    audit = IngestionAudit(rules)
    assert audit.submit(span("a", attrs={"x": 1, "y": 2})) is True
    assert audit.submit(span("b", attrs={"x": 1, "y": 2, "z": 3})) is False
    report = audit.report()
    assert report.reason_count(DropReason.MALFORMED) == 1
    assert report.complete is False


def test_report_summary_is_json_friendly() -> None:
    audit = IngestionAudit(IngestionRules(window_open=False))
    audit.submit(span("a"))
    summary = audit.report().summary()
    assert summary["submitted"] == 1
    assert summary["accepted"] == 0
    assert summary["dropped"] == 1
    assert summary["complete"] is False


def test_drop_reason_coercion_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unknown drop reason"):
        DropReason.coerce("teleported")


def test_sampling_rate_must_be_in_range() -> None:
    with pytest.raises(ValueError, match="sampling_rate"):
        IngestionRules(sampling_rate=1.5)


def test_sampling_configured_flag() -> None:
    assert IngestionRules(sampling_rate=0.5).sampling_configured is True
    assert IngestionRules(sampling_rate=1.0).sampling_configured is False


def test_an_observation_that_is_not_coercible_is_malformed() -> None:
    audit = IngestionAudit()
    assert audit.submit(42) is False
    assert audit.submit("a string") is False
    assert audit.submit(["a", "list"]) is False
    report = audit.report()
    assert report.dropped == 3
    assert report.reason_count(DropReason.MALFORMED) == 3


def test_a_mapping_without_a_span_id_is_malformed() -> None:
    audit = IngestionAudit()
    assert audit.submit({"trace_id": "t1"}) is False
    assert audit.submit({"span_id": "a"}) is False
    assert audit.report().reason_count(DropReason.MALFORMED) == 2


def test_the_audit_keeps_the_rules_it_was_given() -> None:
    rules = IngestionRules(sampling_rate=0.25)
    assert IngestionAudit(rules).rules is rules
