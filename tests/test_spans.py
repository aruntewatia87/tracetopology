"""Tests for the span model."""

from __future__ import annotations

import pytest

from tracetopology import Span, SpanKind, SpanStatus


def test_duration_is_computed_from_timestamps() -> None:
    span = Span("s1", None, "t1", "call", SpanKind.LLM, start_ns=1_000_000, end_ns=3_500_000)
    assert span.duration_ns == 2_500_000
    assert span.duration_ms == 2.5


def test_duration_never_goes_negative() -> None:
    span = Span("s1", None, "t1", "call", start_ns=5_000, end_ns=1_000)
    assert span.duration_ns == 0
    assert span.duration_ms == 0.0


def test_missing_token_attributes_default_to_zero() -> None:
    span = Span("s1", None, "t1", "call", SpanKind.LLM)
    assert span.input_tokens == 0
    assert span.output_tokens == 0
    assert span.cached_tokens == 0
    assert span.total_tokens == 0


def test_tokens_are_read_from_vendor_attribute_spellings() -> None:
    span = Span(
        "s1",
        None,
        "t1",
        "call",
        SpanKind.LLM,
        attributes={
            "gen_ai.usage.input_tokens": 120,
            "gen_ai.usage.output_tokens": 40,
            "gen_ai.request.model": "gpt-4o",
        },
    )
    assert span.input_tokens == 120
    assert span.output_tokens == 40
    assert span.total_tokens == 160
    assert span.model == "gpt-4o"


def test_cached_tokens_are_read_independently() -> None:
    span = Span(
        "s1",
        None,
        "t1",
        "call",
        SpanKind.LLM,
        attributes={"llm.token_count.prompt": 500, "llm.token_count.cached": 400},
    )
    assert span.input_tokens == 500
    assert span.cached_tokens == 400


def test_missing_model_is_none_and_cost_is_zero() -> None:
    span = Span("s1", None, "t1", "call", SpanKind.LLM)
    assert span.model is None
    assert span.cost_usd == 0.0


def test_malformed_token_attributes_do_not_crash() -> None:
    span = Span(
        "s1",
        None,
        "t1",
        "call",
        SpanKind.LLM,
        attributes={"llm.token_count.prompt": "not-a-number", "cost.usd": "nope"},
    )
    assert span.input_tokens == 0
    assert span.cost_usd == 0.0


def test_string_token_attributes_are_coerced() -> None:
    span = Span(
        "s1",
        None,
        "t1",
        "call",
        SpanKind.LLM,
        attributes={"llm.token_count.prompt": "512", "llm.token_count.completion": "64"},
    )
    assert span.input_tokens == 512
    assert span.output_tokens == 64


def test_span_kind_and_status_are_coerced_from_strings() -> None:
    span = Span("s1", None, "t1", "call", "LLM", status="ERROR")
    assert span.kind is SpanKind.LLM
    assert span.status is SpanStatus.ERROR
    assert span.is_error is True


def test_unknown_span_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown span kind"):
        SpanKind.coerce("teleport")


def test_unknown_span_status_raises() -> None:
    with pytest.raises(ValueError, match="unknown span status"):
        SpanStatus.coerce("maybe")


def test_root_and_parent_helpers() -> None:
    root = Span("r", None, "t1", "root")
    child = Span("c", "r", "t1", "child")
    assert root.is_root is True
    assert child.is_root is False


def test_attribute_accessor_uses_default() -> None:
    span = Span("s1", None, "t1", "call", attributes={"a": 1})
    assert span.attribute("a") == 1
    assert span.attribute("missing") is None
    assert span.attribute("missing", 7) == 7


def test_a_nan_cost_is_treated_as_zero() -> None:
    span = Span("s1", None, "t1", "call", SpanKind.LLM, attributes={"cost.usd": float("nan")})
    assert span.cost_usd == 0.0


def test_a_negative_cost_is_clamped_to_zero() -> None:
    span = Span("s1", None, "t1", "call", SpanKind.LLM, attributes={"cost.usd": -5.0})
    assert span.cost_usd == 0.0


def test_a_boolean_token_attribute_is_ignored() -> None:
    span = Span(
        "s1",
        None,
        "t1",
        "call",
        SpanKind.LLM,
        attributes={"llm.token_count.prompt": True},
    )
    assert span.input_tokens == 0


def test_a_none_attribute_value_is_handled() -> None:
    span = Span("s1", None, "t1", "call", SpanKind.LLM, attributes={"llm.token_count.prompt": None})
    assert span.input_tokens == 0


def test_a_blank_model_name_is_none() -> None:
    span = Span("s1", None, "t1", "call", SpanKind.LLM, attributes={"gen_ai.request.model": "  "})
    assert span.model is None


def test_a_float_token_attribute_is_truncated() -> None:
    span = Span(
        "s1",
        None,
        "t1",
        "call",
        SpanKind.LLM,
        attributes={"llm.token_count.prompt": 12.9},
    )
    assert span.input_tokens == 12
