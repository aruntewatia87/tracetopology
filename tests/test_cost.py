"""Tests for price books and per-trace cost attribution."""

from __future__ import annotations

import pytest

from tracetopology import (
    ModelPrice,
    PriceBook,
    Span,
    SpanKind,
    attribute_trace_cost,
    build_trace_tree,
)

MS = 1_000_000


def book() -> PriceBook:
    """A small, hand-checkable price book."""
    return PriceBook(
        prices={
            "alpha": ModelPrice("alpha", 0.001, 0.002),
            "beta": ModelPrice("beta", 0.010, 0.020, 0.0001),
        },
        aliases={"alpha-v2": "alpha", "vendor/alpha": "alpha"},
    )


def llm_span(
    span_id: str,
    parent: str | None,
    *,
    model: str = "alpha",
    prompt: int = 0,
    completion: int = 0,
    cached: int = 0,
    cost: float | None = None,
    kind: SpanKind = SpanKind.LLM,
    start_ms: float = 0.0,
    end_ms: float = 10.0,
    name: str = "llm",
) -> Span:
    """Build a span carrying token counts."""
    attributes: dict = {
        "gen_ai.request.model": model,
        "llm.token_count.prompt": prompt,
        "llm.token_count.completion": completion,
    }
    if cached:
        attributes["llm.token_count.cached"] = cached
    if cost is not None:
        attributes["cost.usd"] = cost
    return Span(
        span_id=span_id,
        parent_span_id=parent,
        trace_id="t1",
        name=name,
        kind=kind,
        start_ns=int(start_ms * MS),
        end_ns=int(end_ms * MS),
        attributes=attributes,
    )


def test_price_book_resolves_aliases() -> None:
    prices = book()
    assert prices.resolve("alpha-v2") == "alpha"
    assert prices.resolve("vendor/alpha") == "alpha"
    assert prices.resolve("ALPHA") == "alpha"
    assert prices.resolve("nope") is None
    assert prices.resolve(None) is None


def test_price_book_membership() -> None:
    prices = book()
    assert "alpha" in prices
    assert "alpha-v2" in prices
    assert "unknown-model" not in prices
    assert len(prices) == 2
    assert prices.models() == ["alpha", "beta"]


def test_default_price_book_resolves_common_aliases() -> None:
    prices = PriceBook.default()
    assert prices.resolve("openai/gpt-4o") == "gpt-4o"
    assert prices.resolve("gpt-4o-latest") == "gpt-4o"
    assert prices.resolve("anthropic/claude-3-5-sonnet-20241022") == "claude-3-5-sonnet"


def test_per_span_cost_matches_the_hand_computed_formula() -> None:
    # alpha: $0.001/1k input, $0.002/1k output.
    # 1000 prompt tokens -> 1000/1000 * 0.001 = $0.001
    #  500 output tokens ->  500/1000 * 0.002 = $0.001
    # total = $0.002
    spans = [llm_span("a", None, model="alpha", prompt=1000, completion=500)]
    breakdown = attribute_trace_cost(spans, book())
    entry = breakdown.span_cost("a")
    assert entry is not None
    assert entry.input_cost == pytest.approx(0.001)
    assert entry.output_cost == pytest.approx(0.001)
    assert entry.total_cost == pytest.approx(0.002)
    assert entry.source == "price_book"


def test_cached_tokens_are_billed_at_the_cheaper_rate() -> None:
    # beta: $0.010/1k uncached input, $0.0001/1k cached input.
    # 1000 prompt tokens of which 800 are cached:
    #   uncached 200 -> 200/1000 * 0.0100 = $0.002
    #   cached   800 -> 800/1000 * 0.0001 = $0.00008
    #   total            = $0.00208
    spans = [llm_span("a", None, model="beta", prompt=1000, cached=800)]
    breakdown = attribute_trace_cost(spans, book())
    entry = breakdown.span_cost("a")
    assert entry is not None
    assert entry.input_tokens == 200
    assert entry.cached_tokens == 800
    assert entry.input_cost == pytest.approx(0.002)
    assert entry.cached_cost == pytest.approx(0.00008)
    assert entry.total_cost == pytest.approx(0.00208)


def test_cached_tokens_are_never_double_counted() -> None:
    spans = [llm_span("a", None, model="alpha", prompt=1000, cached=1500)]
    breakdown = attribute_trace_cost(spans, book())
    entry = breakdown.span_cost("a")
    assert entry is not None
    assert entry.input_tokens == 0
    assert entry.cached_tokens == 1000
    assert entry.total_tokens == 1000


def test_the_sum_invariant_holds() -> None:
    spans = [
        llm_span("a", None, model="alpha", prompt=1000, completion=200),
        llm_span("b", "a", model="beta", prompt=500, completion=100),
        llm_span("c", "a", kind=SpanKind.TOOL, cost=0.05, name="tool"),
    ]
    breakdown = attribute_trace_cost(spans, book())
    assert sum(entry.total_cost for entry in breakdown.per_span) == pytest.approx(
        breakdown.total_cost
    )
    assert sum(breakdown.cost_by_kind.values()) == pytest.approx(breakdown.total_cost)
    assert sum(breakdown.cost_by_model.values()) == pytest.approx(breakdown.total_cost)


def test_cost_breakdown_rejects_a_broken_invariant() -> None:
    from tracetopology.cost import CostBreakdown

    with pytest.raises(ValueError, match="invariant violated"):
        CostBreakdown(per_span=[], total_cost=1.0)


def test_cost_by_kind_and_model_are_reported() -> None:
    spans = [
        llm_span("a", None, model="alpha", prompt=1000, completion=0),
        llm_span("b", "a", model="beta", prompt=1000, completion=0),
        llm_span("c", "a", kind=SpanKind.RETRIEVAL, cost=0.25, name="search"),
    ]
    breakdown = attribute_trace_cost(spans, book())
    assert breakdown.cost_by_kind["llm"] == pytest.approx(0.001 + 0.010)
    assert breakdown.cost_by_kind["retrieval"] == pytest.approx(0.25)
    assert breakdown.cost_by_model["beta"] == pytest.approx(0.010)
    assert breakdown.tokens_by_model["alpha"] == 1000


def test_recorded_cost_is_used_when_the_model_is_unknown() -> None:
    spans = [llm_span("a", None, model="mystery", prompt=100, cost=0.42)]
    breakdown = attribute_trace_cost(spans, book())
    entry = breakdown.span_cost("a")
    assert entry is not None
    assert entry.total_cost == pytest.approx(0.42)
    assert entry.source == "attribute"
    assert breakdown.unpriced_spans == []


def test_unknown_model_without_a_recorded_cost_is_reported_as_unpriced() -> None:
    spans = [llm_span("a", None, model="mystery", prompt=100)]
    breakdown = attribute_trace_cost(spans, book())
    assert breakdown.unpriced_spans == ["a"]
    assert breakdown.total_cost == 0.0


def test_retry_costs_five_times_but_only_one_is_on_the_critical_path() -> None:
    # A planner fans out to five identical tool retries, then one final LLM call.
    spans = [
        llm_span("root", None, kind=SpanKind.AGENT, start_ms=0, end_ms=100, name="planner"),
    ]
    for index in range(5):
        spans.append(
            llm_span(
                f"retry{index}",
                "root",
                kind=SpanKind.TOOL,
                cost=0.10,
                start_ms=index * 10,
                end_ms=index * 10 + 5,
                name="web_search",
            )
        )
    spans.append(
        llm_span(
            "final",
            "root",
            model="alpha",
            prompt=1000,
            completion=1000,
            start_ms=60,
            end_ms=100,
            name="final_llm",
        )
    )
    breakdown = attribute_trace_cost(spans, book())
    # Five retries at $0.10 each, plus the final LLM call.
    assert breakdown.total_cost == pytest.approx(0.50 + 0.003)
    # The critical path runs root -> final_llm only.
    assert breakdown.critical_path_ids == ["root", "final"]
    assert breakdown.critical_path_cost == pytest.approx(0.003)
    assert breakdown.off_path_cost == pytest.approx(0.50)
    assert breakdown.wasted_cost == pytest.approx(0.50)
    assert breakdown.critical_path_share < 0.01


def test_critical_path_cost_plus_off_path_cost_equals_the_total() -> None:
    spans = [
        llm_span("root", None, kind=SpanKind.AGENT, start_ms=0, end_ms=100, name="planner"),
        llm_span("a", "root", model="alpha", prompt=1000, start_ms=0, end_ms=40, name="a"),
        llm_span("b", "root", model="alpha", prompt=1000, start_ms=0, end_ms=80, name="b"),
    ]
    breakdown = attribute_trace_cost(spans, book())
    assert breakdown.critical_path_cost + breakdown.off_path_cost == pytest.approx(
        breakdown.total_cost
    )
    assert breakdown.critical_path_ids == ["root", "b"]


def test_token_based_attribution_counts_tokens_per_model() -> None:
    spans = [
        llm_span("a", None, model="alpha", prompt=100, completion=50),
        llm_span("b", "a", model="beta", prompt=200, completion=25),
    ]
    breakdown = attribute_trace_cost(spans, book())
    assert breakdown.tokens_by_model == {"alpha": 150, "beta": 225}
    assert sum(breakdown.tokens_by_model.values()) == 375


def test_attribution_accepts_a_trace_tree() -> None:
    spans = [llm_span("a", None, model="alpha", prompt=1000)]
    tree = build_trace_tree(spans)
    from_tree = attribute_trace_cost(tree, book())
    from_list = attribute_trace_cost(spans, book())
    assert from_tree.total_cost == pytest.approx(from_list.total_cost)


def test_empty_trace_has_zero_cost() -> None:
    breakdown = attribute_trace_cost([], book())
    assert breakdown.total_cost == 0.0
    assert breakdown.per_span == []
    assert breakdown.critical_path_share == 0.0


def test_breakdown_summary_is_json_friendly() -> None:
    spans = [llm_span("a", None, model="alpha", prompt=1000)]
    summary = attribute_trace_cost(spans, book()).summary()
    assert summary["total_cost"] == pytest.approx(0.001)
    assert summary["cost_by_model"] == {"alpha": pytest.approx(0.001)}
    assert summary["unpriced_spans"] == []


def test_attribution_is_deterministic() -> None:
    spans = [
        llm_span("a", None, model="alpha", prompt=1000, completion=10),
        llm_span("b", "a", model="beta", prompt=500),
    ]
    first = attribute_trace_cost(spans, book())
    second = attribute_trace_cost(list(reversed(spans)), book())
    assert first.total_cost == pytest.approx(second.total_cost)
    assert first.critical_path_ids == second.critical_path_ids


def test_a_cost_resolver_makes_the_tree_agree_with_the_attribution() -> None:
    from tracetopology import span_cost_resolver

    spans = [
        llm_span("a", None, model="alpha", prompt=1000, completion=0, kind=SpanKind.AGENT),
        llm_span("b", "a", model="alpha", prompt=1000, completion=0),
    ]
    prices = book()
    tree = build_trace_tree(spans, cost_resolver=span_cost_resolver(prices))
    breakdown = attribute_trace_cost(spans, prices)
    assert tree.total_cost() == pytest.approx(breakdown.total_cost)
    assert tree.cost_by_kind()["llm"] == pytest.approx(breakdown.cost_by_kind["llm"])


def test_the_cost_resolver_uses_a_recorded_cost_over_the_price_book() -> None:
    from tracetopology import span_cost_resolver

    spans = [llm_span("a", None, model="alpha", prompt=1000, cost=0.99)]
    resolver = span_cost_resolver(book())
    assert resolver(spans[0]) == pytest.approx(0.99)


def test_a_price_book_can_be_built_incrementally() -> None:
    prices = PriceBook()
    prices.add(ModelPrice("gamma", 0.001, 0.002))
    prices.add_alias("g3", "gamma")
    assert prices.resolve("g3") == "gamma"
    assert prices.price_for("gamma") is not None
    assert prices.price_for("missing") is None


def test_a_model_without_a_cache_discount_uses_the_input_price() -> None:
    prices = PriceBook({"solo": ModelPrice("solo", 0.001, 0.002)})
    spans = [llm_span("a", None, model="solo", prompt=1000, cached=1000)]
    breakdown = attribute_trace_cost(spans, prices)
    entry = breakdown.span_cost("a")
    assert entry is not None
    assert entry.cached_cost == pytest.approx(0.001)
