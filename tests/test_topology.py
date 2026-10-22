"""Tests for trace tree construction and topology analysis."""

from __future__ import annotations

import pytest

from tracetopology import (
    LLMAgentTopology,
    Span,
    SpanKind,
    TraceTopologyError,
    build_trace_tree,
)

MS = 1_000_000


def span(
    span_id: str,
    parent: str | None,
    *,
    name: str = "span",
    kind: SpanKind = SpanKind.CHAIN,
    start_ms: float = 0.0,
    end_ms: float = 10.0,
    cost: float = 0.0,
    attributes: dict | None = None,
    model: str | None = None,
) -> Span:
    """Build a span with millisecond timestamps."""
    attrs = dict(attributes or {})
    if model is not None:
        attrs["gen_ai.request.model"] = model
    if cost:
        attrs["cost.usd"] = cost
    return Span(
        span_id=span_id,
        parent_span_id=parent,
        trace_id="t1",
        name=name,
        kind=kind,
        start_ns=int(start_ms * MS),
        end_ns=int(end_ms * MS),
        attributes=attrs,
    )


def linear_trace() -> list[Span]:
    """A -> B -> C, a simple chain."""
    return [
        span("a", None, name="root", kind=SpanKind.AGENT, start_ms=0, end_ms=100),
        span("b", "a", name="middle", kind=SpanKind.TOOL, start_ms=10, end_ms=60),
        span("c", "b", name="leaf", kind=SpanKind.LLM, start_ms=20, end_ms=50),
    ]


def test_parent_child_links_are_correct() -> None:
    tree = build_trace_tree(linear_trace())
    assert tree.root.span.span_id == "a"
    assert [c.span.span_id for c in tree.root.children] == ["b"]
    assert [c.span.span_id for c in tree.root.children[0].children] == ["c"]


def test_walk_is_pre_order() -> None:
    tree = build_trace_tree(
        [
            span("a", None, name="root"),
            span("b", "a", name="left"),
            span("c", "a", name="right"),
            span("d", "b", name="left-left"),
        ]
    )
    assert [node.span.span_id for node in tree.walk()] == ["a", "b", "d", "c"]


def test_depth_is_correct() -> None:
    tree = build_trace_tree(linear_trace())
    assert tree.depth() == 2
    assert tree.root.children[0].depth == 1
    assert tree.root.children[0].children[0].depth == 2


def test_orphans_attach_to_a_synthetic_root() -> None:
    tree = build_trace_tree(
        [
            span("orphan1", "missing", name="o1"),
            span("orphan2", "also-missing", name="o2"),
        ]
    )
    assert tree.root.is_synthetic is True
    assert [c.span.span_id for c in tree.root.children] == ["orphan1", "orphan2"]
    assert len(tree) == 2


def test_orphan_does_not_hide_a_real_root() -> None:
    tree = build_trace_tree(
        [
            span("a", None, name="root"),
            span("orphan", "nope", name="o"),
        ]
    )
    # The orphan's parent was never exported, so it gets its own synthetic
    # root rather than being silently re-parented under the real root span.
    assert tree.root.is_synthetic is True
    assert [c.span.span_id for c in tree.root.children] == ["a", "orphan"]


def test_duplicate_span_ids_raise_a_clear_error() -> None:
    with pytest.raises(TraceTopologyError, match="duplicate span_id"):
        build_trace_tree([span("a", None), span("a", None)])


def test_parent_cycle_raises_a_clear_error() -> None:
    with pytest.raises(TraceTopologyError, match="cycle detected"):
        build_trace_tree([span("a", "b"), span("b", "a")])


def test_self_parent_cycle_raises_a_clear_error() -> None:
    with pytest.raises(TraceTopologyError, match="cycle detected"):
        build_trace_tree([span("a", "a")])


def test_empty_trace_produces_an_empty_root() -> None:
    tree = build_trace_tree([])
    assert tree.root.is_synthetic is True
    assert len(tree) == 0
    assert tree.depth() == 0
    assert tree.total_cost() == 0.0
    assert tree.leaf_count() == 0


def test_total_cost_and_tokens_equal_the_sum_over_leaves() -> None:
    spans = [
        span("a", None, name="root", cost=0.01, start_ms=0, end_ms=100),
        span("b", "a", name="mid", cost=0.02, start_ms=0, end_ms=50),
        span("c", "a", name="leaf", cost=0.03, start_ms=0, end_ms=20),
    ]
    tree = build_trace_tree(spans)
    assert tree.total_cost() == pytest.approx(0.06)
    assert tree.cost_by_kind()["chain"] == pytest.approx(0.06)
    assert tree.leaf_count() == 2


def test_cost_by_kind_totals_match_the_grand_total() -> None:
    spans = [
        span("a", None, name="root", kind=SpanKind.AGENT, cost=0.01, end_ms=100),
        span("b", "a", name="t", kind=SpanKind.TOOL, cost=0.02, end_ms=50),
        span("c", "a", name="l", kind=SpanKind.LLM, cost=0.03, end_ms=20),
    ]
    tree = build_trace_tree(spans)
    by_kind = tree.cost_by_kind()
    assert sum(by_kind.values()) == pytest.approx(tree.total_cost())
    assert by_kind["llm"] == pytest.approx(0.03)
    assert by_kind["tool"] == pytest.approx(0.02)


def test_cost_by_model_groups_and_sorts() -> None:
    spans = [
        span("a", None, name="l1", kind=SpanKind.LLM, cost=0.01, model="gpt-4o"),
        span("b", None, name="l2", kind=SpanKind.LLM, cost=0.05, model="claude"),
        span("c", None, name="l3", kind=SpanKind.LLM, cost=0.02, model="gpt-4o"),
    ]
    tree = build_trace_tree(spans)
    by_model = tree.cost_by_model()
    assert list(by_model) == ["claude", "gpt-4o"]
    assert by_model["gpt-4o"] == pytest.approx(0.03)


def test_self_duration_excludes_child_time() -> None:
    tree = build_trace_tree(
        [
            span("a", None, name="root", start_ms=0, end_ms=100),
            span("b", "a", name="child", start_ms=10, end_ms=50),
        ]
    )
    assert tree.root.duration_ms == pytest.approx(100.0)
    assert tree.root.children[0].duration_ms == pytest.approx(40.0)
    assert tree.root.self_duration_ms == pytest.approx(60.0)


def test_critical_path_picks_the_slowest_chain() -> None:
    # root(0-100) -> fast(10-20) and slow(30-90)
    tree = build_trace_tree(
        [
            span("a", None, name="root", start_ms=0, end_ms=100),
            span("fast", "a", name="fast", start_ms=10, end_ms=20),
            span("slow", "a", name="slow", start_ms=30, end_ms=90),
        ]
    )
    path = [node.span.span_id for node in tree.critical_path()]
    assert path == ["a", "slow"]
    # root self = 100 - (10 + 60) = 30ms; slow self = 60ms; total 90ms.
    assert tree.critical_path_duration_ms() == pytest.approx(30.0 + 60.0)


def test_critical_path_is_a_max_sum_chain_not_a_greedy_one() -> None:
    # root(0-100) has children: A(0-50) with a slow grandchild, B(0-40).
    # Greedy-by-self-time picks B (self 40) but the max sum is A + grandchild.
    tree = build_trace_tree(
        [
            span("root", None, name="root", start_ms=0, end_ms=100),
            span("a", "root", name="a", start_ms=0, end_ms=50),
            span("a_child", "a", name="a_child", start_ms=0, end_ms=45),
            span("b", "root", name="b", start_ms=0, end_ms=40),
        ]
    )
    path = [node.span.span_id for node in tree.critical_path()]
    assert path == ["root", "a", "a_child"]


def test_critical_path_of_a_single_span() -> None:
    tree = build_trace_tree([span("only", None, start_ms=0, end_ms=25)])
    assert [n.span.span_id for n in tree.critical_path()] == ["only"]
    assert tree.critical_path_duration_ms() == pytest.approx(25.0)


def test_render_tree_produces_expected_indented_text() -> None:
    tree = build_trace_tree(
        [
            span("a", None, name="root", kind=SpanKind.AGENT, start_ms=0, end_ms=100),
            span("b", "a", name="tool", kind=SpanKind.TOOL, start_ms=10, end_ms=50),
            span("c", "a", name="llm", kind=SpanKind.LLM, start_ms=60, end_ms=100),
        ]
    )
    assert tree.render_tree() == (
        "root [agent]  100.00ms  self=20.00ms  cost=$0.000000\n"
        "|-- tool [tool]  40.00ms  self=40.00ms  cost=$0.000000\n"
        "`-- llm [llm]  40.00ms  self=40.00ms  cost=$0.000000"
    )


def test_find_by_kind_returns_matching_nodes() -> None:
    tree = build_trace_tree(
        [
            span("a", None, name="root", kind=SpanKind.AGENT),
            span("b", "a", name="tool", kind=SpanKind.TOOL),
            span("c", "a", name="tool2", kind=SpanKind.TOOL),
        ]
    )
    tools = tree.find_by_kind(SpanKind.TOOL)
    assert [n.span.name for n in tools] == ["tool", "tool2"]
    assert tree.find_by_kind("llm") == []


def test_to_flame_dict_shape() -> None:
    tree = build_trace_tree(
        [
            span("a", None, name="root", start_ms=0, end_ms=100),
            span("b", "a", name="child", start_ms=0, end_ms=25),
        ]
    )
    flame = tree.to_flame_dict()
    assert flame["name"] == "root"
    assert flame["value"] == pytest.approx(100.0)
    assert flame["children"][0]["name"] == "child"
    assert flame["children"][0]["value"] == pytest.approx(25.0)


def test_node_path_is_ancestor_joined() -> None:
    tree = build_trace_tree(linear_trace())
    leaf = tree.root.children[0].children[0]
    assert leaf.path == "a/b/c"


def test_find_returns_node_or_none() -> None:
    tree = build_trace_tree(linear_trace())
    assert tree.find("b") is not None
    assert tree.find("zzz") is None


def test_retry_loop_detection_flags_identical_tool_calls() -> None:
    spans = [
        span("a", None, name="agent", kind=SpanKind.AGENT, end_ms=100),
        span(
            "t1",
            "a",
            name="web_search",
            kind=SpanKind.TOOL,
            attributes={"tool.arguments": "q=weather"},
        ),
        span(
            "t2",
            "a",
            name="web_search",
            kind=SpanKind.TOOL,
            attributes={"tool.arguments": "q=weather"},
        ),
        span(
            "t3",
            "a",
            name="web_search",
            kind=SpanKind.TOOL,
            attributes={"tool.arguments": "q=weather"},
        ),
        span(
            "t4", "a", name="other", kind=SpanKind.TOOL, attributes={"tool.arguments": "q=weather"}
        ),
    ]
    tree = build_trace_tree(spans)
    loops = LLMAgentTopology(tree).retry_loops()
    assert len(loops) == 1
    assert loops[0].tool_name == "web_search"
    assert loops[0].occurrences == 3
    assert loops[0].span_ids == ["t1", "t2", "t3"]


def test_retry_loop_ignores_differing_arguments() -> None:
    spans = [
        span("a", None, name="agent", kind=SpanKind.AGENT),
        span(
            "t1", "a", name="web_search", kind=SpanKind.TOOL, attributes={"tool.arguments": "q=a"}
        ),
        span(
            "t2", "a", name="web_search", kind=SpanKind.TOOL, attributes={"tool.arguments": "q=b"}
        ),
    ]
    tree = build_trace_tree(spans)
    assert LLMAgentTopology(tree).retry_loops() == []


def test_tool_fan_out_is_counted_per_agent() -> None:
    spans = [
        span("a", None, name="planner", kind=SpanKind.AGENT, end_ms=100),
        span("t1", "a", name="s1", kind=SpanKind.TOOL),
        span("t2", "a", name="s2", kind=SpanKind.TOOL),
        span("t3", "a", name="s3", kind=SpanKind.TOOL),
        span("b", None, name="other", kind=SpanKind.AGENT, end_ms=50),
        span("t4", "b", name="s4", kind=SpanKind.TOOL),
    ]
    tree = build_trace_tree(spans)
    topology = LLMAgentTopology(tree)
    assert topology.fan_out() == {"planner": 3, "other": 1}
    assert topology.tool_calls_by_agent()["planner"] == 3


def test_cost_concentration_finds_the_most_expensive_span() -> None:
    spans = [
        span("a", None, name="root", kind=SpanKind.AGENT, cost=0.01),
        span("b", "a", name="cheap", kind=SpanKind.TOOL, cost=0.02),
        span("c", "a", name="expensive", kind=SpanKind.LLM, cost=0.50),
    ]
    tree = build_trace_tree(spans)
    topology = LLMAgentTopology(tree)
    top = topology.cost_concentration()
    assert top is not None
    assert top.span.name == "expensive"
    assert topology.cost_concentration_share() == pytest.approx(0.50 / 0.53)


def test_cost_concentration_is_none_for_an_empty_trace() -> None:
    tree = build_trace_tree([])
    assert LLMAgentTopology(tree).cost_concentration() is None
    assert LLMAgentTopology(tree).cost_concentration_share() == 0.0


def test_tokens_by_model_groups_the_tokens() -> None:
    spans = [
        span(
            "a",
            None,
            name="l1",
            kind=SpanKind.LLM,
            model="gpt-4o",
            attributes={"llm.token_count.prompt": 100},
        ),
        span(
            "b",
            None,
            name="l2",
            kind=SpanKind.LLM,
            model="gpt-4o",
            attributes={"llm.token_count.prompt": 50},
        ),
        span(
            "c",
            None,
            name="l3",
            kind=SpanKind.LLM,
            model="claude",
            attributes={"llm.token_count.prompt": 10},
        ),
    ]
    tree = build_trace_tree(spans)
    assert list(tree.tokens_by_model()) == ["gpt-4o", "claude"]
    assert tree.tokens_by_model()["gpt-4o"] == 150
    assert tree.tokens_by_model()["claude"] == 10


def test_to_span_tree_includes_the_kind() -> None:
    tree = build_trace_tree([span("a", None, name="root", kind=SpanKind.AGENT)])
    assert tree.to_span_tree()["kind"] == "agent"


def test_a_synthetic_root_renders_without_a_cost_column() -> None:
    tree = build_trace_tree([span("orphan", "missing", name="o1")])
    rendered = tree.render_tree()
    assert rendered == "(root)\n`-- o1  10.00ms  self=10.00ms  cost=$0.000000"


def test_the_critical_path_ignores_a_synthetic_root() -> None:
    tree = build_trace_tree([span("orphan", "missing", name="o1", end_ms=25.0)])
    assert [n.span.span_id for n in tree.critical_path()] == ["orphan"]
