"""Turn a flat list of spans into a queryable trace topology.

A trace is a tree of spans linked by ``parent_span_id``. Real traces are
messy: parents may be missing (orphans), ids may repeat (retries exported
twice), and a broken exporter can even produce a cycle. This module builds
the tree defensively and exposes the aggregate signals an on-call engineer
actually needs: where the time went, where the money went, and which
branches are duplicated work.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from tracetopology.spans import Span, SpanKind

__all__ = [
    "LLMAgentTopology",
    "RetryLoop",
    "TraceNode",
    "TraceTree",
    "build_trace_tree",
]

SYNTHETIC_ROOT_ID = "__synthetic_root__"
SYNTHETIC_ROOT_NAME = "(root)"
_ORPHAN_KEY = "__orphans__"


class TraceTopologyError(ValueError):
    """Raised when a span collection cannot be turned into a tree."""


@dataclass(slots=True)
class TraceNode:
    """A span placed in the trace tree together with its rollups.

    Attributes:
        span: The underlying span.
        children: Direct children of this node.
        depth: Distance from the root (the synthetic root counts as depth 0).
        path: Slash-joined ancestor path ending with this node's span id.
        aggregate_cost: Sum of ``cost_usd`` over this node's whole subtree.
        aggregate_tokens: Sum of total tokens over this node's whole subtree.
        self_duration_ms: Duration of this span minus the time its children
            consumed, i.e. the time spent doing work that is not delegated.
        downward_score: Best achievable total self time from this node down
            to a leaf. Used to compute the critical path.
        cost: Dollar cost of this span alone. Zero unless a cost resolver
            was supplied to :func:`build_trace_tree`.
    """

    span: Span
    children: list[TraceNode] = field(default_factory=list)
    depth: int = 0
    path: str = ""
    aggregate_cost: float = 0.0
    aggregate_tokens: int = 0
    self_duration_ms: float = 0.0
    downward_score: float = 0.0
    cost: float = 0.0

    @property
    def span_id(self) -> str:
        """Id of the wrapped span."""
        return self.span.span_id

    @property
    def name(self) -> str:
        """Name of the wrapped span."""
        return self.span.name

    @property
    def kind(self) -> SpanKind:
        """Kind of the wrapped span."""
        return self.span.kind

    @property
    def duration_ms(self) -> float:
        """Wall-clock duration of the wrapped span in milliseconds."""
        return self.span.duration_ms

    @property
    def is_synthetic(self) -> bool:
        """``True`` for the synthetic root that holds orphaned spans."""
        return self.span.span_id == SYNTHETIC_ROOT_ID

    def walk(self) -> Iterator[TraceNode]:
        """Yield this node and all descendants in pre-order."""
        yield self
        for child in self.children:
            yield from child.walk()

    def to_span_tree(self) -> dict[str, Any]:
        """Return the subtree as a ``{name, value, children}`` mapping."""
        return {
            "name": self.span.name,
            "value": self.span.duration_ms,
            "kind": self.span.kind.value,
            "children": [child.to_span_tree() for child in self.children],
        }

    def to_flame_dict(self) -> dict[str, Any]:
        """Return the subtree as a flame-graph ready ``{name, value, children}``."""
        return {
            "name": self.span.name,
            "value": self.span.duration_ms,
            "children": [child.to_flame_dict() for child in self.children],
        }


@dataclass(slots=True)
class RetryLoop:
    """A repeated call to the same tool with the same arguments.

    Attributes:
        tool_name: Name of the repeated tool/span.
        arguments: The repeated argument payload.
        occurrences: Number of times the call appears.
        span_ids: Ids of the repeated spans, in tree order.
        total_duration_ms: Combined duration of every occurrence.
        total_cost: Combined cost of every occurrence.
    """

    tool_name: str
    arguments: str
    occurrences: int
    span_ids: list[str] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_cost: float = 0.0

    @property
    def wasted_cost(self) -> float:
        """Cost of every occurrence except the last one."""
        return self.total_cost / self.occurrences * (self.occurrences - 1)


@dataclass(slots=True)
class TraceTree:
    """A rooted tree of spans with aggregate rollups.

    Attributes:
        root: The root :class:`TraceNode`. Always present, even for an empty
            trace, so callers never have to null-check.
    """

    root: TraceNode

    def __len__(self) -> int:
        """Number of real (non-synthetic) spans in the tree."""
        return sum(1 for node in self.walk() if not node.is_synthetic)

    def walk(self) -> Iterator[TraceNode]:
        """Yield every node in pre-order (parents before children)."""
        yield from self.root.walk()

    def nodes(self) -> list[TraceNode]:
        """Return every node in pre-order as a list."""
        return list(self.walk())

    def find_by_kind(self, kind: SpanKind | str) -> list[TraceNode]:
        """Return every node whose span has the given kind."""
        target = SpanKind.coerce(kind)
        return [node for node in self.walk() if node.span.kind is target and not node.is_synthetic]

    def find(self, span_id: str) -> TraceNode | None:
        """Return the node with the given span id, or ``None``."""
        for node in self.walk():
            if node.span.span_id == span_id:
                return node
        return None

    def depth(self) -> int:
        """Maximum depth of the tree, counting the root as depth 0."""
        return max((node.depth for node in self.walk()), default=0)

    def total_cost(self) -> float:
        """Sum of the cost of every real span in the tree."""
        return sum(node.cost for node in self.walk() if not node.is_synthetic)

    def total_tokens(self) -> int:
        """Sum of the tokens of every real span in the tree."""
        return sum(node.span.total_tokens for node in self.walk() if not node.is_synthetic)

    def cost_by_kind(self) -> dict[str, float]:
        """Cost grouped by span kind, sorted by descending cost."""
        totals: dict[str, float] = {}
        for node in self.walk():
            if node.is_synthetic:
                continue
            key = node.span.kind.value
            totals[key] = totals.get(key, 0.0) + node.cost
        return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0])))

    def cost_by_model(self) -> dict[str, float]:
        """Cost grouped by model, sorted by descending cost."""
        totals: dict[str, float] = {}
        for node in self.walk():
            if node.is_synthetic:
                continue
            model = node.span.model or "unknown"
            totals[model] = totals.get(model, 0.0) + node.cost
        return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0])))

    def tokens_by_model(self) -> dict[str, int]:
        """Tokens grouped by model, sorted by descending token count."""
        totals: dict[str, int] = {}
        for node in self.walk():
            if node.is_synthetic:
                continue
            model = node.span.model or "unknown"
            totals[model] = totals.get(model, 0) + node.span.total_tokens
        return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0])))

    def leaf_count(self) -> int:
        """Number of leaves (spans without children) in the tree."""
        return sum(1 for node in self.walk() if not node.children and not node.is_synthetic)

    def critical_path(self) -> list[TraceNode]:
        """Return the chain of spans that dominates end-to-end latency.

        The critical path is the root-to-leaf chain with the largest total
        *self* time, where self time is the span duration minus the duration
        consumed by its children. A branch that fans out into slow children is
        therefore not mistaken for the bottleneck: only the time actually
        spent on the path counts.

        The chain is found with a bottom-up dynamic program: every node stores
        the best achievable self-time sum from itself down to a leaf, and the
        parent keeps the child that maximises it. This is exact, not greedy.

        A synthetic root may start the descent but is not part of the
        returned chain, because it is an artefact rather than a real span.
        Ties are broken by child order, making the result deterministic.
        """
        chain: list[TraceNode] = []
        current: TraceNode | None = self.root
        while current is not None:
            chain.append(current)
            current = self._max_score_child(current)
        if chain and chain[0].is_synthetic:
            chain = chain[1:]
        return chain

    def critical_path_duration_ms(self) -> float:
        """Total self time of the critical path."""
        return sum(node.self_duration_ms for node in self.critical_path())

    def render_tree(self) -> str:
        """Render the tree as indented ASCII, flame-chart style."""
        lines: list[str] = []
        self._render(self.root, lines, prefix="", is_last=True, is_root=True)
        return "\n".join(lines)

    def to_span_tree(self) -> dict[str, Any]:
        """Return the whole tree as a ``{name, value, children}`` mapping."""
        return self.root.to_span_tree()

    def to_flame_dict(self) -> dict[str, Any]:
        """Return the whole tree as a flame-graph ready mapping."""
        return self.root.to_flame_dict()

    # -- internals -------------------------------------------------------

    def _max_score_child(self, node: TraceNode) -> TraceNode | None:
        """Return the child of ``node`` with the largest downward score."""
        if not node.children:
            return None
        return max(node.children, key=lambda child: child.downward_score)

    def _render(
        self,
        node: TraceNode,
        lines: list[str],
        *,
        prefix: str,
        is_last: bool,
        is_root: bool = False,
    ) -> None:
        """Append the ASCII rendering of ``node`` (and its children) to ``lines``."""
        if is_root:
            connector = ""
            child_prefix = ""
        else:
            connector = "`-- " if is_last else "|-- "
            child_prefix = prefix + ("    " if is_last else "|   ")
        if node.is_synthetic:
            lines.append(f"{prefix}{connector}{node.span.name}")
        else:
            label = node.span.name
            if node.span.kind is not SpanKind.CHAIN:
                label = f"{label} [{node.span.kind.value}]"
            lines.append(
                f"{prefix}{connector}{label}  {node.duration_ms:.2f}ms  "
                f"self={node.self_duration_ms:.2f}ms  cost=${node.aggregate_cost:.6f}"
            )
        for index, child in enumerate(node.children):
            self._render(
                child,
                lines,
                prefix=child_prefix,
                is_last=index == len(node.children) - 1,
            )


def build_trace_tree(
    spans: Sequence[Span] | Iterable[Span],
    *,
    cost_resolver: Callable[[Span], float] | None = None,
) -> TraceTree:
    """Build a :class:`TraceTree` from a flat collection of spans.

    Orphans (spans whose parent is not present) are attached to a synthetic
    root so no data is lost. Duplicate span ids and parent cycles are
    rejected with a :class:`TraceTopologyError` instead of looping forever.

    Args:
        spans: The spans of a single trace.
        cost_resolver: Optional callable returning the dollar cost of a span.
            When omitted, the cost recorded on the span itself is used. The
            resolver lets a price book drive the tree's cost rollups, so the
            tree and :mod:`tracetopology.cost` never disagree.

    Returns:
        The assembled tree.

    Raises:
        TraceTopologyError: If a span id is duplicated or the parent links
            contain a cycle.
    """
    span_list = list(spans)
    resolver = cost_resolver or (lambda span: span.cost_usd)
    if not span_list:
        root = TraceNode(
            span=Span(
                span_id=SYNTHETIC_ROOT_ID,
                parent_span_id=None,
                trace_id="",
                name=SYNTHETIC_ROOT_NAME,
                kind=SpanKind.CHAIN,
            ),
            depth=0,
            path=SYNTHETIC_ROOT_ID,
        )
        return TraceTree(root=root)

    by_id: dict[str, Span] = {}
    for span in span_list:
        if span.span_id in by_id:
            raise TraceTopologyError(f"duplicate span_id: {span.span_id!r}")
        by_id[span.span_id] = span

    # A parent chain that never reaches a missing parent is a cycle.
    for span in span_list:
        seen: set[str] = set()
        current = span
        while current.parent_span_id is not None:
            if current.span_id in seen:
                raise TraceTopologyError(f"cycle detected at span_id {current.span_id!r}")
            seen.add(current.span_id)
            parent = by_id.get(current.parent_span_id)
            if parent is None:
                break
            current = parent

    children_of: dict[str | None, list[Span]] = {}
    for span in span_list:
        parent_id = span.parent_span_id
        if parent_id is not None and parent_id not in by_id:
            # Orphan: its real parent was not exported. It is kept in its own
            # bucket so it is never re-parented under an unrelated root span.
            children_of.setdefault(_ORPHAN_KEY, []).append(span)
            continue
        children_of.setdefault(parent_id, []).append(span)

    root_spans = children_of.get(None, [])
    orphan_spans = children_of.get(_ORPHAN_KEY, [])

    if not orphan_spans and len(root_spans) == 1:
        real_root_span = root_spans[0]
        root = TraceNode(
            span=real_root_span,
            depth=0,
            path=real_root_span.span_id,
            cost=resolver(real_root_span),
        )
        root_children = children_of.get(real_root_span.span_id, [])
    else:
        synthetic = Span(
            span_id=SYNTHETIC_ROOT_ID,
            parent_span_id=None,
            trace_id=span_list[0].trace_id,
            name=SYNTHETIC_ROOT_NAME,
            kind=SpanKind.CHAIN,
        )
        root = TraceNode(span=synthetic, depth=0, path=SYNTHETIC_ROOT_ID)
        root_children = list(root_spans) + list(orphan_spans)

    root.children = [
        _make_node(
            child, depth=1, parent_path=root.path, children_of=children_of, resolver=resolver
        )
        for child in root_children
    ]
    _finalise(root)
    return TraceTree(root=root)


def _make_node(
    span: Span,
    *,
    depth: int,
    parent_path: str,
    children_of: dict[str | None, list[Span]],
    resolver: Callable[[Span], float],
) -> TraceNode:
    """Create a node and recursively attach its children."""
    path = f"{parent_path}/{span.span_id}" if parent_path else span.span_id
    node = TraceNode(span=span, depth=depth, path=path, cost=resolver(span))
    node.children = [
        _make_node(
            child, depth=depth + 1, parent_path=path, children_of=children_of, resolver=resolver
        )
        for child in children_of.get(span.span_id, [])
    ]
    return node


def _finalise(node: TraceNode) -> None:
    """Compute aggregate rollups bottom-up for ``node`` and its descendants."""
    child_cost = 0.0
    child_tokens = 0
    child_duration = 0.0
    best_downward = 0.0
    for child in node.children:
        _finalise(child)
        child_cost += child.aggregate_cost
        child_tokens += child.aggregate_tokens
        child_duration += child.duration_ms
        best_downward = max(best_downward, child.downward_score)
    node.aggregate_cost = node.cost + child_cost
    node.aggregate_tokens = node.span.total_tokens + child_tokens
    node.self_duration_ms = max(0.0, node.duration_ms - child_duration)
    node.downward_score = node.self_duration_ms + best_downward


class LLMAgentTopology:
    """Agent-specific analysis: fan-out, retry loops, and cost concentration.

    Args:
        tree: The trace tree to analyse.
    """

    def __init__(self, tree: TraceTree) -> None:
        self._tree = tree

    def tool_calls_by_agent(self) -> dict[str, int]:
        """Number of tool-call descendants per agent span."""
        counts: dict[str, int] = {}
        for node in self._tree.walk():
            if node.is_synthetic or node.span.kind is not SpanKind.AGENT:
                continue
            counts[node.span.name] = counts.get(node.span.name, 0) + _count_kind(
                node, SpanKind.TOOL
            )
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def fan_out(self) -> dict[str, int]:
        """Number of direct tool children per agent span."""
        counts: dict[str, int] = {}
        for node in self._tree.walk():
            if node.is_synthetic or node.span.kind is not SpanKind.AGENT:
                continue
            direct = sum(1 for child in node.children if child.span.kind is SpanKind.TOOL)
            if direct:
                counts[node.span.name] = direct
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def retry_loops(self, *, min_occurrences: int = 2) -> list[RetryLoop]:
        """Detect repeated calls to the same tool with identical arguments.

        Two calls are considered retries when they share a tool name and an
        identical serialised argument payload. The argument payload is read
        from the usual tool-call attribute names.

        Args:
            min_occurrences: Minimum number of identical calls to report.

        Returns:
            Detected retry loops, longest first.
        """
        groups: dict[tuple[str, str], list[TraceNode]] = {}
        for node in self._tree.walk():
            if node.is_synthetic:
                continue
            key = (node.span.name, _tool_arguments(node.span))
            groups.setdefault(key, []).append(node)

        loops: list[RetryLoop] = []
        for (name, arguments), nodes in groups.items():
            if len(nodes) < min_occurrences:
                continue
            loops.append(
                RetryLoop(
                    tool_name=name,
                    arguments=arguments,
                    occurrences=len(nodes),
                    span_ids=[node.span.span_id for node in nodes],
                    total_duration_ms=sum(node.duration_ms for node in nodes),
                    total_cost=sum(node.span.cost_usd for node in nodes),
                )
            )
        loops.sort(key=lambda loop: (-loop.occurrences, loop.tool_name, loop.arguments))
        return loops

    def cost_concentration(self) -> TraceNode | None:
        """Return the single span that accounts for the most cost."""
        best: TraceNode | None = None
        for node in self._tree.walk():
            if node.is_synthetic:
                continue
            if best is None or node.cost > best.cost:
                best = node
        return best

    def cost_concentration_share(self) -> float:
        """Fraction of the total trace cost held by the most expensive span."""
        total = self._tree.total_cost()
        if total <= 0.0:
            return 0.0
        top = self.cost_concentration()
        if top is None:
            return 0.0
        return top.cost / total


def _count_kind(node: TraceNode, kind: SpanKind) -> int:
    """Count the descendants of ``node`` (inclusive) with the given kind."""
    return sum(1 for child in node.walk() if child.span.kind is kind)


def _tool_arguments(span: Span) -> str:
    """Serialise the tool-call arguments of a span into a stable string."""
    for key in ("tool.arguments", "tool.call.arguments", "llm.tool.arguments", "arguments"):
        if key in span.attributes:
            return _stable_repr(span.attributes[key])
    return ""


def _stable_repr(value: Any) -> str:
    """Return a deterministic textual representation of ``value``."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}={_stable_repr(v)}" for k, v in sorted(value.items())) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_repr(item) for item in value) + "]"
    return repr(value)
