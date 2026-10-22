"""Exact cost attribution for LLM traces.

The central promise of this module is an *invariant*: the attributed total is
exactly the sum of the per-span costs. Every rollup is derived from the same
per-span numbers, so a report can never quietly disagree with itself.

It also separates two questions that agent operators constantly confuse:

* **Total cost** - what the whole trace cost, including retries and
  speculative branches that were thrown away.
* **Critical-path cost** - what the slowest path cost, which is the cost of
  the work that actually determined when the user got an answer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tracetopology.spans import Span
from tracetopology.topology import TraceTree, build_trace_tree

__all__ = [
    "CostBreakdown",
    "ModelPrice",
    "PriceBook",
    "SpanCost",
    "attribute_trace_cost",
    "span_cost_resolver",
]


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Price of one model in US dollars per 1,000 tokens.

    Attributes:
        model: Canonical model name.
        input_per_1k: Dollar cost of 1,000 uncached input tokens.
        output_per_1k: Dollar cost of 1,000 output tokens.
        cached_input_per_1k: Dollar cost of 1,000 cached input tokens. Falls
            back to ``input_per_1k`` when the model has no cache discount.
    """

    model: str
    input_per_1k: float
    output_per_1k: float
    cached_input_per_1k: float | None = None

    @property
    def effective_cached_per_1k(self) -> float:
        """Cached input price, defaulting to the regular input price."""
        if self.cached_input_per_1k is None:
            return self.input_per_1k
        return self.cached_input_per_1k


class PriceBook:
    """A model -> price table with alias resolution.

    Args:
        prices: Mapping of canonical model name to :class:`ModelPrice`.
        aliases: Mapping of alternative model names (or vendor prefixes) to
            canonical model names.
    """

    def __init__(
        self,
        prices: Mapping[str, ModelPrice] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        self._prices: dict[str, ModelPrice] = dict(prices or {})
        self._aliases: dict[str, str] = dict(aliases or {})

    def add(self, price: ModelPrice) -> None:
        """Register a model price."""
        self._prices[price.model] = price

    def add_alias(self, alias: str, model: str) -> None:
        """Register an alias pointing at a canonical model name."""
        self._aliases[alias] = model

    def resolve(self, model: str | None) -> str | None:
        """Resolve a model name through the alias map, case-insensitively."""
        if model is None:
            return None
        if model in self._prices:
            return model
        if model in self._aliases:
            return self._aliases[model]
        lowered = model.strip().lower()
        for candidate in self._prices:
            if candidate.lower() == lowered:
                return candidate
        for alias, target in self._aliases.items():
            if alias.lower() == lowered:
                return target
        return None

    def price_for(self, model: str | None) -> ModelPrice | None:
        """Return the price entry for a model, or ``None`` when unknown."""
        resolved = self.resolve(model)
        if resolved is None:
            return None
        return self._prices.get(resolved)

    def models(self) -> list[str]:
        """Sorted list of the canonical model names in the book."""
        return sorted(self._prices)

    def __len__(self) -> int:
        """Number of canonical models in the book."""
        return len(self._prices)

    def __contains__(self, model: object) -> bool:
        """``True`` when ``model`` resolves to a known price entry."""
        return isinstance(model, str) and self.resolve(model) is not None

    @classmethod
    def default(cls) -> PriceBook:
        """Return a price book with representative public model prices.

        The numbers are illustrative reference values, not a live price feed.
        """
        return cls(
            prices={
                "gpt-4o": ModelPrice("gpt-4o", 0.0025, 0.0100, 0.00125),
                "gpt-4o-mini": ModelPrice("gpt-4o-mini", 0.00015, 0.00060, 0.000075),
                "claude-3-5-sonnet": ModelPrice("claude-3-5-sonnet", 0.0030, 0.0150, 0.00030),
                "claude-3-haiku": ModelPrice("claude-3-haiku", 0.00025, 0.00125, 0.00003),
                "llama-3.1-70b": ModelPrice("llama-3.1-70b", 0.00060, 0.00060),
            },
            aliases={
                "gpt-4o-2024-08-06": "gpt-4o",
                "gpt-4o-latest": "gpt-4o",
                "openai/gpt-4o": "gpt-4o",
                "gpt-4o-mini-latest": "gpt-4o-mini",
                "anthropic/claude-3-5-sonnet-20241022": "claude-3-5-sonnet",
                "claude-3-5-sonnet-latest": "claude-3-5-sonnet",
                "claude-3-5-haiku-latest": "claude-3-haiku",
                "meta-llama/Llama-3.1-70B-Instruct": "llama-3.1-70b",
            },
        )


@dataclass(slots=True)
class SpanCost:
    """The cost of a single span, with the arithmetic that produced it.

    Attributes:
        span_id: Id of the span.
        name: Name of the span.
        kind: Kind of the span.
        model: Model the span used, when known.
        input_tokens: Uncached input tokens billed.
        cached_tokens: Cached input tokens billed at the discounted rate.
        output_tokens: Output tokens billed.
        input_cost: Dollar cost of the uncached input tokens.
        cached_cost: Dollar cost of the cached input tokens.
        output_cost: Dollar cost of the output tokens.
        recorded_cost: Dollar cost carried by the span itself, used verbatim
            because an observed cost beats a static price-table estimate.
        source: Where the cost came from: ``"attribute"`` when the span
            carried its own cost, ``"price_book"`` when it was derived, or
            ``"unknown"`` when neither was available.
    """

    span_id: str
    name: str
    kind: str
    model: str | None
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    input_cost: float
    cached_cost: float
    output_cost: float
    source: str
    recorded_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        """Total dollar cost of the span."""
        return self.input_cost + self.cached_cost + self.output_cost + self.recorded_cost

    @property
    def total_tokens(self) -> int:
        """Total tokens billed for the span."""
        return self.input_tokens + self.cached_tokens + self.output_tokens


@dataclass(slots=True)
class CostBreakdown:
    """The full cost picture of one trace.

    Attributes:
        per_span: Cost of every span, in tree order.
        total_cost: Total cost of the trace.
        cost_by_kind: Cost grouped by span kind.
        cost_by_model: Cost grouped by model.
        tokens_by_model: Tokens grouped by model.
        critical_path_cost: Cost of the spans on the critical path.
        critical_path_ids: Ids of the spans on the critical path.
        off_path_cost: Cost of the spans that are not on the critical path.
        unpriced_spans: Ids of spans for which no price could be determined.
    """

    per_span: list[SpanCost] = field(default_factory=list)
    total_cost: float = 0.0
    cost_by_kind: dict[str, float] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    tokens_by_model: dict[str, int] = field(default_factory=dict)
    critical_path_cost: float = 0.0
    critical_path_ids: list[str] = field(default_factory=list)
    off_path_cost: float = 0.0
    unpriced_spans: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Assert the core invariant: the parts sum to the whole."""
        parts = sum(span.total_cost for span in self.per_span)
        if abs(parts - self.total_cost) > 1e-9:
            raise ValueError(
                "cost attribution invariant violated: "
                f"per-span sum {parts!r} != total {self.total_cost!r}"
            )
        kind_total = sum(self.cost_by_kind.values())
        if abs(kind_total - self.total_cost) > 1e-9:
            raise ValueError(
                "cost attribution invariant violated: "
                f"cost_by_kind sum {kind_total!r} != total {self.total_cost!r}"
            )

    @property
    def wasted_cost(self) -> float:
        """Cost of the work that did not contribute to the critical path."""
        return self.off_path_cost

    @property
    def critical_path_share(self) -> float:
        """Fraction of the total cost carried by the critical path."""
        if self.total_cost <= 0.0:
            return 0.0
        return self.critical_path_cost / self.total_cost

    def span_cost(self, span_id: str) -> SpanCost | None:
        """Return the per-span cost entry for ``span_id``."""
        for span in self.per_span:
            if span.span_id == span_id:
                return span
        return None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the breakdown."""
        return {
            "total_cost": self.total_cost,
            "cost_by_kind": dict(self.cost_by_kind),
            "cost_by_model": dict(self.cost_by_model),
            "critical_path_cost": self.critical_path_cost,
            "off_path_cost": self.off_path_cost,
            "unpriced_spans": list(self.unpriced_spans),
        }


def _span_cost(span: Span, price_book: PriceBook) -> SpanCost:
    """Compute the cost of one span.

    A cost recorded on the span always wins: an observed cost is ground
    truth, while a price-book figure is an estimate that can be stale. The
    price book is only used for spans that carry no cost of their own.
    """
    model = price_book.resolve(span.model) or span.model
    input_tokens = span.input_tokens
    cached_tokens = min(span.cached_tokens, input_tokens)
    output_tokens = span.output_tokens
    uncached_input = max(0, input_tokens - cached_tokens)
    price = price_book.price_for(span.model)

    recorded = span.cost_usd
    if recorded > 0.0:
        return SpanCost(
            span_id=span.span_id,
            name=span.name,
            kind=span.kind.value,
            model=model,
            input_tokens=uncached_input,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            input_cost=0.0,
            cached_cost=0.0,
            output_cost=0.0,
            source="attribute",
            recorded_cost=recorded,
        )

    if price is not None:
        input_cost = uncached_input / 1000.0 * price.input_per_1k
        cached_cost = cached_tokens / 1000.0 * price.effective_cached_per_1k
        output_cost = output_tokens / 1000.0 * price.output_per_1k
        source = "price_book"
    else:
        input_cost = 0.0
        cached_cost = 0.0
        output_cost = 0.0
        source = "unknown"

    return SpanCost(
        span_id=span.span_id,
        name=span.name,
        kind=span.kind.value,
        model=model,
        input_tokens=uncached_input,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        cached_cost=cached_cost,
        output_cost=output_cost,
        source=source,
    )


def span_cost_resolver(price_book: PriceBook) -> Callable[[Span], float]:
    """Return a resolver that prices a single span with ``price_book``.

    The resolver follows the same precedence as
    :func:`attribute_trace_cost`: a cost recorded on the span wins, and the
    price book is used only when the span carries none. Passing this
    resolver to :func:`~tracetopology.topology.build_trace_tree` makes the
    tree's cost rollups agree with the attribution breakdown.

    Args:
        price_book: The price table to price spans with.

    Returns:
        A callable mapping a :class:`~tracetopology.spans.Span` to dollars.
    """

    def resolve(span: Span) -> float:
        return _span_cost(span, price_book).total_cost

    return resolve


def attribute_trace_cost(
    trace: TraceTree | Sequence[Span] | Iterable[Span],
    price_book: PriceBook,
) -> CostBreakdown:
    """Attribute the cost of a trace down to the individual span.

    The result is internally consistent by construction: the total is the sum
    of the per-span entries, and the per-kind rollup sums to the same total.
    The assertion in :class:`CostBreakdown` enforces it.

    Args:
        trace: A :class:`~tracetopology.topology.TraceTree` or a flat list of
            spans belonging to one trace.
        price_book: The price table used to price spans that do not carry
            their own cost.

    Returns:
        The full cost breakdown of the trace.
    """
    tree = trace if isinstance(trace, TraceTree) else build_trace_tree(trace)
    nodes = [node for node in tree.walk() if not node.is_synthetic]

    per_span: list[SpanCost] = []
    cost_by_kind: dict[str, float] = {}
    cost_by_model: dict[str, float] = {}
    tokens_by_model: dict[str, int] = {}
    unpriced: list[str] = []

    for node in nodes:
        span = node.span
        entry = _span_cost(span, price_book)
        per_span.append(entry)
        cost_by_kind[entry.kind] = cost_by_kind.get(entry.kind, 0.0) + entry.total_cost
        model_key = entry.model or "unknown"
        cost_by_model[model_key] = cost_by_model.get(model_key, 0.0) + entry.total_cost
        tokens_by_model[model_key] = tokens_by_model.get(model_key, 0) + entry.total_tokens
        if entry.source == "unknown":
            unpriced.append(entry.span_id)

    total = sum(entry.total_cost for entry in per_span)

    critical_ids = [node.span.span_id for node in tree.critical_path()]
    critical_set = set(critical_ids)
    critical_cost = sum(entry.total_cost for entry in per_span if entry.span_id in critical_set)

    breakdown = CostBreakdown(
        per_span=per_span,
        total_cost=total,
        cost_by_kind=dict(sorted(cost_by_kind.items(), key=lambda i: (-i[1], i[0]))),
        cost_by_model=dict(sorted(cost_by_model.items(), key=lambda i: (-i[1], i[0]))),
        tokens_by_model=dict(sorted(tokens_by_model.items(), key=lambda i: (-i[1], i[0]))),
        critical_path_cost=critical_cost,
        critical_path_ids=critical_ids,
        off_path_cost=total - critical_cost,
        unpriced_spans=unpriced,
    )
    return breakdown
