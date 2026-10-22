"""Span model for OpenTelemetry-style LLM traces.

The model is deliberately flat and provider-agnostic: a span is a timed unit of
work with a parent pointer, a semantic kind, and a free-form attribute bag.
Everything downstream (topology, cardinality control, cost attribution) is
derived from that shape.

LLM helpers read well-known attribute spellings so that traces exported from
OpenLLMetry, OpenInference, Langfuse, or plain OpenTelemetry all resolve
without per-vendor code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "Span",
    "SpanKind",
    "SpanStatus",
]

# Attribute spellings understood by the LLM helpers. First match wins, so the
# canonical OpenTelemetry-ish names come first and vendor fallbacks follow.
_INPUT_TOKEN_KEYS: tuple[str, ...] = (
    "llm.token_count.prompt",
    "gen_ai.usage.input_tokens",
    "llm.prompt_tokens",
    "input_tokens",
)
_OUTPUT_TOKEN_KEYS: tuple[str, ...] = (
    "llm.token_count.completion",
    "gen_ai.usage.output_tokens",
    "llm.completion_tokens",
    "output_tokens",
)
_CACHED_TOKEN_KEYS: tuple[str, ...] = (
    "llm.token_count.cached",
    "gen_ai.usage.cache_read_input_tokens",
    "llm.cached_tokens",
    "cached_tokens",
)
_MODEL_KEYS: tuple[str, ...] = (
    "gen_ai.request.model",
    "llm.model",
    "model",
)
_COST_KEYS: tuple[str, ...] = (
    "cost.usd",
    "llm.cost_usd",
    "gen_ai.cost.usd",
    "cost_usd",
)


class SpanKind(str, Enum):
    """The semantic role a span plays inside an LLM trace."""

    LLM = "llm"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    AGENT = "agent"
    CHAIN = "chain"
    HTTP = "http"

    @classmethod
    def coerce(cls, value: SpanKind | str) -> SpanKind:
        """Return ``value`` as a :class:`SpanKind`, accepting loose strings."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"unknown span kind: {value!r}") from exc


class SpanStatus(str, Enum):
    """Outcome of a span."""

    OK = "ok"
    ERROR = "error"

    @classmethod
    def coerce(cls, value: SpanStatus | str) -> SpanStatus:
        """Return ``value`` as a :class:`SpanStatus`, accepting loose strings."""
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        if text in {"ok", "unset", "success"}:
            return cls.OK
        if text in {"error", "err", "failure"}:
            return cls.ERROR
        raise ValueError(f"unknown span status: {value!r}")


def _coerce_int(value: Any) -> int:
    """Best-effort conversion of an attribute value to a non-negative int."""
    if value is None or isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return 0
    return max(0, number)


def _coerce_float(value: Any) -> float:
    """Best-effort conversion of an attribute value to a non-negative float."""
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, number)


def _first(attributes: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first present value among ``keys``."""
    for key in keys:
        if key in attributes:
            return attributes[key]
    return None


@dataclass(slots=True)
class Span:
    """A single timed unit of work inside a trace.

    Attributes:
        span_id: Unique id of this span within its trace.
        parent_span_id: Id of the enclosing span, or ``None`` for a root.
        trace_id: Id shared by every span of the same trace.
        name: Human readable span name.
        kind: Semantic role of the span.
        start_ns: Start timestamp in nanoseconds since the Unix epoch.
        end_ns: End timestamp in nanoseconds since the Unix epoch.
        attributes: Free-form attribute bag (tokens, model, cost, ...).
        events: Timestamped events recorded on the span.
        status: Outcome of the span.
        tenant_id: Optional tenant the span belongs to, used for isolation rules.
    """

    span_id: str
    parent_span_id: str | None
    trace_id: str
    name: str
    kind: SpanKind = SpanKind.CHAIN
    start_ns: int = 0
    end_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: SpanStatus = SpanStatus.OK
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        """Normalise the loosely typed ``kind`` and ``status`` fields."""
        self.kind = SpanKind.coerce(self.kind)
        self.status = SpanStatus.coerce(self.status)
        if self.attributes is None:
            self.attributes = {}
        if self.events is None:
            self.events = []

    @property
    def duration_ns(self) -> int:
        """Duration of the span in nanoseconds (never negative)."""
        return max(0, self.end_ns - self.start_ns)

    @property
    def duration_ms(self) -> float:
        """Duration of the span in milliseconds."""
        return self.duration_ns / 1_000_000.0

    @property
    def is_root(self) -> bool:
        """``True`` when the span has no parent pointer."""
        return self.parent_span_id is None

    @property
    def is_error(self) -> bool:
        """``True`` when the span status is :attr:`SpanStatus.ERROR`."""
        return self.status is SpanStatus.ERROR

    @property
    def input_tokens(self) -> int:
        """Prompt/input tokens reported by the span (``0`` when absent)."""
        return _coerce_int(_first(self.attributes, _INPUT_TOKEN_KEYS))

    @property
    def output_tokens(self) -> int:
        """Completion/output tokens reported by the span (``0`` when absent)."""
        return _coerce_int(_first(self.attributes, _OUTPUT_TOKEN_KEYS))

    @property
    def cached_tokens(self) -> int:
        """Cached prompt tokens reported by the span (``0`` when absent).

        Cached tokens are a *subset* of the input tokens: they are billed at a
        discounted rate but still count towards the prompt size.
        """
        return _coerce_int(_first(self.attributes, _CACHED_TOKEN_KEYS))

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens of the span."""
        return self.input_tokens + self.output_tokens

    @property
    def model(self) -> str | None:
        """Model identifier reported by the span (``None`` when absent)."""
        value = _first(self.attributes, _MODEL_KEYS)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @property
    def cost_usd(self) -> float:
        """Dollar cost recorded on the span (``0.0`` when absent).

        This is the *observed* cost. :mod:`tracetopology.cost` can also derive
        a cost from a price book when the span does not carry one.
        """
        return _coerce_float(_first(self.attributes, _COST_KEYS))

    def attribute(self, name: str, default: Any = None) -> Any:
        """Return an attribute value, falling back to ``default``."""
        return self.attributes.get(name, default)
