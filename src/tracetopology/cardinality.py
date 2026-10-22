"""Cardinality control for Prometheus metrics built from LLM traces.

LLM telemetry is high-cardinality *by nature*: every request carries a
different prompt hash, a different model revision, a different session id.
Push all of that onto metric labels and the series count multiplies:

    series = product over labels of (number of distinct values)

Ten labels with fifty values each is already 97 billion series. The
explosion is multiplicative, so removing one bad label is worth far more
than removing a hundred values from one label.

This module makes the explosion visible (:class:`LabelSet` and
:class:`CardinalityReport`) and then removes it (:class:`LabelPolicy` and
:class:`CardinalityController`).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from tracetopology.spans import Span

__all__ = [
    "CardinalityController",
    "CardinalityReport",
    "CounterFamily",
    "HistogramFamily",
    "HistogramSeries",
    "LabelPolicy",
    "LabelSet",
    "MetricBuilder",
    "MetricBundle",
    "MetricSeries",
]

DEFAULT_BUCKETS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


@dataclass(slots=True)
class LabelSet:
    """The label values observed for a metric family.

    A label set maps a label name to the *set* of values seen for it. The
    number of time series a metric can produce is the product of the per
    label value counts, which is exactly how Prometheus cardinality grows.

    Attributes:
        values: Mapping of label name to the set of observed values.
    """

    values: dict[str, set[str]] = field(default_factory=dict)

    def add(self, name: str, value: str) -> None:
        """Record that ``value`` was observed for the label ``name``."""
        self.values.setdefault(name, set()).add(value)

    def update(self, labels: Mapping[str, str]) -> None:
        """Record every ``(name, value)`` pair of ``labels``."""
        for name, value in labels.items():
            self.add(name, value)

    def value_count(self, name: str) -> int:
        """Number of distinct values seen for the label ``name``."""
        return len(self.values.get(name, ()))

    def label_count(self) -> int:
        """Total number of distinct ``(label, value)`` pairs."""
        return sum(len(vals) for vals in self.values.values())

    def cardinality_estimate(self) -> int:
        """Estimated number of time series: the product of per-label counts.

        An empty label set describes a single, unlabelled series, so the
        empty product is ``1``.
        """
        total = 1
        for vals in self.values.values():
            total *= len(vals)
        return total

    def merge(self, other: LabelSet) -> None:
        """Merge the values of ``other`` into this label set."""
        for name, vals in other.values.items():
            self.values.setdefault(name, set()).update(vals)

    def sorted_values(self) -> dict[str, list[str]]:
        """Return the label values as sorted lists, for deterministic output."""
        return {name: sorted(vals) for name, vals in sorted(self.values.items())}

    def __len__(self) -> int:
        """Number of distinct labels (not values)."""
        return len(self.values)


@dataclass(frozen=True, slots=True)
class LabelPolicy:
    """Rules describing which labels may be used and how far they may vary.

    The policy applies to *label names*. For span attributes the label name is
    the attribute key (for example ``prompt.hash``); for the metric families
    built by :class:`MetricBuilder` the label names are the fixed metric
    labels (``model``, ``tenant_id``, ``kind``, ``status``, ``direction``).

    Attributes:
        allowed_labels: When non-empty, only these label names survive.
        denied_labels: Label names that are always dropped. Denials win over
            the allow list.
        drop_patterns: Regular expressions; a label whose name matches any
            pattern is dropped.
        max_values_per_label: Maximum number of distinct values a single
            label may take. ``0`` disables the cap.
        hash_remaining: When a label exceeds the cap, either bucket the
            values into a bounded number of hash buckets (``True``, keeping
            the label but losing precision) or drop the label entirely
            (``False``).
    """

    allowed_labels: tuple[str, ...] = ()
    denied_labels: tuple[str, ...] = ()
    drop_patterns: tuple[str, ...] = ()
    max_values_per_label: int = 50
    hash_remaining: bool = True

    def __post_init__(self) -> None:
        """Normalise the sequence fields to tuples and validate the cap."""
        object.__setattr__(self, "allowed_labels", tuple(self.allowed_labels))
        object.__setattr__(self, "denied_labels", tuple(self.denied_labels))
        object.__setattr__(self, "drop_patterns", tuple(self.drop_patterns))
        if self.max_values_per_label < 0:
            raise ValueError("max_values_per_label must be >= 0")

    def permits(self, name: str) -> bool:
        """Return ``True`` when the label ``name`` is allowed by this policy."""
        return self.drop_reason(name) is None

    def drop_reason(self, name: str) -> str | None:
        """Return why the label ``name`` is dropped, or ``None`` if it is kept."""
        if name in self.denied_labels:
            return "denied"
        if any(re.search(pattern, name) for pattern in self.drop_patterns):
            return "pattern"
        if self.allowed_labels and name not in self.allowed_labels:
            return "not_allow_listed"
        return None


def stable_hash_bucket(label: str, value: str, buckets: int) -> str:
    """Map ``value`` to one of ``buckets`` stable hash buckets.

    The hash is a BLAKE2b digest, so it is deterministic across processes and
    Python versions (unlike :func:`hash`).
    """
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    digest = hashlib.blake2b(
        f"{label}\x00{value}".encode(),
        digest_size=8,
    ).digest()
    index = int.from_bytes(digest, "big") % buckets
    return f"bucket_{index:04d}"


@dataclass(slots=True)
class CardinalityReport:
    """The before/after picture of a label space.

    Attributes:
        before_labels: Number of distinct ``(label, value)`` pairs before
            the policy was applied.
        after_labels: Number of distinct ``(label, value)`` pairs after the
            policy was applied.
        estimated_series_before: Product of the per-label value counts
            before the policy was applied.
        estimated_series_after: Product of the per-label value counts after
            the policy was applied.
        dropped_attributes: Attribute names that were dropped entirely.
        high_cardinality_labels: Label names that exceeded the value cap.
        hashed_labels: Label names whose values were bucketed rather than
            dropped.
        value_counts_before: Per-label distinct value counts before.
        value_counts_after: Per-label distinct value counts after.
    """

    before_labels: int = 0
    after_labels: int = 0
    estimated_series_before: int = 0
    estimated_series_after: int = 0
    dropped_attributes: list[str] = field(default_factory=list)
    high_cardinality_labels: list[str] = field(default_factory=list)
    hashed_labels: list[str] = field(default_factory=list)
    value_counts_before: dict[str, int] = field(default_factory=dict)
    value_counts_after: dict[str, int] = field(default_factory=dict)

    def reduction_pct(self) -> float:
        """Percentage reduction in estimated series count (``0.0`` if none)."""
        before = self.estimated_series_before
        if before <= 0:
            return 0.0
        return (before - self.estimated_series_after) / before * 100.0

    def label_reduction_pct(self) -> float:
        """Percentage reduction in the number of distinct label values."""
        if self.before_labels <= 0:
            return 0.0
        return (self.before_labels - self.after_labels) / self.before_labels * 100.0

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the report."""
        return {
            "before_labels": self.before_labels,
            "after_labels": self.after_labels,
            "estimated_series_before": self.estimated_series_before,
            "estimated_series_after": self.estimated_series_after,
            "reduction_pct": round(self.reduction_pct(), 4),
            "dropped_attributes": list(self.dropped_attributes),
            "high_cardinality_labels": list(self.high_cardinality_labels),
            "hashed_labels": list(self.hashed_labels),
        }


class CardinalityController:
    """Applies a :class:`LabelPolicy` to span attributes.

    Args:
        policy: The rules to apply.
    """

    def __init__(self, policy: LabelPolicy | None = None) -> None:
        self._policy = policy or LabelPolicy()

    @property
    def policy(self) -> LabelPolicy:
        """The policy in force."""
        return self._policy

    def raw_label_set(self, spans: Iterable[Span]) -> LabelSet:
        """Return every span attribute as a label set, with no filtering.

        This is the "if you just dumped everything" baseline.
        """
        label_set = LabelSet()
        for span in spans:
            for name, value in span.attributes.items():
                label_set.add(str(name), _label_value(value))
        return label_set

    def sanitise_label_set(
        self, label_set: LabelSet
    ) -> tuple[LabelSet, list[str], list[str], list[str]]:
        """Apply the policy to a label set.

        Args:
            label_set: The raw label space.

        Returns:
            A tuple of ``(sanitised, dropped, high_cardinality, hashed)``.
        """
        policy = self._policy
        sanitised = LabelSet()
        dropped: list[str] = []
        high_cardinality: list[str] = []
        hashed: list[str] = []

        for name in sorted(label_set.values):
            reason = policy.drop_reason(name)
            if reason is not None:
                dropped.append(name)
                continue
            values = label_set.values[name]
            cap = policy.max_values_per_label
            if cap and len(values) > cap:
                high_cardinality.append(name)
                if policy.hash_remaining:
                    hashed.append(name)
                    for value in values:
                        sanitised.add(name, stable_hash_bucket(name, value, cap))
                else:
                    dropped.append(name)
                continue
            for value in values:
                sanitised.add(name, value)
        return sanitised, dropped, high_cardinality, hashed

    def report(self, spans: Iterable[Span]) -> CardinalityReport:
        """Build the before/after cardinality report for a set of spans.

        The "before" picture is what you get by turning every span attribute
        into a metric label - the naive approach that blows up a metrics
        backend. The "after" picture applies the policy.
        """
        span_list = list(spans)
        raw = self.raw_label_set(span_list)
        sanitised, dropped, high_cardinality, hashed = self.sanitise_label_set(raw)
        return self._build_report(raw, sanitised, dropped, high_cardinality, hashed)

    def report_for_labels(self, label_sets: Iterable[LabelSet]) -> CardinalityReport:
        """Build the report for an explicit set of label spaces.

        This is what a metrics builder uses: the label spaces are the ones
        that actually become Prometheus series, so the report describes the
        explosion the backend would really suffer.

        Args:
            label_sets: The label spaces to assess.

        Returns:
            The before/after cardinality report.
        """
        raw = LabelSet()
        for label_set in label_sets:
            raw.merge(label_set)
        sanitised, dropped, high_cardinality, hashed = self.sanitise_label_set(raw)
        return self._build_report(raw, sanitised, dropped, high_cardinality, hashed)

    def _build_report(
        self,
        raw: LabelSet,
        sanitised: LabelSet,
        dropped: list[str],
        high_cardinality: list[str],
        hashed: list[str],
    ) -> CardinalityReport:
        """Assemble a :class:`CardinalityReport` from a before/after pair."""
        return CardinalityReport(
            before_labels=raw.label_count(),
            after_labels=sanitised.label_count(),
            estimated_series_before=raw.cardinality_estimate(),
            estimated_series_after=sanitised.cardinality_estimate(),
            dropped_attributes=sorted(dropped),
            high_cardinality_labels=sorted(high_cardinality),
            hashed_labels=sorted(hashed),
            value_counts_before={k: len(v) for k, v in sorted(raw.values.items())},
            value_counts_after={k: len(v) for k, v in sorted(sanitised.values.items())},
        )

    def sanitise_labels(
        self,
        labels: Mapping[str, str],
        *,
        value_counts: Mapping[str, int] | None = None,
    ) -> dict[str, str]:
        """Apply the policy to a single set of metric label values.

        Args:
            labels: The label values of one time series.
            value_counts: Observed distinct value count per label, used to
                decide whether the cap is exceeded.

        Returns:
            The sanitised label values.
        """
        policy = self._policy
        counts = value_counts or {}
        result: dict[str, str] = {}
        for name in sorted(labels):
            value = labels[name]
            if not policy.permits(name):
                continue
            cap = policy.max_values_per_label
            if cap and counts.get(name, 0) > cap:
                if policy.hash_remaining:
                    result[name] = stable_hash_bucket(name, value, cap)
                continue
            result[name] = value
        return result


def _label_value(value: Any) -> str:
    """Render an arbitrary attribute value as a label string."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


@dataclass(slots=True)
class MetricSeries:
    """A single counter/gauge time series.

    Attributes:
        labels: The label values of the series.
        value: The value of the series.
    """

    labels: dict[str, str]
    value: float = 0.0


@dataclass(slots=True)
class HistogramSeries:
    """A single histogram time series.

    Attributes:
        labels: The label values of the series (without ``le``).
        bucket_counts: Cumulative count per bucket upper bound.
        total_count: Total number of observations.
        total_sum: Sum of all observed values.
    """

    labels: dict[str, str]
    bucket_counts: dict[float, int] = field(default_factory=dict)
    total_count: int = 0
    total_sum: float = 0.0


def _format_value(value: float) -> str:
    """Format a float the way Prometheus expects."""
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value)}.0"
    return repr(float(value))


def _format_labels(labels: Mapping[str, str], *, last: str | None = None) -> str:
    """Render a label set as Prometheus label text (``""`` when empty).

    Args:
        labels: The label values.
        last: An optional label name that must be rendered last. Prometheus
            conventionally puts the histogram bound ``le`` last.
    """
    if not labels:
        return ""
    ordered = sorted(labels)
    if last is not None and last in labels:
        ordered = [name for name in ordered if name != last] + [last]
    parts = []
    for name in ordered:
        escaped = str(labels[name]).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
        parts.append(f'{name}="{escaped}"')
    return "{" + ",".join(parts) + "}"


@dataclass(slots=True)
class CounterFamily:
    """A Prometheus counter family.

    Attributes:
        name: Metric name.
        help: Human readable description.
        series: The time series of this family.
    """

    name: str
    help: str
    series: list[MetricSeries] = field(default_factory=list)

    def value(self, labels: Mapping[str, str]) -> float:
        """Return the value of the series with the given labels."""
        key = dict(labels)
        for entry in self.series:
            if entry.labels == key:
                return entry.value
        return 0.0

    def render(self) -> str:
        """Render the family in the Prometheus text exposition format."""
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for entry in sorted(self.series, key=lambda s: tuple(sorted(s.labels.items()))):
            lines.append(f"{self.name}{_format_labels(entry.labels)} {_format_value(entry.value)}")
        return "\n".join(lines)


@dataclass(slots=True)
class HistogramFamily:
    """A Prometheus histogram family.

    Attributes:
        name: Metric name.
        help: Human readable description.
        buckets: The bucket upper bounds, in ascending order.
        series: The time series of this family.
    """

    name: str
    help: str
    buckets: tuple[float, ...] = DEFAULT_BUCKETS
    series: list[HistogramSeries] = field(default_factory=list)

    def render(self) -> str:
        """Render the family in the Prometheus text exposition format."""
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for entry in sorted(self.series, key=lambda s: tuple(sorted(s.labels.items()))):
            for bound in self.buckets:
                labels = dict(entry.labels)
                labels["le"] = _format_value(bound)
                count = entry.bucket_counts.get(bound, 0)
                lines.append(f"{self.name}_bucket{_format_labels(labels, last='le')} {int(count)}")
            labels = dict(entry.labels)
            labels["le"] = "+Inf"
            lines.append(
                f"{self.name}_bucket{_format_labels(labels, last='le')} {int(entry.total_count)}"
            )
            base = _format_labels(entry.labels)
            lines.append(f"{self.name}_sum{base} {_format_value(entry.total_sum)}")
            lines.append(f"{self.name}_count{base} {int(entry.total_count)}")
        return "\n".join(lines)


@dataclass(slots=True)
class MetricBundle:
    """A set of metric families plus the cardinality report that produced it.

    Attributes:
        families: The metric families, in emission order.
        cardinality: The cardinality report for the observed spans.
    """

    families: list[CounterFamily | HistogramFamily] = field(default_factory=list)
    cardinality: CardinalityReport | None = None

    def render(self) -> str:
        """Render every family in the Prometheus text exposition format."""
        return "\n\n".join(family.render() for family in self.families)

    def family(self, name: str) -> CounterFamily | HistogramFamily | None:
        """Return the family with the given name, or ``None``."""
        for fam in self.families:
            if fam.name == name:
                return fam
        return None


class MetricBuilder:
    """Builds Prometheus metric families from spans.

    Each metric family has a fixed, small label space. The cardinality
    controller is applied to that space, so an operator can cap the number of
    distinct ``model`` values or forbid ``tenant_id`` on a shared cluster.

    Args:
        controller: The cardinality controller used to bound label values.
            A default controller with a default policy is used when omitted.
        buckets: Histogram bucket upper bounds, in seconds.
    """

    #: The label space of each metric family.
    LABEL_SPACE: ClassVar[dict[str, tuple[str, ...]]] = {
        "llm_requests_total": ("tenant_id", "model", "kind", "status"),
        "llm_tokens_total": ("tenant_id", "model", "kind", "direction"),
        "llm_request_duration_seconds": ("tenant_id", "model", "kind"),
        "llm_cost_usd_total": ("tenant_id", "model", "kind"),
    }

    def __init__(
        self,
        controller: CardinalityController | None = None,
        *,
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        self._controller = controller or CardinalityController()
        self._buckets = tuple(sorted(float(b) for b in buckets))

    @property
    def controller(self) -> CardinalityController:
        """The cardinality controller in use."""
        return self._controller

    def build(self, spans: Iterable[Span]) -> MetricBundle:
        """Build the metric families for a collection of spans.

        Args:
            spans: The spans to aggregate.

        Returns:
            The metric bundle, including the cardinality report.
        """
        span_list = list(spans)

        requests: list[tuple[dict[str, str], float]] = []
        tokens: list[tuple[dict[str, str], float]] = []
        cost: list[tuple[dict[str, str], float]] = []
        durations: list[tuple[dict[str, str], float]] = []

        for span in span_list:
            requests.append((_labels_for(span, self.LABEL_SPACE["llm_requests_total"]), 1.0))
            cost.append((_labels_for(span, self.LABEL_SPACE["llm_cost_usd_total"]), span.cost_usd))
            durations.append(
                (
                    _labels_for(span, self.LABEL_SPACE["llm_request_duration_seconds"]),
                    span.duration_ms / 1000.0,
                )
            )
            for direction, count in (
                ("input", span.input_tokens),
                ("output", span.output_tokens),
            ):
                if count <= 0:
                    continue
                labels = _labels_for(span, self.LABEL_SPACE["llm_tokens_total"])
                labels["direction"] = direction
                tokens.append((labels, float(count)))

        # The cardinality report describes the label space that actually
        # becomes Prometheus series, which is what a backend would suffer.
        report = self._controller.report_for_labels(
            [_label_set_of(rows) for rows in (requests, tokens, durations, cost)]
        )

        request_series = self._counter_series("llm_requests_total", requests)
        token_series = self._counter_series("llm_tokens_total", tokens)
        cost_series = self._counter_series("llm_cost_usd_total", cost)
        duration_series = self._histogram_series("llm_request_duration_seconds", durations)

        bundle = MetricBundle(
            families=[
                CounterFamily(
                    name="llm_requests_total",
                    help="Total number of LLM trace spans observed.",
                    series=list(request_series.values()),
                ),
                CounterFamily(
                    name="llm_tokens_total",
                    help="Total number of tokens processed, by direction.",
                    series=list(token_series.values()),
                ),
                HistogramFamily(
                    name="llm_request_duration_seconds",
                    help="End-to-end duration of observed spans in seconds.",
                    buckets=self._buckets,
                    series=list(duration_series.values()),
                ),
                CounterFamily(
                    name="llm_cost_usd_total",
                    help="Total observed cost in US dollars.",
                    series=list(cost_series.values()),
                ),
            ],
            cardinality=report,
        )
        return bundle

    def _counter_series(
        self, family: str, rows: list[tuple[dict[str, str], float]]
    ) -> dict[tuple[tuple[str, str], ...], MetricSeries]:
        """Aggregate ``(labels, value)`` rows into counter time series."""
        counts = self._value_counts(rows)
        series: dict[tuple[tuple[str, str], ...], MetricSeries] = {}
        for labels, value in rows:
            safe = self._controller.sanitise_labels(labels, value_counts=counts)
            key = _key(safe)
            entry = series.setdefault(key, MetricSeries(labels=safe))
            entry.value += value
        return series

    def _histogram_series(
        self, family: str, rows: list[tuple[dict[str, str], float]]
    ) -> dict[tuple[tuple[str, str], ...], HistogramSeries]:
        """Aggregate ``(labels, observation)`` rows into histogram series."""
        counts = self._value_counts(rows)
        series: dict[tuple[tuple[str, str], ...], HistogramSeries] = {}
        for labels, observation in rows:
            safe = self._controller.sanitise_labels(labels, value_counts=counts)
            key = _key(safe)
            entry = series.setdefault(
                key,
                HistogramSeries(
                    labels=safe,
                    bucket_counts=dict.fromkeys(self._buckets, 0),
                ),
            )
            entry.total_count += 1
            entry.total_sum += observation
            for bound in self._buckets:
                if observation <= bound:
                    entry.bucket_counts[bound] += 1
        return series

    def _value_counts(self, rows: list[tuple[dict[str, str], float]]) -> dict[str, int]:
        """Count the distinct values observed for each label, over the batch.

        Computing this over the whole batch before sanitising keeps the cap
        decision independent of the order in which spans arrive.
        """
        per_label: dict[str, set[str]] = {}
        for labels, _ in rows:
            for label, value in labels.items():
                per_label.setdefault(label, set()).add(value)
        return {label: len(values) for label, values in per_label.items()}


def _labels_for(span: Span, names: Sequence[str]) -> dict[str, str]:
    """Resolve the metric label values of ``span`` for the given label names."""
    return {name: _metric_label_value(span, name) for name in names}


def _label_set_of(rows: list[tuple[dict[str, str], float]]) -> LabelSet:
    """Return the label space described by a list of label rows."""
    label_set = LabelSet()
    for labels, _ in rows:
        label_set.update(labels)
    return label_set


def _key(labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Return a hashable, order-independent key for a label set."""
    return tuple(sorted(labels.items()))


def _metric_label_value(span: Span, name: str) -> str:
    """Resolve the value of a metric label from a span."""
    if name == "tenant_id":
        return span.tenant_id or "unknown"
    if name == "model":
        return span.model or "unknown"
    if name == "kind":
        return span.kind.value
    if name == "status":
        return span.status.value
    if name == "direction":
        return "input"
    if name in span.attributes:
        return _label_value(span.attributes[name])
    return "unknown"
