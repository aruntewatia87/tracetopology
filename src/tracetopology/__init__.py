"""tracetopology - LLM trace analysis: topology, cardinality, cost, ingestion."""

from tracetopology.cardinality import (
    CardinalityController,
    CardinalityReport,
    CounterFamily,
    HistogramFamily,
    HistogramSeries,
    LabelPolicy,
    LabelSet,
    MetricBuilder,
    MetricBundle,
    MetricSeries,
    stable_hash_bucket,
)
from tracetopology.cost import (
    CostBreakdown,
    ModelPrice,
    PriceBook,
    SpanCost,
    attribute_trace_cost,
    span_cost_resolver,
)
from tracetopology.ingestion import (
    DropReason,
    IngestionAudit,
    IngestionReport,
    IngestionRules,
    audit_observations,
)
from tracetopology.spans import Span, SpanKind, SpanStatus
from tracetopology.topology import (
    LLMAgentTopology,
    RetryLoop,
    TraceNode,
    TraceTopologyError,
    TraceTree,
    build_trace_tree,
)

__version__ = "0.1.0"

__all__ = [
    "CardinalityController",
    "CardinalityReport",
    "CostBreakdown",
    "CounterFamily",
    "DropReason",
    "HistogramFamily",
    "HistogramSeries",
    "IngestionAudit",
    "IngestionReport",
    "IngestionRules",
    "LLMAgentTopology",
    "LabelPolicy",
    "LabelSet",
    "MetricBuilder",
    "MetricBundle",
    "MetricSeries",
    "ModelPrice",
    "PriceBook",
    "RetryLoop",
    "Span",
    "SpanCost",
    "SpanKind",
    "SpanStatus",
    "TraceNode",
    "TraceTopologyError",
    "TraceTree",
    "__version__",
    "attribute_trace_cost",
    "audit_observations",
    "build_trace_tree",
    "span_cost_resolver",
    "stable_hash_bucket",
]
