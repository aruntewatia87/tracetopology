# tracetopology

> LLM trace analysis: topology, cardinality control, exact cost attribution, and a no-silent-drop ingestion guarantee.

[![CI](https://github.com/aruntewatia87/tracetopology/actions/workflows/ci.yml/badge.svg)](https://github.com/aruntewatia87/tracetopology/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

LLM telemetry breaks the assumptions that general-purpose tracing tools were
built on. A single agent request produces a trace with a dozen nested spans,
each carrying a different prompt hash, session id and request id. Point a
Prometheus exporter at that data and the label space multiplies until the
metrics backend falls over. Ask for "the cost of this trace" and no
dashboard can answer, because retries and speculative branches are billed
exactly like the work that produced the answer.

`tracetopology` turns a flat list of OpenTelemetry-style spans into four
things an on-call engineer can actually use: a **queryable topology**, a
**cardinality-safe metric set**, an **exact per-trace cost attribution**, and
an **ingestion audit that proves nothing was silently dropped**. It is
dependency-free, deterministic, and offline.

---

## ✨ Highlights
- **Topology from flat spans** - builds a tree from `parent_span_id` links, attaches orphans to a synthetic root instead of losing them, and rejects duplicate ids and parent cycles with a clear error rather than looping forever.
- **The critical path** - the root-to-leaf chain with the largest total *self* time, found exactly with a bottom-up dynamic program. It answers "why was this slow?" for a 14-call agent trace.
- **Retry-loop detection** - spots the same tool called repeatedly with identical arguments, the classic agent bug, and prices the wasted work.
- **Cardinality control with visible math** - `LabelSet.cardinality_estimate()` is the *product* of per-label value counts, so you can see the explosion (12.8 trillion series) and the fix (48) before you ship it.
- **Hash bucketing, not silent truncation** - a label over its cap is bucketed into a bounded number of stable BLAKE2b buckets, so the series count is bounded without losing the label.
- **Cost attribution with an enforced invariant** - the per-span sum is asserted to equal the reported total, so the report can never disagree with itself. Cached tokens are billed at the discounted rate.
- **Total cost vs critical-path cost** - an agent that retries 5 times costs 5x but only 1 call is on the critical path. Both numbers are surfaced, and their difference is the waste.
- **No-silent-drop guarantee** - every rejected observation is counted with a reason, and `complete` is `False` if anything was dropped for a reason other than the sampling rate you deliberately configured.

---

## 🚀 Quickstart
```bash
uv sync --all-extras --dev
uv run pytest -q
```

---

## 🔧 API Tour

### Build a trace tree and render it

```python
from tracetopology import Span, SpanKind, build_trace_tree

spans = [
    Span("root", None, "t1", "support_agent.plan", SpanKind.AGENT, 0, 1_420_000_000),
    Span(
        "tool",
        "root",
        "t1",
        "web_search",
        SpanKind.TOOL,
        10_000_000,
        310_000_000,
        attributes={"cost.usd": 0.012},
    ),
    Span(
        "llm",
        "root",
        "t1",
        "llm.compose",
        SpanKind.LLM,
        1_190_000_000,
        1_420_000_000,
        attributes={
            "gen_ai.request.model": "claude-3-5-sonnet",
            "llm.token_count.prompt": 2400,
            "llm.token_count.completion": 320,
        },
    ),
]

tree = build_trace_tree(spans)
print(tree.render_tree())
print(tree.depth(), tree.leaf_count(), tree.total_tokens())
print([n.span.name for n in tree.critical_path()])
```

`build_trace_tree` accepts an optional `cost_resolver`, so the tree's cost
rollups and the attribution breakdown are computed from the same numbers.

### Attribute the cost of a trace

```python
from tracetopology import PriceBook, attribute_trace_cost

breakdown = attribute_trace_cost(spans, PriceBook.default())
print(breakdown.total_cost)  # every dollar, attributed
print(breakdown.cost_by_kind)  # llm vs tool vs retrieval
print(breakdown.cost_by_model)  # per model
print(breakdown.critical_path_cost)  # what the slow path cost
print(breakdown.off_path_cost)  # what the waste cost
print(breakdown.unpriced_spans)  # spans with no price and no cost
```

The `CostBreakdown` constructor asserts that the per-span sum equals the
total, and that the per-kind rollup sums to the same total. A broken
breakdown raises `ValueError` instead of shipping a wrong number.

### Bound the label space

```python
from tracetopology import CardinalityController, LabelPolicy

controller = CardinalityController(
    LabelPolicy(
        allowed_labels=("tenant_id", "model", "kind"),
        denied_labels=("user.email",),
        drop_patterns=(r"\.id$",),
        max_values_per_label=50,
        hash_remaining=True,
    )
)
report = controller.report(spans)
print(report.estimated_series_before, "->", report.estimated_series_after)
print(report.reduction_pct())
print(report.dropped_attributes, report.high_cardinality_labels, report.hashed_labels)
```

### Build Prometheus metrics

```python
from tracetopology import MetricBuilder

bundle = MetricBuilder(controller).build(spans)
print(bundle.render())
```

Produces `llm_requests_total`, `llm_tokens_total`,
`llm_request_duration_seconds` (a real histogram with cumulative `le`
buckets) and `llm_cost_usd_total`, in the Prometheus text exposition format.

### Audit ingestion so nothing vanishes

```python
from tracetopology import IngestionAudit, IngestionRules

audit = IngestionAudit(IngestionRules(sampling_rate=0.5, allowed_tenants=("acme",)))
for observation in observations:
    audit.submit(observation)

report = audit.report()
print(report.submitted, report.accepted, report.dropped)
print(report.drop_reasons)  # {'sampled': 10}
print(report.dropped_fraction)
print(report.complete)  # True: sampling was deliberate
```

If a span is malformed or a tenant is rejected, `drop_reasons` records the
reason and `complete` becomes `False`. That is the difference between "we
sampled" and "we lost your data".

---

## 🧠 How it works

### Cardinality explodes multiplicatively

Prometheus stores one time series per distinct *combination* of label values.
If a metric carries labels `L1 ... Ln` and label `Li` takes `vi` distinct
values, the number of series is not the sum but the **product**:

```
series = v1 * v2 * ... * vn
```

This is why cardinality control is not a tidying exercise but a survival
skill. In the demo, 200 request spans carry eight attributes each. The
naive "turn every attribute into a label" approach yields

```
200 (prompt.hash) * 200 (request.id) * 200 (user.email) * 200 (deployment.instance)
  * 200 (llm.token_count.prompt) * 20 (session.id) * 2 (model) * 1 (completion)
  = 12,800,000,000,000 series
```

Twelve trillion series from two hundred requests. The explosion is
multiplicative, so the leverage is also multiplicative: dropping a single
label with 200 values divides the total by 200. `LabelSet.cardinality_estimate()`
computes exactly this product, and `CardinalityReport` reports it before and
after a policy is applied, so the reduction is a measured number rather than
a hope.

Two mechanisms bound the space. The **allow list** removes labels outright:
a metric family declares the dimensions it needs (`tenant_id`, `model`,
`kind`, `status`, `direction`) and nothing else is ever exported. The
**value cap** handles labels that are legitimately useful but unbounded: once
a label exceeds `max_values_per_label`, its values are folded into a bounded
number of stable BLAKE2b hash buckets. The label survives, the series count
is capped, and the mapping is deterministic across processes - so the
buckets remain aggregatable in PromQL rather than scattering into thousands
of one-off series.

### The critical path is a max-sum chain, not the longest chain

Latency in a trace is not additive down the tree: a parent span overlaps its
children. The naive "deepest path" or "slowest single span" heuristics both
point at the wrong span. What an on-call engineer wants is the chain of
work that actually determined when the answer arrived.

The toolkit defines **self time** as the time a span spent that its children
did not:

```
self_time(node) = duration(node) - sum(duration(child) for child in node.children)
```

Self time is what remains after the delegated work is removed, so a parent
that merely waited on three slow children has near-zero self time and is not
blamed for their latency. The critical path is then the root-to-leaf chain
that maximises the total self time:

```
critical_path = argmax over root-to-leaf chains of sum(self_time(node) for node in chain)
```

Because the objective is a plain sum over levels, a bottom-up dynamic
program computes it exactly: every node stores the best achievable self-time
sum from itself down to a leaf, and each parent keeps the child that
maximises it. This is *not* greedy - consider a root with a child `A` of
self time 5 that has a grandchild of self time 40, and a sibling `B` of self
time 40. Greedy-by-self-time picks `B` (40 > 5), but the true maximum is
`A -> grandchild` (45). The DP gets it right, and the test suite asserts
exactly this case.

The same chain gives the **critical-path cost**. An agent that retries a
failing tool five times pays for all five calls, but only the last one is on
the path that determined the response time. `CostBreakdown` reports both
`critical_path_cost` and `off_path_cost`, and their difference is the waste:
in the demo, $0.0158 of the $0.0608 total is on the critical path and
$0.0450 is off it.

### Cost attribution is a sum, and the sum is asserted

Every span gets exactly one `SpanCost`, derived either from the cost the
span recorded itself or from a `PriceBook` lookup. All the rollups -
`total_cost`, `cost_by_kind`, `cost_by_model` - are computed from that same
list of per-span numbers, and the `CostBreakdown` constructor asserts both
that the per-span sum equals the total and that the per-kind rollup equals
it too. A report that would disagree with itself raises `ValueError` at
construction time rather than reaching a dashboard.

Two details matter for real bills. **Cached tokens are a subset of the input
tokens**, not an addition, so they are billed at the discounted rate and
never double-counted. And a **recorded cost always beats the price book**:
an observed cost is ground truth, while a static price table is an estimate
that goes stale whenever a vendor changes pricing.

### Ingestion drops must be explainable

The failure mode this toolkit exists to prevent is the one described in
`langfuse/langfuse#16112`: OTLP traces were accepted with HTTP 200, written
to S3, and never enqueued for processing. Every signal looked healthy while
the data was simply gone. A monitoring pipeline that quietly discards data
is *worse* than one that crashes, because the crash gets noticed.

`IngestionAudit` therefore records every rejection with a reason:
`sampled`, `window_closed`, `malformed`, `tenant_isolation` or
`quota_exceeded`. The report carries `submitted`, `accepted`, `dropped`,
`drop_reasons` and `dropped_fraction`, and the constructor asserts the
accounting invariant that the drop reasons sum to exactly the number of
observations dropped.

The `complete` flag is the actionable bit. A drop caused by a sampling rate
you configured is **expected** and leaves `complete` as `True`. Any other
reason - a malformed observation, a closed window, a tenant-isolation
rejection, an exhausted quota - makes `complete` `False`. That single boolean
is what a health check should assert: it turns "the pipeline is up" into "the
pipeline is up *and* it kept everything it was supposed to keep".

Sampling itself is deterministic. Rather than a random number generator, the
decision uses a stable BLAKE2b hash of the span's identity, so a given trace
is always sampled or never sampled for a given configuration, and the test
suite never flakes.

---

## Why this exists

These are real, open problems in the LLM observability ecosystem, not
hypotheticals.

- **[vllm-project/vllm#60044](https://github.com/vllm-project/vllm/issues/60044)** (open) - *"Per-request prefix-cache miss attribution (prompt divergence vs. eviction)"*. To tell a prompt-divergence miss from an eviction miss you must classify every cache miss per request. That classification is high-cardinality by construction, which is exactly the label-explosion problem this toolkit's `CardinalityController` exists to solve.

- **[langfuse/langfuse#16112](https://github.com/langfuse/langfuse/issues/16112)** (closed) - *"OTLP-native traces accepted (200 OK) and written to S3, but never enqueued"*. **This issue is closed, but it documents a real failure mode worth designing against**: the pipeline reported success while the data was lost. It is the reason `IngestionReport.complete` exists and why drops are counted by reason rather than swallowed.

- **[langfuse/langfuse#12873](https://github.com/langfuse/langfuse/issues/12873)** (open) - RFC for session-boundary behavioral drift monitoring. Session boundaries require a session-scoped view of a trace, which is the topology this toolkit builds.

- **[open-telemetry/semantic-conventions#4240](https://github.com/open-telemetry/semantic-conventions/issues/4240)** (open) - *"Define semantic conventions for span-derived metrics"*. The conventions for turning spans into metrics are not settled, so tooling that derives metrics from spans cannot assume a standard answer. `tracetopology` makes its label space explicit and configurable rather than assuming one.

- **[open-telemetry/semantic-conventions#3142](https://github.com/open-telemetry/semantic-conventions/issues/3142)** (open) - *"Flagging of attributes with PII"*, and **[#877](https://github.com/open-telemetry/semantic-conventions/issues/877)** (open, since 2023) - *"Guidelines for redacting sensitive information"*. Trace sanitization remains unresolved upstream, so `LabelPolicy.denied_labels` and `drop_patterns` are the practical mechanism for keeping PII out of a metrics backend today.

The gap these leave: a GitHub search for *"llm trace cardinality"* returns **zero
repositories**, and *"llm multi tenant telemetry isolation"* returns **one**.
The tooling for understanding LLM traces as graphs, for keeping their labels
safe, for pricing them exactly, and for proving nothing was lost along the
way does not exist as a coherent whole.

---

## 📊 Example output

Verbatim output of `uv run python examples/quickstart.py`:

```
1. TRACE TOPOLOGY
=================
support_agent.plan [agent]  1420.00ms  self=0.00ms  cost=$0.060800
|-- web_search [tool]  300.00ms  self=300.00ms  cost=$0.012000
|-- kb_lookup [tool]  120.00ms  self=120.00ms  cost=$0.004000
|-- ticket_lookup [tool]  65.00ms  self=65.00ms  cost=$0.002000
|-- web_search [tool]  320.00ms  self=320.00ms  cost=$0.012000
|-- web_search [tool]  330.00ms  self=330.00ms  cost=$0.012000
|-- vector_store.query [retrieval]  190.00ms  self=190.00ms  cost=$0.003000
`-- llm.compose [llm]  230.00ms  self=230.00ms  cost=$0.012000

spans=8  depth=1  leaves=7
total tokens=4100

2. AGENT TOPOLOGY
=================
tool fan-out per agent : {'support_agent.plan': 5}
tool calls per agent    : {'support_agent.plan': 5}
cost concentration      : web_search ($0.0120, 19.7% of trace)
retry loops detected    :
  - web_search called 2x with args 'q=acme refund status' (wasted $0.0120)

3. COST ATTRIBUTION
===================
total attributed cost   : $0.060800
by kind:
  tool       $0.042000  ( 69.1%)
  llm        $0.012000  ( 19.7%)
  agent      $0.003800  (  6.3%)
  retrieval  $0.003000  (  4.9%)
by model:
  unknown              $0.045000  ( 74.0%)
  claude-3-5-sonnet    $0.012000  ( 19.7%)
  gpt-4o               $0.003800  (  6.3%)

critical path           : planner -> tool_search_retry2
critical path cost      : $0.015800
off-path (wasted) cost  : $0.045000
critical path share     : 25.99%
invariant (sum==total)  : 0.060800 == 0.060800

4. CARDINALITY CONTROL
======================
naive - every span attribute becomes a metric label:
  distinct label values   : 1023
  estimated series        : 12,800,000,000,000
  label values per label  :
    deployment.instance          200
    gen_ai.request.model         2
    llm.token_count.completion   1
    llm.token_count.prompt       200
    prompt.hash                  200
    request.id                   200
    session.id                   20
    user.email                   200

bounded - only the fixed metric label space is used:
  label values per label  : {'direction': 2, 'kind': 1, 'model': 2, 'status': 1, 'tenant_id': 2}
  estimated series        : 8 -> 8
  reduction               : 0.00%

bounded - and anything above the cap is hashed into 50 buckets:
  estimated series        : 12,800,000,000,000 -> 48
  reduction               : 100.00%
  dropped attributes      : ['gen_ai.request.model', 'llm.token_count.completion', 'llm.token_count.prompt', 'prompt.hash', 'request.id', 'session.id', 'user.email']
  high cardinality labels : ['deployment.instance']
  hashed labels           : ['deployment.instance']

5. PROMETHEUS METRICS
=====================
# HELP llm_requests_total Total number of LLM trace spans observed.
# TYPE llm_requests_total counter
llm_requests_total{kind="agent",model="gpt-4o",status="ok",tenant_id="acme"} 1.0
llm_requests_total{kind="llm",model="claude-3-5-sonnet",status="ok",tenant_id="acme"} 1.0
llm_requests_total{kind="retrieval",model="unknown",status="ok",tenant_id="acme"} 1.0
llm_requests_total{kind="tool",model="unknown",status="ok",tenant_id="acme"} 5.0

# HELP llm_tokens_total Total number of tokens processed, by direction.
# TYPE llm_tokens_total counter
llm_tokens_total{direction="input",kind="agent",model="gpt-4o",tenant_id="acme"} 1200.0
llm_tokens_total{direction="input",kind="llm",model="claude-3-5-sonnet",tenant_id="acme"} 2400.0
llm_tokens_total{direction="output",kind="agent",model="gpt-4o",tenant_id="acme"} 180.0
llm_tokens_total{direction="output",kind="llm",model="claude-3-5-sonnet",tenant_id="acme"} 320.0

# HELP llm_request_duration_seconds End-to-end duration of observed spans in seconds.
# TYPE llm_request_duration_seconds histogram
llm_request_duration_seconds_bucket{kind="agent",model="gpt-4o",tenant_id="acme",le="0.05"} 0
llm_request_duration_seconds_bucket{kind="agent",model="gpt-4o",tenant_id="acme",le="0.1"} 0
llm_request_duration_seconds_bucket{kind="agent",model="gpt-4o",tenant_id="acme",le="0.25"} 0
llm_request_duration_seconds_bucket{kind="agent",model="gpt-4o",tenant_id="acme",le="0.5"} 0
llm_request_duration_seconds_bucket{kind="agent",model="gpt-4o",tenant_id="acme",le="1.0"} 0
llm_request_duration_seconds_bucket{kind="agent",model="gpt-4o",tenant_id="acme",le="2.5"} 1
llm_request_duration_seconds_bucket{kind="agent",model="gpt-4o",tenant_id="acme",le="5.0"} 1
llm_request_duration_seconds_bucket{kind="agent",model="gpt-4o",tenant_id="acme",le="10.0"} 1
llm_request_duration_seconds_bucket{kind="agent",model="gpt-4o",tenant_id="acme",le="30.0"} 1
llm_request_duration_seconds_bucket{kind="agent",model="gpt-4o",tenant_id="acme",le="60.0"} 1
llm_request_duration_seconds_bucket{kind="agent",model="gpt-4o",tenant_id="acme",le="+Inf"} 1
llm_request_duration_seconds_sum{kind="agent",model="gpt-4o",tenant_id="acme"} 1.42
llm_request_duration_seconds_count{kind="agent",model="gpt-4o",tenant_id="acme"} 1
llm_request_duration_seconds_bucket{kind="llm",model="claude-3-5-sonnet",tenant_id="acme",le="0.05"} 0
llm_request_duration_seconds_bucket{kind="llm",model="claude-3-5-sonnet",tenant_id="acme",le="0.1"} 0
llm_request_duration_seconds_bucket{kind="llm",model="claude-3-5-sonnet",tenant_id="acme",le="0.25"} 1
llm_request_duration_seconds_bucket{kind="llm",model="claude-3-5-sonnet",tenant_id="acme",le="0.5"} 1
llm_request_duration_seconds_bucket{kind="llm",model="claude-3-5-sonnet",tenant_id="acme",le="1.0"} 1
llm_request_duration_seconds_bucket{kind="llm",model="claude-3-5-sonnet",tenant_id="acme",le="2.5"} 1
llm_request_duration_seconds_bucket{kind="llm",model="claude-3-5-sonnet",tenant_id="acme",le="5.0"} 1
llm_request_duration_seconds_bucket{kind="llm",model="claude-3-5-sonnet",tenant_id="acme",le="10.0"} 1
llm_request_duration_seconds_bucket{kind="llm",model="claude-3-5-sonnet",tenant_id="acme",le="30.0"} 1
llm_request_duration_seconds_bucket{kind="llm",model="claude-3-5-sonnet",tenant_id="acme",le="60.0"} 1
llm_request_duration_seconds_bucket{kind="llm",model="claude-3-5-sonnet",tenant_id="acme",le="+Inf"} 1
llm_request_duration_seconds_sum{kind="llm",model="claude-3-5-sonnet",tenant_id="acme"} 0.23
llm_request_duration_seconds_count{kind="llm",model="claude-3-5-sonnet",tenant_id="acme"} 1
llm_request_duration_seconds_bucket{kind="retrieval",model="unknown",tenant_id="acme",le="0.05"} 0
llm_request_duration_seconds_bucket{kind="retrieval",model="unknown",tenant_id="acme",le="0.1"} 0
llm_request_duration_seconds_bucket{kind="retrieval",model="unknown",tenant_id="acme",le="0.25"} 1
llm_request_duration_seconds_bucket{kind="retrieval",model="unknown",tenant_id="acme",le="0.5"} 1
llm_request_duration_seconds_bucket{kind="retrieval",model="unknown",tenant_id="acme",le="1.0"} 1
llm_request_duration_seconds_bucket{kind="retrieval",model="unknown",tenant_id="acme",le="2.5"} 1
llm_request_duration_seconds_bucket{kind="retrieval",model="unknown",tenant_id="acme",le="5.0"} 1
llm_request_duration_seconds_bucket{kind="retrieval",model="unknown",tenant_id="acme",le="10.0"} 1
llm_request_duration_seconds_bucket{kind="retrieval",model="unknown",tenant_id="acme",le="30.0"} 1
llm_request_duration_seconds_bucket{kind="retrieval",model="unknown",tenant_id="acme",le="60.0"} 1
llm_request_duration_seconds_bucket{kind="retrieval",model="unknown",tenant_id="acme",le="+Inf"} 1
llm_request_duration_seconds_sum{kind="retrieval",model="unknown",tenant_id="acme"} 0.19
llm_request_duration_seconds_count{kind="retrieval",model="unknown",tenant_id="acme"} 1
llm_request_duration_seconds_bucket{kind="tool",model="unknown",tenant_id="acme",le="0.05"} 0
llm_request_duration_seconds_bucket{kind="tool",model="unknown",tenant_id="acme",le="0.1"} 1
llm_request_duration_seconds_bucket{kind="tool",model="unknown",tenant_id="acme",le="0.25"} 2
llm_request_duration_seconds_bucket{kind="tool",model="unknown",tenant_id="acme",le="0.5"} 5
llm_request_duration_seconds_bucket{kind="tool",model="unknown",tenant_id="acme",le="1.0"} 5
llm_request_duration_seconds_bucket{kind="tool",model="unknown",tenant_id="acme",le="2.5"} 5
llm_request_duration_seconds_bucket{kind="tool",model="unknown",tenant_id="acme",le="5.0"} 5
llm_request_duration_seconds_bucket{kind="tool",model="unknown",tenant_id="acme",le="10.0"} 5
llm_request_duration_seconds_bucket{kind="tool",model="unknown",tenant_id="acme",le="30.0"} 5
llm_request_duration_seconds_bucket{kind="tool",model="unknown",tenant_id="acme",le="60.0"} 5
llm_request_duration_seconds_bucket{kind="tool",model="unknown",tenant_id="acme",le="+Inf"} 5
llm_request_duration_seconds_sum{kind="tool",model="unknown",tenant_id="acme"} 1.135
llm_request_duration_seconds_count{kind="tool",model="unknown",tenant_id="acme"} 5

# HELP llm_cost_usd_total Total observed cost in US dollars.
# TYPE llm_cost_usd_total counter
llm_cost_usd_total{kind="agent",model="gpt-4o",tenant_id="acme"} 0.0
llm_cost_usd_total{kind="llm",model="claude-3-5-sonnet",tenant_id="acme"} 0.0
llm_cost_usd_total{kind="retrieval",model="unknown",tenant_id="acme"} 0.003
llm_cost_usd_total{kind="tool",model="unknown",tenant_id="acme"} 0.042

6. INGESTION AUDIT
==================
healthy run (sampling is deliberate):
  submitted               : 20
  accepted                : 10
  dropped                 : 10
  drop reasons            : {'sampled': 10}
  dropped fraction        : 0.500
  complete                : True

broken run (data was lost for a reason nobody asked for):
  submitted               : 6
  accepted                : 4
  dropped                 : 2
  drop reasons            : {'malformed': 1, 'tenant_isolation': 1}
  dropped fraction        : 0.333
  complete                : False
  drop details:
    malformed          span_id and trace_id are required
    tenant_isolation   tenant 'evilcorp' is not allowed
```

---

## 🧪 Testing
```bash
uv sync --all-extras --dev
uv run pytest -q --cov=tracetopology --cov-report=term-missing
uv run ruff check .
```

---

## 📄 License
MIT © 2026 Arun Tewatia
