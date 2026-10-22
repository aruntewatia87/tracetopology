"""The no-silent-drop guarantee for telemetry ingestion.

A monitoring pipeline that quietly discards data is *worse* than one that
crashes, because the crash is noticed. The `langfuse/langfuse#16112` issue
documents exactly that failure mode: OTLP traces were accepted with HTTP 200,
written to S3, and never enqueued for processing. Everything looked healthy
while the traces were simply gone.

This module makes every drop explicit and countable. Nothing disappears
without a reason, and a report tells you whether the run was *complete* - that
is, whether anything was dropped for a reason other than the sampling rate
you deliberately configured.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tracetopology.spans import Span

__all__ = [
    "DropReason",
    "IngestionAudit",
    "IngestionReport",
    "IngestionRules",
]


class DropReason(str, Enum):
    """Why an observation was not accepted.

    Attributes:
        SAMPLED: Dropped by a configured sampling policy. Expected, so it
            does not make the run incomplete.
        WINDOW_CLOSED: The ingestion window had already closed.
        MALFORMED: The observation could not be parsed or was structurally
            invalid.
        TENANT_ISOLATION: A tenant-isolation rule rejected the observation.
        QUOTA_EXCEEDED: The per-tenant quota was exhausted.
    """

    SAMPLED = "sampled"
    WINDOW_CLOSED = "window_closed"
    MALFORMED = "malformed"
    TENANT_ISOLATION = "tenant_isolation"
    QUOTA_EXCEEDED = "quota_exceeded"

    @classmethod
    def coerce(cls, value: DropReason | str) -> DropReason:
        """Return ``value`` as a :class:`DropReason`, accepting loose strings."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"unknown drop reason: {value!r}") from exc

    @property
    def is_expected(self) -> bool:
        """``True`` when the drop is an intended, configured behaviour."""
        return self is DropReason.SAMPLED


@dataclass(slots=True)
class IngestionRules:
    """The policies that govern an ingestion run.

    Attributes:
        sampling_rate: Fraction of observations to keep, in ``[0, 1]``. A
            value of ``1.0`` keeps everything.
        window_open: Whether the ingestion window is still accepting data.
        allowed_tenants: When non-empty, only these tenants are accepted.
        tenant_quotas: Maximum number of observations accepted per tenant.
        max_attributes: Maximum number of attributes a valid span may carry.
    """

    sampling_rate: float = 1.0
    window_open: bool = True
    allowed_tenants: tuple[str, ...] = ()
    tenant_quotas: Mapping[str, int] = field(default_factory=dict)
    max_attributes: int = 0

    def __post_init__(self) -> None:
        """Validate the sampling rate and normalise the tenant tuple."""
        if not 0.0 <= self.sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be in [0, 1]")
        object.__setattr__(self, "allowed_tenants", tuple(self.allowed_tenants))

    @property
    def sampling_configured(self) -> bool:
        """``True`` when sampling is deliberately less than everything."""
        return self.sampling_rate < 1.0


@dataclass(slots=True)
class IngestionReport:
    """The outcome of an ingestion run.

    Attributes:
        submitted: Number of observations offered to the pipeline.
        accepted: Number of observations accepted.
        dropped: Number of observations rejected.
        drop_reasons: Count of drops per reason.
    """

    submitted: int = 0
    accepted: int = 0
    dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Assert the accounting invariant: accepted + dropped == submitted."""
        if self.accepted + self.dropped != self.submitted:
            raise ValueError(
                "ingestion invariant violated: "
                f"accepted {self.accepted} + dropped {self.dropped} != submitted {self.submitted}"
            )
        total = sum(self.drop_reasons.values())
        if total != self.dropped:
            raise ValueError(
                "ingestion invariant violated: "
                f"drop reasons sum to {total} but {self.dropped} observations were dropped"
            )

    @property
    def dropped_fraction(self) -> float:
        """Fraction of the submission that was dropped."""
        if self.submitted <= 0:
            return 0.0
        return self.dropped / self.submitted

    @property
    def complete(self) -> bool:
        """``True`` when nothing was dropped for an unexpected reason.

        Drops caused by a configured sampling rate are expected and do not
        make the run incomplete. Every other reason does.
        """
        for reason, count in self.drop_reasons.items():
            if count <= 0:
                continue
            if not DropReason.coerce(reason).is_expected:
                return False
        return True

    def reason_count(self, reason: DropReason | str) -> int:
        """Number of observations dropped for the given reason."""
        return self.drop_reasons.get(DropReason.coerce(reason).value, 0)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the report."""
        return {
            "submitted": self.submitted,
            "accepted": self.accepted,
            "dropped": self.dropped,
            "drop_reasons": dict(sorted(self.drop_reasons.items())),
            "dropped_fraction": round(self.dropped_fraction, 6),
            "complete": self.complete,
        }


@dataclass(slots=True)
class _Drop:
    """An internal record of a single dropped observation."""

    reason: DropReason
    detail: str


class IngestionAudit:
    """Accepts observations and records exactly why each rejection happened.

    The audit is deterministic: the same observations with the same rules
    always produce the same report. Sampling uses a stable hash of the
    observation rather than a random number generator, so a trace is either
    always sampled or never sampled for a given run.

    Args:
        rules: The policies governing the run.
    """

    def __init__(self, rules: IngestionRules | None = None) -> None:
        self._rules = rules or IngestionRules()
        self._drops: list[_Drop] = []
        self._accepted: list[Span] = []
        self._submitted = 0
        self._tenant_counts: dict[str, int] = {}

    @property
    def rules(self) -> IngestionRules:
        """The rules governing this run."""
        return self._rules

    @property
    def accepted_spans(self) -> list[Span]:
        """The spans that were accepted, in submission order."""
        return list(self._accepted)

    def submit(self, observation: Any) -> bool:
        """Submit a single observation to the pipeline.

        Args:
            observation: A :class:`~tracetopology.spans.Span` or a mapping
                that can be coerced into one.

        Returns:
            ``True`` when the observation was accepted.
        """
        self._submitted += 1

        if not self._rules.window_open:
            self._record(DropReason.WINDOW_CLOSED, "ingestion window closed")
            return False

        try:
            span = _coerce_span(observation)
        except Exception as exc:  # any parse failure is a drop, not a crash
            self._record(DropReason.MALFORMED, str(exc))
            return False

        if self._rules.max_attributes and len(span.attributes) > self._rules.max_attributes:
            self._record(
                DropReason.MALFORMED,
                f"{len(span.attributes)} attributes exceeds limit {self._rules.max_attributes}",
            )
            return False

        if self._rules.allowed_tenants and (span.tenant_id or "") not in (
            self._rules.allowed_tenants
        ):
            self._record(
                DropReason.TENANT_ISOLATION,
                f"tenant {span.tenant_id!r} is not allowed",
            )
            return False

        tenant = span.tenant_id or ""
        quota = self._rules.tenant_quotas.get(tenant)
        used = self._tenant_counts.get(tenant, 0)
        if quota is not None and used >= quota:
            self._record(
                DropReason.QUOTA_EXCEEDED,
                f"tenant {tenant!r} exceeded quota of {quota}",
            )
            return False

        if not self._keep(span):
            self._record(DropReason.SAMPLED, "sampled out")
            return False

        self._tenant_counts[tenant] = used + 1
        self._accepted.append(span)
        return True

    def submit_many(self, observations: Iterable[Any]) -> int:
        """Submit a batch of observations and return how many were accepted."""
        accepted = 0
        for observation in observations:
            if self.submit(observation):
                accepted += 1
        return accepted

    def report(self) -> IngestionReport:
        """Return the report for the observations submitted so far."""
        reasons: dict[str, int] = {}
        for drop in self._drops:
            reasons[drop.reason.value] = reasons.get(drop.reason.value, 0) + 1
        return IngestionReport(
            submitted=self._submitted,
            accepted=len(self._accepted),
            dropped=len(self._drops),
            drop_reasons=dict(sorted(reasons.items())),
        )

    def drop_details(self) -> list[tuple[str, str]]:
        """Return ``(reason, detail)`` pairs for every dropped observation."""
        return [(drop.reason.value, drop.detail) for drop in self._drops]

    def _record(self, reason: DropReason, detail: str) -> None:
        """Record a dropped observation."""
        self._drops.append(_Drop(reason=reason, detail=detail))

    def _keep(self, span: Span) -> bool:
        """Decide whether the span survives the sampling policy."""
        rate = self._rules.sampling_rate
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        bucket = _stable_bucket(f"{span.trace_id}/{span.span_id}", 10_000)
        return bucket < rate


def _stable_bucket(key: str, buckets: int) -> float:
    """Map ``key`` deterministically into ``[0, 1)``."""
    import hashlib

    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _coerce_span(observation: Any) -> Span:
    """Coerce ``observation`` into a :class:`Span`, raising on invalid data."""
    if isinstance(observation, Span):
        return observation
    if isinstance(observation, Mapping):
        payload = dict(observation)
        span_id = payload.pop("span_id", None)
        trace_id = payload.pop("trace_id", None)
        if not span_id or not trace_id:
            raise ValueError("span_id and trace_id are required")
        return Span(
            span_id=str(span_id),
            parent_span_id=payload.pop("parent_span_id", None),
            trace_id=str(trace_id),
            name=str(payload.pop("name", "unnamed")),
            kind=payload.pop("kind", "chain"),
            start_ns=int(payload.pop("start_ns", 0) or 0),
            end_ns=int(payload.pop("end_ns", 0) or 0),
            attributes=dict(payload.pop("attributes", {}) or {}),
            events=list(payload.pop("events", []) or []),
            status=payload.pop("status", "ok"),
            tenant_id=payload.pop("tenant_id", None),
        )
    raise TypeError(f"cannot coerce {type(observation).__name__} into a Span")


def audit_observations(
    observations: Sequence[Any],
    rules: IngestionRules | None = None,
) -> tuple[IngestionReport, list[Span]]:
    """Convenience helper: audit a batch and return ``(report, accepted)``.

    Args:
        observations: The observations to submit.
        rules: The policies governing the run.

    Returns:
        The report and the accepted spans.
    """
    audit = IngestionAudit(rules)
    audit.submit_many(observations)
    return audit.report(), audit.accepted_spans
