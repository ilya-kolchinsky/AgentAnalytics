from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SpanEvent:
    name: str
    time_ms: int
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time_ms: int
    end_time_ms: int
    status: str = "UNSET"  # OK/ERROR/UNSET
    kind: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)


@dataclass
class Trace:
    trace_id: str
    start_time_ms: int
    end_time_ms: int
    status: str = "UNSET"
    attributes: Dict[str, Any] = field(default_factory=dict)
    spans: List[Span] = field(default_factory=list)


@dataclass
class TimeWindow:
    start_time_ms: int
    end_time_ms: int


@dataclass
class SamplingInfo:
    strategy: str = "none"  # none/uniform/reservoir/...
    rate: float = 1.0
    notes: str = ""


@dataclass
class TraceSource:
    name: str = "mlflow"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceBatch:
    traces: List[Trace]
    window: TimeWindow
    source: TraceSource = field(default_factory=TraceSource)
    sampling: SamplingInfo = field(default_factory=SamplingInfo)
    schema: Any = None  # SchemaBindings (kept Any to avoid circular imports)
