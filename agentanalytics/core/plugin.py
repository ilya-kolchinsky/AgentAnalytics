from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .model import TraceBatch
from .view import TraceViewFactory


@dataclass
class MetricRecord:
    name: str
    data: Dict[str, Any]


@dataclass
class Artifact:
    kind: str  # markdown/json/...
    path: str
    description: str


@dataclass
class TraceAnnotation:
    trace_id: str
    annotations: Dict[str, Any]


@dataclass
class Resources:
    embedder: Any


@dataclass
class AnalysisLimits:
    max_traces: Optional[int] = None
    max_text_chars: int = 8000
    random_seed: int = 13


@dataclass
class PluginContext:
    view_factory: TraceViewFactory
    resources: Resources
    limits: AnalysisLimits
    now_ms: int
    output_dir: str
    run_id: str

    def with_output_dir(self, output_dir: str) -> "PluginContext":
        return PluginContext(
            view_factory=self.view_factory,
            resources=self.resources,
            limits=self.limits,
            now_ms=self.now_ms,
            output_dir=output_dir,
            run_id=self.run_id,
        )


@dataclass
class PluginResult:
    metrics: List[MetricRecord] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    annotations: List[TraceAnnotation] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    summary_md: str = ""  # rich summary for UI/CLI


class Plugin(Protocol):
    name: str
    version: str

    def analyze(self, batch: TraceBatch, ctx: PluginContext, config: Dict[str, Any]) -> PluginResult: ...
