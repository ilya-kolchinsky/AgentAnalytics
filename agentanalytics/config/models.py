from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimeframeConfig:
    # Exactly one of these should typically be set.
    last_n_days: Optional[int] = None
    last_n_hours: Optional[int] = None
    last_n_traces: Optional[int] = None
    # Optional explicit range (ISO 8601). If set, overrides last_* settings.
    start_time_iso: Optional[str] = None
    end_time_iso: Optional[str] = None


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_id: str
    query: Optional[str] = None          # raw query string if supported by your MLflow build


@dataclass
class PluginSpec:
    name: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    output_dir: str = "./artifacts"
    overwrite: bool = True


@dataclass
class AppConfig:
    mlflow: MLflowConfig
    timeframe: TimeframeConfig = field(default_factory=TimeframeConfig)
    plugins: List[PluginSpec] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    schema_bindings: Dict[str, Any] = field(default_factory=dict)  # mapping for TraceView extraction