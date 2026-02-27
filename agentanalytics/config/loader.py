from typing import Any, Dict
import yaml

from .models import AppConfig, MLflowConfig, TimeframeConfig, PluginSpec, OutputConfig


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    if "mlflow" not in raw:
        raise ValueError("Config must include 'mlflow' section with at least tracking_uri.")

    ml = raw["mlflow"] or {}
    tf = raw.get("timeframe", {}) or {}
    out = raw.get("output", {}) or {}
    plugins_raw = raw.get("plugins", []) or []

    cfg = AppConfig(
        mlflow=MLflowConfig(
            tracking_uri=str(ml["tracking_uri"]),
            experiment_id=str(ml["experiment_id"]),
            query=ml.get("query"),
        ),
        timeframe=TimeframeConfig(
            last_n_days=tf.get("last_n_days"),
            last_n_hours=tf.get("last_n_hours"),
            last_n_traces=tf.get("last_n_traces"),
            start_time_iso=tf.get("start_time_iso"),
            end_time_iso=tf.get("end_time_iso"),
        ),
        output=OutputConfig(
            output_dir=out.get("output_dir", "./artifacts"),
            overwrite=bool(out.get("overwrite", True)),
        ),
        plugins=[PluginSpec(**p) for p in plugins_raw],
        schema_bindings=raw.get("schema_bindings", {}) or {},
    )
    return cfg