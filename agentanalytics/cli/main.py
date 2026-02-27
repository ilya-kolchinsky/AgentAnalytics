import argparse
import os
import time
import uuid
import json

from agentanalytics.config import load_config
from agentanalytics.core import (
    Runner,
    SchemaBindings,
    DefaultTraceViewFactory,
)
from agentanalytics.core.plugin import Resources, AnalysisLimits, PluginContext
from agentanalytics.adapters.mlflow_adapter import MLflowTraceAdapter
from agentanalytics.plugins.registry import get_plugin, list_plugins
from agentanalytics.plugins.resolver import DictPluginResolver
from agentanalytics.utils.embeddings import make_embedder

import traceback


def _set_status(run_dir: str, run_id: str, status: str, error: dict | None = None):
    path = os.path.join(run_dir, "status.json")
    payload = {"run_id": run_id, "status": status, "run_dir": run_dir}
    if error:
        payload["error"] = error
    _write_json(path, payload)


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _cmd_run(config_path: str, run_dir: str | None, run_id: str | None):
    print("Launching Agent Analytics engine...")

    cfg = load_config(config_path)
    print("Configuration loaded successfully.")

    rid = run_id or uuid.uuid4().hex[:12]

    # Resolve run directory:
    # If run_dir is provided, use it. Otherwise use cfg.output.output_dir/runs/<run_id>
    base_out = cfg.output.output_dir
    if run_dir is None:
        run_dir = os.path.join(base_out, "runs", rid)

    os.makedirs(run_dir, exist_ok=True)

    # Save a copy of config for reproducibility
    cfg_copy_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_copy_path):
        with open(config_path, "r", encoding="utf-8") as src, open(cfg_copy_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())

    print("Run setup complete.")
    _set_status(run_dir, rid, "RUNNING")

    try:

        schema = SchemaBindings.from_dict(cfg.schema_bindings)
        view_factory = DefaultTraceViewFactory(schema=schema)

        embedder = make_embedder(
            prefer_sentence_transformers=True,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_root=os.path.expanduser("~/.cache/agentanalytics"),
            offline=True,
        )
        resources = Resources(embedder=embedder)

        limits = AnalysisLimits(
            max_traces=cfg.timeframe.last_n_traces,
            max_text_chars=8000,
            random_seed=13,
        )

        ctx = PluginContext(
            view_factory=view_factory,
            resources=resources,
            limits=limits,
            now_ms=int(time.time() * 1000),
            output_dir=run_dir,
            run_id=rid,
        )

        # Ingest traces
        print(f"Fetching traces from {cfg.mlflow.tracking_uri}...")
        adapter = MLflowTraceAdapter(tracking_uri=cfg.mlflow.tracking_uri)
        batch = adapter.fetch(
            timeframe=cfg.timeframe,
            schema=schema,
            query=cfg.mlflow.query,
            experiment_id=cfg.mlflow.experiment_id,
        )
        print(f"Successfully extracted {len(batch.traces)} traces.")

        # Run plugins
        plugins = {p: get_plugin(p) for p in list_plugins()}

        print("The following plugins will be executed:")
        for plugin_name in plugins.keys():
            print(f"- {plugin_name}")

        runner = Runner(resolver=DictPluginResolver(plugins=plugins))
        runner.run(
            batch=batch,
            ctx=ctx,
            plugin_specs=[p.__dict__ for p in cfg.plugins],
        )

        _set_status(run_dir, rid, "FINISHED")
        print(f"Done. Run ID: {rid}")
        print(f"Run directory: {os.path.abspath(run_dir)}")

    except Exception as e:
        tb = traceback.format_exc()
        err_path = os.path.join(run_dir, "error.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(tb)

        _set_status(
            run_dir,
            rid,
            "FAILED",
            error={"type": type(e).__name__, "message": str(e), "traceback_artifact": "error.txt"}
        )
        print(f"FAILED. Run ID: {rid}")
        print(tb)

        # Re-raise so server sees non-zero exit code
        raise


def main():
    parser = argparse.ArgumentParser(prog="agentanalytics")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run analytics plugins on MLflow traces")
    p_run.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    p_run.add_argument("--run-dir", default=None, help="Directory to write this run's outputs")
    p_run.add_argument("--run-id", default=None, help="Optional run_id (otherwise random)")

    args = parser.parse_args()
    if args.cmd == "run":
        _cmd_run(args.config, args.run_dir, args.run_id)


if __name__ == "__main__":
    main()
