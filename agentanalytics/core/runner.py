import os
import json
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Any, Protocol

from .plugin import Plugin, PluginContext, PluginResult, Artifact
from .model import TraceBatch


class PluginResolver(Protocol):
    def get(self, name: str) -> Plugin: ...


@dataclass
class Runner:
    resolver: PluginResolver

    def run(self, batch: TraceBatch, ctx: PluginContext, plugin_specs: List[Dict[str, Any]]) -> PluginResult:
        """
        Runs enabled plugins and writes a run_manifest.json with plugin-scoped artifacts.
        Artifacts are expected to be written under:
          <run_dir>/plugins/<plugin_name>/...
        """
        merged = PluginResult()
        run_started_ms = int(time.time() * 1000)

        manifest = {
            "run_id": ctx.run_id,
            "started_ms": run_started_ms,
            "finished_ms": None,
            "trace_count": len(batch.traces),
            "plugins": [],
        }

        os.makedirs(ctx.output_dir, exist_ok=True)
        os.makedirs(os.path.join(ctx.output_dir, "plugins"), exist_ok=True)

        for spec in plugin_specs:
            name = spec["name"]
            enabled = bool(spec.get("enabled", True))
            cfg = spec.get("config", {}) or {}
            if not enabled:
                continue

            plugin_out_dir = os.path.join(ctx.output_dir, "plugins", name)
            os.makedirs(plugin_out_dir, exist_ok=True)
            plugin_ctx = ctx.with_output_dir(plugin_out_dir)

            plugin_entry = {
                "name": name,
                "version": None,
                "elapsed_sec": None,
                "metrics": [],
                "artifacts": [],
                "annotations": 0,
                "summary_md": "",
                "status": "RUNNING",
                "error": None,
            }

            t0 = time.time()
            try:
                plugin = self.resolver.get(name)
                plugin_entry["version"] = getattr(plugin, "version", "unknown")

                print(f"Running plugin {name}...", end='')
                res = plugin.analyze(batch, plugin_ctx, cfg)
                dt = time.time() - t0
                print("completed.")

                plugin_artifacts = []
                for a in res.artifacts:
                    rel = os.path.relpath(a.path, ctx.output_dir).replace("\\", "/")
                    plugin_artifacts.append({"kind": a.kind, "relpath": rel, "description": a.description})

                plugin_entry.update({
                    "elapsed_sec": dt,
                    "metrics": [m.name for m in res.metrics],
                    "artifacts": plugin_artifacts,
                    "annotations": len(res.annotations),
                    "summary_md": res.summary_md or "",
                    "status": "OK",
                })

                merged.metrics.extend(res.metrics)
                merged.artifacts.extend(res.artifacts)
                merged.annotations.extend(res.annotations)

            except Exception as e:
                dt = time.time() - t0
                tb = traceback.format_exc()

                # Write per-plugin error file so UI can show it
                err_path = os.path.join(plugin_out_dir, "error.txt")
                with open(err_path, "w", encoding="utf-8") as f:
                    f.write(tb)

                plugin_entry.update({
                    "elapsed_sec": dt,
                    "status": "FAILED",
                    "error": {"type": type(e).__name__, "message": str(e),
                              "artifact": os.path.relpath(err_path, ctx.output_dir).replace("\\", "/")},
                    "summary_md": f"❌ Plugin failed: `{type(e).__name__}` — {str(e)}",
                    "artifacts": [
                        {"kind": "text", "relpath": os.path.relpath(err_path, ctx.output_dir).replace("\\", "/"),
                         "description": "Plugin error traceback"}],
                })
                # Continue to next plugin (best effort)

            manifest["plugins"].append(plugin_entry)

        manifest["finished_ms"] = int(time.time() * 1000)

        manifest_path = os.path.join(ctx.output_dir, "run_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        merged.artifacts.append(Artifact(kind="json", path=manifest_path, description="Run manifest"))
        return merged
