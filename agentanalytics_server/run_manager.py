import os
import json
import uuid
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@dataclass
class RunPaths:
    run_id: str
    run_dir: str

    @property
    def config_path(self) -> str:
        return os.path.join(self.run_dir, "config.yaml")

    @property
    def stdout_path(self) -> str:
        return os.path.join(self.run_dir, "stdout.log")

    @property
    def status_path(self) -> str:
        return os.path.join(self.run_dir, "status.json")

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.run_dir, "run_manifest.json")


class RunManager:
    def __init__(self, runs_dir: str):
        self.runs_dir = os.path.abspath(runs_dir)
        os.makedirs(self.runs_dir, exist_ok=True)

    def new_run(self, config_yaml: str, run_id: Optional[str] = None) -> RunPaths:
        rid = run_id or uuid.uuid4().hex[:12]
        run_dir = os.path.join(self.runs_dir, rid)
        os.makedirs(run_dir, exist_ok=True)

        _write_text(os.path.join(run_dir, "config.yaml"), config_yaml)
        _write_json(os.path.join(run_dir, "status.json"), {"run_id": rid, "status": "QUEUED", "run_dir": run_dir})

        return RunPaths(run_id=rid, run_dir=run_dir)

    @staticmethod
    def start_run_subprocess(rp: RunPaths) -> None:
        """
        Starts the CLI in a subprocess and returns immediately.
        Writes stdout/stderr to stdout.log. Status file updated best-effort.
        """
        _write_json(rp.status_path, {"run_id": rp.run_id, "status": "RUNNING", "run_dir": rp.run_dir})

        # Important: call the CLI entrypoint `agentanalytics` if installed, else python -m ...
        cmd = ["agentanalytics", "run", "-c", rp.config_path, "--run-dir", rp.run_dir, "--run-id", rp.run_id]

        with open(rp.stdout_path, "ab") as out:
            # Start detached-ish process
            proc = subprocess.Popen(
                cmd,
                stdout=out,
                stderr=subprocess.STDOUT,
                cwd=rp.run_dir,
            )

            _write_json(rp.status_path, {
                "run_id": rp.run_id,
                "status": "RUNNING",
                "run_dir": rp.run_dir,
                "pid": proc.pid,
            })

        def _watch():
            code = proc.wait()
            # If CLI already marked FINISHED/FAILED, respect it
            status = _read_json(rp.status_path) or {}
            if status.get("status") in ("FINISHED", "FAILED"):
                return
            if code == 0:
                _write_json(rp.status_path,
                            {"run_id": rp.run_id, "status": "FINISHED", "run_dir": rp.run_dir, "pid": proc.pid})
            else:
                _write_json(rp.status_path, {
                    "run_id": rp.run_id,
                    "status": "FAILED",
                    "run_dir": rp.run_dir,
                    "pid": proc.pid,
                    "error": {"type": "ProcessExit", "message": f"CLI exited with code {code}"},
                })

        threading.Thread(target=_watch, daemon=True).start()

    def list_runs(self) -> List[str]:
        if not os.path.isdir(self.runs_dir):
            return []
        # run folders are direct children
        out = []
        for name in os.listdir(self.runs_dir):
            p = os.path.join(self.runs_dir, name)
            if os.path.isdir(p):
                out.append(name)
        out.sort(reverse=True)
        return out

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        run_dir = os.path.join(self.runs_dir, run_id)
        status = _read_json(os.path.join(run_dir, "status.json")) or {"run_id": run_id, "status": "UNKNOWN"}
        manifest = _read_json(os.path.join(run_dir, "run_manifest.json")) or {}
        plugins = [p.get("name") for p in manifest.get("plugins", [])] if manifest else []
        return {
            "run_id": run_id,
            "status": status.get("status", "UNKNOWN"),
            "run_dir": run_dir,
            "started_ms": manifest.get("started_ms"),
            "finished_ms": manifest.get("finished_ms"),
            "trace_count": manifest.get("trace_count"),
            "plugins": [p for p in plugins if p],
        }

    def read_manifest(self, run_id: str) -> Dict[str, Any]:
        run_dir = os.path.join(self.runs_dir, run_id)
        m = _read_json(os.path.join(run_dir, "run_manifest.json"))
        if m is None:
            return {}
        return m

    def read_status(self, run_id: str) -> Dict[str, Any]:
        run_dir = os.path.join(self.runs_dir, run_id)
        s = _read_json(os.path.join(run_dir, "status.json"))
        if s is None:
            return {"run_id": run_id, "status": "UNKNOWN"}
        return s

    def artifact_path(self, run_id: str, relpath: str) -> str:
        # Prevent directory traversal
        relpath = relpath.lstrip("/").replace("\\", "/")
        full = os.path.abspath(os.path.join(self.runs_dir, run_id, relpath))
        root = os.path.abspath(os.path.join(self.runs_dir, run_id))
        if not full.startswith(root):
            raise ValueError("Invalid artifact path")
        return full
