import os
import asyncio
from typing import Optional

import yaml
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from ..models import RunCreateRequest, RunCreateResponse, RunSummary
from ..run_manager import RunManager

router = APIRouter()


def _mgr(request: Request) -> RunManager:
    runs_dir = request.app.state.runs_dir
    return RunManager(runs_dir=runs_dir)


@router.post("/runs", response_model=RunCreateResponse)
async def create_run(request: Request, payload: RunCreateRequest):
    mgr = _mgr(request)

    if payload.config_yaml:
        config_yaml = payload.config_yaml
    elif payload.config_json is not None:
        config_yaml = yaml.safe_dump(payload.config_json, sort_keys=False)
    else:
        raise HTTPException(status_code=400, detail="Provide config_yaml or config_json")

    rp = mgr.new_run(config_yaml=config_yaml, run_id=payload.run_id)
    mgr.start_run_subprocess(rp)

    return RunCreateResponse(run_id=rp.run_id, run_dir=rp.run_dir, status="RUNNING")


@router.post("/runs/upload", response_model=RunCreateResponse)
async def create_run_upload(request: Request, file: UploadFile = File(...), run_id: Optional[str] = None):
    mgr = _mgr(request)
    data = await file.read()
    config_yaml = data.decode("utf-8")

    rp = mgr.new_run(config_yaml=config_yaml, run_id=run_id)
    mgr.start_run_subprocess(rp)
    return RunCreateResponse(run_id=rp.run_id, run_dir=rp.run_dir, status="RUNNING")


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(request: Request):
    mgr = _mgr(request)
    out = []
    for rid in mgr.list_runs():
        s = mgr.get_run_summary(rid)
        out.append(RunSummary(**{
            "run_id": s["run_id"],
            "status": s["status"],
            "started_ms": s.get("started_ms"),
            "finished_ms": s.get("finished_ms"),
            "trace_count": s.get("trace_count"),
            "plugins": s.get("plugins", []),
        }))
    return out


@router.get("/runs/{run_id}")
async def get_run(request: Request, run_id: str):
    mgr = _mgr(request)
    summary = mgr.get_run_summary(run_id)
    status = mgr.read_status(run_id)
    manifest = mgr.read_manifest(run_id)
    return {"summary": summary, "status": status, "manifest": manifest}


@router.get("/runs/{run_id}/artifacts/{relpath:path}")
async def get_artifact(request: Request, run_id: str, relpath: str):
    mgr = _mgr(request)
    try:
        full = mgr.artifact_path(run_id, relpath)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid artifact path")

    if not os.path.exists(full):
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(full)


@router.get("/runs/{run_id}/logs")
async def get_logs(request: Request, run_id: str):
    """
    Returns the full stdout.log (if present) as text.
    """
    mgr = _mgr(request)
    full = mgr.artifact_path(run_id, "stdout.log")
    if not os.path.exists(full):
        raise HTTPException(status_code=404, detail="No logs yet")
    return FileResponse(full, media_type="text/plain")


@router.get("/runs/{run_id}/events")
async def stream_logs_sse(request: Request, run_id: str):
    """
    Server-Sent Events stream that tails stdout.log.
    The UI can subscribe and show live logs.
    """

    mgr = _mgr(request)
    log_path = mgr.artifact_path(run_id, "stdout.log")

    async def event_gen():
        last_pos = 0
        while True:
            if await request.is_disconnected():
                return

            if os.path.exists(log_path):
                with open(log_path, "rb") as f:
                    f.seek(last_pos)
                    chunk = f.read()
                    last_pos = f.tell()
                if chunk:
                    text = chunk.decode("utf-8", errors="replace")
                    # SSE "data:" lines
                    for line in text.splitlines():
                        yield f"data: {line}\n\n"

            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
