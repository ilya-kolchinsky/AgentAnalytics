from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class RunCreateRequest(BaseModel):
    # One of:
    config_yaml: Optional[str] = None
    config_json: Optional[Dict[str, Any]] = None

    # Optional: if provided, stored and used as run_id
    run_id: Optional[str] = None


class RunCreateResponse(BaseModel):
    run_id: str
    run_dir: str
    status: str


class RunSummary(BaseModel):
    run_id: str
    status: str
    started_ms: Optional[int] = None
    finished_ms: Optional[int] = None
    trace_count: Optional[int] = None
    plugins: List[str] = Field(default_factory=list)
