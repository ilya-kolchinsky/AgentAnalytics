from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol
import json

from .model import Trace
from .schema import SchemaBindings


@dataclass
class ErrorInfo:
    type: Optional[str] = None
    message: Optional[str] = None


@dataclass
class ToolCall:
    name: str
    args: Optional[Any] = None
    result: Optional[Any] = None
    span_id: Optional[str] = None
    status: Optional[str] = None


class TraceView(Protocol):
    trace_id: str

    def query_text(self) -> Optional[str]: ...
    def final_answer(self) -> Optional[str]: ...
    def error(self) -> Optional[ErrorInfo]: ...
    def tool_calls(self) -> List[ToolCall]: ...


class TraceViewFactory(Protocol):
    def make(self, trace: Trace) -> TraceView: ...


# ---------- helpers: MLflow JSON blobs (traceInputs/Outputs, spanInputs/Outputs) ----------

def _json_load_maybe(x: Any) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # MLflow stores JSON strings; parse safely
        try:
            val = json.loads(s)
            return val if isinstance(val, dict) else {"_value": val}
        except Exception:
            return None
    return None


def _extract_mlflow_io(attrs: Dict[str, Any], *, is_trace: bool) -> Dict[str, Any]:
    """
    Returns merged key-values from mlflow.{trace|span}Inputs/Outputs JSON blobs, if present.
    """
    out: Dict[str, Any] = {}
    if is_trace:
        inp = _json_load_maybe(attrs.get("mlflow.traceInputs"))
        outp = _json_load_maybe(attrs.get("mlflow.traceOutputs"))
    else:
        inp = _json_load_maybe(attrs.get("mlflow.spanInputs"))
        outp = _json_load_maybe(attrs.get("mlflow.spanOutputs"))

    if inp:
        out.update(inp)
    if outp:
        out.update(outp)
    return out


def _first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _as_text(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return None
    s = str(v).strip()
    return s if s else None


class DefaultTraceView:
    def __init__(self, trace: Trace, schema: SchemaBindings):
        self._t = trace
        self._s = schema
        self.trace_id = trace.trace_id

        # Precompute merged IO maps for trace + each span
        self._trace_io = _extract_mlflow_io(trace.attributes, is_trace=True)

    def _trace_search(self, keys: List[str]) -> Optional[Any]:
        # 1) direct trace attributes
        v = _first_present(self._t.attributes, keys)
        if v not in (None, ""):
            return v
        # 2) traceInputs/Outputs decoded map
        v = _first_present(self._trace_io, keys)
        if v not in (None, ""):
            return v
        return None

    def _span_search(self, keys: List[str]) -> Optional[Any]:
        # Search each span: direct attrs first, then decoded spanInputs/Outputs
        for sp in self._t.spans:
            v = _first_present(sp.attributes, keys)
            if v not in (None, ""):
                return v
            sp_io = _extract_mlflow_io(sp.attributes, is_trace=False)
            v = _first_present(sp_io, keys)
            if v not in (None, ""):
                return v
        return None

    def query_text(self) -> Optional[str]:
        v = self._trace_search(self._s.query_text_keys)
        if v is None:
            v = self._span_search(self._s.query_text_keys)
        return _as_text(v)

    def final_answer(self) -> Optional[str]:
        v = self._trace_search(self._s.final_answer_keys)
        if v is None:
            v = self._span_search(self._s.final_answer_keys)
        return _as_text(v)

    def error(self) -> Optional[ErrorInfo]:
        """
        Error detection strategy (in order):
        1) Look for explicit exception fields in trace/span attrs or mlflow.*Inputs/Outputs blobs.
        2) If missing, infer errors from tool spans:
           - tool.status == "ERROR"
           - tool_result contains {"error": "<TYPE>", "detail": "<desc>"} (or similar)
        3) If still none, fall back to trace.status containing "ERROR".
        """
        # 1) explicit exception keys (existing behavior)
        et = self._trace_search(self._s.error_type_keys) or self._span_search(self._s.error_type_keys)
        em = self._trace_search(self._s.error_message_keys) or self._span_search(self._s.error_message_keys)
        if et is not None or em is not None:
            return ErrorInfo(type=_as_text(et), message=_as_text(em))

        # 2) infer from tool spans (your trace format)
        # Prefer the *last* failing tool call as the likely root cause.
        failing: List[ToolCall] = []
        for tc in self.tool_calls():
            # tc.status comes from tool.status attribute (see tool_calls() impl)
            if (tc.status or "").upper() == "ERROR":
                failing.append(tc)
                continue
            # some instrumentations omit tool.status but include tool_result.error
            if isinstance(tc.result, dict) and ("error" in tc.result):
                failing.append(tc)

        if failing:
            tc = failing[-1]  # last failing tool in the trace
            err_type = None
            err_detail = None

            if isinstance(tc.result, dict):
                err_type = tc.result.get("error") or tc.result.get("error_type")
                err_detail = tc.result.get("detail") or tc.result.get("message") or tc.result.get("error_message")

            # if no structured detail, still return something useful
            etxt = _as_text(err_type) or "TOOL_ERROR"
            msg_parts = [f"tool={tc.name}"]
            if err_detail:
                msg_parts.append(str(err_detail))
            return ErrorInfo(type=etxt, message=" | ".join(msg_parts))

        # 3) last-resort fallback: trace status indicates ERROR
        if "ERROR" in (self._t.status or "").upper():
            return ErrorInfo(type=None, message="Trace status indicates ERROR (no exception/tool error fields found)")

        return None

    def tool_calls(self) -> List[ToolCall]:
        out: List[ToolCall] = []
        for sp in self._t.spans:
            # Tool name can be in attributes or implied by span name
            name = _first_present(sp.attributes, self._s.tool_name_keys)

            if name is None:
                # also check mlflow.spanInputs/Outputs blobs for tool name (some instrumentations place it there)
                sp_io = _extract_mlflow_io(sp.attributes, is_trace=False)
                name = _first_present(sp_io, self._s.tool_name_keys)

            if name is None and (sp.name.lower().startswith("tool:") or sp.name.lower().startswith("tool ")):
                name = sp.name.split(":", 1)[-1].strip() if ":" in sp.name else sp.name

            if name is None:
                continue

            # Tool args/result: first direct attributes by binding keys, then span IO blobs
            args = _first_present(sp.attributes, self._s.tool_args_keys)
            res = _first_present(sp.attributes, self._s.tool_result_keys)

            sp_io = _extract_mlflow_io(sp.attributes, is_trace=False)
            if args is None:
                args = _first_present(sp_io, self._s.tool_args_keys) or sp_io.get("tool_args") or sp_io.get("args")
            if res is None:
                res = _first_present(sp_io, self._s.tool_result_keys) or sp_io.get("tool_result") or sp_io.get("result")

            out.append(
                ToolCall(
                    name=str(name),
                    args=args,
                    result=res,
                    span_id=sp.span_id,
                    status=str(
                        sp.attributes.get("tool.status")
                        or sp.attributes.get("tool_status")
                        or sp.attributes.get("status")
                        or sp.status
                        or ""
                    ),
                )
            )
        return out


class DefaultTraceViewFactory:
    def __init__(self, schema: SchemaBindings):
        self._schema = schema

    def make(self, trace: Trace) -> TraceView:
        return DefaultTraceView(trace, self._schema)