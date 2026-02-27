import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agentanalytics.config.models import TimeframeConfig
from agentanalytics.core.model import TraceBatch, Trace, Span, TimeWindow, TraceSource
from agentanalytics.core.schema import SchemaBindings


def _now_ms() -> int:
    return int(time.time() * 1000)


def _parse_timeframe(tf: TimeframeConfig) -> TimeWindow:
    """
    Returns a TimeWindow, but note: MLflow search_traces does NOT accept start/end kwargs.
    We'll use this only to construct a filter_string (when timeframe is constrained).
    """
    now = _now_ms()

    if tf.start_time_iso and tf.end_time_iso:
        import datetime as dt
        st = int(dt.datetime.fromisoformat(tf.start_time_iso).timestamp() * 1000)
        et = int(dt.datetime.fromisoformat(tf.end_time_iso).timestamp() * 1000)
        return TimeWindow(start_time_ms=st, end_time_ms=et)

    if tf.last_n_hours is not None:
        return TimeWindow(start_time_ms=now - int(tf.last_n_hours) * 3600_000, end_time_ms=now)

    if tf.last_n_days is not None:
        return TimeWindow(start_time_ms=now - int(tf.last_n_days) * 86400_000, end_time_ms=now)

    # No time constraint
    return TimeWindow(start_time_ms=0, end_time_ms=now)


def _has_time_constraint(tf: TimeframeConfig) -> bool:
    return bool(
        (tf.start_time_iso and tf.end_time_iso)
        or (tf.last_n_hours is not None)
        or (tf.last_n_days is not None)
    )


def _combine_filters(a: Optional[str], b: Optional[str]) -> Optional[str]:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return None
    if a and not b:
        return a
    if b and not a:
        return b
    return f"{a} AND {b}"


def _time_filter(window: TimeWindow) -> str:
    # MLflow trace search supports trace.timestamp_ms filtering (start time in ms) :contentReference[oaicite:3]{index=3}
    return f"trace.timestamp_ms >= {window.start_time_ms} AND trace.timestamp_ms <= {window.end_time_ms}"


@dataclass
class MLflowTraceAdapter:
    tracking_uri: str

    def fetch(
            self,
            timeframe: TimeframeConfig,
            schema: SchemaBindings,
            experiment_id: str,
            query: Optional[str] = None,
    ) -> TraceBatch:
        """
        Best-effort MLflow trace fetch:
        - Uses filter_string (NOT start/end kwargs) for time constraints and user filters
        - Tries mlflow.search_traces(..., return_type="list") first
        - Falls back to MlflowClient.search_traces for older versions / different builds
        """
        window = _parse_timeframe(timeframe)

        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except Exception as e:
            raise RuntimeError("mlflow is not installed. Install with: pip install -e '.[mlflow]'") from e

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_id)

        # Build filter_string:
        # - Only add timestamp filter if user asked for a time constraint
        # - Also combine with user-provided query (e.g., tag filters)
        filter_string = query
        if _has_time_constraint(timeframe):
            filter_string = _combine_filters(filter_string, _time_filter(window))

        max_results = int(timeframe.last_n_traces) if timeframe.last_n_traces else None

        # When limiting results, order by most recent first
        order_by = ["timestamp_ms DESC"] if max_results else None  # supported ordering key :contentReference[oaicite:4]{index=4}

        traces_payload = None

        # Strategy 1: mlflow.search_traces (prefer list of Trace objects)
        try:
            kwargs: Dict[str, Any] = {}
            if filter_string:
                kwargs["filter_string"] = filter_string
            if max_results:
                kwargs["max_results"] = max_results
            if order_by:
                kwargs["order_by"] = order_by

            # Prefer list output when supported (MLflow 2.21.1+) :contentReference[oaicite:5]{index=5}
            try:
                kwargs["return_type"] = "list"
                traces_payload = mlflow.search_traces(**kwargs)  # type: ignore[attr-defined]
            except TypeError:
                # Older versions may not support return_type; fall back to DataFrame or client API
                kwargs.pop("return_type", None)
                traces_payload = mlflow.search_traces(**kwargs)  # type: ignore[attr-defined]
        except Exception:
            traces_payload = None

        # Strategy 2: MlflowClient().search_traces (often returns list output, supports pagination)
        if traces_payload is None:
            try:
                client = MlflowClient()
                kwargs = {}
                # Some builds require experiment_ids; if your server does, add them in config later.
                if filter_string:
                    kwargs["filter_string"] = filter_string
                if max_results:
                    kwargs["max_results"] = max_results
                if order_by:
                    kwargs["order_by"] = order_by

                traces_payload = client.search_traces(**kwargs)  # type: ignore[attr-defined]
            except Exception as e:
                raise RuntimeError(
                    "Could not fetch traces. Your MLflow build may not expose tracing APIs as expected.\n"
                    f"Tracking URI: {self.tracking_uri}\n"
                    f"Filter string used: {filter_string!r}\n"
                    f"Original error: {e}"
                ) from e

        traces = self._convert_payload_to_traces(traces_payload)
        return TraceBatch(
            traces=traces,
            window=window,
            source=TraceSource(name="mlflow", details={"tracking_uri": self.tracking_uri, "filter_string": filter_string}),
            schema=schema,
        )

    def _convert_payload_to_traces(self, payload: Any) -> List[Trace]:
        """
        Correctly handles MLflow Trace objects where:
          - trace.info is TraceInfo (ids, status, timing, tags, metadata)
          - trace.data is TraceData (spans)
        """
        # 1) If payload is already list of MLflow Trace objects
        if isinstance(payload, list) and payload:
            first = payload[0]
            if hasattr(first, "info") and hasattr(first, "data"):
                return [self._from_mlflow_trace_obj(t) for t in payload]

        # 2) If payload is a paged object, try to list() it
        try:
            if not isinstance(payload, (list, dict)) and hasattr(payload, "__iter__"):
                as_list = list(payload)
                if as_list and hasattr(as_list[0], "info") and hasattr(as_list[0], "data"):
                    return [self._from_mlflow_trace_obj(t) for t in as_list]
        except Exception:
            pass

        # 3) DataFrame-like (mlflow.search_traces() default return type)
        rows: List[Dict[str, Any]] = []
        try:
            if hasattr(payload, "to_pandas"):
                payload = payload.to_pandas()
            if "pandas" in str(type(payload)).lower():
                rows = payload.to_dict(orient="records")  # type: ignore
        except Exception:
            pass

        # 4) dict/list fallback
        if not rows:
            if isinstance(payload, list):
                rows = [r if isinstance(r, dict) else {"_raw": r} for r in payload]
            elif isinstance(payload, dict):
                rows = payload.get("traces") if isinstance(payload.get("traces"), list) else [payload]
            else:
                try:
                    rows = list(payload)  # type: ignore
                    rows = [r if isinstance(r, dict) else {"_raw": r} for r in rows]
                except Exception:
                    rows = []

        out: List[Trace] = []
        for i, r in enumerate(rows):
            # Many MLflow DataFrames include a `trace` column that contains a Trace object.
            t = r.get("trace")
            if t is not None and hasattr(t, "info") and hasattr(t, "data"):
                out.append(self._from_mlflow_trace_obj(t))
                continue

            # Otherwise, attempt to parse flattened columns (best effort).
            out.append(self._from_flat_row(r, fallback_id=f"trace_{i}"))

        return out

    def _from_mlflow_trace_obj(self, t: Any) -> Trace:
        """
        Properly parse MLflow Trace object: t.info (TraceInfo) + t.data (TraceData).
        """
        info = getattr(t, "info", None)
        data = getattr(t, "data", None)

        # ---- TraceInfo ----
        trace_id = str(
            getattr(info, "trace_id", None) or getattr(info, "request_id", None) or getattr(info, "id", None))

        timestamp_ms = int(getattr(info, "timestamp_ms", None) or 0)
        status = str(getattr(info, "status", None) or getattr(info, "state", None) or "UNSET")

        execution_time_ms = getattr(info, "execution_time_ms", None) or getattr(info, "execution_duration", None)
        end_time_ms = timestamp_ms + int(execution_time_ms) if execution_time_ms is not None else timestamp_ms

        attrs: Dict[str, Any] = {}

        tags = getattr(info, "tags", None)
        if isinstance(tags, dict):
            # keep tags accessible; TraceView can look for tag.* or plain keys depending on bindings
            attrs.update({f"tag.{k}": v for k, v in tags.items()})

        meta = getattr(info, "trace_metadata", None) or getattr(info, "metadata", None) or getattr(info,
                                                                                                   "request_metadata",
                                                                                                   None)
        if isinstance(meta, dict):
            attrs.update(meta)

        # ---- TraceData (Spans) ----
        spans: List[Span] = []
        data_spans = getattr(data, "spans", None) if data is not None else None
        if data_spans is None:
            data_spans = []

        for sp in data_spans:
            spans.append(self._from_mlflow_span_obj(sp, default_trace_id=trace_id, default_start=timestamp_ms,
                                                    default_end=end_time_ms))

        return Trace(
            trace_id=trace_id,
            start_time_ms=timestamp_ms,
            end_time_ms=end_time_ms,
            status=status,
            attributes=attrs,
            spans=spans,
        )

    def _from_mlflow_span_obj(self, sp: Any, default_trace_id: str, default_start: int, default_end: int) -> Span:
        """
        Span objects also have a documented schema; we parse defensively.
        """
        span_id = str(getattr(sp, "span_id", None) or getattr(sp, "id", None) or "")
        parent = getattr(sp, "parent_id", None) or getattr(sp, "parent_span_id", None)
        name = str(getattr(sp, "name", None) or "span")

        st = int(getattr(sp, "start_time_ms", None) or default_start)
        et = int(getattr(sp, "end_time_ms", None) or default_end)

        sstatus = str(getattr(sp, "status", None) or "UNSET")
        kind = getattr(sp, "span_type", None) or getattr(sp, "type", None) or getattr(sp, "kind", None)

        sattrs = getattr(sp, "attributes", None) or {}
        if not isinstance(sattrs, dict):
            sattrs = {}

        # Events are optional; your current analytics doesn't depend on them, so keep empty for now.
        return Span(
            span_id=span_id or f"{default_trace_id}_span_unknown",
            parent_span_id=parent,
            name=name,
            start_time_ms=st,
            end_time_ms=et,
            status=sstatus,
            kind=str(kind) if kind is not None else None,
            attributes=dict(sattrs),
            events=[],
        )

    def _from_flat_row(self, r: Dict[str, Any], fallback_id: str) -> Trace:
        """
        Best-effort parser for DataFrame rows where fields may be flattened.
        Not all MLflow builds expose the same columns.
        """
        now = _now_ms()
        # trace_id might be in trace_id/request_id, or in nested info-like columns
        trace_id = str(r.get("trace_id") or r.get("request_id") or r.get("id") or fallback_id)

        # MLflow docs use trace.timestamp_ms; DataFrame often includes timestamp_ms
        start = int(r.get("timestamp_ms") or r.get("trace_timestamp_ms") or now)

        duration = r.get("execution_time_ms") or r.get("execution_duration")
        end = start + int(duration) if duration is not None else int(r.get("end_time_ms") or start)

        status = str(r.get("status") or r.get("state") or "UNSET")

        attrs: Dict[str, Any] = {}
        if isinstance(r.get("tags"), dict):
            attrs.update({f"tag.{k}": v for k, v in r["tags"].items()})
        if isinstance(r.get("metadata"), dict):
            attrs.update(r["metadata"])
        if isinstance(r.get("trace_metadata"), dict):
            attrs.update(r["trace_metadata"])

        spans_payload = r.get("spans") or []
        spans: List[Span] = []
        if isinstance(spans_payload, list):
            for j, sp in enumerate(spans_payload):
                if not isinstance(sp, dict):
                    continue
                spans.append(
                    Span(
                        span_id=str(sp.get("span_id") or sp.get("id") or f"{trace_id}_span_{j}"),
                        parent_span_id=sp.get("parent_id") or sp.get("parent_span_id"),
                        name=str(sp.get("name") or sp.get("operation") or "span"),
                        start_time_ms=int(sp.get("start_time_ms") or start),
                        end_time_ms=int(sp.get("end_time_ms") or end),
                        status=str(sp.get("status") or "UNSET"),
                        kind=sp.get("kind") or sp.get("type"),
                        attributes=dict(sp.get("attributes") or sp.get("tags") or {}),
                        events=[],
                    )
                )

        return Trace(
            trace_id=trace_id,
            start_time_ms=start,
            end_time_ms=end,
            status=status,
            attributes=attrs,
            spans=spans,
        )
