import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List
from collections import Counter, defaultdict

from ..core.plugin import PluginResult, MetricRecord, Artifact
from ..core.model import TraceBatch
from ..core.view import TraceView
from ..utils.textnorm import norm_numbers, shorten


def _is_tool_error(tc) -> bool:
    # Tool span status is best signal
    if (tc.status or "").upper() == "ERROR":
        return True
    # Some instrumentations store tool.status in args/result dicts; check lightly
    if isinstance(tc.result, dict) and str(tc.result.get("status", "")).upper() == "ERROR":
        return True
    return False


def _error_type(tc) -> str:
    if isinstance(tc.result, dict):
        et = tc.result.get("error") or tc.result.get("error_type") or ""
        return str(et).strip() or "UNKNOWN"
    return "UNKNOWN"


def _args_invalid(tc) -> bool:
    # Prefer explicit error type from tool_result
    if isinstance(tc.result, dict):
        et = str(tc.result.get("error") or tc.result.get("error_type") or "").upper()
        if et == "INVALID_ARGUMENTS":
            return True
    # fallback heuristic
    if tc.args is None:
        return True
    if isinstance(tc.args, dict) and len(tc.args) == 0:
        return True
    return False


@dataclass
class ToolReliabilityAuditor:
    name: str = "tool_reliability_auditor"
    version: str = "0.1.0"

    def analyze(self, batch: TraceBatch, ctx, config: Dict[str, Any]) -> PluginResult:
        top_k_tools = int(config.get("top_k_tools", 20))
        include_examples = bool(config.get("include_examples", True))
        max_examples = int(config.get("max_examples", 5))

        views: List[TraceView] = [ctx.view_factory.make(t) for t in batch.traces]

        tool_stats = defaultdict(lambda: {
            "calls": 0,
            "errors": 0,
            "invalid_args": 0,
            "trace_errors": 0,
            "tool_signatures": Counter(),
            "error_snippets": Counter(),
            "example_trace_ids": [],
        })

        total_tool_calls = 0
        total_error_calls = 0

        error_type_counts = Counter()
        tools_by_error_type = defaultdict(lambda: Counter())  # err_type -> tool -> count
        examples_by_error_type = defaultdict(list)  # err_type -> [trace_id...]

        for v in views:
            terr = v.error() is not None
            tool_calls = v.tool_calls()
            for tc in tool_calls:
                total_tool_calls += 1
                st = tool_stats[tc.name]
                st["calls"] += 1
                if terr:
                    st["trace_errors"] += 1

                is_err = _is_tool_error(tc)
                if is_err:
                    st["errors"] += 1
                    total_error_calls += 1

                if _args_invalid(tc):
                    st["invalid_args"] += 1

                # Tool signature = name only for now; later you can add args schema fingerprint
                st["tool_signatures"][tc.name] += 1

                if is_err:
                    if isinstance(tc.result, dict):
                        et = str(tc.result.get("error") or tc.result.get("error_type") or "")
                        detail = str(tc.result.get("detail") or tc.result.get("message") or "")
                        sig = f"{et}: {detail}".strip()
                        if sig:
                            st["error_snippets"][norm_numbers(sig)[:180]] += 1

                if include_examples and len(st["example_trace_ids"]) < max_examples:
                    st["example_trace_ids"].append(v.trace_id)

                if is_err:
                    etype = _error_type(tc).upper()
                    error_type_counts[etype] += 1
                    tools_by_error_type[etype][tc.name] += 1
                    if include_examples and len(examples_by_error_type[etype]) < max_examples:
                        examples_by_error_type[etype].append(v.trace_id)

        # Build table
        tools = []
        for tool, st in tool_stats.items():
            calls = st["calls"]
            errors = st["errors"]
            inv = st["invalid_args"]
            erate = errors / calls if calls else 0.0
            irate = inv / calls if calls else 0.0
            top_err = st["error_snippets"].most_common(1)[0][0] if st["error_snippets"] else ""
            tools.append({
                "tool": tool,
                "calls": calls,
                "errors": errors,
                "error_rate": erate,
                "invalid_args": inv,
                "invalid_args_rate": irate,
                "top_error_snippet": top_err,
                "example_trace_ids": st["example_trace_ids"],
            })

        tools.sort(key=lambda x: (x["errors"], x["calls"]), reverse=True)
        top_tools = tools[:top_k_tools]

        if total_error_calls == 0:
            top_failing = []
        else:
            top_failing = [t["tool"] for t in top_tools[:3]]

        summary = {
            "headline": "No tool failures detected" if total_error_calls == 0 else f"{total_error_calls} failed tool calls detected",
            "kpis": {
                "total_tool_calls": total_tool_calls,
                "tools_seen": len(tools),
                "failed_tool_calls": total_error_calls,
                "overall_error_rate": (total_error_calls / total_tool_calls) if total_tool_calls else 0.0,
                "error_types": error_type_counts.most_common(),
                "top_tools_by_error_type": {
                    et: tools_by_error_type[et].most_common(10)
                    for et, _ in error_type_counts.most_common()
                },
                "example_trace_ids_by_error_type": dict(examples_by_error_type),
            },
            "bullets": (
                ["All tool spans were OK in this dataset."]
                if total_error_calls == 0
                else [f"Top failing tools: {', '.join(top_failing)}"]
            ),
            "top_items": top_failing,
        }

        # Write artifacts
        os.makedirs(ctx.output_dir, exist_ok=True)
        json_path = os.path.join(ctx.output_dir, "tool_reliability.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "top_tools": top_tools,
                "all_tools": tools,
            }, f, indent=2)

        md_path = os.path.join(ctx.output_dir, "tool_reliability_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Tool Call Reliability Report\n\n")
            f.write(f"- Tool calls observed: **{total_tool_calls}**\n")
            f.write(f"- Tools seen: **{len(tools)}**\n")
            f.write(f"- Overall tool error rate: **{summary['kpis']['overall_error_rate']:.2%}**\n\n")
            f.write("## Top failing tools\n\n")
            if total_error_calls == 0:
                f.write("No failed tool calls were detected in this dataset.\n\n")
            else:
                for i, t in enumerate(top_tools, 1):
                    f.write(f"### {i}. `{t['tool']}` — {t['errors']} errors / {t['calls']} calls ({t['error_rate']:.2%})\n\n")
                    f.write(f"- Invalid args rate: **{t['invalid_args_rate']:.2%}** ({t['invalid_args']} / {t['calls']})\n")
                    if t["top_error_snippet"]:
                        f.write(f"- Most common error snippet: `{shorten(t['top_error_snippet'], 180)}`\n")
                    if t["example_trace_ids"]:
                        f.write("- Example trace IDs:\n")
                        for tid in t["example_trace_ids"][:max_examples]:
                            f.write(f"  - `{tid}`\n")
                    f.write("\n")

            f.write("## Top error types\n\n")
            if total_error_calls == 0:
                f.write("✅ No tool errors detected.\n\n")
            else:
                for et, c in error_type_counts.most_common(10):
                    f.write(f"- `{et}`: **{c}**\n")
                f.write("\n")

            f.write("## Top tools per error type\n\n")
            if total_error_calls == 0:
                f.write("No tool errors, so nothing to break down.\n\n")
            else:
                for et, c in error_type_counts.most_common(10):
                    f.write(f"### `{et}` — {c} occurrences\n\n")
                    top_tools_by_type = tools_by_error_type[et].most_common(8)
                    for tool, cnt in top_tools_by_type:
                        f.write(f"- `{tool}`: {cnt}\n")
                    ex = examples_by_error_type.get(et, [])
                    if ex:
                        f.write("\nExample trace IDs:\n")
                        for tid in ex[:max_examples]:
                            f.write(f"- `{tid}`\n")
                    f.write("\n")

            f.write("## Suggestions\n\n")
            f.write("- Tools with high invalid-args rates: add schema validation + examples; log structured argument errors.\n")
            f.write("- Tools with high error rates: inspect auth/timeouts/retries; verify tool contracts match agent output.\n")

        top_types = [et for et, _ in error_type_counts.most_common(3)]
        if total_error_calls == 0:
            summary_md = (
                "✅ **No failed tool calls detected**\n\n"
                f"- Tool calls observed: **{total_tool_calls}**\n"
                f"- Tools seen: **{len(tools)}**\n"
            )
        else:
            top_list = "\n".join(
                [f"    - `{t['tool']}`: **{t['errors']}** errors (**{t['error_rate'] * 100:.1f}%**)" for t in
                 top_tools[:3]])
            summary_md = (
                f"⚠️ **{total_error_calls} failed tool calls detected**\n\n"
                f"- Overall error rate: **{(total_error_calls / total_tool_calls) * 100:.1f}%**\n"
                f"- Top failing tools:\n{top_list}\n"
                f"- Top error types: {', '.join(top_types) if top_types else '-'}\n"
            )

        return PluginResult(
            metrics=[
                MetricRecord(name="tool.reliability.summary", data=summary),
                MetricRecord(name="tool.reliability.top_tools", data={"top_tools": top_tools}),
            ],
            artifacts=[
                Artifact(kind="markdown", path=md_path, description="Tool reliability report"),
                Artifact(kind="json", path=json_path, description="Tool reliability JSON"),
            ],
            summary=summary,
            summary_md=summary_md,
        )
