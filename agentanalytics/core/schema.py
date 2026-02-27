from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SchemaBindings:
    # Keys in canonical Trace/Span attributes to locate semantic fields.
    # These are *suggestions*; TraceView can fall back to heuristics.
    query_text_keys: List[str] = field(
        default_factory=lambda: ["user.query_text", "query_text", "input.query", "prompt"]
    )
    final_answer_keys: List[str] = field(
        default_factory=lambda: ["agent.final_answer", "final_answer", "output.text", "response"]
    )
    error_type_keys: List[str] = field(default_factory=lambda: ["error.type", "exception.type"])
    error_message_keys: List[str] = field(default_factory=lambda: ["error.message", "exception.message"])
    tool_name_keys: List[str] = field(default_factory=lambda: ["tool.name", "tool_name"])
    tool_args_keys: List[str] = field(default_factory=lambda: ["tool.args", "tool.arguments", "tool_args"])
    tool_result_keys: List[str] = field(default_factory=lambda: ["tool.result", "tool_output", "tool.response"])

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SchemaBindings":
        sb = SchemaBindings()
        for k, v in (d or {}).items():
            if hasattr(sb, k) and isinstance(v, list):
                setattr(sb, k, v)
        return sb
