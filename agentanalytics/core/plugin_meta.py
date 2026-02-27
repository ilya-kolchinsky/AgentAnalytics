from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PluginParam:
    key: str
    type: str                     # "int" | "float" | "bool" | "str" | "enum" | "json"
    title: str
    description: str = ""
    default: Any = None
    required: bool = False
    enum: Optional[List[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class PluginMeta:
    name: str
    version: str
    title: str
    description: str
    params: List[PluginParam] = field(default_factory=list)

    # Optional: semantic requirements (what TraceView fields you assume)
    requires: List[str] = field(default_factory=list)

    def defaults_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for p in self.params:
            if p.default is not None:
                d[p.key] = p.default
        return d
