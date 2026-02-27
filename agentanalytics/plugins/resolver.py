from dataclasses import dataclass
from typing import Dict

from agentanalytics.core.plugin import Plugin


@dataclass
class DictPluginResolver:
    plugins: Dict[str, Plugin]

    def get(self, name: str) -> Plugin:
        if name not in self.plugins:
            raise KeyError(f"Unknown plugin '{name}'. Available: {list(self.plugins.keys())}")
        return self.plugins[name]
