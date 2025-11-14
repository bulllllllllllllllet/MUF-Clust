"""插件抽象基类与接口说明"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


class Plugin(Protocol):
    name: str

    def supports(self, task: str) -> bool:
        """中文说明：返回插件是否支持某任务（如 'segmentation'）。"""
        ...

    def apply(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """中文说明：执行插件逻辑并返回结果。"""
        ...


@dataclass
class ExamplePlugin:
    name: str = "example"

    def supports(self, task: str) -> bool:
        return task in {"segmentation", "unmixing"}

    def apply(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        return {"plugin_result": "placeholder"}