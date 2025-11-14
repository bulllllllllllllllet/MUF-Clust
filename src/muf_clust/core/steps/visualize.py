"""可视化步骤骨架（5.x）"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from ...utils.logging import log_info


@dataclass
class VisualizeStepSkeleton:
    name: str = "visualize_skeleton"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log_info("运行可视化骨架：占位不做实际绘图")
        return {"visualize": {"status": "skipped"}}