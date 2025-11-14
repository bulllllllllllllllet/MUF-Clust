"""分割步骤骨架（3.1/3.2）"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from ...utils.logging import log_info


@dataclass
class SegmentationStepSkeleton:
    name: str = "segmentation_skeleton"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log_info("运行分割骨架：占位不做实际分割")
        return {"segmentation": {"status": "skipped"}}