"""特征提取步骤骨架（3.3/3.4）"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from ...utils.logging import log_info


@dataclass
class FeatureStepSkeleton:
    name: str = "features_skeleton"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log_info("运行特征骨架：占位不做实际特征提取")
        return {"features": {"status": "skipped"}}