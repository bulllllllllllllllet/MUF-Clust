"""聚类步骤骨架（4.x）"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from ...utils.logging import log_info


@dataclass
class ClusterStepSkeleton:
    name: str = "cluster_skeleton"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log_info("运行聚类骨架：占位不做实际聚类")
        return {"cluster": {"status": "skipped"}}