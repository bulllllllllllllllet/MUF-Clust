"""高层管线 API

中文注释说明：
- 该模块提供稳定对外接口，屏蔽内部实现细节。
- 日志输出遵循分隔标记规范；绘图内使用英文标签。
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from ..utils.logging import log_info
from ..core.pipeline import Pipeline, Step


def run_preprocess(image_path: Optional[str] = None,
                   dataset_dir: Optional[str] = None,
                   output_dir: str = "outputs",
                   qc: str = "basic",
                   seed: int = 42,
                   options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """执行 2.2 预处理与 QC 步骤（对齐 guide.md）。

    中文说明：
    - `image_path` 与 `dataset_dir` 二选一；若均提供，优先处理 `image_path`。
    - 返回字典包含产物路径与摘要信息。
    """
    log_info(f"开始预处理与QC：image_path={image_path} dataset_dir={dataset_dir} qc={qc}")

    steps: list[Step] = []
    try:
        from ..core.steps.preprocess import PreprocessStep
        steps.append(PreprocessStep(qc_level=qc, seed=seed, options=options or {}))
    except Exception:
        # 中文日志：若步骤实现暂未提供，提示骨架执行
        from ..core.steps.preprocess import PreprocessStepSkeleton
        steps.append(PreprocessStepSkeleton(qc_level=qc, seed=seed, options=options or {}))

    pipeline = Pipeline(steps=steps)
    context = {
        "image_path": image_path,
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
    }
    result = pipeline.run(context)
    return result


def run_full_pipeline(input_path: str,
                      output_dir: str = "outputs",
                      config_path: Optional[str] = None,
                      seed: int = 42,
                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """执行完整管线：分割→特征→降维→聚类→可视化与导出。

    中文说明：
    - 该接口会顺序调用核心步骤，具体实现可逐步迁移。
    - 返回的字典包含各阶段产物路径、统计与参数。
    """
    log_info(f"开始完整管线：input_path={input_path} config={config_path}")

    steps: list[Step] = []
    try:
        from ..core.steps.preprocess import PreprocessStep
        steps.append(PreprocessStep(qc_level=options.get("qc", "basic") if options else "basic",
                                    seed=seed, options=options or {}))
    except Exception:
        from ..core.steps.preprocess import PreprocessStepSkeleton
        steps.append(PreprocessStepSkeleton(qc_level=options.get("qc", "basic") if options else "basic",
                                            seed=seed, options=options or {}))

    # 留出其他步骤骨架（segmentation/features/cluster/visualize），后续填充实现
    try:
        from ..core.steps.segmentation import SegmentationStepSkeleton
        from ..core.steps.features import FeatureStepSkeleton
        from ..core.steps.cluster import ClusterStepSkeleton
        from ..core.steps.visualize import VisualizeStepSkeleton
        steps += [SegmentationStepSkeleton(), FeatureStepSkeleton(), ClusterStepSkeleton(), VisualizeStepSkeleton()]
    except Exception:
        pass

    pipeline = Pipeline(steps=steps)
    context = {
        "input_path": input_path,
        "output_dir": output_dir,
        "config_path": config_path,
    }
    result = pipeline.run(context)
    return result