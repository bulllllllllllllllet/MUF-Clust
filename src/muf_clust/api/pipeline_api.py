"""高层管线 API

中文注释说明：
- 该模块提供稳定对外接口，屏蔽内部实现细节。
- 日志输出遵循分隔标记规范；绘图内使用英文标签。
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import os

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
        "prefer_low_res": not bool((options or {}).get("high_res", True)),
        "ref_channel": (options or {}).get("ref_channel", "DAPI"),
        "drift_max_tiles": int((options or {}).get("drift_max_tiles", 0)),
        "use_cellpose": bool((options or {}).get("use_cellpose", True)),
        "only_tile_x": (options or {}).get("only_tile_x"),
        "only_tile_y": (options or {}).get("only_tile_y"),
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

    # 分割→特征→聚类→可视化（分割已实现，其余保持骨架）
    try:
        from ..core.steps.segmentation import SegmentationStep
        from ..core.steps.features import FeatureStepSkeleton
        from ..core.steps.cluster import ClusterStepSkeleton
        from ..core.steps.visualize import VisualizeStepSkeleton
        steps += [SegmentationStep(), FeatureStepSkeleton(), ClusterStepSkeleton(), VisualizeStepSkeleton()]
    except Exception:
        pass

    pipeline = Pipeline(steps=steps)
    # 兼容预处理步骤的入参：自动将 input_path 映射为 image_path 或 dataset_dir
    is_dir = os.path.isdir(input_path)
    context = {
        "input_path": input_path,
        "image_path": (None if is_dir else input_path),
        "dataset_dir": (input_path if is_dir else None),
        "output_dir": output_dir,
        "config_path": config_path,
        "prefer_low_res": bool((options or {}).get("prefer_low_res", False)),
        "ref_channel": (options or {}).get("ref_channel", "DAPI"),
        "drift_max_tiles": int((options or {}).get("drift_max_tiles", 0)),
        "cancer_type": (options or {}).get("cancer_type"),
        "tile_size": int((options or {}).get("tile_size", 1024)),
        "use_cellpose": bool((options or {}).get("use_cellpose", True)),
        "cellpose_model": (options or {}).get("cellpose_model") or "nuclei",
        "cellpose_gpu": bool((options or {}).get("cellpose_gpu", True)),
        "cellpose_diameter": (options or {}).get("cellpose_diameter"),
        "cellpose_batch_size": (options or {}).get("cellpose_batch_size"),
        "only_tile_x": (options or {}).get("only_tile_x"),
        "only_tile_y": (options or {}).get("only_tile_y"),
    }
    result = pipeline.run(context)
    return result


def run_segmentation(input_path: str,
                     output_dir: str = "outputs",
                     options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """仅执行分割步骤（解耦预处理）。

    中文说明：
    - 只调用分割步骤，直接在原始 DAPI 图像上进行 Cellpose 分割。
    - 返回分割输出目录与统计信息。

    默认参数说明（若 `options` 未提供对应键，将使用如下默认）：
    - `prefer_low_res=False`：默认使用高分辨率层进行读取与分割。
    - `cancer_type=None`：通道配置类别（为空则在下游使用全局默认）。
    - `tile_size=1024`：分割时的 tile 尺寸（像素）。
    - `use_cellpose=True`：启用 Cellpose 进行分割。
    - `cellpose_model='nuclei'`：Cellpose 模型类型（核分割）。
    - `cellpose_gpu=True`：启用 GPU（若不可用则在下游报错）。
    - `cellpose_diameter=None`：核直径（像素，None 由 Cellpose自动/默认推断）。
    - `cellpose_batch_size=None`：推理批大小（None 使用库默认）。
    - `only_tile_x=None` / `only_tile_y=None`：仅处理指定 tile 坐标（None 处理全部）。
    """
    log_info(f"开始分割（segmentation-only）：input_path={input_path}")

    steps: list[Step] = []
    from ..core.steps.segmentation import SegmentationStep
    steps.append(SegmentationStep())

    pipeline = Pipeline(steps=steps)
    is_dir = os.path.isdir(input_path)
    context = {
        "input_path": input_path,
        "image_path": (None if is_dir else input_path),
        "dataset_dir": (input_path if is_dir else None),
        "output_dir": output_dir,
        "prefer_low_res": bool((options or {}).get("prefer_low_res", False)),
        "cancer_type": (options or {}).get("cancer_type"),
        "tile_size": int((options or {}).get("tile_size", 1024)),
        "use_cellpose": bool((options or {}).get("use_cellpose", True)),
        "cellpose_model": (options or {}).get("cellpose_model") or "nuclei",
        "cellpose_gpu": bool((options or {}).get("cellpose_gpu", True)),
        "cellpose_diameter": (options or {}).get("cellpose_diameter"),
        "cellpose_batch_size": (options or {}).get("cellpose_batch_size"),
        "only_tile_x": (options or {}).get("only_tile_x"),
        "only_tile_y": (options or {}).get("only_tile_y"),
    }
    result = pipeline.run(context)
    return result
