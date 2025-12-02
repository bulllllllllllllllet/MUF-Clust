"""高层管线 API

中文注释说明：
- 该模块提供稳定对外接口，屏蔽内部实现细节。
- 日志输出遵循分隔标记规范；绘图内使用英文标签。
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import os

import numpy as np
import pandas as pd

from ..utils.logging import log_info, log_warn, log_error
from ..core.pipeline import Pipeline, Step
from ..core.steps.preprocess import list_images
from ..core.steps.cluster import _zscore_matrix, _pca, _kmeans


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


def run_features(image_path: str,
                 seg_out_dir: str,
                 roi_tiles_dir: str,
                 options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """仅执行特征提取步骤。

    中文说明：
    - 基于已有分割输出（ROI标签 tiles）进行强度特征汇总。
    - 需提供原始影像路径、分割输出目录与 ROI tiles 目录。
    """
    log_info(f"开始特征提取（features-only）：image_path={image_path} seg_out_dir={seg_out_dir}")

    steps: list[Step] = []
    from ..core.steps.features import FeatureStep
    steps.append(FeatureStep())

    pipeline = Pipeline(steps=steps)
    context = {
        "image_path": image_path,
        "segmentation": {
            "out_dir": seg_out_dir,
            "paths": {"roi_tiles_dir": roi_tiles_dir},
        },
        "prefer_low_res": bool((options or {}).get("prefer_low_res", False)),
        "cancer_type": (options or {}).get("cancer_type"),
        "only_tile_x": (options or {}).get("only_tile_x"),
        "only_tile_y": (options or {}).get("only_tile_y"),
        "num_workers": int((options or {}).get("num_workers", 1)),
    }
    result = pipeline.run(context)
    return result


def run_features_folder(dataset_dir: str,
                        seg_root: str,
                        options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """文件夹模式执行特征提取。

    中文说明：
    - 递归遍历 dataset_dir 下的 qptiff/tiff 图像；
    - 根据图像 basename 匹配 seg_root/<basename>/roi_tiles；
    - 若检测到已有 features_long.csv 和 features_matrix_mean.csv 则跳过该样本。
    """
    log_info(f"开始特征提取（文件夹模式）：dataset_dir={dataset_dir} seg_root={seg_root}")

    from ..core.steps.features import FeatureStep

    images = list_images(dataset_dir)
    if not images:
        log_error("文件夹模式：未在 dataset_dir 中找到 qptiff/tiff 图像")
        raise RuntimeError("未发现可进行特征提取的图像文件")

    total_cells = 0
    processed = 0
    failed: list[str] = []

    for idx, img in enumerate(images, 1):
        sample_name = os.path.splitext(os.path.basename(img))[0]
        seg_out_dir = os.path.join(seg_root, sample_name)
        roi_dir = os.path.join(seg_out_dir, "roi_tiles")
        features_dir = os.path.join(seg_out_dir, "features")
        long_csv = os.path.join(features_dir, "features_long.csv")
        matrix_csv = os.path.join(features_dir, "features_matrix_mean.csv")

        if os.path.isdir(features_dir) and os.path.isfile(long_csv) and os.path.isfile(matrix_csv):
            log_info(f"跳过已完成样本：{sample_name}（检测到 features_long.csv 与 features_matrix_mean.csv）")
            continue

        if not os.path.isdir(roi_dir):
            log_warn(f"跳过样本：{sample_name}（缺少 ROI tiles 目录：{roi_dir}）")
            failed.append(img)
            continue

        log_info(f"[{idx}/{len(images)}] 特征提取：{img}")
        steps: list[Step] = [FeatureStep()]
        pipeline = Pipeline(steps=steps)
        context = {
            "image_path": img,
            "segmentation": {
                "out_dir": seg_out_dir,
                "paths": {"roi_tiles_dir": roi_dir},
            },
            "prefer_low_res": bool((options or {}).get("prefer_low_res", False)),
            "cancer_type": (options or {}).get("cancer_type"),
            "only_tile_x": (options or {}).get("only_tile_x"),
            "only_tile_y": (options or {}).get("only_tile_y"),
            "num_workers": int((options or {}).get("num_workers", 1)),
        }
        try:
            res = pipeline.run(context)
            feats = res.get("features", {})
            total_cells += int(feats.get("cells", 0))
            processed += 1
        except Exception as e:
            log_error(f"处理失败：{img}，错误：{e}")
            failed.append(img)
            continue

    return {
        "features": {
            "status": "ok",
            "mode": "folder",
            "root": seg_root,
            "dataset_dir": dataset_dir,
            "images": len(images),
            "processed": processed,
            "cells": total_cells,
            "failed": failed,
        }
    }


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

    # 分割→特征→聚类→可视化（分割与特征已实现，其余保持骨架）
    try:
        from ..core.steps.segmentation import SegmentationStep
        from ..core.steps.features import FeatureStep
        from ..core.steps.cluster import ClusterStepSkeleton
        from ..core.steps.visualize import VisualizeStepSkeleton
        steps += [SegmentationStep(), FeatureStep(), ClusterStepSkeleton(), VisualizeStepSkeleton()]
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


def run_cluster_from_csv(matrix_csv: str,
                         seed: int = 42,
                         k: Optional[int] = None) -> Dict[str, Any]:
    """对已有特征矩阵 CSV 执行 Z-score→PCA→KMeans 聚类。

    中文说明：
    - 读取由 FeatureStep 写出的 features_matrix_mean.csv；
    - 对通道列做 Z-score 标准化与 PCA 降维；
    - 在 PCA 空间内运行 KMeans 聚类；
    - 将 cluster/置信度/PCA 坐标写回新的 *_cluster.csv 文件。
    """
    log_info(f"开始基于 CSV 的聚类：matrix_csv={matrix_csv}")

    df = pd.read_csv(matrix_csv)
    if df.empty or df.shape[0] == 0:
        log_warn("输入特征矩阵为空，跳过聚类")
        return {"cluster": {"status": "skipped", "reason": "empty_features", "matrix_csv": matrix_csv}}

    non_feature_cols = [c for c in df.columns if c.lower() in {"cell_id", "tile_x", "tile_y", "label", "x", "y", "centroid_x", "centroid_y", "local_x", "local_y"}]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    if len(feature_cols) == 0:
        log_warn("未在 CSV 中找到特征列，跳过聚类")
        return {"cluster": {"status": "skipped", "reason": "no_feature_columns", "matrix_csv": matrix_csv}}

    x = df[feature_cols].to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_norm = _zscore_matrix(x)

    n_components = min(20, x_norm.shape[0], x_norm.shape[1])
    scores, explained_ratio = _pca(x_norm, n_components=n_components)

    n_cells = x.shape[0]
    if k is None:
        if n_cells < 20:
            k_val = max(2, min(4, n_cells))
        elif n_cells < 100:
            k_val = 6
        else:
            k_val = 8
    else:
        k_val = int(max(2, min(k, n_cells)))

    labels, centers = _kmeans(scores, k=k_val, n_init=5, max_iter=100, seed=int(seed))

    if centers.size > 0:
        dists = np.linalg.norm(scores[:, None, :] - centers[None, :, :], axis=2)
        sorted_d = np.sort(dists, axis=1)
        d1 = sorted_d[:, 0]
        d2 = sorted_d[:, 1] if sorted_d.shape[1] > 1 else (sorted_d[:, 0] + 1e-6)
        confidence = 1.0 - np.clip(d1 / (d2 + 1e-6), 0.0, 1.0)
    else:
        confidence = np.ones((n_cells,), dtype=float)

    df["cluster"] = labels.astype(int)
    df["confidence"] = confidence.astype(float)
    for i in range(scores.shape[1]):
        df[f"pca_{i+1}"] = scores[:, i]

    out_csv = matrix_csv.replace(".csv", "_cluster.csv")
    df.to_csv(out_csv, index=False)

    log_info(f"CSV 聚类完成：cells={n_cells}, k={k_val}, 输入={matrix_csv}, 输出={out_csv}")

    return {
        "cluster": {
            "status": "ok",
            "k": int(k_val),
            "cells": int(n_cells),
            "labels_csv": out_csv,
            "explained_variance_ratio": explained_ratio.tolist(),
            "matrix_csv": matrix_csv,
        }
    }


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
        "num_workers": int((options or {}).get("num_workers", 1)),
        "folder_mode": bool(is_dir),
        "only_roi_tiles": bool(is_dir),
    }
    result = pipeline.run(context)
    return result
