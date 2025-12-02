"""聚类步骤骨架（4.x）"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from ...utils.logging import log_info, log_warn, log_error


def _zscore_matrix(x: np.ndarray) -> np.ndarray:
    """对特征矩阵按列做 Z-score 标准化，忽略全零/常数列。"""
    if x.size == 0:
        return x
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


def _pca(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """使用 SVD 实现 PCA，返回 (scores, explained_variance_ratio)。"""
    if x.size == 0:
        return x, np.array([])
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    n = x.shape[0]
    explained_var = (s ** 2) / max(n - 1, 1)
    total_var = explained_var.sum() if explained_var.size > 0 else 1.0
    explained_ratio = explained_var / total_var
    k = max(1, min(n_components, vt.shape[0]))
    scores = u[:, :k] * s[:k]
    return scores, explained_ratio[:k]


def _kmeans(x: np.ndarray, k: int, n_init: int = 5, max_iter: int = 100, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """简易 KMeans 实现，返回 (labels, centers)。"""
    if x.size == 0 or k <= 0:
        return np.zeros((x.shape[0],), dtype=int), np.empty((0, x.shape[1]))
    rng = np.random.default_rng(seed)
    best_inertia = np.inf
    best_labels = None
    best_centers = None
    for _ in range(max(1, n_init)):
        indices = rng.choice(x.shape[0], size=min(k, x.shape[0]), replace=False)
        centers = x[indices].astype(float, copy=True)
        labels = np.zeros((x.shape[0],), dtype=int)
        for _ in range(max_iter):
            # E-step: 分配标签
            dists = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # M-step: 更新中心
            for j in range(centers.shape[0]):
                mask = labels == j
                if not np.any(mask):
                    continue
                centers[j] = x[mask].mean(axis=0)
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels if best_labels is not None else np.zeros((x.shape[0],), dtype=int), best_centers if best_centers is not None else np.empty((0, x.shape[1]))


@dataclass
class ClusterStepSkeleton:
    name: str = "cluster"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        features = context.get("features") or {}
        matrix_csv = features.get("features_matrix_csv")
        if not matrix_csv:
            log_warn("缺少特征矩阵 CSV（features_matrix_csv），聚类步骤跳过")
            return {"cluster": {"status": "skipped", "reason": "no_features_matrix"}}

        try:
            df = pd.read_csv(matrix_csv)
        except Exception as e:
            log_error(f"读取特征矩阵失败: {matrix_csv}, 错误: {e}")
            raise

        if df.empty or df.shape[0] == 0:
            log_warn("特征矩阵为空，聚类步骤跳过")
            return {"cluster": {"status": "skipped", "reason": "empty_features"}}

        # 约定：非特征列（索引/坐标等）保留在前面，其余数值列作为聚类特征
        non_feature_cols = [c for c in df.columns if c.lower() in {"cell_id", "tile_x", "tile_y", "label", "x", "y", "centroid_x", "centroid_y"}]
        feature_cols = [c for c in df.columns if c not in non_feature_cols]
        if len(feature_cols) == 0:
            log_warn("特征矩阵中未找到数值特征列，聚类步骤跳过")
            return {"cluster": {"status": "skipped", "reason": "no_feature_columns"}}

        x = df[feature_cols].to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_norm = _zscore_matrix(x)

        # PCA 降维，最多保留 20 维或样本/特征数的较小值
        n_components = min(20, x_norm.shape[0], x_norm.shape[1])
        scores, explained_ratio = _pca(x_norm, n_components=n_components)

        # 选择聚类数 K：默认 8，若细胞数较少则减小
        n_cells = x.shape[0]
        if n_cells < 20:
            k = max(2, min(4, n_cells))
        elif n_cells < 100:
            k = 6
        else:
            k = 8

        seed = int(context.get("seed", 42))
        labels, centers = _kmeans(scores, k=k, n_init=5, max_iter=100, seed=seed)

        # 置信度：用距离最近/次近中心的比值构造一个简单指标（0-1 趋势）
        if centers.size > 0:
            dists = np.linalg.norm(scores[:, None, :] - centers[None, :, :], axis=2)
            sorted_d = np.sort(dists, axis=1)
            d1 = sorted_d[:, 0]
            d2 = sorted_d[:, 1] if sorted_d.shape[1] > 1 else (sorted_d[:, 0] + 1e-6)
            confidence = 1.0 - np.clip(d1 / (d2 + 1e-6), 0.0, 1.0)
        else:
            confidence = np.ones((n_cells,), dtype=float)

        # 将聚类结果和 PCA 坐标写回 DataFrame
        df["cluster"] = labels.astype(int)
        df["confidence"] = confidence.astype(float)
        for i in range(scores.shape[1]):
            df[f"pca_{i+1}"] = scores[:, i]

        # 保存聚类结果 CSV
        out_dir = features.get("features_dir") or context.get("output_dir") or "outputs"
        out_csv = matrix_csv.replace(".csv", "_cluster.csv")
        try:
            df.to_csv(out_csv, index=False)
        except Exception as e:
            log_error(f"保存聚类结果 CSV 失败: {out_csv}, 错误: {e}")
            raise

        log_info(f"4.x 聚类分析完成：cells={n_cells}, k={k}, 输入={matrix_csv}, 输出={out_csv}")

        cluster_info = {
            "status": "ok",
            "k": int(k),
            "cells": int(n_cells),
            "labels_csv": out_csv,
            "explained_variance_ratio": explained_ratio.tolist(),
        }
        return {"cluster": cluster_info}