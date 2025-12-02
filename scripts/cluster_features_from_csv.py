#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""递归收集 features_matrix_mean.csv 并统一聚类的独立脚本。

用法示例：

    python cluster_features_from_csv.py \
        --root /nfs5/zyh/MUF-Clust/outputs \
        --seed 42

脚本逻辑：
- 从 --root 指定的目录开始，递归查找所有名为 "features_matrix_mean.csv" 的文件；
- 读取每个 CSV，并为其中的每一行细胞记录添加一个 `__source_csv` 字段标记来源路径；
- 将所有样本的细胞特征拼接成一个大的 DataFrame；
- 对特征列执行 Z-score 标准化 → PCA 降维 → KMeans 聚类；
- 将 cluster/置信度/PCA 坐标写入合并后的表，并保存为一个新的 CSV 文件；
- 默认输出路径为 <root>/combined_features_cluster.csv，可通过 --output 覆盖。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

# 确保可以从源码目录导入 muf_clust（src 布局）
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from muf_clust.utils.logging import log_info, log_warn, log_error
from muf_clust.core.steps.cluster import _zscore_matrix, _pca, _kmeans


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="递归收集 features_matrix_mean.csv 并统一聚类")
    p.add_argument(
        "--root",
        required=True,
        help="递归搜索 features_matrix_mean.csv 的根目录（通常为 outputs 根目录）",
    )
    p.add_argument(
        "--output",
        required=False,
        default=None,
        help="合并聚类结果输出 CSV 路径（默认 <root>/combined_features_cluster.csv）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于 KMeans 聚类，可复现实验）",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="聚类数 K（默认按细胞总数自动选择：<20→2-4，<100→6，其它→8）",
    )
    return p.parse_args()


def _find_feature_csvs(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn == "features_matrix_mean.csv":
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def main() -> None:
    args = parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        log_error(f"根目录不存在或不是目录: {root}")
        raise SystemExit(1)

    # 处理输出路径：
    # - 若未指定 --output，则默认写入 <root>/combined_features_cluster.csv；
    # - 若 --output 是已存在的目录，则在该目录下写入 combined_features_cluster.csv；
    # - 否则将 --output 视为完整文件路径。
    if args.output:
        output_arg = os.path.abspath(args.output)
        if os.path.isdir(output_arg):
            out_csv = os.path.join(output_arg, "combined_features_cluster.csv")
        else:
            out_csv = output_arg
    else:
        out_csv = os.path.join(root, "combined_features_cluster.csv")

    log_info(f"递归搜索 features_matrix_mean.csv：root={root}")
    csv_paths = _find_feature_csvs(root)
    if not csv_paths:
        log_warn("在指定根目录下未找到任何 features_matrix_mean.csv 文件，退出")
        raise SystemExit(0)

    log_info(f"共找到 {len(csv_paths)} 个 features_matrix_mean.csv 文件，将合并后统一聚类")

    dfs = []
    total_cells = 0
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            log_warn(f"读取 CSV 失败，跳过: {path}, 错误: {e}")
            continue
        if df.empty or df.shape[0] == 0:
            log_warn(f"CSV 为空，跳过: {path}")
            continue
        n = df.shape[0]
        df["__source_csv"] = path
        dfs.append(df)
        total_cells += n
        log_info(f"  载入 {path}，细胞数={n}，累计细胞数={total_cells}")

    if not dfs:
        log_warn("未成功载入任何非空特征矩阵，退出")
        raise SystemExit(0)

    big = pd.concat(dfs, ignore_index=True)
    n_cells = big.shape[0]
    log_info(f"合并后总细胞数={n_cells}，开始执行 Z-score → PCA → KMeans 聚类")

    # 确定非特征列：索引/坐标等不参与聚类
    non_feature_cols = {
        "cell_id",
        "tile_x",
        "tile_y",
        "label",
        "x",
        "y",
        "centroid_x",
        "centroid_y",
        "local_x",
        "local_y",
        "cluster",
        "confidence",
        "__source_csv",
    }
    feature_cols = [c for c in big.columns if c not in non_feature_cols]
    if not feature_cols:
        log_warn("合并后的表中未找到可用于聚类的特征列，退出")
        raise SystemExit(0)

    x = big[feature_cols].to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_norm = _zscore_matrix(x)

    n_components = min(20, x_norm.shape[0], x_norm.shape[1])
    scores, explained_ratio = _pca(x_norm, n_components=n_components)

    if args.k is None:
        if n_cells < 20:
            k_val = max(2, min(4, n_cells))
        elif n_cells < 100:
            k_val = 6
        else:
            k_val = 8
    else:
        k_val = int(max(2, min(args.k, n_cells)))

    log_info(f"开始 KMeans 聚类：cells={n_cells}, k={k_val}, seed={args.seed}")
    labels, centers = _kmeans(scores, k=k_val, n_init=5, max_iter=100, seed=int(args.seed))

    if centers.size > 0:
        dists = np.linalg.norm(scores[:, None, :] - centers[None, :, :], axis=2)
        sorted_d = np.sort(dists, axis=1)
        d1 = sorted_d[:, 0]
        d2 = sorted_d[:, 1] if sorted_d.shape[1] > 1 else (sorted_d[:, 0] + 1e-6)
        confidence = 1.0 - np.clip(d1 / (d2 + 1e-6), 0.0, 1.0)
    else:
        confidence = np.ones((n_cells,), dtype=float)

    big["cluster"] = labels.astype(int)
    big["confidence"] = confidence.astype(float)
    for i in range(scores.shape[1]):
        big[f"pca_{i+1}"] = scores[:, i]

    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    big.to_csv(out_csv, index=False)
    log_info(f"统一聚类完成：cells={n_cells}, k={k_val}, 输出文件={out_csv}")


if __name__ == "__main__":
    main()
