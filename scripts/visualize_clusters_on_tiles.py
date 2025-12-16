#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""根据 combined_features_cluster.csv 将聚类后的细胞点绘制回 raw tile 图像上。

用法示例：

    python visualize_clusters_on_tiles.py \
        --cluster-csv /nfs5/zyh/MUF-Clust/cluster_outputs/combined_features_cluster.csv \
        --roi-root /nfs5/zyh/MUF-Clust/outputs/segmentation \
        --output-dir /nfs5/zyh/MUF-Clust/cluster_outputs/cluster_tiles

脚本逻辑：
- 从 --cluster-csv 读取统一聚类后的细胞特征表（包含 tile_x/tile_y/local_x/local_y/cluster/__source_csv 等列）；
- 通过 __source_csv 推断 sample 名称（与 segmentation 下的子目录对应，如 102_Scan1 等）；
- 按 sample、tile_x、tile_y 分组，构造 raw tile 图像路径：
    <roi-root>/<sample>/raw_tiles/raw_<tile_x>_<tile_y>.png
- 在每个 raw tile 上用散点图绘制 local_x/local_y 坐标的细胞点，不同 cluster 使用不同颜色；
- 将渲染结果保存为 PNG 到 --output-dir/<sample>/cluster_<tile_x>_<tile_y>.png。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd

# 确保可以从源码目录导入 muf_clust（src 布局）
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from muf_clust.utils.logging import log_info, log_warn, log_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="根据聚类结果在 raw tile 上绘制细胞散点图")
    p.add_argument(
        "--cluster-csv",
        required=False,
        default="/nfs5/zyh/MUF-Clust/cluster_outputs/combined_features_cluster.csv",
        help="统一聚类结果 CSV 路径（默认为项目下 cluster_outputs/combined_features_cluster.csv）",
    )
    p.add_argument(
        "--roi-root",
        required=False,
        default="/nfs5/zyh/MUF-Clust/outputs/segmentation",
        help="segmentation 结果根目录（其下应包含 <sample>/raw_tiles/raw_<x>_<y>.png）",
    )
    p.add_argument(
        "--output-dir",
        required=False,
        default="/nfs5/zyh/MUF-Clust/cluster_outputs/cluster_tiles",
        help="聚类可视化输出根目录（每个 sample 一个子目录）",
    )
    p.add_argument(
        "--max-cells-per-tile",
        type=int,
        default=0,
        help="单个 tile 最多绘制的细胞数（0 或负数表示不过滤，全部绘制）",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="输出 PNG 的 DPI（默认 150）",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="如目标 PNG 已存在，是否覆盖重写（默认不覆盖，直接跳过）",
    )
    return p.parse_args()


def extract_sample(path: str) -> str:
    """从 __source_csv 路径中推断 sample 名称。

    假设路径类似：
        /.../outputs/<sample>/tile_x_y/features/features_matrix_mean.csv
    或至少包含 `outputs/<sample>/...` 片段。
    """

    parts = path.split(os.sep)
    if "outputs" in parts:
        idx = parts.index("outputs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # 回退：使用上上级目录名
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


def _ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _build_roi_path(roi_root: str, sample: str, tile_x: int, tile_y: int) -> str:
    """构造 raw tile PNG 的路径。"""
    return os.path.join(roi_root, sample, "raw_tiles", f"raw_{tile_x}_{tile_y}.png")


def _load_image(path: str) -> np.ndarray:
    try:
        img = plt.imread(path)
    except Exception as e:
        raise RuntimeError(f"读取图像失败: {path}, 错误: {e}")
    if img.ndim == 2:  # 灰度 → 伪 RGB
        img = np.stack([img] * 3, axis=-1)
    return img


def main() -> None:
    args = parse_args()

    cluster_csv = os.path.abspath(args.cluster_csv)
    roi_root = os.path.abspath(args.roi_root)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isfile(cluster_csv):
        log_error(f"聚类结果 CSV 不存在: {cluster_csv}")
        raise SystemExit(1)
    if not os.path.isdir(roi_root):
        log_error(f"segmentation/raw_tiles 根目录不存在或不是目录: {roi_root}")
        raise SystemExit(1)

    _ensure_dir(output_dir)

    log_info(f"读取聚类结果 CSV: {cluster_csv}")
    df = pd.read_csv(cluster_csv)
    if df.empty:
        log_warn("聚类结果表为空，退出")
        raise SystemExit(0)

    required_cols = {"tile_x", "tile_y", "local_x", "local_y", "cluster", "__source_csv"}
    missing = sorted(list(required_cols - set(df.columns)))
    if missing:
        log_error(f"聚类结果缺少必要列: {missing}")
        raise SystemExit(1)

    log_info(f"总细胞数: {len(df)}")

    # 推断 sample 名称
    df["sample"] = df["__source_csv"].astype(str).apply(extract_sample)

    # 推断全局聚类类别数，用于构造离散 colormap
    try:
        n_clusters = int(df["cluster"].max()) + 1
    except Exception:
        n_clusters = int(df["cluster"].nunique())
    if n_clusters <= 0:
        log_error("无法推断聚类类别数（cluster 列为空或异常）")
        raise SystemExit(1)

    cmap = plt.cm.get_cmap("tab20", n_clusters)
    boundaries = np.arange(-0.5, n_clusters + 0.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    max_cells = int(args.max_cells_per_tile or 0)
    dpi = int(args.dpi or 150)

    # 按 sample、tile_x、tile_y 分组
    group_cols = ["sample", "tile_x", "tile_y"]
    grouped = df.groupby(group_cols)

    total_tiles = len(grouped)
    log_info(f"需要绘制的 tile 数量: {total_tiles}")

    n_done = 0
    n_skipped_no_image = 0
    n_skipped_existing = 0

    for (sample, tx, ty), g in grouped:
        try:
            tx_int = int(tx)
            ty_int = int(ty)
        except Exception:
            log_warn(f"无法解析 tile 坐标为整数，跳过: sample={sample}, tile_x={tx}, tile_y={ty}")
            continue

        roi_path = _build_roi_path(roi_root, sample, tx_int, ty_int)
        if not os.path.isfile(roi_path):
            log_warn(f"找不到 raw tile 图像，跳过该 tile: sample={sample}, tile_x={tx_int}, tile_y={ty_int}, roi={roi_path}")
            n_skipped_no_image += 1
            continue

        out_sample_dir = os.path.join(output_dir, sample)
        _ensure_dir(out_sample_dir)
        out_png = os.path.join(out_sample_dir, f"cluster_{tx_int}_{ty_int}.png")

        if os.path.isfile(out_png) and not args.overwrite:
            log_info(f"输出已存在且未指定 --overwrite，跳过: {out_png}")
            n_skipped_existing += 1
            continue

        cells_df = g.copy()
        if max_cells > 0 and len(cells_df) > max_cells:
            cells_df = cells_df.sample(n=max_cells, random_state=42)

        try:
            img = _load_image(roi_path)
        except Exception as e:
            log_warn(str(e))
            n_skipped_no_image += 1
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)

        clusters = cells_df["cluster"].to_numpy(dtype=int)
        sc = ax.scatter(
            cells_df["local_x"].to_numpy(dtype=float),
            cells_df["local_y"].to_numpy(dtype=float),
            c=clusters,
            s=5,
            cmap=cmap,
            norm=norm,
            alpha=0.7,
        )

        ax.set_title(f"{sample} tile {tx_int}_{ty_int} (n={len(cells_df)})")
        ax.set_xticks([])
        ax.set_yticks([])

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("cluster")
        cbar.set_ticks(range(n_clusters))

        fig.tight_layout()
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        log_info(f"完成绘制: sample={sample}, tile_x={tx_int}, tile_y={ty_int}, cells={len(cells_df)}, 输出={out_png}")
        n_done += 1

    log_info(
        f"全部完成：总 tile={total_tiles}, 成功绘制={n_done}, "
        f"缺少 ROI 图像跳过={n_skipped_no_image}, 已存在输出跳过={n_skipped_existing}"
    )


if __name__ == "__main__":
    main()
