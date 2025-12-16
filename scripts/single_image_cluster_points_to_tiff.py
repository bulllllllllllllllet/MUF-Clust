#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""将单图聚类 CSV 中的点坐标还原为整幅 tiff 标签图。

用法示例：

    uv run single_image_cluster_points_to_tiff.py \
        --image-path /path/to/sample.qptiff \
        --output-root /nfs5/zyh/MUF-Clust/outputs

或显式指定聚类结果 CSV：

    uv run single_image_cluster_points_to_tiff.py \
        --image-path /path/to/sample.qptiff \
        --cluster-csv /path/to/features_matrix_mean_cluster.csv

约定：
- 输入 CSV 为 single_image_seg_feat_cluster.py 最终输出的 *_cluster.csv；
- CSV 中需要包含列：tile_x, tile_y, local_x, local_y, cluster；
- 全局像素坐标按以下方式计算：
    global_x = tile_x + local_x
    global_y = tile_y + local_y
- 输出 tiff 与所选分辨率层的图像尺寸完全一致：
    * 默认使用高分辨率层（与 --high-res 行为一致）；
    * 若指定 --low-res，则使用最低分辨率层尺寸。
- 输出像素值：0 为背景，cluster 标签写入为 (cluster + 1)，方便与背景区分。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import tifffile  # type: ignore
except Exception:  # pragma: no cover
    tifffile = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="将单图聚类 CSV 还原为整幅 tiff 标签图")
    p.add_argument(
        "--image-path",
        required=True,
        help="原始图像路径（qptiff/tif/tiff 等）",
    )
    p.add_argument(
        "--cluster-csv",
        required=False,
        default=None,
        help="聚类结果 CSV 路径（默认为根据 image-path 和 output-root 自动推断 *_cluster.csv）",
    )
    p.add_argument(
        "--output-root",
        required=False,
        default="/nfs5/zyh/MUF-Clust/outputs",
        help="运行 single_image_seg_feat_cluster.py 时使用的输出根目录（用于自动推断 *_cluster.csv 路径）",
    )
    p.add_argument(
        "--out-tiff",
        required=False,
        default=None,
        help="输出 tiff 文件路径（默认与 *_cluster.csv 同目录，后缀改为 _cluster_map.tiff）",
    )
    p.add_argument(
        "--high-res",
        action="store_true",
        default=True,
        help="使用最高分辨率层尺寸（默认 True）",
    )
    p.add_argument(
        "--low-res",
        dest="high_res",
        action="store_false",
        help="改为使用最低分辨率层尺寸",
    )
    return p.parse_args()


def _get_image_shape(path: str, prefer_low_res: bool = False) -> Tuple[int, int]:
    """读取 qptiff/tiff 的某一 pyramid 层尺寸 (H, W)。

    逻辑与分割模块中 _get_dapi_shape 一致，只是无需选择通道。
    """
    if tifffile is None:
        raise RuntimeError("需要安装 tifffile 以读取 QPTIFF/TIFF 图像")

    with tifffile.TiffFile(path) as tf:  # type: ignore
        s = tf.series[0]
        levels = getattr(s, "levels", [s])
        lvl_idx = (len(levels) - 1) if (prefer_low_res and len(levels) > 1) else 0
        lvl = levels[lvl_idx]
        axes = getattr(lvl, "axes", "")
        shape = lvl.shape
        axes_str = axes if isinstance(axes, str) else ""
        if axes_str and ("Y" in axes_str and "X" in axes_str):
            y_idx = axes_str.find("Y")
            x_idx = axes_str.find("X")
            h = int(shape[y_idx])
            w = int(shape[x_idx])
        else:
            h = int(shape[-2])
            w = int(shape[-1])
        return h, w


def _infer_cluster_csv(image_path: str, output_root: str) -> str:
    image_path = os.path.abspath(image_path)
    output_root = os.path.abspath(output_root)
    sample_name = os.path.splitext(os.path.basename(image_path))[0]

    seg_out_dir = os.path.join(output_root, "segmentation", sample_name)
    features_dir = os.path.join(seg_out_dir, "features")
    matrix_csv = os.path.join(features_dir, "features_matrix_mean.csv")
    cluster_csv = matrix_csv.replace(".csv", "_cluster.csv")
    return cluster_csv


def _load_cluster_points(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到聚类结果 CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"tile_x", "tile_y", "local_x", "local_y", "cluster"}
    missing = required_cols.difference({c.lower() for c in df.columns})
    if missing:
        raise ValueError(
            f"CSV 中缺少必要列（大小写不敏感）：{sorted(missing)}，"
            f"实际列：{list(df.columns)}"
        )

    # 统一取列名（支持大小写差异）
    def _col(name: str) -> str:
        for c in df.columns:
            if c.lower() == name:
                return c
        raise KeyError(name)

    tx_col = _col("tile_x")
    ty_col = _col("tile_y")
    lx_col = _col("local_x")
    ly_col = _col("local_y")
    cl_col = _col("cluster")

    return df[[tx_col, ty_col, lx_col, ly_col, cl_col]].rename(
        columns={
            tx_col: "tile_x",
            ty_col: "tile_y",
            lx_col: "local_x",
            ly_col: "local_y",
            cl_col: "cluster",
        }
    )


def _points_to_full_label_image(
    df: pd.DataFrame,
    height: int,
    width: int,
) -> np.ndarray:
    """将聚类点映射到整幅图像坐标，生成 H×W 的 uint16 标签图。

    背景值为 0，每个点位置写入 (cluster + 1)。
    """
    img = np.zeros((height, width), dtype=np.uint8)

    # 向量化计算全局坐标
    gx = (df["tile_x"].to_numpy(dtype=float) + df["local_x"].to_numpy(dtype=float))
    gy = (df["tile_y"].to_numpy(dtype=float) + df["local_y"].to_numpy(dtype=float))
    cl = df["cluster"].to_numpy(dtype=int)

    # 四舍五入到最近像素，并裁剪到合法范围
    gx = np.rint(gx).astype(np.int64)
    gy = np.rint(gy).astype(np.int64)
    gx = np.clip(gx, 0, width - 1)
    gy = np.clip(gy, 0, height - 1)

    vals = _cluster_to_vals(cl)

    # 在每个点周围绘制半径约 1.5 像素的小圆（3x3 邻域内的圆形掩膜）
    radius = 1.5
    r2 = radius * radius
    offsets = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx * dx + dy * dy <= r2:
                offsets.append((dy, dx))

    # 对每个偏移量进行向量化写入
    for dy, dx in offsets:
        yy = gy + dy
        xx = gx + dx
        yy = np.clip(yy, 0, height - 1)
        xx = np.clip(xx, 0, width - 1)
        img[yy, xx] = vals

    return img


def _cluster_to_vals(
    clusters: np.ndarray,
) -> np.ndarray:
    if clusters.size == 0:
        return clusters.astype(np.uint8)

    uniq, inv = np.unique(clusters, return_inverse=True)
    k = int(uniq.size)
    if k > 255:
        raise ValueError(f"类别数过多，无法编码到 0..255：K={k} > 255")
    return (inv.astype(np.uint16) + 1).astype(np.uint8)


def _build_label_pyramid(
    df: pd.DataFrame,
    height: int,
    width: int,
    max_levels: int = 4,
    downscale: int = 2,
    min_size: int = 512,
) -> list[np.ndarray]:
    gx = (df["tile_x"].to_numpy(dtype=float) + df["local_x"].to_numpy(dtype=float))
    gy = (df["tile_y"].to_numpy(dtype=float) + df["local_y"].to_numpy(dtype=float))
    cl = df["cluster"].to_numpy(dtype=int)

    gx = np.rint(gx).astype(np.int64)
    gy = np.rint(gy).astype(np.int64)
    gx = np.clip(gx, 0, width - 1)
    gy = np.clip(gy, 0, height - 1)
    vals = _cluster_to_vals(cl)

    levels: list[np.ndarray] = []
    levels.append(
        _points_to_full_label_image(
            df,
            height=height,
            width=width,
        )
    )

    it = range(1, max_levels)
    if tqdm is not None:
        it = tqdm(it, total=max_levels - 1, desc="build pyramid", unit="level")  # type: ignore

    scale = 1
    for _ in it:
        scale *= downscale
        h = height // scale
        w = width // scale
        if h < min_size or w < min_size:
            break
        small = np.zeros((h, w), dtype=np.uint8)
        sy = gy // scale
        sx = gx // scale
        sy = np.clip(sy, 0, h - 1)
        sx = np.clip(sx, 0, w - 1)
        small[sy, sx] = vals
        levels.append(small)

    return levels


def _save_pyramidal_wsi_tiff(path: str, levels: list[np.ndarray]) -> None:
    if tifffile is None:
        raise RuntimeError("需要安装 tifffile 以写出 WSI TIFF")

    if not levels:
        raise ValueError("写出 WSI TIFF 需要至少一层图像")

    base = levels[0]
    tile_size = 512
    options = dict(
        tile=(tile_size, tile_size),
        compression="deflate",
        photometric="minisblack",
        metadata=None,
    )

    subifds = len(levels) - 1 if len(levels) > 1 else None
    with tifffile.TiffWriter(path) as tif:  # type: ignore
        tif.write(
            base,
            subifds=subifds,
            dtype=base.dtype,
            **options,
        )
        it = levels[1:]
        if tqdm is not None:
            it = tqdm(it, total=len(levels) - 1, desc="write pyramid", unit="level")  # type: ignore
        for lvl in it:
            tif.write(
                lvl,
                subfiletype=1,
                dtype=lvl.dtype,
                **options,
            )


def main() -> None:
    args = _parse_args()

    image_path = os.path.abspath(args.image_path)
    if not os.path.isfile(image_path):
        print(f"[错误] 图像路径不存在或不是文件: {image_path}", file=sys.stderr)
        raise SystemExit(1)

    # 解析/推断聚类 CSV 路径
    if args.cluster_csv:
        cluster_csv = os.path.abspath(args.cluster_csv)
    else:
        cluster_csv = _infer_cluster_csv(image_path, args.output_root)

    if not os.path.isfile(cluster_csv):
        print(
            f"[错误] 未找到聚类结果 CSV: {cluster_csv}\n"
            f"请检查：\n"
            f"  1) 是否已运行 single_image_seg_feat_cluster.py;\n"
            f"  2) --output-root 是否与当时运行时保持一致;\n"
            f"  3) 或通过 --cluster-csv 显式指定 *_cluster.csv。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # 输出 tiff 路径
    if args.out_tiff:
        out_tiff = os.path.abspath(args.out_tiff)
    else:
        base, _ = os.path.splitext(cluster_csv)
        out_tiff = base + "_cluster_map.tiff"

    os.makedirs(os.path.dirname(out_tiff), exist_ok=True)

    # 读取图像尺寸（与分割/特征提取保持相同的分辨率层选择逻辑）
    prefer_low_res = not bool(args.high_res)
    try:
        h, w = _get_image_shape(image_path, prefer_low_res=prefer_low_res)
    except Exception as e:  # pragma: no cover
        print(f"[错误] 读取图像尺寸失败: {e}", file=sys.stderr)
        raise SystemExit(1)

    # 读取聚类点
    try:
        df = _load_cluster_points(cluster_csv)
    except Exception as e:
        print(f"[错误] 读取聚类 CSV 失败: {e}", file=sys.stderr)
        raise SystemExit(1)

    if df.empty:
        print(f"[警告] 聚类 CSV 为空，没有可写入的点: {cluster_csv}", file=sys.stderr)
        raise SystemExit(0)

    # 写出 tiff
    if tifffile is None:  # pragma: no cover
        print("[错误] 缺少 tifffile 依赖，无法写出 tiff 文件", file=sys.stderr)
        raise SystemExit(1)

    try:
        print("[信息] 正在构建金字塔...", file=sys.stderr)
        levels = _build_label_pyramid(
            df,
            height=h,
            width=w,
        )
        print("[信息] 正在写出 WSI TIFF...", file=sys.stderr)
        _save_pyramidal_wsi_tiff(out_tiff, levels)
    except Exception as e:  # pragma: no cover
        print(f"[错误] 保存 tiff 失败: {out_tiff}, 错误: {e}", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"完成：\n"
        f"  图像: {image_path}\n"
        f"  聚类 CSV: {cluster_csv}\n"
        f"  输出标签 tiff: {out_tiff}\n"
        f"  尺寸: {h}x{w}"
    )


if __name__ == "__main__":
    main()
