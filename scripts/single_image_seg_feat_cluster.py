#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""针对单张图像执行完整流程：分割 → 特征提取 → CSV 聚类。

用法示例：

    uv run single_image_seg_feat_cluster.py \
        --image-path /nfs5/yj/MIHC/dataset/KidneyCancer/NO2/11/Scan1/11_Scan1.qptiff \
        --output-root /nfs5/zyh/MUF-Clust/single_image_output \
        --cancer-type KidneyCancer \
        --num-workers 16

脚本逻辑：
- 对指定的单张图像调用 run_segmentation() 做整图分割，生成 ROI tiles；
- 基于该 ROI tiles 目录调用 run_features() 做特征提取，写出 features_long.csv / features_matrix_mean.csv；
- 对该图的 features_matrix_mean.csv 调用 run_cluster_from_csv()，执行 Z-score → PCA → KMeans 聚类；
- 最终输出 *_cluster.csv，包含 cluster/置信度/PCA 坐标。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# 确保可以从源码目录导入 muf_clust（src 布局）
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from muf_clust.config import DEFAULT_CANCER_TYPE, get_runtime_defaults
from muf_clust.api.pipeline_api import run_segmentation, run_features, run_cluster_from_csv
from muf_clust.utils.logging import log_info, log_warn, log_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对单张图像执行分割→特征提取→聚类的完整流程")
    p.add_argument(
        "--image-path",
        required=True,
        help="单张原始图像路径（qptiff/tif/tiff 等）",
    )
    p.add_argument(
        "--output-root",
        required=False,
        default=None,
        help="输出根目录（默认为 /nfs5/zyh/MUF-Clust/outputs）",
    )
    p.add_argument(
        "--cancer-type",
        required=False,
        default=DEFAULT_CANCER_TYPE,
        choices=["KidneyCancer", "BladderCancer"],
        help="通道配置类别",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="分割与窗口读取的 tile 大小（像素），默认使用运行时配置中的 tile_size",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="传给分割与特征提取的 num_workers（内部多线程处理 tiles/像素）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于 CSV 聚类的 KMeans，可复现实验）",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="聚类数 K（默认按细胞数自动选择：<20→2-4，<100→6，其它→8）",
    )
    p.add_argument(
        "--high-res",
        action="store_true",
        default=True,
        help="使用高分辨率层读取通道（默认 True）",
    )
    p.add_argument(
        "--low-res",
        dest="high_res",
        action="store_false",
        help="改为使用低分辨率层读取通道",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    image_path = os.path.abspath(args.image_path)
    if not os.path.isfile(image_path):
        log_error(f"图像路径不存在或不是文件: {image_path}")
        raise SystemExit(1)

    defaults = get_runtime_defaults(args.cancer_type)

    output_root = args.output_root or "/nfs5/zyh/MUF-Clust/outputs"
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)

    prefer_low_res = not bool(args.high_res)
    tile_size = int(args.tile_size or defaults.tile_size or 1024)
    num_workers = max(1, int(args.num_workers))

    sample_name = os.path.splitext(os.path.basename(image_path))[0]

    log_info(f"单图完整流程: 图像={image_path}")
    log_info(f"输出根目录: {output_root}")
    log_info(f"cancer_type={args.cancer_type}, tile_size={tile_size}, prefer_low_res={prefer_low_res}, num_workers={num_workers}")

    # 1) 分割：整图分割，生成 ROI tiles
    try:
        log_info("[1/3] 开始分割 (run_segmentation)")
        seg_result = run_segmentation(
            input_path=image_path,
            output_dir=output_root,
            options={
                "cancer_type": args.cancer_type,
                "prefer_low_res": prefer_low_res,
                "tile_size": tile_size,
                "use_cellpose": True,
                "num_workers": num_workers,
            },
        )
    except Exception as e:
        log_error(f"分割阶段失败: 图像={image_path}, 错误: {e}")
        raise SystemExit(1)

    seg = seg_result.get("segmentation", {}) or {}
    seg_out_dir = seg.get("out_dir") or os.path.join(output_root, "segmentation", sample_name)
    paths = seg.get("paths", {}) or {}
    roi_dir = paths.get("roi_tiles_dir") or os.path.join(seg_out_dir, "roi_tiles")

    if not os.path.isdir(roi_dir):
        log_error(f"找不到 ROI tiles 目录，无法继续特征提取: roi_dir={roi_dir}")
        raise SystemExit(1)

    log_info(f"分割完成: seg_out_dir={seg_out_dir}, roi_dir={roi_dir}, cells={seg.get('count')}, tiles={seg.get('tiles')}")

    # 2) 特征提取：基于 ROI tiles 计算特征
    try:
        log_info("[2/3] 开始特征提取 (run_features)")
        feat_result = run_features(
            image_path=image_path,
            seg_out_dir=seg_out_dir,
            roi_tiles_dir=roi_dir,
            options={
                "cancer_type": args.cancer_type,
                "prefer_low_res": prefer_low_res,
                "num_workers": num_workers,
            },
        )
    except Exception as e:
        log_error(
            f"特征提取阶段失败: 图像={image_path}, seg_out_dir={seg_out_dir}, roi_dir={roi_dir}, 错误: {e}"
        )
        raise SystemExit(1)

    feats = feat_result.get("features", {}) or {}
    features_dir = feats.get("features_dir") or os.path.join(seg_out_dir, "features")
    matrix_csv = feats.get("features_matrix_csv") or os.path.join(features_dir, "features_matrix_mean.csv")

    if not os.path.isfile(matrix_csv):
        log_error(f"未找到特征矩阵 CSV，无法进行聚类: matrix_csv={matrix_csv}")
        raise SystemExit(1)

    log_info(
        f"特征提取完成: features_dir={features_dir}, "
        f"matrix_csv={matrix_csv}, cells={feats.get('cells')}"
    )

    # 3) 聚类：对该图的特征矩阵执行 Z-score → PCA → KMeans
    try:
        log_info("[3/3] 开始基于 CSV 的聚类 (run_cluster_from_csv)")
        cluster_result = run_cluster_from_csv(
            matrix_csv=matrix_csv,
            seed=int(args.seed),
            k=(int(args.k) if args.k is not None else None),
        )
    except Exception as e:
        log_error(f"聚类阶段失败: matrix_csv={matrix_csv}, 错误: {e}")
        raise SystemExit(1)

    cluster_info = cluster_result.get("cluster", {}) or {}
    out_csv = cluster_info.get("labels_csv") or matrix_csv.replace(".csv", "_cluster.csv")

    log_info(
        f"单图完整流程完成: 图像={image_path}, "
        f"cells={cluster_info.get('cells')}, k={cluster_info.get('k')}, "
        f"聚类结果 CSV={out_csv}"
    )


if __name__ == "__main__":
    main()
