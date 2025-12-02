#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""对原始图像随机抽取少量 tile，执行窗口分割并对该 tile 做单 tile 特征提取。

用法示例：

    python sample_roi_features.py \
        --images-root /nfs5/yj/MIHC/dataset/KidneyCancer \
        --output-root /nfs5/zyh/MUF-Clust/outputs \
        --tiles-per-image 2

脚本逻辑：
- 遍历 images-root 下的每个原始图像（qptiff/tif/tiff 等），视为一个样本；
- 使用 tile_size 在整图上构建规则 tile 网格，从中随机抽取 N 个 tile（默认 2 个）；
- 对每个被抽中的 tile 坐标 (tile_x, tile_y)，调用 run_segmentation() 的窗口模式
  （only_tile_x/only_tile_y、only_roi_tiles=True），仅在该窗口内进行 ROI 分割；
- 随后对同一 tile 调用 run_features()，仅处理该 tile（only_tile_x/only_tile_y），
  并将 seg_out_dir 指向 output-root/<sample>/tile_<x>_<y>；
- 特征输出位于 output-root/<sample>/tile_<x>_<y>/features/ 下。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Optional, List, Tuple

try:
    import tifffile  # type: ignore
except Exception:
    tifffile = None  # type: ignore

# 确保可以从源码目录导入 muf_clust（src 布局）
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from muf_clust.config import DEFAULT_CANCER_TYPE, get_runtime_defaults
from muf_clust.api.pipeline_api import run_features, run_segmentation
from muf_clust.utils.logging import log_info, log_warn, log_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对原始图像随机抽取若干 tile，执行窗口分割并对该 tile 做特征提取")
    p.add_argument(
        "--images-root",
        required=True,
        help="原始图像根目录（包含 qptiff/tif/tiff）；脚本将对每张图随机抽取若干 tile 进行分割+特征提取",
    )
    p.add_argument(
        "--output-root",
        required=False,
        default=None,
        help="特征输出根目录（默认为 /nfs5/zyh/MUF-Clust/outputs）",
    )
    p.add_argument(
        "--cancer-type",
        required=False,
        default=DEFAULT_CANCER_TYPE,
        choices=["KidneyCancer", "BladderCancer"],
        help="通道配置类别",
    )
    p.add_argument(
        "--tiles-per-image",
        type=int,
        default=2,
        help="每张图随机抽取的 tile 数量",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于抽样 tile，可复现实验）",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="传给分割与特征提取的 num_workers（并行处理 tiles/像素）",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="分割与窗口读取的 tile 大小（像素），默认使用运行时配置中的 tile_size",
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


def _parse_tile_xy(filename: str) -> tuple[int, int]:
    """从文件名中解析倒数第二、第三段为 tile_x, tile_y（通用 "*_x_y.ext" 形式）。"""
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"无法从文件名解析 tile 坐标: {filename}")
    try:
        x = int(parts[-2])
        y = int(parts[-1])
    except Exception as e:
        raise ValueError(f"解析 tile 坐标失败: {filename}, 错误: {e}")
    return x, y


IMAGE_EXTS = {".qptiff", ".qptif", ".tif", ".tiff"}


def _iter_images(root: str) -> List[str]:
    items: List[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            _name, ext = os.path.splitext(fn)
            if ext.lower() in IMAGE_EXTS:
                items.append(os.path.join(dirpath, fn))
    items.sort()
    return items


def _get_image_shape(path: str, prefer_low_res: bool) -> Tuple[int, int]:
    if tifffile is None:
        raise RuntimeError("需要安装 tifffile 以读取图像尺寸")
    with tifffile.TiffFile(path) as tf:  # type: ignore
        s = tf.series[0]
        levels = getattr(s, "levels", [s])
        lvl_idx = (len(levels) - 1) if (prefer_low_res and len(levels) > 1) else 0
        lvl = levels[lvl_idx]
        axes = getattr(lvl, "axes", "")
        shape = lvl.shape
        axes_str = axes if isinstance(axes, str) and axes else ""
        if axes_str and ("Y" in axes_str and "X" in axes_str):
            y_idx = axes_str.find("Y")
            x_idx = axes_str.find("X")
            h = int(shape[y_idx])
            w = int(shape[x_idx])
        else:
            h = int(shape[-2])
            w = int(shape[-1])
        return h, w


def main() -> None:
    args = parse_args()

    random.seed(args.seed)

    defaults = get_runtime_defaults(args.cancer_type)
    images_root = os.path.abspath(args.images_root)
    if not os.path.isdir(images_root):
        log_error(f"原始图像根目录不存在或不是目录: {images_root}")
        raise SystemExit(1)

    output_root = args.output_root or "/nfs5/zyh/MUF-Clust/outputs"
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)

    prefer_low_res = not bool(args.high_res)
    tile_size = int(args.tile_size or defaults.tile_size or 1024)
    tiles_per_image = max(1, int(args.tiles_per_image))

    log_info(f"原始图像根目录: {images_root}")
    log_info(f"输出根目录: {output_root}")
    log_info(f"每张图随机抽取 tile 数量: {tiles_per_image}, tile_size={tile_size}, prefer_low_res={prefer_low_res}")

    images = _iter_images(images_root)
    if not images:
        log_warn("在原始图像根目录下未找到任何 qptiff/tif/tiff 文件，退出")
        return

    for idx, image_path in enumerate(images, 1):
        sample_name = os.path.splitext(os.path.basename(image_path))[0]
        log_info(f"[{idx}/{len(images)}] 处理图像: {image_path} (样本={sample_name})")

        try:
            h, w = _get_image_shape(image_path, prefer_low_res=prefer_low_res)
        except Exception as e:
            log_error(f"读取图像尺寸失败，跳过该图像: {image_path}, 错误: {e}")
            continue

        if h <= 0 or w <= 0:
            log_warn(f"图像尺寸异常，跳过: {image_path}, shape=({h}, {w})")
            continue

        xs = list(range(0, w, tile_size))
        ys = list(range(0, h, tile_size))
        if not xs or not ys:
            log_warn(f"无法基于 tile_size={tile_size} 在图像上构建 tile 网格，跳过: {image_path}")
            continue

        grid: List[Tuple[int, int]] = [(x, y) for y in ys for x in xs]
        k = min(tiles_per_image, len(grid))
        chosen = random.sample(grid, k)
        log_info(f"样本 {sample_name} 总 tile 数 {len(grid)}，随机选取 {k} 个: {chosen}")

        for j, (tx, ty) in enumerate(chosen, 1):
            log_info(
                f"  ({j}/{k}) 开始单 tile 分割+特征提取: 样本={sample_name}, "
                f"tile_x={tx}, tile_y={ty}, image_path={image_path}, tile_size={tile_size}"
            )

            try:
                seg_result = run_segmentation(
                    input_path=image_path,
                    output_dir=output_root,
                    options={
                        "cancer_type": args.cancer_type,
                        "prefer_low_res": prefer_low_res,
                        "tile_size": tile_size,
                        "use_cellpose": True,
                        "only_tile_x": tx,
                        "only_tile_y": ty,
                        "only_roi_tiles": True,
                        "num_workers": int(args.num_workers),
                    },
                )
            except Exception as e:
                log_error(
                    f"  ({j}/{k}) 窗口分割失败: 样本={sample_name}, tile_x={tx}, tile_y={ty}, "
                    f"image_path={image_path}, 错误: {e}"
                )
                continue

            seg = seg_result.get("segmentation", {}) or {}
            out_dir_seg = seg.get("out_dir") or os.path.join(output_root, "segmentation", sample_name)
            paths = seg.get("paths", {}) or {}
            roi_dir = paths.get("roi_tiles_dir") or os.path.join(out_dir_seg, "roi_tiles")

            if not os.path.isdir(roi_dir):
                log_warn(
                    f"  ({j}/{k}) 找不到 ROI tiles 目录，跳过特征提取: 样本={sample_name}, "
                    f"roi_dir={roi_dir}"
                )
                continue

            seg_out_dir = os.path.join(output_root, sample_name, f"tile_{tx}_{ty}")
            os.makedirs(seg_out_dir, exist_ok=True)

            log_info(
                f"  ({j}/{k}) 开始单 tile 特征提取: 样本={sample_name}, "
                f"tile_x={tx}, tile_y={ty}, image_path={image_path}, roi_dir={roi_dir}, seg_out_dir={seg_out_dir}"
            )

            try:
                _ = run_features(
                    image_path=image_path,
                    seg_out_dir=seg_out_dir,
                    roi_tiles_dir=roi_dir,
                    options={
                        "cancer_type": args.cancer_type,
                        "prefer_low_res": prefer_low_res,
                        "only_tile_x": tx,
                        "only_tile_y": ty,
                        "num_workers": int(args.num_workers),
                    },
                )
            except Exception as e:
                log_error(
                    f"  ({j}/{k}) 单 tile 特征提取失败: 样本={sample_name}, "
                    f"tile_x={tx}, tile_y={ty}, seg_out_dir={seg_out_dir}, 错误: {e}"
                )
                continue

            log_info(
                f"  ({j}/{k}) 完成单 tile 分割+特征提取: 样本={sample_name}, tile_x={tx}, tile_y={ty}, "
                f"features_dir={os.path.join(seg_out_dir, 'features')}"
            )

    log_info("全部图像的随机单 tile 分割+特征提取完成")


if __name__ == "__main__":
    main()
