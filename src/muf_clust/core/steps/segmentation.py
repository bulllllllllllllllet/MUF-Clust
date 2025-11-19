"""分割步骤实现（3.1/3.2）"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from ...utils.logging import log_info, log_error
from ...utils.paths import ensure_dir
from ...config import get_config, DEFAULT_CANCER_TYPE

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from skimage import measure  # type: ignore
except Exception:
    measure = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # type: ignore

try:
    import tifffile  # type: ignore
except Exception:
    tifffile = None  # type: ignore

try:
    from cellpose import models  # type: ignore
except Exception:
    models = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

import csv
from .preprocess import read_qptiff_stack, list_images, tile_iter

# 中文说明：
# - 本文件实现分割步骤：Cellpose（默认）。
# - 依赖项均按“可选导入”处理，缺失库时会提供中文错误提示并终止。


#  将输入图像线性缩放到 0–255 的 8bit，用于可视化与存盘
def _to_uint8(x: np.ndarray) -> np.ndarray:
    #  将任意浮点图归一化到 8bit，便于可视化与 OpenCV 处理
    m = float(np.min(x))
    M = float(np.max(x))
    if M <= m:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - m) / (M - m)
    return (y * 255.0).astype(np.uint8)


#  读取 QPTIFF 的 DAPI 通道（可选低分辨率层），返回二维 float32 图像
def _read_dapi_full(path: str, dapi_index0: int, prefer_low_res: bool = True) -> np.ndarray:
    #  从 QPTIFF 文件读取指定的 DAPI 通道，返回二维浮点图；支持多层级低分辨率选择
    if tifffile is None:
        #  缺少 tifffile 依赖时直接报错
        raise RuntimeError("需要安装 tifffile 以读取 QPTIFF")
    with tifffile.TiffFile(path) as tf:  # type: ignore
        s = tf.series[0]
        #  获取多层级信息（如金字塔各分辨率层），默认优先选择低分辨率层以加速
        levels = getattr(s, "levels", [s])
        lvl_idx = (len(levels) - 1) if (prefer_low_res and len(levels) > 1) else 0
        lvl = levels[lvl_idx]
        #  读取该层的轴标记与数据数组
        axes = getattr(lvl, "axes", "")
        arr = lvl.asarray()
        axes_str = axes if isinstance(axes, str) else ""
        #  确定通道轴字符（优先 C；有些文件用 S 代表 Sample/Channel）
        ch_char = "C" if (axes_str and "C" in axes_str) else ("S" if (axes_str and "S" in axes_str) else None)
        ch_idx = (axes_str.find(ch_char) if ch_char else None)
        #  若存在通道轴且通道数量>1，则按 dapi_index0 选择该通道
        if ch_idx is not None and arr.shape[ch_idx] > 1:
            sel = [slice(None)] * arr.ndim
            sel[ch_idx] = slice(dapi_index0, dapi_index0 + 1)
            sub = np.asarray(arr[tuple(sel)])
            sub = np.squeeze(sub)
        elif arr.ndim == 3:
            #  三维数组但未标注通道轴时，按最后一维选择
            sub = np.asarray(arr[..., dapi_index0])
        else:
            #  否则直接使用该层数据
            sub = np.asarray(arr)
        #  确保输出为二维数组
        if sub.ndim != 2:
            sub = np.squeeze(sub)
        #  统一为 float32 便于后续计算
        return sub.astype(np.float32)


 


@dataclass
class SegmentationStep:
    name: str = "segmentation"

    #  执行分割流程：解析输入与配置，运行 Cellpose，保存 tiles/掩码，并汇总核质心
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        image_path, out_dir, dapi, cp_model = _prepare_and_load(context)
        mask_dir, raw_dir = _prepare_output_dirs(out_dir)
        stats = _segment_tiles_and_save(dapi, cp_model, context, raw_dir, mask_dir)
        cent_path = _write_centroids_csv(out_dir, stats["centroids"]) if stats["centroids"] else os.path.join(out_dir, "nuclei_centroids.csv")
        log_info(f"分割完成，核数量：{stats['count']}，tiles={stats['tiles']}")
        return {
            "segmentation": {
                "status": "ok",
                "out_dir": out_dir,
                "count": stats["count"],
                "tiles": stats["tiles"],
                "paths": {
                    "raw_tiles_dir": raw_dir,
                    "mask_tiles_dir": mask_dir,
                    "centroids_csv": cent_path,
                },
            }
        }


#  准备输入路径与输出目录；选择 DAPI 通道；初始化 Cellpose 模型
def _prepare_and_load(context: Dict[str, Any]):
    image_path = context.get("image_path") or context.get("input_path")
    if image_path and os.path.isdir(image_path):
        imgs = list_images(image_path)
        image_path = imgs[0] if imgs else None
    if not image_path or not os.path.isfile(image_path):
        log_error("未找到可分割的图像路径")
        raise RuntimeError("缺少图像路径")
    output_root = context.get("output_dir", "outputs")
    sample = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(output_root, "segmentation", sample)
    ensure_dir(out_dir)
    log_info(f"开始分割：{image_path}")
    cancer_type = context.get("cancer_type", DEFAULT_CANCER_TYPE)
    cfgs = get_config(cancer_type) or []
    prefer_lr = bool(context.get("prefer_low_res", True))
    use_cellpose = bool(context.get("use_cellpose", True))
    log_info(f"准备读取 DAPI 通道（prefer_low_res={prefer_lr}）")
    import time
    t0 = time.time()
    try:
        dapi_idx = next(((c.index - 1) for c in cfgs if c.is_dapi), 0)
        dapi = _read_dapi_full(image_path, dapi_idx, prefer_low_res=prefer_lr)
    except Exception:
        stack = read_qptiff_stack(image_path, prefer_low_res=prefer_lr)
        C = stack.shape[0]
        dapi_idx = next(((c.index - 1) for c in cfgs if c.is_dapi and (c.index - 1) < C), 0)
        dapi = stack[dapi_idx]
    log_info(f"DAPI 读取完成，尺寸 {dapi.shape[0]}x{dapi.shape[1]}，耗时 {time.time() - t0:.2f}s")
    log_info("使用原始 DAPI 进行分割（解耦预处理与分割）")
    if not use_cellpose:
        log_error("未启用 Cellpose，且已移除传统分割路径")
        raise RuntimeError("未启用 Cellpose")
    if models is None:
        log_error("未安装 cellpose，无法使用 Cellpose 进行分割")
        raise RuntimeError("需要安装 cellpose 库")
    try:
        gpu_ok = bool(models.use_gpu())  # type: ignore
    except Exception:
        gpu_ok = False
    if bool(context.get("cellpose_gpu", True)):
        if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():  # type: ignore
            log_error("未检测到可用CUDA/GPU，请检查 PyTorch CUDA 安装与可见设备")
            raise RuntimeError("GPU 不可用：请配置 CUDA 并确保设备可见")
        gpu_ok = True
    model_name = str(context.get("cellpose_model", "nuclei"))
    if hasattr(models, "CellposeModel"):
        cp_model = models.CellposeModel(gpu=gpu_ok, pretrained_model=model_name)  # type: ignore
    elif hasattr(models, "Cellpose"):
        cp_model = models.Cellpose(gpu=gpu_ok, model_type=model_name)  # type: ignore
    else:
        log_error("不支持的 cellpose 版本：缺少 CellposeModel/Cellpose 类")
        raise RuntimeError("cellpose 版本不兼容")
    log_info(f"使用 Cellpose 进行分割（gpu={gpu_ok}，model={model_name}，tile=1024）")
    return image_path, out_dir, dapi, cp_model


#  创建分割结果子目录（原始 tile 与掩码 tile）
def _prepare_output_dirs(out_dir: str):
    raw_dir = os.path.join(out_dir, "raw_tiles")
    mask_dir = os.path.join(out_dir, "mask_tiles")
    ensure_dir(raw_dir)
    ensure_dir(mask_dir)
    return mask_dir, raw_dir


#  按 tiles 运行 Cellpose 分割，保存可视化，并记录每个核的质心坐标
def _segment_tiles_and_save(dapi: np.ndarray, cp_model, context: Dict[str, Any], raw_dir: str, mask_dir: str):
    tile_size = int(context.get("tile_size", 1024))
    H, W = dapi.shape
    rects = tile_iter(H, W, tile_size)
    only_x = context.get("only_tile_x")
    only_y = context.get("only_tile_y")
    if only_x is not None or only_y is not None:
        rects = [(y, x, h, w) for (y, x, h, w) in rects if (only_x is None or x == int(only_x)) and (only_y is None or y == int(only_y))]
    log_info(f"开始按tiles处理：尺寸 {H}x{W}，tile_size={tile_size}，总计 {len(rects)} tiles")
    all_centroids = []
    total_count = 0
    for idx, (y0, x0, h, w) in enumerate(rects, 1):
        log_info(f"[{idx}/{len(rects)}] 开始处理 tile (x={x0}, y={y0}, h={h}, w={w})")
        dp = dapi[y0:y0+h, x0:x0+w]
        try:
            diam = context.get("cellpose_diameter", None)
            bsz = context.get("cellpose_batch_size", None)
            if bsz is not None and hasattr(cp_model, "batch_size"):
                try:
                    setattr(cp_model, "batch_size", int(bsz))
                except Exception:
                    pass
            res = cp_model.eval(dp, channels=[0, 0], diameter=diam)  # type: ignore
            masks = res[0] if isinstance(res, (list, tuple)) else res
            lb = np.asarray(masks, dtype=np.int32)
        except Exception as e:
            log_error(f"Cellpose 分割失败：{e}")
            raise
        if cv2 is not None:
            cv2.imwrite(os.path.join(raw_dir, f"raw_{x0}_{y0}.png"), _to_uint8(dapi[y0:y0+h, x0:x0+w]))
            cv2.imwrite(os.path.join(mask_dir, f"mask_{x0}_{y0}.png"), ((lb > 0).astype(np.uint8)) * 255)
        elif plt is not None:
            fig_raw = plt.figure(figsize=(5, 5))
            plt.imshow(_to_uint8(dapi[y0:y0+h, x0:x0+w]), cmap="gray")
            plt.axis("off")
            fig_raw.savefig(os.path.join(raw_dir, f"raw_{x0}_{y0}.png"), dpi=150)
            plt.close(fig_raw)
            fig_mask = plt.figure(figsize=(5, 5))
            plt.imshow((lb > 0).astype(np.uint8), cmap="gray")
            plt.axis("off")
            fig_mask.savefig(os.path.join(mask_dir, f"mask_{x0}_{y0}.png"), dpi=150)
            plt.close(fig_mask)
        if measure is not None:
            props = measure.regionprops(lb)
            for p in props:
                cy, cx = p.centroid
                #  记录每个核的质心坐标（global_* 为全图坐标；tile_* 为 tile 左上角；local_* 为 tile 内坐标）
                all_centroids.append({
                    "global_x": float(x0 + cx),
                    "global_y": float(y0 + cy),
                    "tile_x": int(x0),
                    "tile_y": int(y0),
                    "local_x": float(cx),
                    "local_y": float(cy),
                })
            detected = len(props)
        else:
            detected = int(np.max(lb)) if lb.size > 0 else 0
        total_count += detected
        log_info(f"[{idx}/{len(rects)}] 完成 tile (x={x0}, y={y0})，检测核 {detected}")
    return {"centroids": all_centroids, "count": total_count, "tiles": len(rects)}


#  将每个核的质心坐标写出为 CSV 文件（列：global_x/global_y/tile_x/tile_y/local_x/local_y）
def _write_centroids_csv(out_dir: str, centroids: list[dict]):
    cent_path = os.path.join(out_dir, "nuclei_centroids.csv")
    with open(cent_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=["global_x", "global_y", "tile_x", "tile_y", "local_x", "local_y"])
        wtr.writeheader()
        for it in centroids:
            wtr.writerow(it)
    return cent_path