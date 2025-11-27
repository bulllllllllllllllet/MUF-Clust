"""分割步骤实现（3.1/3.2）"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...utils.logging import log_info, log_error, log_warn
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


# DAPI-only 细胞区域识别的常量（近似扩张系数与最近邻限制系数）
ROI_SCALE = 1.8
ROI_NEAREST_LIMIT = 0.6


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

    #  执行分割流程（支持单图与文件夹模式）。
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        dataset_dir = context.get("dataset_dir")
        input_path = context.get("input_path")
        folder_mode = bool(context.get("folder_mode", False))
        is_dir = bool(dataset_dir) or (bool(input_path) and os.path.isdir(str(input_path)))

        if folder_mode and is_dir:
            base_dir = dataset_dir or str(input_path)
            images = list_images(base_dir)
            if not images:
                log_error("文件夹模式：未找到 qptiff/tiff 图像")
                raise RuntimeError("未发现可分割的图像文件")
            output_root = context.get("output_dir", "outputs")
            seg_root = os.path.join(output_root, "segmentation")
            ensure_dir(seg_root)
            log_info(f"文件夹模式：发现 {len(images)} 个图像，输出根目录：{seg_root}")
            total_cells = 0
            total_tiles = 0
            failed: List[str] = []
            for idx, img in enumerate(images, 1):
                sample_name = os.path.splitext(os.path.basename(img))[0]
                out_dir = os.path.join(seg_root, sample_name)
                roi_dir = os.path.join(out_dir, "roi_tiles")
                cent_csv = os.path.join(out_dir, "nuclei_centroids.csv")
                if os.path.isdir(roi_dir) and os.path.isfile(cent_csv):
                    try:
                        has_roi = any((fn.startswith("roi_") or fn.startswith("roi_label_")) for fn in os.listdir(roi_dir))
                    except Exception:
                        has_roi = False
                    if has_roi:
                        log_info(f"跳过已完成样本：{sample_name}（检测到 roi_tiles 与 nuclei_centroids.csv）")
                        continue
                log_info(f"[{idx}/{len(images)}] 处理 {img}")
                ctx2 = dict(context)
                ctx2["image_path"] = img
                ctx2["dataset_dir"] = None
                ctx2["only_roi_tiles"] = True
                try:
                    image_path, out_dir, dapi, cp_model = _prepare_and_load(ctx2)
                    mask_dir, raw_dir, roi_dir = _prepare_output_dirs(out_dir, only_roi=True)
                    stats = _segment_tiles_and_save(dapi, cp_model, ctx2, raw_dir, mask_dir, roi_dir)
                    cent_path = _write_centroids_csv(out_dir, stats["centroids"]) if stats["centroids"] else os.path.join(out_dir, "nuclei_centroids.csv")
                    total_cells += stats["count"]
                    total_tiles += stats["tiles"]
                    log_info(f"=== 图像分割完成 ===\n样本：{sample_name}\n输出目录：{out_dir}\nROI目录：{os.path.join(out_dir, 'roi_tiles')}\n核数量：{stats['count']}，tiles：{stats['tiles']}")
                except Exception as e:
                    log_error(f"处理失败：{img}，错误：{e}")
                    failed.append(img)
                    continue
            return {
                "segmentation": {
                    "status": "ok",
                    "mode": "folder",
                    "out_dir": seg_root,
                    "count": total_cells,
                    "tiles": total_tiles,
                    "images": len(images),
                    "failed": failed,
                    "paths": {"root": seg_root},
                }
            }
        else:
            image_path, out_dir, dapi, cp_model = _prepare_and_load(context)
            only_roi = bool(context.get("only_roi_tiles", False))
            mask_dir, raw_dir, roi_dir = _prepare_output_dirs(out_dir, only_roi=only_roi)
            stats = _segment_tiles_and_save(dapi, cp_model, context, raw_dir, mask_dir, roi_dir)
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
                        "roi_tiles_dir": roi_dir,
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
    prefer_lr = bool(context.get("prefer_low_res", False))
    use_cellpose = bool(context.get("use_cellpose", True))
    log_info(f"准备读取 DAPI 通道（prefer_low_res={prefer_lr}）")
    import time
    t0 = time.time()
    try:
        dapi_idx = next(((c.index - 1) for c in cfgs if c.is_dapi), 0)
        only_x = context.get("only_tile_x")
        only_y = context.get("only_tile_y")
        if only_x is not None and only_y is not None:
            tile_size = int(context.get("tile_size", 1024))
            x0 = int(only_x)
            y0 = int(only_y)
            dapi = _read_dapi_window(image_path, dapi_idx, x0, y0, tile_size, tile_size, prefer_low_res=prefer_lr)
            context["window_mode"] = True
            context["tile_offset_x"] = x0
            context["tile_offset_y"] = y0
            log_info(f"窗口化读取 DAPI：区域 ({x0},{y0}) 大小 {dapi.shape[1]}x{dapi.shape[0]}")
        else:
            dapi = _read_dapi_full(image_path, dapi_idx, prefer_low_res=prefer_lr)
    except Exception:
        stack = read_qptiff_stack(image_path, prefer_low_res=prefer_lr)
        C = stack.shape[0]
        dapi_idx = next(((c.index - 1) for c in cfgs if c.is_dapi and (c.index - 1) < C), 0)
        only_x = context.get("only_tile_x")
        only_y = context.get("only_tile_y")
        if only_x is not None and only_y is not None:
            tile_size = int(context.get("tile_size", 1024))
            x0 = int(only_x)
            y0 = int(only_y)
            y1 = min(stack.shape[-2], y0 + tile_size)
            x1 = min(stack.shape[-1], x0 + tile_size)
            dapi = np.asarray(stack[dapi_idx, y0:y1, x0:x1], dtype=np.float32)
            context["window_mode"] = True
            context["tile_offset_x"] = x0
            context["tile_offset_y"] = y0
            log_info(f"窗口化读取 DAPI（回退栈）：区域 ({x0},{y0}) 大小 {dapi.shape[1]}x{dapi.shape[0]}")
        else:
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


#  创建分割结果子目录（原始 tile、核掩码、细胞ROI掩码）
def _prepare_output_dirs(out_dir: str, only_roi: bool):
    roi_dir = os.path.join(out_dir, "roi_tiles")
    ensure_dir(roi_dir)
    if only_roi:
        return None, None, roi_dir
    raw_dir = os.path.join(out_dir, "raw_tiles")
    mask_dir = os.path.join(out_dir, "mask_tiles")
    ensure_dir(raw_dir)
    ensure_dir(mask_dir)
    return mask_dir, raw_dir, roi_dir


#  按 tiles 运行 Cellpose 分割，保存可视化，并记录每个核的质心坐标；基于 DAPI 构造细胞 ROI（固定半径+Voronoi）
def _segment_tiles_and_save(dapi: np.ndarray, cp_model, context: Dict[str, Any], raw_dir: str, mask_dir: str, roi_dir: str):
    tile_size = int(context.get("tile_size", 1024))
    H, W = dapi.shape
    window_mode = bool(context.get("window_mode", False))
    if window_mode:
        rects = [(0, 0, H, W)]
        base_x0 = int(context.get("tile_offset_x", 0))
        base_y0 = int(context.get("tile_offset_y", 0))
    else:
        rects = tile_iter(H, W, tile_size)
        base_x0 = 0
        base_y0 = 0
    # 窗口模式下已按指定偏移生成单一 rect，无需再按 only_tile 过滤；整图模式才执行过滤
    only_x = context.get("only_tile_x")
    only_y = context.get("only_tile_y")
    if (only_x is not None or only_y is not None) and (not window_mode):
        rects = [(y, x, h, w) for (y, x, h, w) in rects if (only_x is None or x == int(only_x)) and (only_y is None or y == int(only_y))]
    log_info(f"开始按tiles处理：尺寸 {H}x{W}，tile_size={tile_size}，总计 {len(rects)} tiles")
    all_centroids = []
    total_count = 0
    num_workers = int(context.get("num_workers", 1))
    if num_workers > 1:
        log_info(f"并行分割：workers={num_workers}")
        def _process_one(idx: int, y0: int, x0: int, h: int, w: int):
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
                if raw_dir is not None:
                    cv2.imwrite(os.path.join(raw_dir, f"raw_{base_x0 + x0}_{base_y0 + y0}.png"), _to_uint8(dp))
                if mask_dir is not None:
                    cv2.imwrite(os.path.join(mask_dir, f"mask_{base_x0 + x0}_{base_y0 + y0}.png"), ((lb > 0).astype(np.uint8)) * 255)
            elif plt is not None:
                if raw_dir is not None:
                    fig_raw = plt.figure(figsize=(5, 5)); plt.imshow(_to_uint8(dp), cmap="gray"); plt.axis("off"); fig_raw.savefig(os.path.join(raw_dir, f"raw_{base_x0 + x0}_{base_y0 + y0}.png"), dpi=150); plt.close(fig_raw)
                if mask_dir is not None:
                    fig_mask = plt.figure(figsize=(5, 5)); plt.imshow((lb > 0).astype(np.uint8), cmap="gray"); plt.axis("off"); fig_mask.savefig(os.path.join(mask_dir, f"mask_{base_x0 + x0}_{base_y0 + y0}.png"), dpi=150); plt.close(fig_mask)
            detected = 0
            cents_local: List[Dict[str, Any]] = []
            if measure is not None:
                props = measure.regionprops(lb)
                for p in props:
                    cy, cx = p.centroid
                    cents_local.append({
                        "global_x": float(base_x0 + x0 + cx),
                        "global_y": float(base_y0 + y0 + cy),
                        "tile_x": int(x0),
                        "tile_y": int(y0),
                        "local_x": float(cx),
                        "local_y": float(cy),
                    })
                roi_lb = _build_cell_roi_from_props(props, h, w)
                if cv2 is not None:
                    cv2.imwrite(os.path.join(roi_dir, f"roi_{base_x0 + x0}_{base_y0 + y0}.png"), ((roi_lb > 0).astype(np.uint8)) * 255)
                    cv2.imwrite(os.path.join(roi_dir, f"roi_label_{base_x0 + x0}_{base_y0 + y0}.png"), roi_lb.astype(np.uint16))
                    try:
                        roi_u8 = np.clip(roi_lb.astype(np.float32), 0, None)
                        vmax = float(np.max(roi_u8))
                        if vmax > 0:
                            roi_u8 = (roi_u8 / vmax * 255.0).astype(np.uint8)
                        colored = cv2.applyColorMap(roi_u8, cv2.COLORMAP_PARULA)
                        cv2.imwrite(os.path.join(roi_dir, f"roi_label_colored_{base_x0 + x0}_{base_y0 + y0}.png"), colored)
                    except Exception:
                        pass
                elif plt is not None:
                    fig_roi = plt.figure(figsize=(5, 5)); plt.imshow((roi_lb > 0).astype(np.uint8), cmap="gray"); plt.axis("off"); fig_roi.savefig(os.path.join(roi_dir, f"roi_{base_x0 + x0}_{base_y0 + y0}.png"), dpi=150); plt.close(fig_roi)
                    fig_roi_lbl = plt.figure(figsize=(5, 5)); plt.imshow(roi_lb, cmap="nipy_spectral"); plt.axis("off"); fig_roi_lbl.savefig(os.path.join(roi_dir, f"roi_label_{base_x0 + x0}_{base_y0 + y0}.png"), dpi=150); plt.close(fig_roi_lbl)
                detected = len(props)
            else:
                detected = int(np.max(lb)) if lb.size > 0 else 0
            return idx, y0, x0, detected, cents_local
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            fut_map = {}
            for idx, (y0, x0, h, w) in enumerate(rects, 1):
                fut = ex.submit(_process_one, idx, y0, x0, h, w)
                fut_map[fut] = (idx, y0, x0)
            for fut in as_completed(fut_map):
                idx, y0, x0, detected, cents_local = fut.result()
                total_count += detected
                all_centroids.extend(cents_local)
                log_info(f"[{idx}/{len(rects)}] 完成 tile (x={x0}, y={y0})，检测核 {detected}")
    else:
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
                if raw_dir is not None:
                    cv2.imwrite(os.path.join(raw_dir, f"raw_{base_x0 + x0}_{base_y0 + y0}.png"), _to_uint8(dapi[y0:y0+h, x0:x0+w]))
                if mask_dir is not None:
                    cv2.imwrite(os.path.join(mask_dir, f"mask_{base_x0 + x0}_{base_y0 + y0}.png"), ((lb > 0).astype(np.uint8)) * 255)
            elif plt is not None:
                if raw_dir is not None:
                    fig_raw = plt.figure(figsize=(5, 5))
                    plt.imshow(_to_uint8(dapi[y0:y0+h, x0:x0+w]), cmap="gray")
                    plt.axis("off")
                    fig_raw.savefig(os.path.join(raw_dir, f"raw_{base_x0 + x0}_{base_y0 + y0}.png"), dpi=150)
                    plt.close(fig_raw)
                if mask_dir is not None:
                    fig_mask = plt.figure(figsize=(5, 5))
                    plt.imshow((lb > 0).astype(np.uint8), cmap="gray")
                    plt.axis("off")
                    fig_mask.savefig(os.path.join(mask_dir, f"mask_{base_x0 + x0}_{base_y0 + y0}.png"), dpi=150)
                    plt.close(fig_mask)
            if measure is not None:
                props = measure.regionprops(lb)
                for p in props:
                    cy, cx = p.centroid
                    all_centroids.append({
                        "global_x": float(base_x0 + x0 + cx),
                        "global_y": float(base_y0 + y0 + cy),
                        "tile_x": int(x0),
                        "tile_y": int(y0),
                        "local_x": float(cx),
                        "local_y": float(cy),
                    })
                roi_lb = _build_cell_roi_from_props(props, h, w)
                if cv2 is not None:
                    cv2.imwrite(os.path.join(roi_dir, f"roi_{base_x0 + x0}_{base_y0 + y0}.png"), ((roi_lb > 0).astype(np.uint8)) * 255)
                    cv2.imwrite(os.path.join(roi_dir, f"roi_label_{base_x0 + x0}_{base_y0 + y0}.png"), roi_lb.astype(np.uint16))
                    try:
                        roi_u8 = np.clip(roi_lb.astype(np.float32), 0, None)
                        vmax = float(np.max(roi_u8))
                        if vmax > 0:
                            roi_u8 = (roi_u8 / vmax * 255.0).astype(np.uint8)
                        colored = cv2.applyColorMap(roi_u8, cv2.COLORMAP_PARULA)
                        cv2.imwrite(os.path.join(roi_dir, f"roi_label_colored_{base_x0 + x0}_{base_y0 + y0}.png"), colored)
                    except Exception:
                        pass
                elif plt is not None:
                    fig_roi = plt.figure(figsize=(5, 5))
                    plt.imshow((roi_lb > 0).astype(np.uint8), cmap="gray")
                    plt.axis("off")
                    fig_roi.savefig(os.path.join(roi_dir, f"roi_{base_x0 + x0}_{base_y0 + y0}.png"), dpi=150)
                    plt.close(fig_roi)
                    fig_roi_lbl = plt.figure(figsize=(5, 5))
                    plt.imshow(roi_lb, cmap="nipy_spectral")
                    plt.axis("off")
                    fig_roi_lbl.savefig(os.path.join(roi_dir, f"roi_label_{base_x0 + x0}_{base_y0 + y0}.png"), dpi=150)
                    plt.close(fig_roi_lbl)
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

# 根据核的形态属性生成细胞 ROI（DAPI-only）：
# 1) 估计每个核的等效半径 r_nuc = sqrt(area/pi)
# 2) 设定细胞半径 R = min(ROI_SCALE*r_nuc, ROI_NEAREST_LIMIT*最近邻距离)
# 3) 在每个核质心为圆心的圆盘内进行 Voronoi 分配：像素归属距离最近的质心（限制在该圆盘内）
def _build_cell_roi_from_props(props: List[Any], h: int, w: int) -> np.ndarray:
    # 收集标签、质心与面积
    labels: List[int] = []
    cents: List[Tuple[float, float]] = []
    radii: List[float] = []
    for p in props:
        labels.append(int(p.label))
        cy, cx = p.centroid
        cents.append((float(cy), float(cx)))
        area = float(getattr(p, "area", 0.0))
        r_nuc = (np.sqrt(area / np.pi) if area > 0.0 else 4.0)
        radii.append(r_nuc)

    n = len(labels)
    if n == 0:
        return np.zeros((h, w), dtype=np.int32)

    # 计算最近邻距离
    nn_dist: List[float] = [np.inf] * n
    for i in range(n):
        yi, xi = cents[i]
        dmin = np.inf
        for j in range(n):
            if i == j:
                continue
            yj, xj = cents[j]
            d = np.sqrt((yi - yj) ** 2 + (xi - xj) ** 2)
            if d < dmin:
                dmin = d
        nn_dist[i] = (dmin if np.isfinite(dmin) else np.inf)

    # 确定细胞半径上限
    R: List[float] = []
    for i in range(n):
        r_nuc = radii[i]
        dnn = nn_dist[i]
        r1 = ROI_SCALE * r_nuc
        r2 = (ROI_NEAREST_LIMIT * dnn) if np.isfinite(dnn) else r1
        R.append(float(min(r1, r2)))

    # Voronoi 限制分配（仅在各自圆盘内）：最近距离优先
    yy, xx = np.ogrid[:h, :w]
    assign_label = np.zeros((h, w), dtype=np.int32)
    assign_dist2 = np.full((h, w), np.inf, dtype=np.float64)
    for i in range(n):
        cy, cx = cents[i]
        dy = yy - cy
        dx = xx - cx
        d2 = dy * dy + dx * dx
        r2 = R[i] * R[i]
        mask = d2 <= r2
        better = mask & (d2 < assign_dist2)
        if np.any(better):
            assign_label[better] = labels[i]
            assign_dist2[better] = d2[better]
    return assign_label.astype(np.int32)

# 窗口化读取 DAPI 通道的指定区域（仅加载目标 tile 对应的切片），返回二维 float32 图像
def _read_dapi_window(path: str, dapi_index0: int, x0: int, y0: int, w: int, h: int, prefer_low_res: bool = False) -> np.ndarray:
    if tifffile is None:
        raise RuntimeError("需要安装 tifffile 以窗口化读取 QPTIFF")
    try:
        import zarr as _z  # type: ignore
    except Exception:
        _z = None  # type: ignore
    with tifffile.TiffFile(path) as tf:  # type: ignore
        s = tf.series[0]
        levels = getattr(s, "levels", [s])
        lvl_idx = (len(levels) - 1) if (prefer_low_res and len(levels) > 1) else 0
        lvl = levels[lvl_idx]
        axes = getattr(lvl, "axes", "")
        shape = lvl.shape
        arr = s.aszarr(level=lvl_idx)
        try:
            _ = arr.ndim  # zarr-like
        except AttributeError:
            if _z is None:
                raise RuntimeError("缺少 zarr 依赖，无法窗口化读取")
            arr = _z.open(arr)
        axes_str = axes if isinstance(axes, str) and axes else ""
        y_idx = axes_str.find("Y") if (axes_str and "Y" in axes_str) else (arr.ndim - 2)
        x_idx = axes_str.find("X") if (axes_str and "X" in axes_str) else (arr.ndim - 1)
        ch_char = "C" if (axes_str and "C" in axes_str) else ("S" if (axes_str and "S" in axes_str) else None)
        ch_idx = axes_str.find(ch_char) if ch_char else None
        # 边界裁剪
        full_h = shape[y_idx] if y_idx >= 0 else shape[-2]
        full_w = shape[x_idx] if x_idx >= 0 else shape[-1]
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(full_w, x0 + int(w)); y1 = min(full_h, y0 + int(h))
        # 构造切片
        slices = [slice(None)] * arr.ndim
        if y_idx is not None:
            slices[y_idx] = slice(y0, y1)
        if x_idx is not None:
            slices[x_idx] = slice(x0, x1)
        if ch_idx is not None:
            slices[ch_idx] = dapi_index0
        patch = np.asarray(arr[tuple(slices)])
        if patch.ndim != 2:
            patch = np.squeeze(patch)
        return np.asarray(patch, dtype=np.float32)