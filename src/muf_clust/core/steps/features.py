"""特征提取步骤实现（3.3/3.4）"""

from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from ...utils.logging import log_info, log_warn, log_error
from ...utils.paths import ensure_dir
from ...config import get_config, DEFAULT_CANCER_TYPE
from .preprocess import read_qptiff_stack

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    import tifffile  # type: ignore
except Exception:
    tifffile = None  # type: ignore


#  解析 ROI 标签文件名中的 tile 坐标 (x, y)
def _parse_roi_label_xy(name: str) -> Tuple[int, int]:
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    return int(parts[-2]), int(parts[-1])


#  将输入图像线性缩放到 0–255 的 8bit，用于可视化与存盘
def _to_uint8(x: np.ndarray) -> np.ndarray:
    m = float(np.min(x))
    M = float(np.max(x))
    if M <= m:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - m) / (M - m)
    return (y * 255.0).astype(np.uint8)


#  窗口化读取指定通道的子区域（通用通道索引）
def _read_channel_window(path: str, channel_index0: int, x0: int, y0: int, w: int, h: int, prefer_low_res: bool = False) -> np.ndarray:
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
        arr = s.aszarr(level=lvl_idx)
        try:
            _ = arr.ndim
        except AttributeError:
            if _z is None:
                raise RuntimeError("缺少 zarr 依赖，无法窗口化读取")
            arr = _z.open(arr)
        axes_str = axes if isinstance(axes, str) and axes else ""
        y_idx = axes_str.find("Y") if (axes_str and "Y" in axes_str) else (arr.ndim - 2)
        x_idx = axes_str.find("X") if (axes_str and "X" in axes_str) else (arr.ndim - 1)
        ch_char = "C" if (axes_str and "C" in axes_str) else ("S" if (axes_str and "S" in axes_str) else None)
        ch_idx = axes_str.find(ch_char) if ch_char else None
        full_h = arr.shape[y_idx] if y_idx >= 0 else arr.shape[-2]
        full_w = arr.shape[x_idx] if x_idx >= 0 else arr.shape[-1]
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(full_w, x0 + int(w)); y1 = min(full_h, y0 + int(h))
        slices = [slice(None)] * arr.ndim
        if y_idx is not None:
            slices[y_idx] = slice(y0, y1)
        if x_idx is not None:
            slices[x_idx] = slice(x0, x1)
        if ch_idx is not None:
            slices[ch_idx] = channel_index0
        patch = np.asarray(arr[tuple(slices)])
        if patch.ndim != 2:
            patch = np.squeeze(patch)
        return np.asarray(patch, dtype=np.float32)


#  写出长表 CSV（每行对应一个细胞×一个通道）
def _write_features_long_csv(features_dir: str, rows: List[Dict[str, Any]]) -> str:
    path = os.path.join(features_dir, "features_long.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        keys = ["tile_x","tile_y","label","local_x","local_y","channel_index","channel_name","mean","median","sum","std","mean_bg_corr","median_bg_corr","n_pixels"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


#  拼接为 N×C 特征矩阵（默认使用 mean_bg_corr 作为度量）
def _write_features_matrix_csv(features_dir: str, rows: List[Dict[str, Any]], channel_names: List[str], metric_key: str = "mean_bg_corr") -> str:
    path = os.path.join(features_dir, "features_matrix_mean.csv")
    ids = sorted({(r["tile_x"], r["tile_y"], r["label"]) for r in rows})
    mat_map: Dict[str, Dict[str, Any]] = {}
    for (tx, ty, lb) in ids:
        cid = f"{tx}_{ty}_{lb}"
        entry: Dict[str, Any] = {"cell_id": cid, "tile_x": tx, "tile_y": ty, "label": lb, "local_x": 0.0, "local_y": 0.0}
        for ch in channel_names:
            entry[ch] = 0.0
        mat_map[cid] = entry
    for r in rows:
        key = f"{r['tile_x']}_{r['tile_y']}_{r['label']}"
        e = mat_map.get(key)
        if e is None:
            continue
        ch = r["channel_name"]
        e[ch] = float(r.get(metric_key, 0.0))
        if float(e.get("local_x", 0.0)) == 0.0 and float(e.get("local_y", 0.0)) == 0.0:
            e["local_x"] = float(r.get("local_x", 0.0))
            e["local_y"] = float(r.get("local_y", 0.0))
    cols = ["cell_id","tile_x","tile_y","label","local_x","local_y"] + channel_names
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for cid in sorted(mat_map.keys()):
            w.writerow(mat_map[cid])
    return path


@dataclass
class FeatureStep:
    name: str = "features"

    #  执行特征提取：基于分割生成的 ROI 标签，聚合非 DAPI/AF 通道强度；支持整图与单 tile
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        image_path = context.get("image_path") or context.get("input_path")
        seg = context.get("segmentation", {})
        out_dir = seg.get("out_dir")
        roi_dir = seg.get("paths", {}).get("roi_tiles_dir")
        if not image_path or not out_dir or not roi_dir:
            log_error("缺少分割输出或图像路径，无法进行特征提取")
            raise RuntimeError("缺少必要输入")

        cancer_type = context.get("cancer_type", DEFAULT_CANCER_TYPE)
        cfgs = get_config(cancer_type) or []
        prefer_lr = bool(context.get("prefer_low_res", False))
        only_x = context.get("only_tile_x")
        only_y = context.get("only_tile_y")
        features_dir = ensure_dir(os.path.join(out_dir, "features"))

        ch_specs = [c for c in (cfgs or []) if (not getattr(c, "is_dapi", False)) and (not getattr(c, "is_af", False))]
        if len(ch_specs) == 0:
            return {"features": {"features_dir": features_dir, "features_long_csv": os.path.join(features_dir, "features_long.csv"), "features_matrix_csv": os.path.join(features_dir, "features_matrix_mean.csv"), "cells": 0}}

        files = [f for f in os.listdir(roi_dir) if f.startswith("roi_label_") and f.endswith(".png") and ("colored" not in f)]
        files.sort(key=lambda fn: (_parse_roi_label_xy(fn)[1], _parse_roi_label_xy(fn)[0]))
        total_tiles = len(files)
        log_info(f"开始 3.3 特征提取：通道数={len(ch_specs)}，ROI tiles={total_tiles}（prefer_low_res={prefer_lr}）")

        use_window = (tifffile is not None)
        channel_tiles_dir = ensure_dir(os.path.join(out_dir, "channel_tiles")) if (use_window and only_x is not None and only_y is not None) else None
        stack = None
        if not use_window:
            log_error("缺少 tifffile，无法进行特征提取")
            raise RuntimeError("需要安装 tifffile 以读取 QPTIFF")
        else:
            log_info("采用按 tile 窗口化读取通道数据（避免整图栈）")
        log_info(f"开始遍历 ROI tiles：总数 {total_tiles}")

        rows: List[Dict[str, Any]] = []
        tile_idx = 0
        num_workers = int(context.get("num_workers", 1))
        stack_lock = threading.Lock()
        def _get_stack():
            nonlocal stack
            if stack is None:
                with stack_lock:
                    if stack is None:
                        stack = read_qptiff_stack(image_path, prefer_low_res=prefer_lr)
            return stack
        def _process_tile(fn: str):
            tx, ty = _parse_roi_label_xy(fn)
            if (only_x is not None and tx != int(only_x)) or (only_y is not None and ty != int(only_y)):
                return tx, ty, 0, []
            lbl = cv2.imread(os.path.join(roi_dir, fn), cv2.IMREAD_UNCHANGED) if cv2 is not None else None
            if lbl is None or lbl.ndim != 2:
                return tx, ty, 0, []
            h, w = lbl.shape
            tile_rows: List[Dict[str, Any]] = []
            labels = np.unique(lbl)
            labels = labels[labels > 0]
            for spec in ch_specs:
                idx = int(getattr(spec, "index", 1)) - 1
                try:
                    ch = _read_channel_window(image_path, idx, tx, ty, w, h, prefer_low_res=prefer_lr) if use_window else _get_stack()[idx, ty:ty+h, tx:tx+w]
                except Exception:
                    st = _get_stack()
                    if st is None or idx < 0 or idx >= st.shape[0]:
                        continue
                    ch = st[idx, ty:ty+h, tx:tx+w]
                bg = ch[lbl == 0]
                bg_med = float(np.median(bg)) if bg.size > 0 else 0.0
                for lb in labels:
                    vals = ch[lbl == lb]
                    if vals.size == 0:
                        continue
                    yy, xx = np.nonzero(lbl == lb)
                    cy = float(np.mean(yy)) if yy.size > 0 else 0.0
                    cx = float(np.mean(xx)) if xx.size > 0 else 0.0
                    tile_rows.append({
                        "tile_x": int(tx),
                        "tile_y": int(ty),
                        "label": int(lb),
                        "local_x": float(cx),
                        "local_y": float(cy),
                        "channel_index": idx + 1,
                        "channel_name": str(getattr(spec, "name", f"ch{idx+1}")),
                        "mean": float(np.mean(vals)),
                        "median": float(np.median(vals)),
                        "sum": float(np.sum(vals)),
                        "std": float(np.std(vals)),
                        "mean_bg_corr": float(np.mean(vals) - bg_med),
                        "median_bg_corr": float(np.median(vals) - bg_med),
                        "n_pixels": int(vals.size),
                    })
            return tx, ty, len(tile_rows), tile_rows
        if num_workers > 1:
            log_info(f"并行特征提取：workers={num_workers}")
            done = 0
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futs = [ex.submit(_process_tile, fn) for fn in files]
                for f in as_completed(futs):
                    tx, ty, added, tile_rows = f.result()
                    if added > 0:
                        rows.extend(tile_rows)
                    done += 1
                    log_info(f"[{done}/{total_tiles}] 完成 tile (x={tx}, y={ty})，新增记录 {added}，累计记录 {len(rows)} 行")
        else:
            for fn in files:
                tx, ty = _parse_roi_label_xy(fn)
                if (only_x is not None and tx != int(only_x)) or (only_y is not None and ty != int(only_y)):
                    if only_x is not None or only_y is not None:
                        continue
                lbl = cv2.imread(os.path.join(roi_dir, fn), cv2.IMREAD_UNCHANGED) if cv2 is not None else None
                if lbl is None:
                    raise RuntimeError("需要 OpenCV 读取 ROI 标签图")
                if lbl.ndim != 2:
                    log_warn(f"跳过非二维标签图：{fn}，shape={getattr(lbl, 'shape', None)}")
                    continue
                h, w = lbl.shape
                tile_idx += 1
                start_rows = len(rows)
                log_info(f"[{tile_idx}/{total_tiles}] 特征提取：tile (x={tx}, y={ty}, h={h}, w={w})")
                for spec in ch_specs:
                    idx = int(getattr(spec, "index", 1)) - 1
                    try:
                        ch = _read_channel_window(image_path, idx, tx, ty, w, h, prefer_low_res=prefer_lr) if use_window else _get_stack()[idx, ty:ty+h, tx:tx+w]
                    except Exception:
                        st = _get_stack()
                        if st is None or idx < 0 or idx >= st.shape[0]:
                            continue
                        ch = st[idx, ty:ty+h, tx:tx+w]
                    bg = ch[lbl == 0]
                    bg_med = float(np.median(bg)) if bg.size > 0 else 0.0
                    labels = np.unique(lbl)
                    labels = labels[labels > 0]
                    for lb in labels:
                        vals = ch[lbl == lb]
                        if vals.size == 0:
                            continue
                        yy, xx = np.nonzero(lbl == lb)
                        cy = float(np.mean(yy)) if yy.size > 0 else 0.0
                        cx = float(np.mean(xx)) if xx.size > 0 else 0.0
                        rows.append({
                            "tile_x": int(tx),
                            "tile_y": int(ty),
                            "label": int(lb),
                            "local_x": float(cx),
                            "local_y": float(cy),
                            "channel_index": idx + 1,
                            "channel_name": str(getattr(spec, "name", f"ch{idx+1}")),
                            "mean": float(np.mean(vals)),
                            "median": float(np.median(vals)),
                            "sum": float(np.sum(vals)),
                            "std": float(np.std(vals)),
                            "mean_bg_corr": float(np.mean(vals) - bg_med),
                            "median_bg_corr": float(np.median(vals) - bg_med),
                            "n_pixels": int(vals.size),
                        })
                added = len(rows) - start_rows
                log_info(f"[{tile_idx}/{total_tiles}] 完成 tile (x={tx}, y={ty})，新增记录 {added}，累计记录 {len(rows)} 行")

        long_csv = _write_features_long_csv(features_dir, rows)
        matrix_csv = _write_features_matrix_csv(features_dir, rows, [str(getattr(s, "name", f"ch{getattr(s, 'index', 1)}")) for s in ch_specs], metric_key="mean_bg_corr")
        cells = len({(r["tile_x"], r["tile_y"], r["label"]) for r in rows})
        log_info(f"3.3 特征提取完成：cells={cells}，long_csv={long_csv}，matrix_csv={matrix_csv}")

        return {
            "features": {
                "features_dir": features_dir,
                "features_long_csv": long_csv,
                "features_matrix_csv": matrix_csv,
                "cells": int(cells),
                "channel_tiles_dir": channel_tiles_dir,
            }
        }