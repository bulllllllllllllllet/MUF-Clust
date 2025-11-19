"""预处理与QC步骤实现（对齐 guide.md 2.2）

中文说明：
- 调用现有 `src/muf_clust/preprocess.py` 中的具体实现（list_images/process_one_image），
  将其编排为面向管线的 `Step`，便于通过 `api.run_preprocess` 统一触发。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import math
import json
import random
import numpy as np

from ...utils.logging import log_info, log_warn, log_error
from ...utils.paths import ensure_dir
from ...config import get_config, DEFAULT_CANCER_TYPE, ChannelSpec

try:
    import tifffile  # type: ignore
except Exception:
    tifffile = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from skimage import filters, registration, morphology, util  # type: ignore
except Exception:
    filters = None  # type: ignore
    registration = None  # type: ignore
    morphology = None  # type: ignore
    util = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
except Exception:
    plt = None  # type: ignore
    sns = None  # type: ignore


def list_images(dataset_dir: str, pattern: Tuple[str, ...] = (".qptiff", ".tif", ".tiff")) -> List[str]:
    files: List[str] = []
    import os
    for root, _, fns in os.walk(dataset_dir):
        for fn in fns:
            if any(fn.lower().endswith(p) for p in pattern):
                files.append(os.path.join(root, fn))
    return sorted(files)


def read_qptiff_stack(path: str, prefer_low_res: bool = False) -> np.ndarray:
    if tifffile is None:
        raise RuntimeError("需要安装 tifffile 以按多通道读取 QPTIFF/TIFF")

    from typing import List as _List

    def _extract_yx(arr: np.ndarray, axes: str, ch_idx: Optional[int] = None, ch_index: Optional[int] = None) -> np.ndarray:
        axes = axes or ""
        ndim = arr.ndim
        sel: _List[object] = [0] * ndim
        y_idx = axes.find("Y") if (axes and "Y" in axes) else (ndim - 2 if ndim >= 2 else 0)
        x_idx = axes.find("X") if (axes and "X" in axes) else (ndim - 1 if ndim >= 2 else 0)
        if 0 <= y_idx < ndim:
            sel[y_idx] = slice(None)
        if 0 <= x_idx < ndim:
            sel[x_idx] = slice(None)
        if ch_idx is not None and 0 <= ch_idx < ndim:
            sel[ch_idx] = ch_index if ch_index is not None else 0
        sub = np.asarray(arr[tuple(sel)])
        if sub.ndim != 2:
            sub = np.squeeze(sub)
            if sub.ndim != 2:
                if sub.ndim >= 2:
                    sub = sub.reshape(sub.shape[-2], sub.shape[-1])
                else:
                    sub = sub[np.newaxis, np.newaxis]
        return sub

    def _center_crop_2d(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        H, W = img.shape
        if H == target_h and W == target_w:
            return img
        y0 = max(0, (H - target_h) // 2)
        x0 = max(0, (W - target_w) // 2)
        return img[y0 : y0 + target_h, x0 : x0 + target_w]

    channels_2d: _List[np.ndarray] = []

    with tifffile.TiffFile(path) as tf:  # type: ignore
        for s in tf.series:
            levels = getattr(s, "levels", [s])
            lvl_idx = (len(levels) - 1) if (prefer_low_res and len(levels) > 1) else 0
            lvl = levels[lvl_idx]
            axes = getattr(lvl, "axes", "")
            arr = lvl.asarray()

            axes_str = axes if isinstance(axes, str) else ""
            ch_char = "C" if (axes_str and "C" in axes_str) else ("S" if (axes_str and "S" in axes_str) else None)
            ch_idx = (axes_str.find(ch_char) if ch_char else None)

            if ch_idx is not None and arr.shape[ch_idx] > 1:
                for c in range(arr.shape[ch_idx]):
                    ch2d = _extract_yx(arr, axes_str, ch_idx=ch_idx, ch_index=c)
                    channels_2d.append(ch2d)
            else:
                if arr.ndim == 2:
                    channels_2d.append(arr)
                elif arr.ndim == 3:
                    k = arr.shape[-1]
                    if k > 1 and k <= 16:
                        for c in range(k):
                            ch2d = _extract_yx(arr, axes_str if axes_str else "YXC", ch_idx=arr.ndim - 1, ch_index=c)
                            channels_2d.append(ch2d)
                    else:
                        ch2d = _extract_yx(arr, axes_str, ch_idx=None, ch_index=None)
                        channels_2d.append(ch2d)
                else:
                    ch2d = _extract_yx(arr, axes_str, ch_idx=None, ch_index=None)
                    channels_2d.append(ch2d)

    if len(channels_2d) == 0:
        raise RuntimeError("未能识别到任何通道，请检查文件是否为有效的 QPTIFF/TIFF")

    min_h = min(ch.shape[0] for ch in channels_2d)
    min_w = min(ch.shape[1] for ch in channels_2d)
    stack = [
        _center_crop_2d(np.asarray(ch, dtype=np.float32), min_h, min_w) for ch in channels_2d
    ]
    return np.stack(stack, axis=0)


def normalize_image(img: np.ndarray) -> np.ndarray:
    m, M = float(np.min(img)), float(np.max(img))
    if M - m < 1e-6:
        return np.zeros_like(img)
    return (img - m) / (M - m)


def edge_map(img: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        img8 = (normalize_image(img) * 255).astype(np.uint8)
        img8 = cv2.GaussianBlur(img8, (0, 0), 1.2)
        edges = cv2.Canny(img8, 50, 150)
        return edges.astype(np.float32)
    if filters is not None:
        return filters.sobel(img)
    gy, gx = np.gradient(img)
    return np.hypot(gx, gy)


def tile_iter(H: int, W: int, tile: int) -> List[Tuple[int, int, int, int]]:
    ys = list(range(0, H, tile))
    xs = list(range(0, W, tile))
    rects: List[Tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            rects.append((y, x, min(tile, H - y), min(tile, W - x)))
    return rects


def phase_drift(dapi: np.ndarray, ch: np.ndarray, upsample: int = 10) -> Tuple[float, float]:
    if registration is None:
        raise RuntimeError("需要安装 scikit-image (registration) 才能计算漂移")
    shift, _, _ = registration.phase_cross_correlation(dapi, ch, upsample_factor=upsample)  # type: ignore
    return float(shift[0]), float(shift[1])


def drift_vector_field(ref: np.ndarray, ch: np.ndarray, tile: int = 512, max_tiles: int = 0) -> Dict[str, np.ndarray]:
    H, W = ref.shape
    rects = tile_iter(H, W, tile)
    if max_tiles and max_tiles > 0 and len(rects) > max_tiles:
        step = max(1, int(math.ceil(len(rects) / float(max_tiles))))
        rects = rects[::step]
    centers_y: List[float] = []
    centers_x: List[float] = []
    dys: List[float] = []
    dxs: List[float] = []
    for (y, x, h, w) in rects:
        r = ref[y : y + h, x : x + w]
        c = ch[y : y + h, x : x + w]
        dy, dx = phase_drift(r, c)
        centers_y.append(y + h / 2.0)
        centers_x.append(x + w / 2.0)
        dys.append(dy)
        dxs.append(dx)
    return {
        "centers_y": np.array(centers_y, dtype=np.float32),
        "centers_x": np.array(centers_x, dtype=np.float32),
        "dy": np.array(dys, dtype=np.float32),
        "dx": np.array(dxs, dtype=np.float32),
    }


def plot_drift_quiver(field: Dict[str, np.ndarray], out_path: str, title: str = "Channel drift vector field") -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 8))
    plt.quiver(field["centers_x"], field["centers_y"], field["dx"], -field["dy"], angles='xy', scale_units='xy', scale=1)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def overlay_edges(ref: np.ndarray, ch: np.ndarray) -> np.ndarray:
    r_e = edge_map(ref)
    c_e = edge_map(ch)
    r_n = normalize_image(r_e)
    c_n = normalize_image(c_e)
    rgb = np.zeros((ref.shape[0], ref.shape[1], 3), dtype=np.float32)
    rgb[..., 2] = r_n
    rgb[..., 0] = c_n
    rgb[..., 1] = 0.5 * (r_n + c_n)
    return np.clip(rgb, 0.0, 1.0)


def save_image(img: np.ndarray, out_path: str) -> None:
    if plt is None:
        return
    import os
    dirn = os.path.dirname(out_path)
    if dirn:
        ensure_dir(dirn)
    plt.figure(figsize=(8, 8))
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def spectral_crosstalk_matrix(stack: np.ndarray, exclude_dapi: bool = True) -> np.ndarray:
    C = stack.shape[0]
    vecs: List[np.ndarray] = []
    for ci in range(C):
        vecs.append(stack[ci].reshape(-1))
    mat = np.vstack(vecs)
    corr = np.corrcoef(mat)
    if exclude_dapi:
        pass
    return corr


def plot_corr_heatmap(corr: np.ndarray, labels: List[str], out_path: str, title: str = "Spectral crosstalk matrix heatmap") -> None:
    if plt is None:
        return
    plt.figure(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", xticklabels=labels, yticklabels=labels)  # type: ignore
    else:
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def af_background_correction(af: np.ndarray, radius: int = 10) -> np.ndarray:
    if morphology is None:
        if cv2 is not None:
            bg = cv2.GaussianBlur(af, (0, 0), sigmaX=radius, sigmaY=radius)
            return np.clip(af - bg, 0, None)
        return af
    selem = morphology.disk(radius)  # type: ignore
    af_norm = normalize_image(af)
    af_corr = morphology.white_tophat(af_norm, selem)  # type: ignore
    return af_corr


def laplacian_variance(img: np.ndarray) -> float:
    if cv2 is not None:
        img8 = (normalize_image(img) * 255).astype(np.uint8)
        lap = cv2.Laplacian(img8, cv2.CV_64F)
        return float(lap.var())
    gy, gx = np.gradient(img)
    gyy, _ = np.gradient(gy)
    _, gxx = np.gradient(gx)
    lap = gxx + gyy
    return float(np.var(lap))


def quality_heatmap(img: np.ndarray, tile: int = 512) -> Tuple[np.ndarray, Dict[str, float]]:
    H, W = img.shape
    rects = tile_iter(H, W, tile)
    vals: List[float] = []
    for (y, x, h, w) in rects:
        vals.append(laplacian_variance(img[y : y + h, x : x + w]))
    grid_h = math.ceil(H / tile)
    grid_w = math.ceil(W / tile)
    heat = np.array(vals, dtype=np.float32).reshape(grid_h, grid_w)
    thr = np.percentile(heat, 20)
    low_ratio = float(np.mean(heat < thr))
    return heat, {"threshold": float(thr), "low_quality_ratio": low_ratio}


def save_heatmap(heat: np.ndarray, out_path: str, title: str = "Image quality score heatmap") -> None:
    if plt is None:
        return
    plt.figure(figsize=(6, 5))
    plt.imshow(heat, cmap="viridis")
    plt.colorbar(label="Laplacian variance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def process_one_image(path: str, configs: List[ChannelSpec], out_dir: str, seed: int = 42, tile: int = 512, prefer_low_res: bool = True, ref_channel: Optional[str] = None, drift_max_tiles: int = 0) -> Dict[str, object]:
    random.seed(seed)
    ensure_dir(out_dir)
    stack = read_qptiff_stack(path, prefer_low_res=prefer_low_res)

    C = stack.shape[0]
    use_configs = [c for c in configs if (c.index - 1) < C]
    dapi_idx = next(((c.index - 1) for c in use_configs if c.is_dapi), 0)
    af_idx = next(((c.index - 1) for c in use_configs if c.is_af), None)
    dapi_idx = dapi_idx if dapi_idx < C else 0
    dapi = stack[dapi_idx]
    import os
    report: Dict[str, Any] = {
        "file": os.path.basename(path),
        "shape": list(stack.shape),
        "drift": {},
        "crosstalk": {},
        "af": {},
        "quality": {},
    }

    H0, W0 = stack.shape[-2], stack.shape[-1]
    tile_eff = tile
    if math.ceil(H0 / tile_eff) < 2 or math.ceil(W0 / tile_eff) < 2:
        tile_eff = max(min(H0, W0) // 2, 64)

    ref_name = (configs[dapi_idx].name if dapi_idx < len(configs) else "DAPI")
    if ref_channel:
        for spec in use_configs:
            if spec.name == ref_channel and (spec.index - 1) < C:
                dapi_idx = spec.index - 1
                dapi = stack[dapi_idx]
                ref_name = spec.name
                break

    for spec in use_configs:
        ci = spec.index - 1
        ch = stack[ci]
        field = drift_vector_field(dapi, ch, tile=tile_eff, max_tiles=int(drift_max_tiles))
        max_disp = float(np.max(np.hypot(field["dx"], field["dy"])) if len(field["dx"]) > 0 else 0.0)
        mean_disp = float(np.mean(np.hypot(field["dx"], field["dy"])) if len(field["dx"]) > 0 else 0.0)
        report["drift"][spec.name] = {"max_px": max_disp, "mean_px": mean_disp}
        overlay = overlay_edges(dapi, ch)
        save_image(overlay, os.path.join(out_dir, f"align_overlay_{ref_name}_vs_{spec.name}.png"))
        plot_drift_quiver(field, os.path.join(out_dir, f"drift_quiver_{ref_name}_vs_{spec.name}.png"), title=f"{ref_name} drift vector field")

    labels = [s.name for s in use_configs]
    corr = spectral_crosstalk_matrix(stack)
    report["crosstalk"]["matrix_summary"] = {
        "min": float(np.min(corr)),
        "max": float(np.max(corr)),
        "mean_abs_offdiag": float(np.mean(np.abs(corr - np.diag(np.diag(corr))))),
    }
    plot_corr_heatmap(corr, labels, os.path.join(out_dir, "crosstalk_heatmap.png"))

    if af_idx is not None and af_idx < C:
        af = stack[af_idx]
        af_norm = normalize_image(af)
        af_corr = af_background_correction(af)
        af_corr_norm = normalize_image(af_corr)
        if plt is not None:
            plt.figure(figsize=(6, 4))
            plt.hist(af_norm.reshape(-1), bins=64, alpha=0.6, label="AF_raw")
            plt.hist(af_corr_norm.reshape(-1), bins=64, alpha=0.6, label="AF_corrected")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "af_hist_compare.png"), dpi=180)
            plt.close()
        med_raw = float(np.median(af_norm))
        med_corr = float(np.median(af_corr_norm))
        drop_ratio = (med_raw - med_corr) / (med_raw + 1e-8)
        report["af"] = {"median_raw": med_raw, "median_corrected": med_corr, "median_drop_ratio": float(drop_ratio)}

    heat, qinfo = quality_heatmap(dapi, tile=tile_eff)
    save_heatmap(heat, os.path.join(out_dir, "quality_heatmap.png"))
    report["quality"] = qinfo

    H, W = dapi.shape
    rects = tile_iter(H, W, tile_eff)
    if len(rects) > 0 and plt is not None:
        random.shuffle(rects)
        import os
        for i, (y, x, h, w) in enumerate(rects[:5]):
            any_ch = stack[random.choice([c.index - 1 for c in use_configs])]
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(dapi[y : y + h, x : x + w], cmap="gray")
            axes[0].set_title("DAPI tile")
            axes[0].axis("off")
            axes[1].imshow(any_ch[y : y + h, x : x + w], cmap="gray")
            axes[1].set_title("Marker tile")
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"qc_tile_{i+1}.png"), dpi=180)
            plt.close(fig)

    import os
    with open(os.path.join(out_dir, "qc_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


@dataclass
class PreprocessStep:
    qc_level: str = "basic"
    seed: int = 42
    options: Dict[str, Any] = None
    name: str = "preprocess"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cancer_type = (self.options or {}).get("cancer_type", DEFAULT_CANCER_TYPE)
        tile_size = int((self.options or {}).get("tile_size", 512))
        output_root = context.get("output_dir", "outputs")
        image_path = context.get("image_path")
        dataset_dir = context.get("dataset_dir")
        prefer_low_res = bool(context.get("prefer_low_res", True))

        configs = get_config(cancer_type)
        if not configs:
            log_error(f"未找到通道配置：{cancer_type}")
            raise RuntimeError(f"缺失通道配置：{cancer_type}")

        images: List[str] = []
        if image_path:
            if not os.path.isfile(image_path):
                log_error(f"单图路径不存在：{image_path}")
                raise FileNotFoundError(image_path)
            images = [image_path]
            log_info("单图模式")
            log_info(f"将处理 1 个图像 -> {image_path}")
        else:
            if not dataset_dir:
                log_error("未提供输入路径：需要 --dataset_dir 或 --image_path")
                raise RuntimeError("缺少输入路径")
            images = list_images(dataset_dir)
            if not images:
                log_warn("未在输入目录中发现图像")
                raise RuntimeError("未发现可处理的图像文件")

        out_root = os.path.join(output_root, cancer_type)
        ensure_dir(out_root)

        log_info("开始批量处理")
        log_info(f"发现 {len(images)} 个图像。输出根目录：{out_root}")

        reports: Dict[str, Any] = {}
        failed: List[str] = []
        for idx, img_path in enumerate(images, 1):
            sample_name = os.path.splitext(os.path.basename(img_path))[0]
            out_dir = os.path.join(out_root, sample_name)
            ensure_dir(out_dir)
            log_info(f"[{idx}/{len(images)}] 处理 {img_path} → {out_dir}")
            try:
                rep = process_one_image(
                    img_path,
                    configs,
                    out_dir,
                    seed=self.seed,
                    tile=tile_size,
                    prefer_low_res=prefer_low_res,
                    ref_channel=(context.get("ref_channel") if isinstance(context.get("ref_channel"), str) else None),
                    drift_max_tiles=int(context.get("drift_max_tiles", 0)),
                )
                reports[sample_name] = rep
            except Exception as e:
                log_error("处理失败")
                log_error(f"{img_path}，错误：{e}")
                failed.append(img_path)
                continue

        log_info("预处理与QC完成")

        return {
            "preprocess": {
                "qc_level": self.qc_level,
                "seed": self.seed,
                "cancer_type": cancer_type,
                "count": len(images),
                "success": len(reports),
                "failed": failed,
                "out_root": out_root,
                "reports": reports,
            }
        }


@dataclass
class PreprocessStepSkeleton:
    """当真实实现不可用时的骨架占位，保证 API 可用。"""
    qc_level: str = "basic"
    seed: int = 42
    options: Dict[str, Any] = None
    name: str = "preprocess_skeleton"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log_info("运行预处理骨架：记录参数并返回占位产物")
        # 占位产出，可用于后续步骤测试
        return {
            "preprocess": {
                "qc_level": self.qc_level,
                "seed": self.seed,
                "output_dir": context.get("output_dir", "outputs"),
            }
        }