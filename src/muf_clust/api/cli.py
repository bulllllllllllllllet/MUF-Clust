import argparse
import sys

from ..config import get_runtime_defaults, DEFAULT_CANCER_TYPE
from .pipeline_api import run_preprocess, run_full_pipeline, run_segmentation, run_features, run_features_folder
from ..utils.logging import log_info, log_warn, log_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MUF-Clust 预处理与QC")
    p.add_argument("--dataset_dir", required=False, help="数据集目录（包含 qptiff/tiff）")
    p.add_argument("--image_path", required=False, help="单图路径（qptiff/tiff），提供则仅处理该图像")
    p.add_argument("--cancer_type", required=False, default=DEFAULT_CANCER_TYPE, choices=["KidneyCancer", "BladderCancer"], help="通道配置类别")
    p.add_argument("--output_dir", default=None, help="输出根目录（不提供则使用配置默认）")
    p.add_argument("--tile_size", type=int, default=None, help="QC按tile大小（像素，不提供则使用配置默认）")
    p.add_argument("--seed", type=int, default=None, help="随机种子（不提供则使用配置默认）")
    p.add_argument("--high_res", action="store_true", default=True, help="使用高分辨率层进行QC（默认使用高分辨率层）")
    p.add_argument("--ref_channel", default="DAPI", help="参考通道名（默认 DAPI）")
    p.add_argument("--drift_max_tiles", type=int, default=0, help="限制漂移计算的最大tile数（0表示不限制）")
    p.add_argument("--cellpose_model", required=False, help="Cellpose 预训练模型（'cyto'/'nuclei' 或绝对路径）")
    p.add_argument("--cellpose_gpu", dest="cellpose_gpu", action="store_true", default=True, help="强制启用GPU进行Cellpose推理（默认开启，GPU不可用则报错）")
    p.add_argument("--no_cellpose_gpu", dest="cellpose_gpu", action="store_false", help="禁用GPU进行Cellpose推理")
    p.add_argument("--only_tile_x", type=int, default=None, help="仅处理指定tile的x坐标")
    p.add_argument("--only_tile_y", type=int, default=None, help="仅处理指定tile的y坐标")
    p.add_argument("--cellpose_diameter", type=float, default=None, help="Cellpose分割直径（像素，提供可显著加速）")
    p.add_argument("--cellpose_batch_size", type=int, default=None, help="Cellpose推理批大小（默认由库决定，如8）")
    p.add_argument("--seg_out_dir", required=False, help="分割输出目录（features-only）")
    p.add_argument("--roi_tiles_dir", required=False, help="ROI标签tiles目录（features-only）")
    p.add_argument("--num_workers", type=int, default=1, help="并行处理的workers数量（分割与特征均生效，>=1）")
    p.add_argument("--run", choices=["preprocess", "segmentation", "features", "full"], default="preprocess", help="运行模式：仅预处理、仅分割、仅特征或完整管线")
    return p.parse_args()


def main():
    args = parse_args()
    defaults = get_runtime_defaults(args.cancer_type)
    orig_image_path = args.image_path
    orig_dataset_dir = args.dataset_dir
    args.dataset_dir = args.dataset_dir or defaults.dataset_dir
    args.image_path = args.image_path or defaults.image_path
    args.output_dir = args.output_dir or defaults.output_root
    args.tile_size = args.tile_size or defaults.tile_size
    args.seed = args.seed or defaults.seed
    try:
        if args.run == "preprocess":
            result = run_preprocess(
                image_path=args.image_path,
                dataset_dir=args.dataset_dir,
                output_dir=args.output_dir,
                qc="basic",
                seed=args.seed,
                options={"cancer_type": args.cancer_type, "tile_size": args.tile_size, "high_res": bool(getattr(args, "high_res", False)), "ref_channel": args.ref_channel, "drift_max_tiles": args.drift_max_tiles, "cellpose_gpu": args.cellpose_gpu, "cellpose_diameter": args.cellpose_diameter, "cellpose_batch_size": args.cellpose_batch_size, "only_tile_x": args.only_tile_x, "only_tile_y": args.only_tile_y},
            )
            prep = result.get("preprocess", {})
            count = prep.get("count", 0)
            out_root = prep.get("out_root", args.output_dir)
            failed = prep.get("failed", [])
            log_info("预处理与QC完成")
            log_info(f"总计 {count} 个图像，输出：{out_root}")
            if failed:
                log_warn("存在处理失败项")
                for p in failed:
                    log_warn(f"失败：{p}")
        elif args.run == "segmentation":
            input_path = args.image_path or args.dataset_dir
            result = run_segmentation(
                input_path=input_path,
                output_dir=args.output_dir,
                options={"cancer_type": args.cancer_type, "high_res": bool(getattr(args, "high_res", False)), "tile_size": args.tile_size, "cellpose_model": args.cellpose_model, "cellpose_gpu": args.cellpose_gpu, "cellpose_diameter": args.cellpose_diameter, "cellpose_batch_size": args.cellpose_batch_size, "only_tile_x": args.only_tile_x, "only_tile_y": args.only_tile_y, "num_workers": args.num_workers},
            )
            seg = result.get("segmentation", {})
            out_dir = seg.get("out_dir", args.output_dir)
            log_info("分割步骤执行完成")
            log_info(f"分割输出：{out_dir}")
        elif args.run == "features":
            folder_mode = bool(orig_dataset_dir) and not bool(orig_image_path)
            if folder_mode:
                if not args.seg_out_dir:
                    log_error("缺少分割输出目录 seg_out_dir（文件夹模式）")
                    sys.exit(2)
                result = run_features_folder(
                    dataset_dir=args.dataset_dir,
                    seg_root=args.seg_out_dir,
                    options={
                        "cancer_type": args.cancer_type,
                        "prefer_low_res": not bool(getattr(args, "high_res", False)),
                        "only_tile_x": args.only_tile_x,
                        "only_tile_y": args.only_tile_y,
                        "num_workers": args.num_workers,
                    },
                )
                feats = result.get("features", {})
                log_info("特征步骤执行完成（文件夹模式）")
                log_info(f"seg_root：{feats.get('root')}")
                log_info(f"数据集目录：{feats.get('dataset_dir')}")
                log_info(f"总图像数：{feats.get('images')}，实际处理：{feats.get('processed')}，细胞总数：{feats.get('cells')}")
                failed = feats.get("failed") or []
                if failed:
                    log_warn("存在特征提取失败/缺少 ROI 的样本")
                    for p in failed:
                        log_warn(f"失败：{p}")
            else:
                if not args.seg_out_dir:
                    log_error("缺少分割输出目录 seg_out_dir")
                    sys.exit(2)
                roi_dir = args.roi_tiles_dir or (args.seg_out_dir.rstrip("/") + "/roi_tiles")
                result = run_features(
                    image_path=args.image_path,
                    seg_out_dir=args.seg_out_dir,
                    roi_tiles_dir=roi_dir,
                    options={"cancer_type": args.cancer_type, "prefer_low_res": not bool(getattr(args, "high_res", False)), "only_tile_x": args.only_tile_x, "only_tile_y": args.only_tile_y, "num_workers": args.num_workers},
                )
                feats = result.get("features", {})
                log_info("特征步骤执行完成")
                log_info(f"特征输出目录：{feats.get('features_dir')}")
                log_info(f"长表CSV：{feats.get('features_long_csv')}")
                log_info(f"矩阵CSV：{feats.get('features_matrix_csv')}")
                log_info(f"细胞数：{feats.get('cells')}")
        else:
            input_path = args.image_path or args.dataset_dir
            result = run_full_pipeline(
                input_path=input_path,
                output_dir=args.output_dir,
                seed=args.seed,
                options={"cancer_type": args.cancer_type, "high_res": bool(getattr(args, "high_res", False)), "ref_channel": args.ref_channel, "drift_max_tiles": args.drift_max_tiles, "tile_size": args.tile_size, "cellpose_model": args.cellpose_model, "cellpose_gpu": args.cellpose_gpu, "cellpose_diameter": args.cellpose_diameter, "cellpose_batch_size": args.cellpose_batch_size, "only_tile_x": args.only_tile_x, "only_tile_y": args.only_tile_y},
            )
            seg = result.get("segmentation", {})
            out_dir = seg.get("out_dir", args.output_dir)
            log_info("完整管线执行完成")
            log_info(f"分割输出：{out_dir}")
    except Exception as e:
        log_error("执行失败")
        log_error(f"错误：{e}")
        sys.exit(2)


if __name__ == "__main__":
    main()