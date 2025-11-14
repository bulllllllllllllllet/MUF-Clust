import argparse
import sys

from ..config import get_runtime_defaults, DEFAULT_CANCER_TYPE
from .pipeline_api import run_preprocess
from ..utils.logging import log_info, log_warn, log_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MUF-Clust 预处理与QC")
    p.add_argument("--dataset_dir", required=False, help="数据集目录（包含 qptiff/tiff）")
    p.add_argument("--image_path", required=False, help="单图路径（qptiff/tiff），提供则仅处理该图像")
    p.add_argument("--cancer_type", required=False, default=DEFAULT_CANCER_TYPE, choices=["KidneyCancer", "BladderCancer"], help="通道配置类别")
    p.add_argument("--output_dir", default=None, help="输出根目录（不提供则使用配置默认）")
    p.add_argument("--tile_size", type=int, default=None, help="QC按tile大小（像素，不提供则使用配置默认）")
    p.add_argument("--seed", type=int, default=None, help="随机种子（不提供则使用配置默认）")
    return p.parse_args()


def main():
    args = parse_args()
    defaults = get_runtime_defaults(args.cancer_type)
    args.dataset_dir = args.dataset_dir or defaults.dataset_dir
    args.image_path = args.image_path or defaults.image_path
    args.output_dir = args.output_dir or defaults.output_root
    args.tile_size = args.tile_size or defaults.tile_size
    args.seed = args.seed or defaults.seed
    try:
        result = run_preprocess(
            image_path=args.image_path,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            qc="basic",
            seed=args.seed,
            options={"cancer_type": args.cancer_type, "tile_size": args.tile_size},
        )
    except Exception as e:
        log_error("执行失败")
        log_error(f"错误：{e}")
        sys.exit(2)

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


if __name__ == "__main__":
    main()