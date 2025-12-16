#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""根据单图聚类结果 CSV 生成 Point JSON 数组。

输入 CSV 要包含以下字段：
- tile_x: tile 左上角在整图坐标中的 x（像素）
- tile_y: tile 左上角在整图坐标中的 y（像素）
- local_x: 细胞在该 tile 内的局部 x（像素）
- local_y: 细胞在该 tile 内的局部 y（像素）
- cluster: 聚类类别（整数）

全局坐标计算公式：
    x = tile_x + local_x
    y = tile_y + local_y

输出为一个 JSON 数组，每个元素形如：
{
  "geometry": {
    "type": "Point",
    "coordinates": [x, y]
  },
  "properties": {
    "label": "类别<cluster值>",
    "note": "<K>分类结果"  # K 为该 CSV 中聚类类别总数
  }
}

用法示例：

    uv run cluster_csv_to_points_json.py \
        --input /nfs5/zyh/MUF-Clust/single_image_output/segmentation/11_Scan1/features/features_matrix_mean_cluster.csv \
        --output /nfs5/zyh/MUF-Clust/single_image_output/segmentation/11_Scan1/features/features_points.json
"""

from __future__ import annotations

# segmentation 模式用法示例：
# uv run scripts/cluster_csv_to_points_json.py \
#   --input /path/to/features_matrix_mean_cluster.csv \
#   --mode segmentation \
#   --output /path/to/segmentation_output
# 将会生成：
#   /path/to/segmentation_output_classes.json
#   /path/to/segmentation_output_points.csv

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="根据聚类结果 CSV 生成 Point JSON 数组")
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="输入的聚类结果 CSV 路径 (例如 *_cluster.csv)",
    )
    p.add_argument(
        "--output",
        "-o",
        required=False,
        default=None,
        help=(
            "输出基础路径: "
            "points 模式下为 NDJSON/JSONLines（默认与输入同名，扩展名改为 _points.ndjson）；"
            "segmentation 模式下会在此基础上生成 _classes.json 和 _points.csv 两个文件"
        ),
    )
    p.add_argument(
        "--mode",
        choices=["points", "segmentation"],
        default="points",
        help="输出模式: points 为 Point NDJSON, segmentation 为分割结果所需的 classes.json 和 points.csv",
    )
    return p.parse_args()


def _infer_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return base + "_points.ndjson"


def load_rows(csv_path: str) -> tuple[List[Dict[str, Any]], Set[int]]:
    rows: List[Dict[str, Any]] = []
    clusters: Set[int] = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"tile_x", "tile_y", "local_x", "local_y", "cluster"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少必要列: {', '.join(sorted(missing))}")

        for row in reader:
            try:
                tile_x = float(row["tile_x"])
                tile_y = float(row["tile_y"])
                local_x = float(row["local_x"])
                local_y = float(row["local_y"])
                cluster = int(row["cluster"])
            except Exception as e:
                raise ValueError(f"解析行失败: {row!r}, 错误: {e}")

            x = tile_x + local_x
            y = tile_y + local_y

            rows.append({
                "x": x,
                "y": y,
                "cluster": cluster,
            })
            clusters.add(cluster)

    return rows, clusters


def build_features(rows: List[Dict[str, Any]], clusters: Set[int]) -> List[Dict[str, Any]]:
    k = len(clusters)
    note = f"k={k}聚类点"

    # 为每个聚类类别分配一个较为区分的颜色
    base_colors = [
        "#E41A1C",  # red
        "#377EB8",  # blue
        "#4DAF4A",  # green
        "#984EA3",  # purple
        "#FF7F00",  # orange
        "#FFFF33",  # yellow
        "#A65628",  # brown
        "#F781BF",  # pink
        "#999999",  # gray
    ]
    sorted_clusters = sorted(clusters)
    num_colors = len(base_colors)

    cluster_meta: Dict[int, Dict[str, Any]] = {}
    for idx, c in enumerate(sorted_clusters):
        color = base_colors[idx % num_colors]
        # class_value: 使用 CSV 中的聚类值；class_name: 直接用该类别值的字符串表示
        cluster_meta[c] = {
            "class_value": int(c),
            "class_name": str(c),
            "class_color": color,
        }

    features: List[Dict[str, Any]] = []
    for r in rows:
        c = int(r["cluster"])
        x = float(r["x"])
        y = float(r["y"])

        meta = cluster_meta[c]

        features.append({
            "geometry": {
                "type": "Point",
                "coordinates": [x, y],
            },
            "properties": {
                "class_value": meta["class_value"],
                "class_name": meta["class_name"],
                "class_color": meta["class_color"],
                "note": note,
            },
        })
    return features


def main() -> None:
    args = parse_args()
    input_csv = os.path.abspath(args.input)

    if not os.path.isfile(input_csv):
        raise SystemExit(f"输入 CSV 不存在: {input_csv}")

    output_json = os.path.abspath(args.output or _infer_output_path(input_csv))

    rows, clusters = load_rows(input_csv)
    if not rows:
        raise SystemExit("输入 CSV 为空或没有有效数据行")

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)

    if args.mode == "points":
        # points 模式：保持原有 NDJSON 行为
        features = build_features(rows, clusters)
        with open(output_json, "w", encoding="utf-8") as f:
            for feature in features:
                line = json.dumps(feature, ensure_ascii=False)
                f.write(line)
                f.write("\n")
        print(f"已生成 JSON: {output_json}, 点数量={len(features)}, 聚类类别数={len(clusters)}")
    else:
        # segmentation 模式：输出两个文件
        base, _ext = os.path.splitext(output_json)
        classes_path = base + "_classes.json"
        points_path = base + "_points.csv"

        # 生成 classes 列表，value / name / color
        base_colors = [
            "#E41A1C",  # red
            "#377EB8",  # blue
            "#4DAF4A",  # green
            "#984EA3",  # purple
            "#FF7F00",  # orange
            "#FFFF33",  # yellow
            "#A65628",  # brown
            "#F781BF",  # pink
            "#999999",  # gray
        ]
        sorted_clusters = sorted(clusters)
        num_colors = len(base_colors)

        classes: List[Dict[str, Any]] = []
        for idx, c in enumerate(sorted_clusters):
            color = base_colors[idx % num_colors]
            classes.append({
                "value": int(c),
                "name": str(c),
                "color": color,
            })

        with open(classes_path, "w", encoding="utf-8") as f:
            json.dump(classes, f, ensure_ascii=False, indent=2)

        # 生成 points CSV: pixel_value,x,y
        with open(points_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pixel_value", "x", "y"])
            for r in rows:
                writer.writerow([
                    int(r["cluster"]),
                    float(r["x"]),
                    float(r["y"]),
                ])

        print(
            f"已生成 segmentation 输出: classes={classes_path}, points={points_path}, 点数量={len(rows)}, 聚类类别数={len(clusters)}",
        )


if __name__ == "__main__":
    main()
