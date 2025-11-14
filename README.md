# MUF-Clust

本仓库用于 MUF-Clust 项目的版本管理与协作。当前为初始提交，占位 README。

## 架构与管理原则

请阅读 `ARCHITECTURE.md` 获取完整分层架构、函数管理原则与与 `guide.md` 的对应关系。核心思想：
- 分层目录：`api/`（稳定接口）、`core/`（管线与步骤）、`plugins/`（可插拔能力）、`utils/`（日志/路径）、`integrations/`（WandB/FiftyOne）、`configs/`（YAML 配置）。
- 日志：控制台输出采用分隔标记；绘图使用英文；注释与调试输出使用中文。
- API：通过 `muf_clust.api.pipeline_api` 提供 `run_preprocess`、`run_full_pipeline`。

## 初始化说明
- 本地已初始化 Git 仓库。
- 后续将逐步补充项目代码与文档。

## 预处理（第一步，依据 guide.md 2.2）

功能涵盖：通道对齐检查、光谱混叠相关矩阵、自发荧光（AF）直方图与背景扣除示例、图像清晰度评分热图。

### 运行依赖

请在运行环境安装如下Python包：
- `numpy`
- `tifffile`
- `scikit-image`
- `opencv-python`
- `matplotlib`
- `seaborn`

### 使用示例

1) 肾癌数据（KidneyCancer）：

```
python -m muf_clust.cli \
  --dataset_dir /nfs5/yj/MIHC/dataset/KidneyCancer \
  --cancer_type KidneyCancer \
  --output_dir outputs/preprocess \
  --tile_size 512
```

2) 膀胱癌数据（BladderCancer）：

```
python -m muf_clust.cli \
  --dataset_dir /nfs5/yj/MIHC/dataset/BladderCancer \
  --cancer_type BladderCancer \
  --output_dir outputs/preprocess \
  --tile_size 512
```

3) 仅处理单张图像（便于快速试跑与验证）：

```
python -m muf_clust.cli \
  --image_path /nfs5/yj/MIHC/dataset/KidneyCancer/sample_001.tiff \
  --cancer_type KidneyCancer \
  --output_dir outputs/preprocess \
  --tile_size 512
```

输出包含：
- `align_overlay_{channel}.png`：DAPI与通道边缘叠加图，用于目视对齐检查。
- `drift_quiver_{channel}.png`：按tile估计的漂移矢量场（建议阈值：平均 ≤ 2 px，最大 ≤ 3 px）。
- `crosstalk_heatmap.png`：通道相关矩阵热图（期望非对角项绝对值 ≤ 0.1）。
- `af_hist_compare.png`：AF校正前后直方图对比（中位数下降比例 ≥ 30%）。
- `quality_heatmap.png`：Laplacian方差热图与低质量比例统计（建议 ≤ 10%）。
- `qc_report.json`：以上指标的数值汇总。

注意：当前为QC与预处理演示实现，qptiff实际通道组织可能需要结合元数据进一步适配。

## 配置示例

- 默认：`src/muf_clust/configs/defaults.yaml`
- 通道映射：`src/muf_clust/configs/channels_kidney.yaml`、`src/muf_clust/configs/channels_bladder.yaml`

## 高层 API 使用（示例）

```python
from muf_clust.api.pipeline_api import run_preprocess
run_preprocess(image_path="/nfs5/yj/MIHC/dataset/KidneyCancer/sample_001.tiff", output_dir="outputs")
```