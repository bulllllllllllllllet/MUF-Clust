# MUF-Clust 架构与函数管理原则（对齐 guide.md）

面向多重荧光图像无监督聚类分析的可扩展分层架构，强调单一职责、可插拔、配置驱动与可观测性（日志/QC/可视化）。本架构与 `guide.md` 的流程一一对应，确保后续算法与工程迭代有统一规范可依。

## 设计原则

- 单一职责与清晰边界：每个模块只做一件事，避免耦合。
- 面向对象与管线化：步骤（Step）封装处理逻辑，管线（Pipeline）负责编排。
- 配置驱动：行为由 YAML/CLI/环境变量驱动，代码逻辑尽量无硬编码。
- 插件化扩展：第三方/可选能力（如 Cellpose、Mesmer、光谱去混）通过插件层注册与启用。
- API 稳定：暴露少量高层 API（`api/`），内部实现可演进。
- 可观测性：统一日志（中文调试、分隔标记）、丰富 QC 可视化（绘图使用英文）。
- 复现友好：种子统一、版本/环境记录、流程参数持久化。

## 分层与目录结构（src/muf_clust）

- `api/` 高层稳定 API（供 CLI/脚本/前端调用）
  - `pipeline_api.py`：`run_preprocess`、`run_full_pipeline` 等入口
- `core/` 核心管线与步骤（对应 guide.md 章节）
  - `pipeline.py`：`Pipeline` 与 `Step` 抽象
  - `steps/`：`preprocess.py`、`segmentation.py`、`features.py`、`cluster.py`、`visualize.py`
- `plugins/` 插件层（可选能力）
  - `base.py` 抽象基类、`registry.py` 注册机制（Cellpose/Mesmer/spectral unmixing 预留）
- `utils/` 工具层
  - `logging.py` 统一日志（分隔标记、中文调试输出）、`paths.py` 路径与命名规范
- `integrations/` 外部集成
  - `wandb.py` 训练/实验记录；`fiftyone.py` 错误样本分析
- `configs/` 配置层（示例 YAML）
  - `defaults.yaml`、`channels_kidney.yaml`、`channels_bladder.yaml`
- `cli.py`/脚本层：命令行入口；保留现有 CLI，与 `api/` 逐步对齐
- 现有 `preprocess.py`：保持兼容，逐步迁移到 `core/steps/preprocess.py`

## 命名与注释规范

- 代码/类/函数使用英文命名；绘图中的标签与图例使用英文。
- 代码注释与终端调试输出统一使用中文。
- 控制台日志使用分隔标记：
  
  ```
  _____
  [INFO] 开始处理 sample_001.tiff
  _____
  ```

## 面向对象抽象

- `Step`：统一步骤接口，包含 `name` 与 `run(context)` 方法。
- `Pipeline`：接受步骤列表，顺序执行，统一日志与上下文传递。
- `Plugin`：统一插件接口，`apply(data, config)` + `supports(task)` + `describe()`。

## 配置管理

- 配置来源优先级：环境变量 > CLI 参数 > 项目 YAML > 内置 `defaults.yaml`。
- 通道映射通过 `channels_*.yaml` 统一；新增面板按同规范扩展。
- 关键开关与默认值对齐 `guide.md`（如 `qc`、`af_mode`、`spectral_unmix`、`seed`）。

## API 设计（稳定层）

- `run_preprocess(image_path|dataset_dir, output_dir, **options)`：仅执行 2.2 预处理与 QC。
- `run_full_pipeline(input_path, output_dir, config)`：执行 3.x～5.x 全流程并输出结果与报告。

返回值以字典形式包含：路径、统计、参数摘要、可视化产物位置。

## 与 guide.md 的步骤映射

- 2.2 预处理/QC → `core/steps/preprocess.py`
- 3.1/3.2 分割与细胞区域 → `core/steps/segmentation.py`（支持插件：Cellpose/Mesmer）
- 3.3/3.4 特征提取与标准化 → `core/steps/features.py`
- 4.x 降维/聚类/评价 → `core/steps/cluster.py`
- 5.x 输出与可视化 → `core/steps/visualize.py`

## 观测与记录

- 日志：`utils.logging` 提供 `log_info/log_debug/log_warn/log_error`，默认中文输出与分隔块格式。
- WandB：使用 `integrations.wandb` 记录指标曲线与可视化快照。
- FiftyOne：使用 `integrations.fiftyone` 汇总错误样本与问题 tile。

## 渐进迁移计划

- 现有 `preprocess.py` 保持稳定；新增的 `core/steps/preprocess.py` 先提供骨架与统一日志。
- 后续将逐步将现有实现拆分到 `steps/` 文件夹，并经 `api/` 暴露。

## 输出目录与命名

- 参照 `guide.md 5.4`，由 `utils.paths` 统一生成：前缀包含 `{sample_id}`，使用两位编号排序，图用 PNG，报告用 Markdown/JSON。

## 快速使用示例（概念性）

```python
from muf_clust.api.pipeline_api import run_preprocess, run_full_pipeline

run_preprocess(image_path=".../sample_001.tiff", output_dir="./outputs", qc="basic")
run_full_pipeline(input_path="/nfs5/yj/MIHC/dataset/KidneyCancer", output_dir="./outputs", config_path="src/muf_clust/configs/defaults.yaml")
```

> 注：当前提交为骨架与规范文档，避免影响既有运行；后续将逐步将现有逻辑迁移到分层结构中。