/nfs5/yj/MIHC/dataset/KidneyCancer 路径下现在存放着肾癌多重荧光的qptiff图以及对应的ppr图片
/nfs5/yj/MIHC/dataset/BladderCancer 路径下现在存放着膀胱癌多重荧光的qptiff图以及对应的ppr图片
下面是 **可直接交付的最终版需求文档**，
我已经将“通道生物学含义”留出了表格占位，后续你只需要把 marker 名称填进去即可。

---

# 多重荧光图像无监督聚类分析需求文档（最终版）

## 1. 项目概述

### 1.1 项目目标

对 qptiff 格式的多重免疫荧光（mIF）全视野图像进行无监督细胞聚类分析。
通过 DAPI 通道检测细胞核，基于各荧光通道的强度构建细胞特征，进行聚类与可视化，最终用于探索组织中的细胞群体组成与差异。

### 1.2 技术路线（总流程）

DAPI 细胞核检测 → 细胞区域定义 → 多通道特征提取 → 特征标准化 → 无监督聚类 → 可视化与结果导出

---

## 2. 数据要求

### 2.1 输入数据

| 项目   | 说明                               |
| ---- | -------------------------------- |
| 图像格式 | qptiff 多通道荧光图像                   |
| 图像来源 | 多重荧光扫描系统（如 Akoya Vectra Polaris） |
| 分辨率  | 支持单细胞级分析的物镜扫描分辨率                 |
| 元数据  | 包含每个通道对应的染料 / 抗体标签信息             |

#### 通道配置表（肾癌）

| 通道编号 | 通道名称（示例：Opal520） | Marker（抗体） | 生物学含义 | 备注   |
| -------- | ------------------------- | -------------- | ---------- | ------ |
| 1        | DAPI                      | -              | 核染色     | 已固定 |
| 2        | Opal480                   | CD20           | 细胞质     |        |
| 3        | Opal520                   | CD3            | 细胞膜     |        |
| 4        | Opal570                   | FOXP3          | 细胞核     |        |
| 5        | Opal620                   | CD23           | 细胞核     |        |
| 6        | Opal690                   | CD8            | 细胞膜     |        |
| 7        | Opal780                   | PAX8s          | 细胞核     |        |
| 8        | SampleAF                  | AF             | 细胞自发荧光（背景） | 默认不参与聚类；经数据检验可并入 |

#### 通道配置表（膀胱癌）

| 通道编号 | 通道名称（示例：Opal520） | Marker（抗体） | 生物学含义 | 备注   |
| -------- | ------------------------- | -------------- | ---------- | ------ |
| 1        | DAPI                      | -              | 核染色     | 已固定 |
| 2        | Opal480                   | PANCK          | 细胞质     |        |
| 3        | Opal520                   | IgM            | 细胞膜     |        |
| 4        | Opal570                   | FOXP3          | 细胞核     |        |
| 5        | Opal620                   | CD4            | 细胞膜     |        |
| 6        | Opal690                   | PD-L1          | 细胞膜     |        |
| 7        | Opal780                   | CXCL13         | 细胞核     |        |
| 8        | SampleAF                  | AF             | 细胞自发荧光（背景） | 默认不参与聚类；经数据检验可并入 |

> 2.2 数据预处理要求

* 通道对齐检查（如存在漂移需配准）
* 光谱混叠处理（若数据已 unmix 可跳过）
* 自发荧光去除（可选）
* 图像质量评分（模糊/曝光异常检测）

可视化检验产出（QC）：

* 通道对齐叠加图：DAPI 与各通道的边缘/特征叠加，量化像素漂移（阈值 ≤ 2 px；超过则建议配准）。
* 通道漂移矢量场：按 tile 计算位移向量，展示最大/平均位移（最大位移 ≤ 3 px）。
* 光谱混叠矩阵热图：unmix 前后通道相关系数/串扰矩阵，非对角项期望 ≤ 0.1。
* 自发荧光去除前后直方图：AF 通道强度分布对比，中位数下降比例 ≥ 30%。
* 图像质量评分热图：基于 Laplacian 方差或 Brenner 梯度的清晰度分数，低质量 tile 比例 ≤ 10%。
* 抽样策略：每样本随机抽取 5 个 tile 用于 QC 展示（固定随机种子以保证复现）。

### 2.3 SampleAF 使用策略与数据驱动决策

结论（一句话）：优先将 SampleAF 作为“校正与质控变量”，而非默认并入聚类特征；但通过数据检验其对生物分群的贡献后再决定是否纳入并比较效果。

不宜直接并入的理由（反对）：
- AF 常为背景信号（基质/胶原/脂褐素等），可能淹没真实标记或产生假相关。
- AF 强度与切片厚度、曝光、批次、组织类型相关，易引入技术性群聚而非生物学群聚。

可能需要并入的理由（支持）：
- AF 能区分组织类型或病变区（疤痕/基质 vs 肿瘤），有助解释空间结构。
- AF 作为特征可辅助识别伪阳性或坏片段（artifact class）。

推荐的流程（数据驱动）：
- 可视化：整图 SampleAF 热图，检查空间分布（局部集中/结构重合/全局噪声）。
- 细胞级量化：为每细胞计算 `AF_raw`、`AF_background_corrected`、`AF_norm`（可按核周区域归一）。
- 相关性分析：计算 AF 与各标记通道及形态特征的 Spearman 相关；若 |ρ| ≥ 0.3 且 p < 0.01，提示严重相关需谨慎。
- 两套聚类并比较：
  * A：不含 AF 的特征集合 → 评估轮廓系数、Calinski-Harabasz、聚类大小分布
  * B：含 AF 的特征集合 → 同样评估
  * 比较：若 B 明显提高质量且更易生物学解释（如可明确分出基质 vs 肿瘤），则可并入，否则剔除。
- AF 作为校正/权重变量（备选）：对其它通道做背景校正（subtract/regress out），或在差异分析/可视化中作为协变量调整。

具体判定规则（可落地）：
- 若 AF 与任一通道 Spearman ρ ≥ 0.3 且 p < 0.01 → 强制执行 spectral unmixing / 背景校正；并进行含/不含 AF 的聚类对比。
- 若包含 AF 后轮廓系数或 CH 指数提升 ≥ 5%，且聚类更贴近已知生物学（实验人员/病理图验真）→ 可并入。
- 否则不直接并入聚类特征，但将 AF 写入输出表作为 QC 字段与可视化层。

实现建议（伪代码）：

`# 1. 计算 AF（每细胞）\nAF_cell = mean(AF_pixels_in_cell) - local_background\n\n# 2. 相关性\nfor ch in marker_channels:\n    rho, p = spearmanr(AF_cell, ch_cell)\n\n# 3. 聚类对比（含/不含 AF）\nclusters_without_AF = cluster(features_without_AF)\nclusters_with_AF = cluster(features_with_AF)\ncompare_metrics(clusters_without_AF, clusters_with_AF)`

额外注意：
- 若已完成 spectral unmixing 且 AF 分离为独立组件，则 AF 值更可靠；若未 unmix，AF 往往污染其它通道，建议先 unmix。
- AF 极高且所有 marker 同时偏高，可能为坏区域（burn/overstain）—可作为敏感指标标注。

### 2.4 术语解释：Spectral Unmixing（光谱去混）

- 现象：不同染料的发射光谱存在重叠，导致通道间串扰（crosstalk），使同一像素的观测强度是多个光谱的线性混合。
- 做什么：基于已知或估计的光谱响应矩阵，用线性/非线性去混算法（常见为线性去混）将混合信号分解为各染料的成分，降低通道污染。
- 为什么重要：能提高标记通道的特异性与定量准确度，减少假相关；AF 组件在 unmix 后更稳定。
- 何时建议：当通道相关性/串扰高（如 |ρ| ≥ 0.3 且 p < 0.01），或观察到明显的跨通道信号泄露时，建议执行 unmix；若输入已完成 unmix，应保留标记并跳过重复处理。

---

## 3. 核心处理流程

### 3.1 细胞核检测（Cellpose）

#### 3.1.1 DAPI预处理（可选）

* 高斯滤波去噪（σ≈1.0）
* 局部对比度增强（CLAHE）
* 背景扣除
* 强度归一化到 [0,1]

#### 3.1.2 Tile 切分与 Cellpose 推理

* 切分：将整图按 `1024×1024` 固定大小切分为 tiles（边缘不足的 tile 按实际尺寸保留）。
* 推理：对每个 tile 使用 Cellpose nuclei 模型（单通道 DAPI）进行核分割；支持 GPU，直径 `diameter=auto`，其余参数按数据集经验值配置。
* 合并：将各 tile 的分割结果映射回整图坐标系，保证标签唯一性；过滤面积较小的碎片（例如 < 20 像素）。

#### 3.1.3 核形态特征记录

* 核质心 (x,y)
* 边界框
* 面积、周长、等效直径
* 椭圆拟合长短轴
 
#### 3.1.4 QC 可视化与阈值建议
 
* dapi_raw.png：DAPI 原始图概览（缩略图 + 局部放大）。
* dapi_preprocessed.png：预处理前后对比（高斯滤波、CLAHE、背景扣除）。
* nuclei_mask_cellpose.png：Cellpose 分割掩码展示（整图或 tile 拼接）。
* nuclei_edge_overlay_cellpose.png：Cellpose 边缘叠加到原始 DAPI 图，检查过/欠分割。
* 模型记录：保存 Cellpose 版本与关键参数（如 diameter、flow/threshold）。
* 指标阈值：
  - 核计数与密度：与经验范围比对（异常密度需复核）。
  - 碎片比例（面积 < 50 像素）：≤ 5%。
  - 抽样 IoU（人工标注对比）：≥ 0.7。

### 3.2 细胞区域定义

基于 3.1 的 Cellpose 核分割结果，提供两种可配置方案：

| 模式    | 描述                            | 适用场景         |
| ----- | ----------------------------- | ------------ |
| 固定半径法 | 区域半径 = 核直径的 1.5–2 倍，按核为种子扩展 | 简单、速度快       |
| 细胞级分割 | 使用 Cellpose whole-cell 模型获取细胞轮廓 | 对膜标记/胞质信号更准确 |

实现约定：整图同样按 `1024×1024` tiles 进行区域推理与合并，保证坐标与标签一致性。

重叠区域处理：若两个细胞区域重叠 > 30%，采用 Voronoi 分割保证每像素只归属一个细胞。
 
#### 3.2.1 QC 可视化与阈值建议
 
* cell_regions_overlay.png：细胞区域边界叠加到原图（随机抽样 100–500 个细胞）。
* overlap_heatmap.png：像素归属冲突热图，标注重叠>30%的热点区域。
* radius_vs_nucleus.png：固定半径法的半径与核直径关系散点图（期望半径在 1.5–2×）。
* 模式一致性：在 Cellpose 模式下，边界与膜/胞质信号对齐误差（平均偏移 ≤ 3 px）。

### 3.3 多通道强度特征提取

对每个细胞在每个非 DAPI 通道提取：

* 平均强度
* 中值强度
* 总强度（可选）
* 背景校正强度
* 标准差（可选）
 
#### 3.3.1 QC 可视化
 
* per_channel_hist_{ch}.png：每通道强度直方图（细胞内 vs 背景对比）。
* per_channel_violin_{marker}.png：按 Marker 的细胞内强度分布（支持子群抽样）。
* roi_bg_boxplot_{ch}.png：细胞 ROI 与邻域背景强度箱线图，检验背景校正是否有效（ROI 中位数应显著高于背景）。

### 3.4 特征向量与标准化

* 构建 **N × C** 细胞特征矩阵（C=通道数目）
* 保留原始强度特征（raw）以及 Z-score 标准化后特征（normalized）
* 异常值通过 IQR 或 Winsorize 处理
 
#### 3.4.1 QC 可视化
 
* raw_vs_normalized_scatter.png：原始 vs 标准化后特征的散点/相关性图（观察是否存在异常缩放）。
* outlier_detection_report.md：异常值比例与处理策略（IQR/Winsorize），建议异常比例 ≤ 3%。

---

## 4. 聚类分析

### 4.1 特征降维

* PCA 用于降噪
* t-SNE 或 UMAP 用于二维可视化（支持随机子采样）
 
#### 4.1.1 QC 可视化
 
* pca_explained_variance.png：累计方差解释曲线（前 N 主成分 ≥ 80%）。
* umap_param_sensitivity.png：UMAP 参数敏感度（n_neighbors/min_dist）的多图对比。
* tsne_reproducibility.png：不同随机种子下 t-SNE 的稳定性对比（布局的一致性定性评估）。

### 4.2 聚类算法

默认：K-means
可选：DBSCAN / GMM
聚类数自动推荐范围：8–12（基于常见免疫细胞/结构细胞亚型数量）

聚类数评价：

* 肘部法
* 轮廓系数
* Calinski-Harabasz 指数
 
#### 4.2.1 QC 可视化
 
* k_elbow_curve.png：K-Means 肘部法曲线，推荐范围 8–12。
* silhouette_distribution.png：不同 K 的轮廓系数分布图。
* ch_index_curve.png：Calinski-Harabasz 指数曲线。
* cluster_stability_ari.png：多次运行的聚类稳定性（Adjusted Rand Index），目标 ≥ 0.85。

---

## 5. 输出与可视化

### 5.1 输出结果

| 字段            | 描述                                 |
| ------------- | ---------------------------------- |
| Cell ID       | 索引编号                               |
| (x,y)         | 质心坐标                               |
| raw 特征        | 每通道原始强度                            |
| normalized 特征 | 标准化后特征                             |
| cluster       | 聚类标签                               |
| confidence    | 聚类置信度（如 GMM posterior probability） |

附加字段（AF 相关，建议加入输出表或单独 JSON）：
- `AF_raw`：细胞内 AF 原始强度
- `AF_bg_corrected`：背景校正后的 AF 强度
- `AF_norm`：归一化 AF 指标（如按细胞面积/核周区域）
- `af_included_in_cluster`：是否将 AF 纳入聚类特征（布尔）
- `spearman_rho_per_marker`：AF 与各通道的相关系数（可另存矩阵）
- `af_artifact_flag`：疑似坏区域标记（布尔/等级）

### 5.2 可视化内容

* 细胞核分割显示图
* 聚类 t-SNE/UMAP 分布图
* 聚类空间重绘叠加在原图上
* 聚类平均特征热图
* 每通道特征分布箱线图/小提琴图
 
### 5.4 QC 输出目录与命名规范
 
建议输出目录结构：
 
* `outputs/{sample_id}/qc/`
  - `01_source_overview.png`
  - `02_dapi_preprocessed.png`
  - `03_nuclei_mask_overlay.png`
  - `04_cell_regions_overlay.png`
  - `05_channel_alignment_vectors.png`
  - `06_unmixing_crosstalk_matrix.png`
  - `07_feature_hist_channel_{ch}.png`
  - `08_feature_violin_{marker}.png`
  - `09_pca_explained_variance.png`
  - `10_umap_tsne_clusters.png`
  - `11_spatial_clusters_overlay.png`
  - `12_cluster_means_heatmap.png`
  - `QC_report.md`
  - `13_af_heatmap.png`
  - `14_af_vs_marker_spearman_matrix.png`
  - `15_cluster_compare_without_AF.png`
  - `16_cluster_compare_with_AF.png`
 
命名规范：
 
* 前缀包含 `{sample_id}`（如 `KC_001_`），并使用两位编号保证浏览排序。
* 通道/Marker 使用 `{ch}` 或 `{marker}` 占位，英文下划线分隔。
* 所有图使用 PNG；报告使用 Markdown（`QC_report.md`），便于自动汇总。

### 5.3 分析报告

* 细胞总数
* 聚类数量与比例
* 各类平均特征解释
* 可疑群体（异常强度、潜在新亚型）提示

---

## 6. 性能要求

| 目标        | 指标                     |
| --------- | ---------------------- |
| 单张整图处理时间  | < 30 min（支持 GPU / 多进程） |
| 内存占用      | < 16GB                 |
| 细胞核分割 IoU | > 0.7（基于人工抽样验证）        |
| 聚类稳定性     | 多次运行一致性 > 85%          |

---

## 7. 风险控制

| 风险     | 影响   | 解决方案                     |
| ------ | ---- | ------------------------ |
| 分割不准   | 特征偏差 | 可切换至 Cellpose / Mesmer   |
| 光谱串扰   | 强度失真 | 允许 spectral unmixing 预处理 |
| 细胞密集重叠 | 聚类误差 | 使用 Voronoi 区域限制像素归属      |

---

## 8. 交付物

* 完整 Python 源码
* 安装与运行说明
* 结构化 API 文档
* 可视化与分析报告模板
 
### 附：运行示例与 QC 开关
 
* 示例命令：
  - `python -m mufclust.run --input /nfs5/yj/MIHC/dataset/KidneyCancer --out ./outputs --qc full --qc-save-png --qc-max-tiles 5 --sample-id KC_001`
* 关键开关约定：
  - `--qc {off|basic|full}`：QC 级别（默认 `basic`）。
  - `--qc-out <path>`：QC 输出目录（默认 `outputs/{sample_id}/qc`）。
  - `--qc-save-png`：保存所有 QC 图为 PNG。
  - `--qc-keep-intermediate`：保留中间结果（掩码、tile 级图等）。
  - `--qc-max-tiles`：限制每步可视化的 tile 数量以控制体积（默认 5）。
  - `--seed <int>`：统一随机种子用于抽样/降维/聚类复现。
  - `--af-mode {skip|qc|feature}`：AF 使用模式（默认 `qc`）；`feature` 将 AF 纳入特征集。
  - `--af-regress`：对其它通道回归/去除 AF 影响后再聚类。
  - `--af-include-if-better`：自动执行含/不含 AF 的聚类并比较，若指标提升≥5%则纳入。
  - `--spectral-unmix {auto|on|off}`：光谱去混策略；`auto` 在高相关/串扰时触发。

## 9. 运行日志（run.log）

* 位置：`outputs/{sample_id}/run.log`
* 内容字段建议：
  - 时间戳、样本ID、输入路径、软件版本/提交号、随机种子
  - 环境信息：CPU/GPU型号与设备ID、内存峰值、库版本
  - 流程节点：预处理/分割/特征/降维/聚类各阶段的开始/结束时间与耗时
  - 参数摘要：关键阈值（如分水岭参数、半径系数、UMAP/TSNE/K值等）
  - QC 抽样：抽取的 5 个 tile 索引/坐标、采样种子
  - 统计指标：分割计数、碎片比例、过分割率、PCA累计方差、轮廓系数、ARI 等
  - 兼容性与回退：是否启用配准、是否跳过 unmix/AF 去除、是否切换 Cellpose/Mesmer
  - 警告与错误：异常通道、文件读取失败、数值溢出、内存不足提示

## 10. 兼容性说明与回退策略

* SampleAF 通道：
  - SampleAF 表示细胞自发荧光（AF），主要用于背景建模与去除；默认不计入聚类特征。
  - 若数据已进行 AF 去除或未采集 AF 通道，可在配置中标记为 `skip_af` 并跳过相关 QC。
* 光谱混叠（unmixing）：
  - 若输入已完成 unmix，保留标记并跳过混叠矩阵 QC；否则执行混叠评估与记录。
* 仪器/通道命名差异：
  - 不同设备（Akoya/其他）通道命名可能不一致，增加 `channels.yaml` 映射以规范化标记与顺序。
* 组织/面板差异：
  - 肾癌/膀胱癌面板不同，需在通道表内补全 Marker 与生物学含义；对于新增面板，按相同规范扩展。
* 性能与资源：
  - 当内存不足或 GPU 不可用时，自动回退到分块/CPU 模式，并在 `run.log` 写入回退记录与建议参数（如减小 tile 大小、禁用高分辨率可视化）。
* 文件与路径：
  - 若存在 ppr 图片辅助文件，路径不一致时应允许可选输入；缺失则自动跳过相关步骤并记录。
* QC 抽样与体积控制：
  - 默认随机抽样 5 个 tile 进行可视化；可通过 `--qc-max-tiles` 调整，建议保持 ≤ 10 以控制输出体积。

---

文档已完成，可以直接用于开发执行。
如果你希望，我可以 **继续为该系统画结构架构图 / 流程图 / 模块依赖关系图 / 数据库表结构**。