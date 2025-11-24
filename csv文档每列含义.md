**features_long.csv**
- `tile_x`：tile 左上角的全图 X 坐标（像素）（src/muf_clust/core/steps/segmentation.py:394）。
- `tile_y`：tile 左上角的全图 Y 坐标（像素）（src/muf_clust/core/steps/segmentation.py:395）。
- `label`：该 tile 内细胞的实例标签 ID（背景为 0，不写入）（src/muf_clust/core/steps/segmentation.py:396）。
- `channel_index`：通道序号（1 开始，对应配置的 `ChannelSpec.index`）（src/muf_clust/core/steps/segmentation.py:397；src/muf_clust/config.py:18-43）。
- `channel_name`：通道名称（来自配置的 `ChannelSpec.name`，如 `Opal480`）（src/muf_clust/core/steps/segmentation.py:398）。
- `mean`：该细胞 ROI 内像素强度的均值（src/muf_clust/core/steps/segmentation.py:399）。
- `median`：该细胞 ROI 内像素强度的中位数（src/muf_clust/core/steps/segmentation.py:400）。
- `sum`：该细胞 ROI 内像素强度的总和（src/muf_clust/core/steps/segmentation.py:401）。
- `std`：该细胞 ROI 内像素强度的标准差（src/muf_clust/core/steps/segmentation.py:402）。
- `mean_bg_corr`：均值的背景校正值（细胞内均值 − 同 tile ROI 外像素中位数）（src/muf_clust/core/steps/segmentation.py:385-386, 403）。
- `median_bg_corr`：中位数的背景校正值（细胞内中位数 − 同 tile ROI 外像素中位数）（src/muf_clust/core/steps/segmentation.py:404）。
- `n_pixels`：该细胞 ROI 的像素数量（src/muf_clust/core/steps/segmentation.py:405）。

示例行解释（你提供的第 7 行）：
- `18432,12288` → 该 tile 的全图坐标。
- `6` → 细胞标签 ID。
- `2,Opal480` → 通道序号与名称（非 DAPI/AF 通道）。
- `mean=10.2798, median=11.0, sum=9920.0, std=5.3930` → ROI 内强度统计。
- `mean_bg_corr=5.2798, median_bg_corr=6.0` → 背景校正后的强度（背景取 ROI 外像素的中位数）。
- `n_pixels=965` → 该细胞 ROI 像素数。

**features_matrix_mean.csv**
- `cell_id`：细胞全局 ID，拼接自 `tile_x_tile_y_label`（src/muf_clust/core/steps/segmentation.py:429）。
- `tile_x`：同上，tile 的 X 坐标（src/muf_clust/core/steps/segmentation.py:429-431）。
- `tile_y`：同上，tile 的 Y 坐标（src/muf_clust/core/steps/segmentation.py:429-431）。
- `label`：细胞标签 ID（src/muf_clust/core/steps/segmentation.py:429-431）。
- 后续每一列为一个非 DAPI/AF 通道的特征值，列名为 `channel_name`（如 `Opal480`、`Opal520`…），默认填入该细胞在该通道的 `mean_bg_corr`（src/muf_clust/core/steps/segmentation.py:424-447）。
  - 若某细胞在某通道无记录，则该列为 `0.0`（初始化为 0；src/muf_clust/core/steps/segmentation.py:430-433）。

**补充说明**
- 背景的定义：同一 tile 下、`roi_label` 为 0 的像素集合的中位数（鲁棒，避免极端值影响）。
- 通道筛选：仅统计非 DAPI、非 AF 通道（src/muf_clust/core/steps/segmentation.py:362-363；src/muf_clust/config.py:18-43）。
- 默认度量：矩阵默认使用 `mean_bg_corr`；如需换成 `median_bg_corr` 或 `mean`，可调整 `metric_key`（src/muf_clust/core/steps/segmentation.py:424）。