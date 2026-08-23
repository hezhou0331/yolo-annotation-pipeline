# Review指南

## 推荐顺序

1. 根目录`README.md`和本目录`快速开始.md`。
2. `configs/atec_objects.yaml`：类别与标签边界。
3. `projects/atec_real/manifests/train_bg01_train.yaml`：场景、实例和关键Mask路径。
4. `scripts/run_auto_annotation_pipeline.sh`：正式自动标注入口。
5. `tools/segment_rgbd_sequence.py`：分段和Mask完整性检查。
6. `tools/annotate_multinstance_project.py`：多实例调度与安全聚合。
7. `tools/propagate_masks_sam2.py`和`tools/sam2_recovery.py`：SAM2传播与恢复。
8. `tools/quality_metrics.py`：质量阈值。
9. `tests/`：回归和安全测试。

## 不需要日常Review

- `third_party/`：固定版本的上游源码，本地存在但不进入主仓库。
- `models/`中的实际权重。
- `projects/*/data`、`datasets`、`assets`和`reports`。
- 缓存、日志、运行输出和Conda环境内部文件。

## 必须冻结的业务规则

- 餐盒与可见内容物是否始终作为同一个实例。
- 垃圾桶是标整桶、桶口ROI还是只使用标定坐标。
- ATEC参赛组别及是否启用Bonus类别。
- 同类多个实物的实例命名和进入/离开画面的有效区间。
