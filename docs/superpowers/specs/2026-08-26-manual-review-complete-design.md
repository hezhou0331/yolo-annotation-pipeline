# 场次人工 Review 完成标记设计

日期：2026-08-26

## 目标

让用户可以明确标记一个完整采集场次已经完成逐帧人工 Review，并在 ATEC App 中单独显示“已经过人工 Review”分组，避免与仅完成关键帧标注的场次混淆。

## 设计

- 标记文件固定为场次目录下：
  `project_reports/manual_review_complete.json`
- 文件记录：schema version、scene、class、完成时间、当时导出报告路径及其修改时间、accepted/review/rejected 计数。
- 标记采用显式按钮写入，不因打开或关闭播放器自动完成，避免只检查部分分段就误报整场完成。
- App 提供“标记当前场次 Review 完成”和“取消 Review 完成标记”。
- 只有场次已有可播放的传播结果和导出报告时才能标记完成。
- 若导出报告在完成标记后被重新生成或修改，标记视为失效，场次自动回到“已标记完成”分组，等待再次人工检查。
- 不移动、复制或删除 RGB-D、Mask、Manifest、YOLO 数据；仅增加一个小型元数据文件。

## 场次分组

1. `needs_manual`：未标记完成；
2. `keyframes_complete`：关键帧完成但尚未完成人工 Review，或旧 Review 标记已经失效；
3. `human_reviewed`：显式标记完成且导出报告未发生后续变化。

`human_reviewed` 保留原有自动流程颜色：绿色表示有 accepted 数据，黄色表示仍存在 review/rejected，红色表示 accepted=0。人工 Review 完成不等于所有帧 accepted。

## 失效规则

完成标记保存导出报告的 `mtime_ns`。读取场次状态时，若当前导出报告路径不同、文件不存在或 `mtime_ns` 不一致，则标记无效。SAM2 局部重传播、A/R/X 修改后重新聚合和整场重跑都会更新导出报告，从而自动使旧标记失效。

## 测试

- 标记文件可写入、读取和取消；
- 没有导出报告时拒绝标记；
- 有效标记令场次进入 `human_reviewed`；
- 导出报告更新后标记失效；
- GUI固定显示三个分组；
- 已 Review 场次只出现在“已经过人工 Review”；
- 取消标记后返回“已标记完成”。
