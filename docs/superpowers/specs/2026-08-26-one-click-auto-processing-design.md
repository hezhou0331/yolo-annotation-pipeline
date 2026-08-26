# 一键自动处理所有场景：设计说明

## 目标

桌面 App 的“一键自动处理所有场景”扫描项目全部场景，逐场完成完整性预检、安全修复、自动分段、SAM2 传播、质量检查和 YOLO11-seg 导出。单场失败不得中止后续场景；停止后保留已完成结果，下次按文件状态继续。

## 安全边界

- 只自动隔离可由 `metadata.json` 证明为未提交的单边 RGB/Depth 孤立文件，不删除原始数据。
- 只有 `atec_capture_session.json` 的场景名、类别、split 和来源 ID 完整且与目录一致时，才自动补建 Manifest。
- 已有 Manifest 必须能解析且 project、classes、instances 结构完整；仅当采集元数据可证明目标值时，才自动修复场景/输出路径、split、来源 ID、名称前缀以及缺失的类别字段，并在同目录保留时间戳备份。实例编号缺失、重复或类别冲突一律转人工。
- 缺失已记录 RGB/Depth、缺失关键帧 Mask、类别/实例不明确、持续传播失败均进入人工清单。
- 已经完成导出的场景不重复处理；已有关键帧 Mask 不覆盖。
- YOLO 正式目录只接收现有质量链判为 accepted 的帧。

## 调度状态

每个场景预检后归入：

- `init`：Manifest 可无歧义补建；完成后进入 `segment`。
- `segment`：完整 RGB-D 和 Manifest 已有，但还没有分段报告。
- `run`：关键帧 Mask 完整，运行现有 `atec-pipeline run`。
- `manual`：不能安全修复或需要人工 Mask/Review。
- `skip`：已有有效导出，无需重复处理。

GUI 使用独立 `QProcess` 执行批处理，使主界面仍可选择和编辑其他场景；批处理当前场景只读。批处理在独立进程组中运行，停止或关闭 App 时终止整棵子进程树，不删除其已写入的有效中间结果。

## 报告

报告写入 `projects/atec_real/project_reports/auto_processing/`，同时维护带时间戳报告和 `latest.json`。每场记录阶段、结果、原因、孤立文件隔离目录、Manifest 路径/备份和退出码；报告采用临时文件后原子替换。
