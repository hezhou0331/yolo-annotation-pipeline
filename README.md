# ATEC YOLO11-seg RGB-D 标注与训练流水线

使用 Orbbec Gemini 336L 采集 RGB-D，以人工关键帧 Mask + SAM2 传播生成实例分割标签，通过质量门控导出 YOLO11-seg，并用薄桌面 App 管理采集、标注、验证、训练和实时识别。

## 当前里程碑（2026-08-24）

已经完成第一个**四类阶段性模型**，不是完整七类模型：

```text
can
watermelon_rind
meal_box
red_paper_bag
```

质量门控固定为：`accepted` 进入 `train/val`，`review` 和 `rejected` 不进入。

| split | 图片/标签 | can | watermelon_rind | meal_box | red_paper_bag |
|---|---:|---:|---:|---:|---:|
| train | 1107 | 369 | 291 | 143 | 304 |
| val | 278 | 82 | 65 | 58 | 73 |

验证集按完整场次/采集视频划分，没有随机拆相邻帧。严格验证未发现跨 split 重复图片、`capture_session_id` 或 `source_video_id` 泄漏。当前只有 `train + val`，外接摄像头作为现场测试，不把 `val` 冒充 `test`。

正式实验：

```text
runs/segment/atec_4class_baseline_20260824
```

配置：`yolo11s-seg.pt`、baseline、100 epochs、batch 4、imgsz 640、device 0。训练因 EarlyStopping 在第 35 轮正常结束，最佳第 5 轮，实际耗时约 16.2 分钟。独立 val 复验：

```text
Mask Precision:   0.838
Mask Recall:      0.890
Mask mAP50:       0.933
Mask mAP50-95:    0.737
```

最佳权重不进入 Git 历史，通过 GitHub Release 作为可选附件提供：

```text
runs/segment/atec_4class_baseline_20260824/weights/best.pt
```

详细结果见[工作日志](docs/zh-CN/工作日志.md)。

## 最快开始

```bash
cd /path/to/YOLO_Annotation_Pipeline
./scripts/atec-app
```

App 保持“薄 App”：负责类别选择、场次管理、后台命令和日志，不重写 Orbbec、SAM2、YOLO 或训练算法。当前支持：

- RGB-D 采集、停止、保存或丢弃；
- 数据完整性检查与安全隔离修复；
- “需要人工标记”和“关键帧已完成”分组；
- 分段关键 Mask、SAM2 传播、YOLO 导出；
- 完整场次级自动准备 `train/val`；
- 严格验证、训练结果发现；
- 使用真实 `best.pt` 启动外接 `/dev/video0` 或 Orbbec RGB 实时识别。

## 队友首次使用（Private 仓库）

仓库所有者先在 GitHub 仓库的 `Settings → Collaborators` 中添加队友。队友接受邀请后执行：

```bash
gh auth login
git clone https://github.com/hezhou0331/yolo-annotation-pipeline.git
cd yolo-annotation-pipeline
./scripts/download_atec_data.sh
./scripts/atec-app
```

`gh auth login` 必须登录已经获得本私有仓库权限的 GitHub 账号。环境安装与相机配置见
[环境配置](docs/zh-CN/环境配置.md)，日常采集、关键帧标记、传播和 Review 见
[App 使用手册](docs/zh-CN/App使用手册.md)。

## 外接摄像头启动 YOLO 实时识别

最短命令（默认使用四类训练结果、`/dev/video0`、`conf=0.25`）：

```bash
cd /path/to/YOLO_Annotation_Pipeline
./scripts/atec-live-yolo
```

完整命令：

```bash
cd /path/to/YOLO_Annotation_Pipeline
~/miniforge3/envs/yolo11/bin/python \
  tools/live_yolo11_seg.py \
  --model runs/segment/atec_4class_baseline_20260824/weights/best.pt \
  --source 0 \
  --conf 0.25 \
  --imgsz 640 \
  --device 0
```

这里的 `--source 0` 对应外接 HD Webcam `/dev/video0`。这是通用四类实例分割识别，不是黄罐专用。按 `Q`、`Esc` 或 `Ctrl+C` 退出。建议先保持 `conf=0.25`；降低到 `0.15` 会明显增加背景误检。

## 数据目录边界

```text
projects/atec_real/data/       RGB-D、关键 Mask 和中间结果
projects/atec_real/datasets/   导出的 YOLO11-seg 数据集
```

两者职责不同，禁止因为名称相近而删除。采集先写入 `data/.staging/<session>/`，确认保存后才移动到 `data/scenes/<class>/<session>/`。

真实数据、数据集、训练输出、模型、`xcx/` 和 `third_party/` 默认由 `.gitignore` 排除，不进入 Git。

### 可选下载真实数据

私有 Git 仓库只保存代码、配置、Manifest 和文档；大体积数据不进入 Git 历史。获得仓库权限的队友需要复现当前采集、Mask Review 或直接使用已导出的
YOLO11-seg 数据集时，可在克隆仓库并完成 `gh auth login` 后运行：

```bash
./scripts/download_atec_data.sh
```

脚本从 GitHub Release 下载 `data-20260824` 的两个分卷，自动合并并校验 SHA-256，然后恢复：

```text
projects/atec_real/data/
projects/atec_real/datasets/
```

为避免覆盖本地采集，已有非空数据目录时脚本会停止；确认需要合并时才使用
`./scripts/download_atec_data.sh --force`。团队快照保留原始 RGB-D 图像和标注内容，只把影响跨机器使用的仓库绝对路径改为相对路径；本机原始数据不会被修改。

## 技术边界

```text
Orbbec RGB-D
→ 自动分段与完整性检查
→ 人工关键帧实例 Mask
→ SAM2 传播
→ 光流辅助重注册与质量门控
→ 可选 FoundationPose 刚体 6D 辅助
→ YOLO11-seg 导出与 Review
→ 场次级 train/val
→ 训练
→ 摄像头现场测试
```

**FoundationPose 不是 YOLO 标签转换器。** SAM2 是当前视频 Mask 传播主线；FoundationPose 只适合拥有正确尺度 CAD/Mesh 的刚体。西瓜皮、餐盒、纸袋等可变形物体优先使用 SAM2 和人工 Review。

## xcx 检测模型与当前分割模型

xcx 权重主要是 YOLO11 **detect**：输出框、类别和置信度，不输出实例像素 Mask。当前正式模型是 YOLO11 **seg**：还输出每个实例的像素 Mask。

xcx 可用于关键帧候选框或后续 `xcx-transfer` A/B，但分割头不能直接复用，类别头也不一致。首版使用官方 `yolo11s-seg.pt` baseline 是低风险基准；是否切换必须由相同 train/val 和现场误检率证明。

## 常用检查

```bash
./scripts/atec-pipeline doctor
./scripts/check_all_environments.sh
./scripts/verify_local_assets.sh
./scripts/atec-pipeline status
./scripts/atec-pipeline validate projects/atec_real/datasets/atec_yolo11_seg/dataset.yaml
./scripts/atec-pipeline smoke-test
```

## 文档

- [App 使用手册](docs/zh-CN/App使用手册.md)
- [项目命令](docs/zh-CN/项目命令.md)
- [环境配置](docs/zh-CN/环境配置.md)
- [工作日志](docs/zh-CN/工作日志.md)
