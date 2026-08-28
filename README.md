# ATEC YOLO11-seg RGB-D 标注与训练流水线

使用 Orbbec Gemini 336L 采集 RGB-D，以人工关键帧 Mask + SAM2 传播生成实例分割标签，通过质量门控导出 YOLO11-seg，并用薄桌面 App 管理采集、标注、验证、训练和实时识别。

## 当前里程碑（2026-08-27）

当前**标注配置为 9 类**。已有正式数据快照仍覆盖前 7 类，当前可直接用于实时测试的训练权重仍是 5 类阶段模型：

```text
0 can
1 watermelon_rind
2 meal_box
3 red_paper_bag
4 blue_bin
5 green_bin
6 red_bin
7 purple_paper_bag
8 sand_bottle
```

`purple_paper_bag`（紫色袋子）和 `sand_bottle`（沙瓶）是比赛干扰项，但在本项目中作为显式实例类别标注。它们目前还没有进入已发布的 train/val 数据和模型；各自拥有独立有效 train/val 场次后，才能称为完整 9 类模型。历史 7 类 Manifest、dataset 和模型仍作为当前 9 类配置的严格连续前缀兼容。

当前导出数据统计（只统计正式 `train/val`，不包含 `_staging`）：

| split | 图片/标签 | can | watermelon_rind | meal_box | red_paper_bag | blue_bin | green_bin | red_bin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 3,431 | 349 | 764 | 403 | 553 | 526 | 461 | 375 |
| val | 532 | 82 | 65 | 58 | 73 | 103 | 87 | 64 |
| 合计 | **3,963** | **431** | **829** | **461** | **626** | **629** | **548** | **439** |

每张正式图片当前对应一个有效实例标签；`review` / `rejected` 帧不进入正式训练目录。验证集按完整场次/采集视频划分，不随机拆分相邻帧；当前没有独立 `test` 集。原始 RGB-D、关键 Mask、传播结果和 Review 报告也会作为可选数据快照发布。

当前可选训练权重：

```text
runs/segment/atec_5class_reviewed_20260826_1744/weights/best.pt
类别：can、watermelon_rind、meal_box、red_paper_bag、blue_bin
```

该权重是五类阶段模型，只能识别上述五类；它不包含 `green_bin`、`red_bin`、`purple_paper_bag` 或 `sand_bottle`。现有七类数据可重新训练七类阶段模型；只有新增两类也具备独立 train/val 后，才训练完整九类模型。权重不进入 Git 历史，通过 [2026-08-26 模型 Release](https://github.com/hezhou0331/yolo-annotation-pipeline/releases/tag/model-20260826) 作为可选附件提供。

## 最快开始

在 VS Code、Codex 或文件管理器中打开仓库根目录，并在该目录的终端直接运行：

```bash
./scripts/atec-app
```

App 保持“薄 App”：负责类别选择、场次管理、后台命令和日志，不重写 Orbbec、SAM2、YOLO 或训练算法。类别来自 `configs/atec_objects.yaml`，当前为 9 类，GUI 不维护第二份硬编码列表。当前支持：

- RGB-D 采集、停止、保存或丢弃；
- 数据完整性检查与安全隔离修复；
- “需要人工标记”和“关键帧已完成”分组；
- 分段关键 Mask、SAM2 传播、YOLO 导出；
- 完整场次级自动准备 `train/val`；
- 严格验证、训练结果发现；
- 使用真实 `best.pt` 启动外接 `/dev/video0` 或 Orbbec RGB 实时识别。

## GitHub 使用手册（队友）

> 当前仓库和数据 Release 均为 **Public**。任何人都可以克隆代码并下载数据；只有被添加为 Collaborator 的队友才能直接向仓库推送代码。

### 1. 给队友写权限（可选）

仓库管理员在 GitHub 网页中打开：

```text
Settings → Collaborators → Add people
```

使用队友的 GitHub 用户名或邮箱发送邀请。队友接受邀请后可以推送分支；不需要写权限的使用者可直接克隆和下载。

### 2. 队友第一次安装

建议使用 Ubuntu/Linux。先安装基础工具：

```bash
sudo apt update
sudo apt install -y git curl zstd python3-pyqt5 python3-yaml
```

公开仓库的克隆与数据下载不要求 GitHub 登录。只有需要推送代码时，才需要配置自己的 GitHub 账号或 SSH Key。

### 3. 克隆代码并恢复团队数据

```bash
git clone https://github.com/hezhou0331/yolo-annotation-pipeline.git
cd yolo-annotation-pipeline
./scripts/download_atec_data.sh
```

下载脚本会自动：

1. 从公开 Release 下载数据分卷；
2. 合并分卷；
3. 校验 SHA-256；
4. 恢复 `projects/atec_real/data/` 和 `projects/atec_real/datasets/`。

默认不会覆盖已有本地数据。如果这个目录已经有采集内容，脚本会停止并提示。确认需要合并时才运行：

```bash
./scripts/download_atec_data.sh --force
```

查看当前数据 Release：

[ATEC data snapshot 2026-08-26](https://github.com/hezhou0331/yolo-annotation-pipeline/releases/tag/data-20260826)

### 4. 配置 Python 环境

完整环境配置见[环境配置](docs/zh-CN/环境配置.md)。本项目通常使用三个环境：

| 环境 | 用途 |
|---|---|
| `orbbec` | Gemini 336L RGB-D 采集 |
| `yolo11` | YOLO11、SAM2、Mask 传播、训练、实时识别 |
| `foundationpose` | FoundationPose 和部分项目检查工具 |

如果只是查看已有数据、运行 App 或 Review，先确保 `yolo11` 环境可用；如果要重新采集 RGB-D，还需要正确配置 `orbbec` 环境和相机 SDK。

仓库迁移到任意目录后不需要修改脚本；入口会根据自身位置定位仓库根目录。若 Conda 环境不在默认位置，可统一覆盖解释器：

```bash
export ATEC_ORBBEC_PY=/path/to/orbbec/bin/python
export ATEC_FP_PY=/path/to/foundationpose/bin/python
export ATEC_YOLO_PY=/path/to/yolo11/bin/python
# 可选：系统 Python 不在默认位置时
export ATEC_APP_PY=/path/to/system/python3
export ATEC_CLI_PY=/path/to/system/python3
```

旧的 `ATEC_ORBBEC_PYTHON`、`ATEC_FP_PYTHON` 和 `ATEC_YOLO_PYTHON` 仍兼容，但新配置统一使用较短的 `*_PY` 名称。

### 5. 启动 App

```bash
./scripts/atec-app
```

App 是薄管理层，不在界面内重复实现算法。它负责调用已有的采集、分段、Mask、SAM2、导出、验证和训练命令。

主流程：

```text
选择物品
→ 选择 train/val/test
→ 采集并保存场次
→ 补齐所有分段关键帧 Mask
→ SAM2 传播
→ 检查连续标注效果
→ 准备 train/val
→ 验证数据集
→ 训练 YOLO11
→ 摄像头实时识别
```

### 6. 采集和标记

1. 在 App 中选择物品。场次列表只显示当前物品；下方分为“未标记完成”和“已标记完成”。
2. 选择数据用途：`train`、`val` 或 `test`。
3. 点击开始采集，确认出现 `Orbbec RGB-D Capture` 预览窗口。
4. 点击停止，或在采集终端按 `Ctrl+C`。
5. 选择保存或丢弃。丢弃只删除本次暂存，不影响已保存场次。
6. 点击“标下一个”，完成当前场次所有缺失的关键帧 Mask。
7. 只有清单全部完成后，才点击“继续自动处理”或“开始传播”。

“已标记完成”只表示必需关键帧 Mask 已保存，不表示每一帧的 SAM2 结果都正确。

### 7. 检查连续标注效果

在已完成关键帧的场次下点击“检查标注效果”，可以按分段播放原始帧，并查看 RGB、半透明 Mask、轮廓、状态和失败原因。

```text
空格       播放/暂停
← / →      前后逐帧
↑ / ↓      上一个/下一个分段
A          当前帧 accepted
R          当前帧 review
X          当前帧 rejected
[ / ]      标记问题区间开始/结束
K          当前帧增加关键帧并重新传播
Q / Esc     退出
```

重要操作：

- **从此帧开始全部拒绝**：当前帧到指定范围末尾不进入训练；
- **在此处增加关键帧并重新传播**：适用于 Mask 从某一帧开始逐渐偏离物品的情况。

`accepted` 才允许进入正式训练数据；`review` 和 `rejected` 默认不会进入训练。拒绝帧不会删除原始 RGB、Depth 或 Mask 文件。

### 8. 准备数据、验证和训练

在 App 中点击“准备训练数据”。程序只汇总已有 accepted 导出的场次，并按完整场次或采集视频切分 `train/val`；未完成场次需要逐场处理，不会由后台批量推进。

训练按钮只有在以下条件满足后才会启用：

- train 和 val 都有有效图片与标签；
- 标签格式和类别 ID 合法；
- accepted/review/rejected 门控通过；
- train/val 没有相同的场次、`capture_session_id` 或 `source_video_id`；
- 数据集验证通过。

当前已发布数据快照包含前七类，但当前可直接下载测试的权重是五类阶段性模型：

```text
can
watermelon_rind
meal_box
red_paper_bag
blue_bin
```

它不是七类或九类完整模型；`green_bin`、`red_bin` 需要重新训练七类阶段模型，紫色袋子和沙瓶还必须先补齐独立 train/val 数据。

### 9. 实时识别

训练生成 `best.pt` 后，可以在 App 中点击“启动实时识别”，也可以直接运行：

```bash
./scripts/atec-live-yolo
```

默认使用外接摄像头 `/dev/video0`。模型训练过哪些类别，就识别哪些类别，不是黄罐专用。

```text
Q / Esc / Ctrl+C   退出实时识别
```

也可以手动指定模型：

```bash
ATEC_YOLO_MODEL=/path/to/best.pt ./scripts/atec-live-yolo
```

### 10. 获取代码更新

队友开始工作前建议先同步：

```bash
git pull --ff-only origin main
```

如果本地有未提交修改，先保存自己的工作，不要直接覆盖。完成修改后建议使用独立分支：

```bash
git switch -c teammate/<short-description>
git add <changed-files>
git commit -m "describe the change"
git push -u origin teammate/<short-description>
```

不要提交以下内容：

```text
projects/atec_real/data/
projects/atec_real/datasets/
models/
runs/
third_party/
xcx/
```

这些内容已经由 `.gitignore` 或 GitHub Release 管理。

### 11. 常见问题

| 现象 | 处理 |
|---|---|
| `gh auth` 无权访问 | 确认已接受 Collaborator 邀请，并使用正确 GitHub 账号登录 |
| 下载脚本提示目录已有数据 | 先备份本地数据；确认需要合并时使用 `--force` |
| App 没有打开相机预览 | 检查 USB、关闭占用相机的程序，再运行环境检查 |
| 一直显示第一个分段 | 查看关键帧清单，逐个完成缺失分段，不要只看整个场次名称 |
| 第一帧或后续帧是 rejected | 查看播放器中的失败原因；必要时增加关键帧并重新传播 |
| 训练按钮不可用 | 先检查 accepted 数量、train/val 是否完整以及数据验证结果 |
| 实时识别误检多 | 先保持 `conf=0.25`；降低置信度会增加背景误检 |

详细命令见[项目命令](docs/zh-CN/项目命令.md)，App 操作细节见[App 使用手册](docs/zh-CN/App使用手册.md)，环境问题见[环境配置](docs/zh-CN/环境配置.md)。

## 外接摄像头启动 YOLO 实时识别

最短命令（默认使用五类阶段模型、外接摄像头、`conf=0.25`）：

```bash
./scripts/atec-live-yolo
```

完整命令：

```bash
"${ATEC_YOLO_PY:-$HOME/miniforge3/envs/yolo11/bin/python}" \
  tools/live_yolo11_seg.py \
  --model runs/segment/atec_5class_reviewed_20260826_1744/weights/best.pt \
  --source 0 \
  --conf 0.25 \
  --imgsz 640 \
  --device 0
```

这里的 `--source 0` 对应外接 HD Webcam `/dev/video0`。这是通用五类实例分割识别，不是黄罐专用。按 `Q`、`Esc` 或 `Ctrl+C` 退出。建议先保持 `conf=0.25`；降低到 `0.15` 会明显增加背景误检。

## 数据目录边界

```text
projects/atec_real/data/       RGB-D、关键 Mask 和中间结果
projects/atec_real/datasets/   导出的 YOLO11-seg 数据集
```

两者职责不同，禁止因为名称相近而删除。采集先写入 `data/.staging/<session>/`，确认保存后才移动到 `data/scenes/<class>/<session>/`。

真实数据、数据集、训练输出、模型、`xcx/` 和 `third_party/` 默认由 `.gitignore` 排除，不进入 Git。

### 可选下载真实数据

Git 仓库只保存代码、配置、Manifest 和文档；大体积数据不进入 Git 历史。需要复现当前采集、Mask Review 或直接使用已导出的
YOLO11-seg 数据集时，克隆公开仓库后直接运行：

```bash
./scripts/download_atec_data.sh
```

脚本从公开 GitHub Release 下载 `data-20260826` 的数据分卷，自动合并并校验 SHA-256，然后恢复：

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
