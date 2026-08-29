# ATEC 桌面 App 使用手册

本项目推荐通过薄桌面 App 完成日常采集、标注、检查、训练和实时识别。App 只管理流程和数据，不重写 Orbbec、SAM2 或 YOLO 算法。

## 1. 启动

在 IDE 或文件管理器中打开仓库根目录，并从该目录的终端运行：

```bash
./scripts/atec-app
```

主流程：

```text
选择类别 → 采集并保存 → 标完所有分段关键帧
→ SAM2 自动处理 → 检查连续 Mask → 准备 train/val
→ 验证 → 训练 → 实时识别
```

当前九类从 `configs/atec_objects.yaml` 读取，App 不维护第二份硬编码类别表：

| 按键 | 类别 |
|---:|---|
| 1 | `can` |
| 2 | `watermelon_rind` |
| 3 | `meal_box` |
| 4 | `red_paper_bag` |
| 5 | `blue_bin` |
| 6 | `green_bin` |
| 7 | `red_bin` |
| 8 | `purple_paper_bag`（紫色袋子，比赛干扰项） |
| 9 | `sand_bottle`（沙瓶，比赛干扰项） |

选择类别后，场次列表只显示该类别，并分为“未标记完成”“已标记完成”和“已经过人工 Review”。“已标记完成”仅表示必需关键帧 Mask 已保存，不代表传播结果全部正确；只有显式确认整场检查完成后，场次才进入“已经过人工 Review”。

当前正式数据已覆盖全部九类，紫色袋子和沙瓶也分别拥有来自独立视频的有效 train/val 标签。历史 4/5/7 类数据、Manifest、权重和 run 仍可读取、Review、验证和回滚，不会被九类模型覆盖。

### 薄 App 与 Debug 边界

```text
GUI：选择、确认、状态、启动/停止、日志
workflow_commands.py：纯命令规划，不导入Qt、不写数据
atec-pipeline：统一命令分发
tools/：采集、SAM2、切分、导出、验证、训练算法
```

点击“继续自动处理”后，界面应立即展开日志并显示“正在启动/运行中：SAM2传播与YOLO导出”。验证集切分也通过 `atec-pipeline split` 子进程执行，GUI 不在自身进程中移动训练文件。若按钮异常，优先查看日志中显示的完整命令、程序路径、退出码和错误信息。

## 2. RGB-D 采集

1. 选择类别和 `train`、`val` 或 `test`。
2. 点击开始采集，确认 `Orbbec RGB-D Capture` 窗口左侧 RGB、右侧深度正常。
3. 点击 App 的停止按钮，或在启动采集的终端按 `Ctrl+C`。
4. 等待元数据收尾，然后选择保存或丢弃。

采集先写入：

```text
projects/atec_real/data/.staging/<session_id>/
```

确认保存后才进入：

```text
projects/atec_real/data/scenes/<class>/<scene>/
```

丢弃只删除本次暂存，不影响已有场次。不要手工混用：

```text
projects/atec_real/data/       RGB-D、关键 Mask 和中间结果
projects/atec_real/datasets/   导出的 YOLO11-seg 数据集
```

## 3. 关键帧 Mask

点击“标下一个”后会打开 Mask 编辑器。

```text
左键       添加多边形点或使用画笔
Enter      应用当前多边形
P / B      多边形 / 画笔模式
E          添加 / 擦除切换
S/Ctrl+S   保存
Q/Esc      退出
```

只有顶部显示 `SAVED` 才算保存成功。点错时可切换擦除模式修正；不要在未保存状态直接退出。

一次录制可能因镜头运动、停顿或画面变化产生多个自动分段。每个分段都可能需要自己的关键 Mask，必须让清单全部显示完成后再传播。

## 4. 自动处理和连续检查

关键帧完整后点击“继续自动处理”，执行 SAM2 传播、质量门控和 YOLO 导出。

状态含义：

- `accepted`：允许进入正式训练标签；
- `review`：需要人工判断，默认不进入训练；
- `rejected`：不进入训练；
- 人工状态保存在场次目录中，不修改原始 RGB、Depth 或自动质量报告。

选中已处理场次后点击“检查当前场次标注效果”，可以选择：

```text
全部分段
每批最多 3 个连续分段
单独一个分段
```

播放器显示 RGB、半透明 Mask、轮廓、当前帧、分段、自动状态、人工状态和原因。

```text
空格       播放/暂停
← / →      前后逐帧
↑ / ↓      上一分段/下一分段
A          当前帧 accepted
R          当前帧 review
X          当前帧 rejected
[ / ]      开始/结束问题区间，区间设为 rejected
K          当前帧增加关键帧并重新传播
Q/Esc      退出
```

播放器按钮：

- `Reject all remaining frames`：从当前帧到本场次结尾全部拒绝；
- `Add keyframe and rerun`：在当前位置修订关键帧并重新传播。

退出后，App 会根据修改自动重新导出；新增关键帧会重新运行 SAM2。原始帧不会因为拒绝而删除。

### 标记整场人工 Review 完成

检查完当前采集场次的全部必要分段后，回到 App 点击：

```text
标记当前场次 Review 完成
```

确认后，该场次会进入“已经过人工 Review”分组。仅关闭播放器不会自动完成，避免只看了一部分视频就被误标为已检查。

如果误点，可以点击：

```text
取消 Review 完成标记
```

该状态只写入：

```text
<scene>/project_reports/manual_review_complete.json
```

它不会移动或修改 RGB-D、Mask、Manifest、YOLO 标签，也不表示所有帧都必须是 `accepted`。列表仍用原来的绿色、黄色或红色显示自动处理质量。标记与当前导出报告绑定；重新传播或重新导出后，旧标记会自动失效，需要重新检查并确认。

## 5. 准备训练数据与训练

点击“准备训练数据”后，App 会汇总全部类别中已经有 accepted 导出的场次，而不只扫描当前界面类别。未完成场次需要逐场处理；切分以完整场次、`capture_session_id` 和 `source_video_id` 为不可拆分单位，禁止把同一视频的相邻帧随机分到 train/val。

训练按钮只有在以下条件满足后解锁：

- train 和 val 均有有效图片与标签；
- 数据集验证通过；
- 没有场次或来源 ID 跨 split 泄漏；
- 标签类别和格式合法。

训练默认使用官方 YOLO11s-seg baseline。当前九类正式数据共有 5,883 个 accepted 图片/标签对，按完整场次分为 train 5,010 和 val 873；九个类别在两边均有独立视频。默认实验名为 `atec_9class_reviewed_20260829`，从 `models/yolo11s-seg.pt` 全新初始化，不加载或续训历史 4/5/7 类权重。

## 6. 实时识别

训练完成后，App 只使用真实生成的 `best.pt`，不会用官方基础权重冒充训练结果。选择外接摄像头或 Orbbec RGB 后点击实时识别。

这是通用 YOLO11-seg 识别：模型训练过哪些类别，就识别哪些类别，不是黄罐专用。

```text
Q / Esc / Ctrl+C   退出识别
```

当前九类默认模型可直接启动：

```bash
./scripts/atec-live-yolo
```

启动器默认读取 `runs/segment/atec_9class_reviewed_20260829/weights/best.pt`，并使用 `/dev/video0`、`conf=0.25`。如需回滚，可通过 `ATEC_YOLO_MODEL` 显式指定保留的历史权重；降低阈值会增加背景误检。

## 7. 常见问题

- **看不到 Orbbec 预览：** 检查 USB 连接，关闭占用相机的程序，运行环境检查。
- **保存键没有反应：** 先点击 Mask 编辑器窗口取得焦点，再按 `S` 或 `Ctrl+S`，确认出现 `SAVED`。
- **一直标第一段：** 查看分段清单，选中缺失项；场次名、实例和分段是不同层级。
- **传播逐渐偏离：** 在播放器中拒绝错误区间，或按 `K` 在偏离前增加关键帧重新传播。
- **训练按钮不可用：** 查看 train/val、数据验证和 accepted 数量，不能用相邻帧伪造验证集。
