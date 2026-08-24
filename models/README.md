# 本地模型权重

模型权重不进入 Git。当前电脑应保留：

```text
models/sam2.1_t.pt
models/yolo11s-seg.pt
models/yolo11n-seg.pt                       # 仅速度备选
xcx/checkpoint/robot_detector_yolo11s_640_luna1600_final.pt
xcx/checkpoint/trash_bins_yolo11s_640_final.pt
xcx/checkpoint/unified_yolo11s_640_15class_final.pt
third_party/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth
third_party/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth
```

当前四类 baseline 初始化权重：

```text
models/yolo11s-seg.pt
SHA256 1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3
```

训练结果：

```text
runs/segment/atec_4class_baseline_20260824/weights/best.pt
```

初始化权重和训练结果都不提交 GitHub。

## 模型角色

- `yolo11s-seg.pt`：官方实例分割初始化，当前 baseline 主线；
- `yolo11n-seg.pt`：只有 `s` 版本速度不足时才做同数据速度 A/B；
- xcx 权重：YOLO11 detect 教师，输出框而非 Mask，可用于关键帧候选或迁移实验；
- FoundationPose 权重：刚体 6D 姿态辅助，不负责 YOLO 标签转换。

xcx detect 的 backbone/neck 可尝试迁入 YOLO11s-seg，但分割头不能直接复用，类别头也不一致。是否采用必须在相同 train/val 上证明优于官方 baseline。

## 校验

```bash
./scripts/verify_local_assets.sh
```

人工确认的哈希在 `models/checksums.sha256`。不要使用 `git add -f` 提交权重。
