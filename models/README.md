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

当前九类全量训练的官方 baseline 初始化权重：

```text
models/yolo11s-seg.pt
SHA256 1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3
```

训练结果：

```text
runs/segment/atec_9class_reviewed_20260829/weights/best.pt
SHA256 94cb56050f121d63e8219287a4c5d0c2108e30078015a6ed624765701b9dfc0f
```

该 run 请求最多 100 个 epoch、`patience=30`，因 EarlyStopping 实际完成 46 个 epoch，最佳为 epoch 16；最终独立 full-val Mask 指标为 `P=0.94597`、`R=0.94736`、`mAP50=0.96011`、`mAP50-95=0.87006`。训练和验收使用 Ultralytics `8.4.126`、Torch `2.6.0+cu124`。

初始化权重和训练结果都不提交 Git 历史；九类 `best.pt` 作为 `atec-9class-reviewed-20260829-best.pt` 通过 [model-20260829 GitHub Release](https://github.com/hezhou0331/yolo-annotation-pipeline/releases/tag/model-20260829) 单独发布，并附同名 `.sha256`。历史 4/5/7 类 run 与权重继续保留，用于回滚和对照，不会被新实验覆盖。

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
