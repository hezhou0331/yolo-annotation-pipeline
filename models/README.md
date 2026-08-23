# 本地模型权重

模型权重不进入Git。请在当前电脑或新环境中放置为：

```text
models/sam2.1_t.pt
models/yolo11n-seg.pt
third_party/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth
third_party/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth
```

来源分别为SAM2、Ultralytics YOLO11和FoundationPose官方发布资源。下载时遵守对应上游许可证与使用条款。

校验：

```bash
./scripts/verify_local_assets.sh
```

不要使用`git add -f`提交权重；FoundationPose scorer权重约190MB，超过GitHub普通Git单文件100MB限制。
