# YOLO Annotation Pipeline

面向ATEC实物识别任务的RGB-D采集、半自动实例分割标注、质量检查和YOLO11-seg训练Pipeline。

> 日常Review只需要关注：`atec_pipeline/`、`configs/`、`scripts/`、`tools/`、`tests/`、`projects/`和`docs/zh-CN/`。模型、真实数据和第三方源码只保留在本机，不进入Git。

## 工作流

```text
Orbbec Gemini 336L RGB-D采集
→ 自动分段与RGB/深度同步检查
→ 人工制作首帧或关键帧实例Mask
→ SAM2传播Mask
→ 光流辅助重注册、质量门控与人工Review
→ 可选FoundationPose提供6D姿态辅助
→ 导出YOLO11-seg数据集
→ 训练与验证YOLO11
```

**FoundationPose不是YOLO标签转换器。** 当前主线依靠SAM2传播实例Mask；FoundationPose只适合具有尺寸正确、坐标可靠CAD/Mesh的刚体，用于6D姿态估计、跨帧辅助跟踪和结果校验。纸袋、餐盒和西瓜皮等可变形目标不应把FoundationPose作为主标注方法。

## 当前状态

- 相机、YOLO11、SAM2、FoundationPose、PyTorch3D、NVDiffRast和PyOrbbecSDK环境此前已通过检查。
- 真实项目现有109帧RGB-D数据，自动划分为11段。
- 当前`0/11`段具备所有活跃实例的完整关键Mask；这是批量标注前的直接阻塞项。
- 真实数据、模型和第三方源码默认被`.gitignore`排除。

## 快速开始

```bash
# 1. 检查本机依赖和模型
./scripts/atec-pipeline doctor
./scripts/bootstrap_third_party.sh --check
./scripts/verify_local_assets.sh

# 2. 检查相机与环境
./scripts/check_all_environments.sh

# 3. 只检查现有项目，不写入正式数据集
./scripts/run_auto_annotation_pipeline.sh \
  projects/atec_real/manifests/train_bg01_train.yaml \
  --dry-run --allow-missing-key-masks

# 4. 关键Mask齐全后执行正式标注
./scripts/run_auto_annotation_pipeline.sh \
  projects/atec_real/manifests/train_bg01_train.yaml

# 5. Review通过后训练YOLO11-seg
./scripts/train_yolo11_seg.sh \
  projects/atec_real/datasets/atec_yolo11_seg/dataset.yaml 100 4
```

新电脑克隆后，先阅读[快速开始](docs/zh-CN/快速开始.md)，再运行：

```bash
./scripts/bootstrap_third_party.sh --install
```

模型权重的名称、位置和SHA256见[`models/README.md`](models/README.md)。

## 文档

- [快速开始](docs/zh-CN/快速开始.md)
- [数据采集与标注流程](docs/zh-CN/数据采集与标注流程.md)
- [环境配置](docs/zh-CN/环境配置.md)
- [当前完成状态](docs/zh-CN/当前完成状态.md)
- [Review指南](docs/zh-CN/Review指南.md)
- [清理记录](docs/zh-CN/清理记录.md)
- [第三方依赖](third_party/README.md)

## 数据与Git边界

`projects/atec_real/data`是原始RGB-D和关键/跟踪Mask；`projects/atec_real/datasets`是YOLO导出结果。两者职责不同，不是重复目录，均在本地保留但不上传GitHub。

本仓库暂不附带开源许可证。ATEC规则PDF位于本机`docs/private/`，不会提交到远端。
