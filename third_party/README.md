# Third-party dependencies

本目录保存运行所需的上游源码，但除本README外不进入主仓库。使用`scripts/bootstrap_third_party.sh`检查或恢复。

| 目录 | 上游仓库 | 固定commit |
|---|---|---|
| FoundationPose | `https://github.com/NVlabs/FoundationPose.git` | `a1b694b` |
| OrbbecSDK_v2 | `https://github.com/orbbec/OrbbecSDK_v2.git` | `b71adc7` |
| pyorbbecsdk | `https://github.com/orbbec/pyorbbecsdk.git` | `0f089c9` |
| BundleSDF | `https://github.com/NVlabs/BundleSDF.git` | `ffa67d4` |
| nvdiffrast | `https://github.com/NVlabs/nvdiffrast.git` | `253ac4f` |
| pytorch3d | `https://github.com/facebookresearch/pytorch3d.git` | `fdaf9bd` |

```bash
./scripts/bootstrap_third_party.sh --check
./scripts/bootstrap_third_party.sh --install
```

源码恢复不等于完成Python/CUDA编译，也不会下载模型权重。环境定义在`environments/`，模型位置在`models/README.md`。

各依赖继续受其上游许可证约束；本仓库不重新授权或复制第三方代码。
