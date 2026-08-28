# ATEC真实项目

正式类别的唯一配置来源是仓库根目录的 `configs/atec_objects.yaml`。本目录不再维护副本；
项目初始化命令会读取该配置，并把当时的类别表写入每个 Manifest，便于历史数据追溯。
历史七类 Manifest 仍可作为当前九类配置的连续前缀使用，但不要在本目录单独修改类别 ID 或名称。

提交到Git的内容主要是 Manifest 和本说明。以下目录均为本地工作区：

- `data/`：RGB-D输入、关键Mask和跟踪Mask；
- `datasets/`：YOLO11-seg导出结果；
- `assets/`：CAD和Mesh；
- `reports/`：运行报告。

这些目录不会上传GitHub。新电脑需要自行恢复数据，或运行项目初始化/采集命令重新创建。
