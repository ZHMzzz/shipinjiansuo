# 恢复 WISE2 OWLv2 物体检索

2026-10-27，按用户明确选择恢复。物体入口使用本地 WISE2 原生 `/demo/search`、OWLv2 图像查询、区域特征、FAISS 与原生排序。移除了此前 Qwen 物体精细匹配的实现及开关；旧 `refine=true` 请求会明确提示刷新页面。

物体框在 WISE API 中保留，MVP 页面隐藏。Qwen 继续负责文字、图文、动作与截图候选召回；局部截图对齐、人脸检索及登记保持。SigLIP 仍停用。

## 索引与上传

`scripts/restore_owl_project.py` 在发布副本上使用 WISE 的 `WiseProject.merge`、`extract-features` 与 `create-index`：

- 复用旧发布版本的 32 个视频、253,009 条 OWLv2 向量，全部向量数据的 SHA-256 与恢复后对应前缀一致，未重算。WISE 重映射向量 ID，并同步区域元数据。
- 为后来新增的 6 个视频提取 9,901 条区域向量。恢复时 38 个视频、262,910 条 OWLv2 向量。
- 检查区域元数据、特征存储、搜索索引的 ID 集合及数量一致，检查媒体、人脸、Qwen 和缩略图指纹不变。
- 后续上传恢复“转码 → OWLv2 与人脸提取 → WISE 建索引 → 新增 Qwen 片段 → 原子发布”。初始建库与模型准备脚本同步恢复 OWLv2。

当前官方 WISE 合并 API 在源代码中仍标记 experimental，所以本项目只在可丢弃的副本上调用，保留活动指针并验证后发布，不直接修改在线版本。

## 实测

用户的两张原图经原生 WISE 接口查询：

| 图片 | 首位结果 | 本次请求耗时 |
| --- | --- | ---: |
| 挖掘机 | 摔倒01_白天工地.mp4，0 秒 | 1.03 秒 |
| 太阳镜 | whistling-LqQYSUb_TJI_000052_000062.mp4，2 秒 | 0.74 秒 |

新闻局部截图仍首位命中抖音20261026-905866.mp4，约 30 秒，对齐相关性 0.984。上述是本机单次请求记录，不代表并发/P95 性能，也不把相似度当作正确概率。

真实增量上传 1.6 秒测试视频成功，新增 207 条 OWLv2 区域向量、3 条人脸向量和 1 个 Qwen 片段；自动发布后能从 WISE 物体接口查到。此测试约 18.94 秒，包含任务与模型状态开销。视频库因此增加一条回归样本，总计 39 视频，263,117 条 OWLv2 向量。原 Qwen 向量与人脸行未变。

42 项自动化测试通过，包括旧模型迁移保留 OWLv2、索引 ID 错配检测和启动要求检查。原中文、图文、动作、人脸、截图、上传去重与失败隔离回归通过。新增上传后重启，活动指针不变，OWLv2 与 InsightFace 仍加载，39 视频的 Qwen 片段就绪，原生物体检索再次通过。浏览器已确认请求实际发送到 `/demo/search` 并指定 OWLv2，精细匹配开关已消失，物体框隐藏，播放跳转正常，视频库显示物体/人脸/Qwen 能力。

另已验证 Qwen 动作重排实际执行成功后，立即调用 OWLv2 仍正常返回太阳镜正确视频首位，模型切换未破坏物体检索。

## 复现与恢复

正常使用只需启动并刷新页面：

```sh
./scripts/start.sh
```

如果恢复一个只有 Qwen 和人脸索引的本地库，可先在线构建副本：

```sh
source scripts/env.sh
python scripts/restore_owl_project.py
```

脚本会优先使用该库迁移记录指向的旧 OWLv2 项目；也可用 `--old-project /absolute/old/project` 指定。找不到可复用旧项目时，使用官方提取链路计算全部缺失视频。完成打印 `STAGED_PROJECT=...` 后，确认上传任务已结束，停止服务并发布返回的路径：

```sh
./scripts/stop.sh
python scripts/restore_owl_project.py --publish /absolute/staged/project
./scripts/start.sh
```

若构建期间视频库更新，发布会拒绝旧基线，须基于最新库重新构建。构建失败删除未发布副本，旧在线库不变。完整初始恢复仍使用 `setup.sh → index.sh → start.sh`。

详细证据在忽略入 Git 的 `logs/owl-restoration-build.log`、`owl-restoration-vector-check.json`、`owl-restored-user-images.json`、`owl-native-upload-verification.json`。用户原图不复制入 Git。
