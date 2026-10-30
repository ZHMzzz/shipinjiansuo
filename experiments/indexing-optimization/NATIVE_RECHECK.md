# 当前 Qwen 新索引方式复核

2026-10-28（America/Los_Angeles）。**可以继续使用入库端 OWLv2 快速处理器＋实际 FP16 权重＋InsightFace CoreML。** 已沿当前真实上传代码在独立项目副本完成对照；没有启用生产优化。按用户最新要求，以当前新 Qwen 索引为目标，不把历史向量兼容作为采用前提。

## 当前方式与接入边界

ADR 0007 将 Qwen 片段向量、元数据和搜索索引统一存到 WISE，不再保存 `action-index`。编码仍为 Qwen3-VL-Embedding 2B、FP16、总像素预算 1048576、8 秒窗口/4 秒步长、每段 16 帧；没有切成原版 FP32 或高像素预算。

当前上传顺序为：转码 → 复制项目 → 暂存 Qwen store → WISE 提取 OWLv2/人脸并建索引 → 恢复 Qwen store → 编码新 Qwen 片段并直接写 WISE → 校验加载。

应保留已有 `suspend_native_store` 暂存/恢复逻辑，阻止 WISE 自动发现 Qwen 后重复按帧计算，不能直接使用旧回放脚本或删除 Qwen store。优化仅在帧提取子进程生效，Qwen 及查询编码保持当前配置；此前查询端快速处理器的小头盔退化结论仍适用。

## 当前真实路径实测

调用当前 `IngestionManager.process_job`，baseline 和组合优化均从同一 41 视频/135 Qwen 片段快照开始，处理同一条 35.44 秒原始上传文件，各跑一次。只把帧提取命令替换为两组共有的计时入口，暂存恢复、转码、复制和建索引使用现有代码。Qwen builder 调用现有 prepare/encode/finish API；发布回调校验加载索引，只切换实验目录指针。

| 阶段 | baseline | 优化组合 |
|---|---:|---:|
| 完整 process_job | 113.58 秒 | 86.42 秒 |
| OWLv2／人脸提取子进程 | 83.93 秒 | 41.81 秒 |
| WISE 帧索引构建 | 3.41 秒 | 6.39 秒 |
| Qwen 准备及新片段编码 | 21.50 秒 | 35.16 秒 |
| Qwen 原生索引写入 | 0.0130 秒 | 0.0136 秒 |
| 校验、加载及实验内发布 | 0.62 秒 | 0.50 秒 |

本轮总耗时减少 **23.91%**，帧提取减少约 **50.19%**。Qwen 配置与新向量相同，但耗时明显波动，原因未定位；不能归因于统一 WISE 存储。建索引耗时也有波动。因此这是单条视频的兼容性复核，不是稳定吞吐基准，旧报告的 35%–38% 不直接套用。

Qwen 加载与四个文字查询预热 12.31 秒，在单条处理计时外。初始实验副本准备、文件入队不计时；process_job 内真实复制、转码与子进程启动计入。在线服务保持运行，并额外常驻实验 Qwen；未测 HTTP 上传、排队或线上并发。

## 新索引验收

- 两组均成功生成 42 视频、143 个 Qwen 片段的原生 WISE 项目，新视频均新增 **8 个 Qwen 片段、77 条人脸向量**。OWLv2 区域数 10,674 → 10,653，不能声称逐区域无损。
- 新生成的 8 个 Qwen 向量逐元素相同，编码配置、片段时间覆盖及读写校验通过。
- “视频中的场景和动作”“骑自行车”“户外慢跑”“红色安全帽写字”四个固定查询的前 10 条 Qwen 结果，片段、顺序和分数一致。这是配置间一致性，不代表这些问题都有语义正确答案。
- Qwen store 在帧提取时确实暂存、编码前恢复；没有 action-index 或未恢复的暂存目录。三种原生索引全部加载成功，数量与数据库一致。
- 正式活动指针、完整项目文件、正式源码/配置、人物登记库的前后指纹完全一致；实验临时项目已清理。

结论：只加速入库的 OWLv2/人脸阶段，Qwen 统一走当前原生 WISE 新索引流程，查询编码保持现状。此次未重新编码整个正式视频库，也未实施生产开关。

证据：[result.json](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/indexing-optimization/native-recheck-20261028/result.json)、[前指纹](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/indexing-optimization/native-recheck-20261028/production-before.json)、[后指纹](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/indexing-optimization/native-recheck-20261028/production-after.json)。脚本：[native_recheck.py](native_recheck.py)。各组日志和计时在 `native-recheck-20261028/baseline/`、`optimized/`。脚本拒绝覆盖已有结果；复测需选用新的输出目录名。
