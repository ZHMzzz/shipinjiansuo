Status: ready-for-agent
Outcome: completed
Type: task

# 保留 2B 与测试结果，清理 8B 非必需文件

用户授权继续使用 2B、暂不切换 8B，保留测试报告和实验结果，并清理 8B 不再需要的文件以释放空间。

已删除两轮实验内的 8B 权重、专用运行时/编译产物、可再生临时帧和 Python 缓存，共七个明确路径，约 9.77 GiB。没有清理共享模型缓存或其他实验。

保留两轮原报告、标注/预览、原始排名/资源记录/日志、输出向量和独立实验索引，以及复现源码和版本来源。377 个保留文件及 97 个受保护文件前后哈希一致，生产指针不变。

交付：`experiments/qwen8b-full-library/CLEANUP.md`、`cleanup-result.json` 和两实验目录 README。清理前完整清单保存为同规格目录下的 `cleanup-plan.json`。

## Comments

- 用户明确授权删除无用的 8B 文件，无需再确认；原报告和结果没有覆盖或改写。
- 磁盘可用空间约由 17 GiB 增至 27 GiB。以后重跑 8B 需重新下载/构建并生成临时帧。
