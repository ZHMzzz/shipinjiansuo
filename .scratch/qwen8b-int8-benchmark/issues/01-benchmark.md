Status: ready-for-agent
Outcome: completed

# 独立测试 8B 8-bit 并与 2B 对照

按 ../spec.md 执行。已开始：环境检查、固定样本和运行时选择。

## Comments

- 本机 Apple M5 / 32GiB；初始磁盘可用约 30GiB。
- 正式 Python：PyTorch 2.8.0、Transformers 4.57.6；不升级。
- 已完成独立 Q8_0 实测、逐段输入对齐、三轮查询与 Metal／进程内存采样。
- 结果：两者 Hit@1 均为 10/12，8B 未解决未戴安全帽误排；Metal 峰值 5.11 → 9.36GiB，片段编码中位数 2.10 → 3.66 秒。
- 42 个正式配置／索引文件及活动项目指针校验未变。结论见 `experiments/qwen8b-int8/REPORT.md`。
