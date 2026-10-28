Status: ready-for-agent
Type: task
Resolution: completed

# 复核当前 Qwen 单套 WISE 存储与索引加速的兼容性

采用当前 IngestionManager.process_job，在独立根目录执行真实上传流程；对照 baseline 与 OWLv2 fast + 实际 FP16 权重 + CoreML。保留 suspend_native_store 保护，核对旧 Qwen 向量及 ID、新片段、三种索引加载、查询结果和正式状态哈希。测试不启用生产优化。

## Comments

- 已确认 ADR 0007：变化是统一存储，编码参数没有切成原版 FP32 或高像素预算。旧实验脚本缺少当前 Qwen store 暂存流程，不直接重用。
- 用户调整验收重点：以后统一用当前新 Qwen 索引；不把历史向量兼容作为采用前提。结论以新增片段写入 WISE、检索可用性、与加速配置配合及实测耗时为准。
- 复核完成：当前真实 process_job 副本对照，35.44 秒视频 113.58→86.42 秒；新增 8 个 Qwen 片段，向量一致，4 个查询 Top10 一致，三种索引校验加载成功。无 action-index，正式前后指纹一致。结果见 `experiments/indexing-optimization/NATIVE_RECHECK.md`；Qwen 耗时存在波动，不承诺稳定提速百分比。未启用生产优化。
