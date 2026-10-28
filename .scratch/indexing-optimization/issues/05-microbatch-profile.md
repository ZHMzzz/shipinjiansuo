# 定位小批量推理未提速的原因
Status: ready-for-agent
Type: task

继续批量 1/2 对比，使用固定输入、交错重复、分阶段同步计时以及 MPS 算子/拷贝/回退统计。检验预处理和数据搬运、模型内部计算、系统负载三类原因。两模型串行，只在隔离实验中插桩；不改变正式配置。xctrace 不可用，使用已验证可输出结果的 PyTorch MPS 内置统计，不把 CPU profiler 调度时间误称 GPU 算子执行时间。

Resolution: completed

## Comments

2026-10-28：完成分阶段交错计时、细/粗模块插桩、四份 MPS 算子统计及独立算子复测，正式指纹一致。Qwen 视觉 SDPA 对同样两片段均调用 384 次，合批未减少；实际 RoPE 的 rotate_half 复测 5.419→6.808ms，得到明确慢点。OWL 独立注意力和驻留输入模型测试不稳定复现变慢，不能宣称锁定单一算子或带宽根因。额外发现 Qwen patch Conv3d 同权重线性投影试验从约 1.53s 到约 2.3ms，存在相对 L2 0.026% 数值差异；仅隔离算子测试，未上线、未完成整段检索回归。完整报告：`experiments/indexing-optimization/MICROBATCH_PROFILE.md`。
