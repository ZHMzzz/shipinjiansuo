# 分开测试官方 OWLv2 SDPA 与视觉推理路径编译
Status: ready-for-agent
Type: task
Resolution: completed

用户要求先分开测试官方 SDPA 和实际视觉路径编译。使用正式 FP16/960px/batch=1 为基线，仅在隔离运行时进行依赖升级与推理测试，不修改正式服务、模型原件或索引。分别报告运行时版本变化、注意力实现变化及编译收益，记录编译首次成本、稳态耗时、内存及输出差异；不得把多项变量合并后的收益归因给单项开关。

## Comments

- 2026-10-28：完成。11 个视频共 44 个冻结预处理画面，使用完全相同权重、FP16、960px、batch=1；仅计模型与排序/搬运/过滤，不含解码、预处理、数据库、Qwen 或发布。
- 正式 Torch 2.8 / Transformers 4.57.6：普通推理 6 轮均值 13.07 秒；实际视觉 forward 的 fullgraph 编译 4 轮均值 45.91 秒，明显回退。
- 隔离 Transformers 5.21.0 / Torch 2.8：eager 20.58 秒、官方 SDPA 12.88 秒（各 4 轮）。两者输出完全一致；相对正式基线收益约 1.4%，旧升级组合不推荐。
- 隔离 Transformers 5.21.0 / Torch 2.14.1：普通 eager 15.20 秒、SDPA 6.05 秒、视觉路径编译 5.62 秒（各 6 轮、两个进程）。对正式基线分别减少 53.71% / 57.02%。每项分别启用；编译器自己融合了 12 层 attention，收益不能叠加。新编译缓存目录首次两个画面需 4.77 秒，SDPA 0.33 秒，此为编译加执行而非纯编译时长。
- 清空未使用 allocator 缓存后的四帧采样：MPS 驱动峰值正式约 1.06 GiB，新 SDPA/编译各约 1.04 GiB，新 eager 约 2.04 GiB；不是精确硬件峰值，RSS 与 MPS 不相加。
- 固定 9 个参考图查询，新 SDPA/编译的视频前五排序均与正式基线一致；SDPA 的独立包图片、编译的挖掘机查询，其画面前五排序/时间点有变化。区域数分别 11335→11349 / 11334，阈值跨越 124 / 117 个 patch。不是逐位无损，也不是全库召回率验证。基线和两候选各自跨进程重复输出完全一致。
- 建议优先对官方 SDPA 做完整入库和全库检索回归。新运行时为本次纯视觉测试移除了 ABI 不兼容的 torchaudio 2.8，不能将整个环境直接替换正式服务。正式依赖、代码、库与 Qwen 不变。
- 脚本：`experiments/indexing-optimization/owl_official_bench.py`、`owl_official_compare.py`；证据：`owl-official-20261028/design.json`、各运行时 `result.json`、`compare-*.json`、`summary.json`、`final-integrity.json`。隔离运行时位于 git 忽略的 `data/model-cache/owl-optimization-runtime/`。
