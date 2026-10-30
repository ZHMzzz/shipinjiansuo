# Qwen3-VL-Embedding 8B Q8_0 与现有 2B FP16 对照

2026-10-27，Apple M5 / 32GiB 统一内存，独立离线实验。

本轮没有观察到足以支持替换的检索收益。8B Q8_0 可以在本机运行，但首位命中数与现有 2B 相同，热查询和片段编码更慢，Metal 分配峰值约增加 83%。建议继续保留现有 2B；若继续验证 8B，应扩大独立标注样本，尤其是当前真实失败案例。

正式配置、模型接入代码、活动项目指针及受保护索引没有变化。实验前后校验的 42 个文件 SHA-256 全部一致；正式库仍为 40 个视频、129 个片段。实验进程已经退出，未安装或启用新的在线服务。

## 实测结果

候选集为已有的 10 段人工标注研究视频，含翻越围栏、围栏旁站立、摔倒、蹲下、正常行走、戴安全帽及头部裸露对照。6 类查询意图各有中文和英文，共 12 条查询。每个模型执行 3 轮查询；计分只使用第一轮，不把重复查询计成新样本。

| 指标 | 现有 2B FP16 | 8B Q8_0 |
| --- | ---: | ---: |
| 首位命中 Hit@1 | 10/12（83.3%） | 10/12（83.3%） |
| 前三位至少一个命中 Hit@3 | 11/12（91.7%） | 11/12（91.7%） |
| 前五位至少一个命中 Hit@5 | 12/12（100%） | 12/12（100%） |
| 前三位召回率，按查询平均 | 84.7% | 87.5% |
| MRR，首个相关结果倒数排名的平均值 | 0.896 | 0.896 |
| 中文首位命中 | 5/6 | 5/6 |
| 预热后片段编码中位数，3 段重复 | 2.10 秒 | 3.66 秒 |
| 热查询中位数，后两轮共 24 次 | 0.061 秒 | 0.238 秒 |
| 热查询 P95，后两轮共 24 次 | 0.118 秒 | 0.349 秒 |
| FAISS 查询中位数 | 0.025 毫秒 | 0.057 毫秒 |
| Metal 分配峰值，50ms 采样 | 5.11GiB | 9.36GiB |
| 进程树 RSS 峰值，100ms 采样 | 2.44GiB | 9.26GiB |

热查询包含文字编码与本实验 FAISS 检索，不含页面、网络请求、正式库元数据回查；每次重新执行编码，没有应用层查询缓存。片段编码包含画面预处理和模型推理，8B 还包含本机进程间传输；均不含原视频解码、OWLv2、人脸提取和正式入库发布。不能把这些时间当成完整视频上传入库耗时。

两边三轮查询的完整排名均一致。重复编码的三段视频与各自首次向量余弦相似度均为 1.0（浮点记录精度内）。

## 哪些查询发生了变化

“作业人员没有佩戴安全帽”：

| 排名 | 2B FP16 | 8B Q8_0 |
| --- | --- | --- |
| 1 | 室外戴帽工人 | 戴帽工人蹲下 |
| 2 | 戴帽工人蹲下 | 室外戴帽工人 |
| 3 | 室内戴帽工人 | 室内戴帽工人 |
| 4 | 头部裸露的摔倒片段 | 头部裸露的摔倒片段 |
| 5 | 头部裸露的行走片段 | 头部裸露的行走片段 |

8B 没有解决这条中文否定条件的误排。英文未戴帽查询两者也都把戴帽片段排在第一位，首个相关结果均在第二位。

有限的改善来自英文“Workers wearing safety helmets.”：8B 将三段戴帽片段全部排进前三，2B 的第三位为头部裸露的摔倒片段。因此平均 Recall@3 上升约 2.8 个百分点，但 Hit@1、Hit@3、Hit@5、MRR 和中文指标均未提高。其他简单动作查询两者均能首位命中。

完整逐查询排名见 [query-comparison.csv](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/qwen8b-int8/query-comparison.csv)，原始相似度见两个模型的 JSON。不同模型的相似度数值不能当成同一概率尺度比较。

## 内存与速度的解释

Metal 主指标：2B 使用 `torch.mps.driver_allocated_memory()`；8B 使用其推理进程中 `MTLDevice.currentAllocatedSize`，均以 50ms 周期采样。它们包含运行时已分配及缓存的 Metal 缓冲区，不代表全部系统 RAM，也不应与 RSS 相加。

同时记录了 macOS `ri_phys_footprint`：2B 进程树峰值约 6.13GiB，8B 约 2.12GiB。**这不表示 8B 更省内存**：llama.cpp 对约 8.8GB 权重采用文件映射，文件支持的驻留页和匿名／压缩内存的计费口径不同；8B 的 RSS 与 Metal 指标明显更高。物理 footprint 原始数据保留供排查，不用它单独给两个运行时排序。

测试前系统已存在大量交换内存使用。最终两轮的开始交换占用分别为约 19.15GiB、22.41GiB，运行中最低可用内存约 4.32GiB、3.51GiB。没有触发 2GiB 可用内存保护。

首轮 2B 热查询中位数为 0.176 秒，补测为 0.061 秒，排名及分数完全相同；8B 两轮热查询中位数约为 0.209、0.238 秒。可见当前机器负载会显著影响时延。因此“最终主测查询约慢 3.9 倍、片段编码约慢 1.7 倍”只描述本轮观测，不是稳定性能承诺。

观测到的模型加载时间为 2B 12.83 秒、8B 4.10 秒；首轮 2B 为 37.92 秒。加载顺序、文件缓存、Metal 内核编译和运行时均不同，尤其 8B 刚下载过权重，不能据此宣称 8B 冷启动更快。模型下载时间不计入加载或推理时间。

## 输入对齐与运行方式

- 2B：直接调用正式 `wise.feature.qwen3vl.Qwen3VLEmbeddingFeatureExtractor`，PyTorch 2.8.0 / MPS / FP16，未改动其源码、模板和参数。
- 8B：社区转换的语言主干和视觉组件均为 Q8_0；小型归一化等张量保留高精度，属于权重量化，不是输出向量压缩。使用 llama.cpp b11944、commit `1b4554eff` 的预编译 Metal 库；实验适配器仅调用其公开 C API。
- 抽取相同的 16 张 RGB 帧，文件名和标签不进入模型。时间元数据复现当前正式编码器的列表输入行为：2fps、成对帧时间标记 `<0.2 seconds>` 至 `<7.2 seconds>`；这是既有行为，不是按短片真实长度重新设计的时间编码。
- 使用同一个默认系统提示词 `Represent the user's input.`、用户／assistant 模板、末尾 EOS 和末 token 池化；输出向量均 L2 归一化。
- 8B 在每次编码时调用现有 `qwen_vl_utils` 做同样的缩放，然后传入 libmtmd。RGB 转为 uint8 在这 10 段样本上最大误差均为 0；数据形状与 SHA-256 记录在 `dataset.json`。
- 对每段逐项核对非视觉 token 序列、8 组成对帧的视觉 token 数和总 token 数。9 段各为 980 token，近方形围栏片段为 1052 token；两边全部一致。第一次直接采用 libmtmd 默认缩放的尝试没有通过检查，已归档在 `unaligned-attempt/`，不计入结果。
- 8B 与 2B 的推理引擎、算子实现和精度仍不同。本轮没有 8B FP16 对照，因此不能单独归因量化损失，也不能将本结果推广为“8B 原版不如 2B”。

这只是 10 段已有研究素材上的探索性对照。6 类意图的中英文并非 12 个独立意图，部分动作和对照来自同一原视频，且小候选集的 Top-5 指标容易达到上限。**未对正式 40 个视频／129 个片段重建实验全库，也未测试完整页面的端到端响应。**

## 复现与文件

在项目根目录执行；所有模型、运行时和结果均写入本实验目录，正式环境依赖不升级：

```sh
source scripts/env.sh
python experiments/qwen8b-int8/setup_runtime.py
python experiments/qwen8b-int8/download_models.py
python experiments/qwen8b-int8/benchmark.py --prepare
python experiments/qwen8b-int8/benchmark.py --backend 2b-fp16
python experiments/qwen8b-int8/benchmark.py --align-frames
python experiments/qwen8b-int8/benchmark.py --backend 8b-q8
python experiments/qwen8b-int8/summarize.py
```

顺序执行两种模型，避免并发影响测量。汇总脚本会检查本次保存的生产快照；若以后正式视频库有正常更新，最后的保护校验将失败，需要为新实验重新保存基线，不能忽略校验错误。

- [summary.json](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/qwen8b-int8/summary.json)：汇总指标和逐段输入对齐检查。
- [2b-fp16.json](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/qwen8b-int8/2b-fp16.json)、[8b-q8.json](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/qwen8b-int8/8b-q8.json)：逐段耗时、逐查询排名、内存快照。
- `*-resources.json`：100ms 进程树与系统内存采样。
- `initial-baseline/`：首次 2B 运行；`encoder-only-timing/`：8B 加入共享预处理计时前的诊断结果，不用于主表片段速度。
- [model-provenance.json](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/qwen8b-int8/model-provenance.json)、[runtime-provenance.json](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/qwen8b-int8/runtime-provenance.json)：版本、下载来源及 SHA-256。
- [production-before.json](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/qwen8b-int8/production-before.json)、[production-after.json](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/qwen8b-int8/production-after.json)：正式配置和索引保护校验。
- `models/`：约 8.80GB 十进制量化权重；整个独立实验目录约 8.6GiB。大权重、运行时、帧和生成向量不进入 Git。

模型来源：[Qwen3-VL-Embedding-8B Q8_0 转换仓库](https://huggingface.co/mradermacher/Qwen3-VL-Embedding-8B-GGUF)，固定 revision `ffa49879fdb91ed1a436fbc84f37b123f714bb13`；两个下载文件均通过仓库 LFS SHA-256 校验。运行时来源：[llama.cpp b11944](https://github.com/ggml-org/llama.cpp/releases/tag/b11944)。
