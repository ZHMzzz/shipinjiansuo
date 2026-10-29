# OWLv2 后续优化分析
Status: ready-for-agent
Type: research
Resolution: completed

在入库进程复用接入后，重新分析 OWLv2 剩余计算开销，并核对官方实现。本轮不修改正式配置。

## Comments

- 2026-10-28：对媒体 40 的 0/5/15/30 秒四帧进行模型已驻留的同步分项计时。普通提取均值 1.1659 秒，插桩 1.2401 秒（额外 6.37%）；插桩总时间内注意力含 QKV 占 70.16%、MLP 18.31%、投影卷积 0.83%。卷积时间嵌套在 embeddings 中，不能重复相加。插桩前后输出完全一致、正式库/人员库/代码指纹一致。
- 证据：`experiments/indexing-optimization/profile_owl_backbone.py` 与 `owl-backbone-20261028.json`。这些是四帧诊断占比，不是完整视频精确拆分或优化收益。
- 新发现：已安装 Transformers 4.57.6 使用显式 attention；官方当前 main 的 OWLv2 已声明 `_supports_sdpa=True` 并接入 attention backend。优先在独立环境测官方实现及新 PyTorch，再单独测实际调用路径编译；当前 WISE 的 model.compile 编译顶层 __call__，提取却直接调用 image_embedder，不能认为配置 compile=true 就覆盖视觉推理。
- 次选：960→768 输入（3600→2304 patch，注意力二次项理论约 40.96%，不是端到端耗时）；需同时适配位置编码和 box grid，并回归小物体/参考图检索。或仅降低物体采样率，需保留人脸采样且验证短暂目标召回。
- 小优化：绕过未使用的全零 class logits；保持原排序的阈值过滤后再搬运向量；缓存设备上的 box bias。这些均尚未测试，头部/搬运总体预算远低于视觉骨干。
- 不优先：增大 batch（之前无稳定收益）、卷积改 Linear（本轮 <1%）、仅调高 objectness 阈值（过滤发生在全部推理之后）。原论文 token dropping/instance selection 是训练优化，明确不用于推理。
- 官方证据：[HF 当前主干](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlv2/modeling_owlv2.py)、[PyTorch 2.14 MPS 改进](https://pytorch.org/blog/pytorch-2-14-release-blog/)、[OWLv2 原论文](https://papers.nips.cc/paper_files/paper/2023/file/e6d58fc68c0f3c36ae6e0e64478a69c0-Paper-Conference.pdf)。主干功能不代表当前安装版本已可用；新环境速度和检索质量均尚未验证。
