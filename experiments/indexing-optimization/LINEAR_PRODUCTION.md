# Qwen 线性投影正式接入

2026-10-28。先将优化前的代码和实验记录提交为 `cd8db34`，已推送 `origin/codex/video-delete`，随后接入新方案并重启本地正式服务。

`scripts/qwen_projection.py` 在 `ActionEngine.encode_segment` 的视频推理调用期间，把 Qwen3-VL-Embedding 2B 全块 Conv3d 转为同权重 Linear。上下文退出时恢复原 forward，包括异常退出；不写第三方包或下载的模型。调用方沿用现有 `search_gate` 串行控制。结构或输入尺寸不符合已验证模型时明确失败，不继续使用未经验证的映射。

新上传视频、缺失片段补建默认生效；仍为批量 1、FP16、16 帧、1048576 像素预算和 8 秒窗口/4 秒步长。查询路径、既有片段向量与 WISE 存储设置不变，不主动重编全库。状态接口 `/mvp-api/actions/status` 新增 `video_patch_projection=linear_same_weights_v1`，表示视频编码采用的实现，并非声称历史向量均已重编。

## 验证

- `source scripts/env.sh && python -m pytest tests -q`：78 项通过。新增测试覆盖数值等价、参数对象及状态保持、异常恢复、结构和输入保护。
- 正式 `encode_segment` 实际处理媒体 1 的 0–8 秒、媒体 19 的 0–7.875 秒、媒体 41 的 20–26.875 秒：输出与此前冻结的实验 Linear 向量逐元素相同。输入文件哈希匹配。
- 同一实例视频处理结束后，原 forward 恢复；清空文字缓存后重新编码的查询向量与处理前逐元素相同。
- 重启前确认无活跃上传/补建。服务 PID 17369 → 93269，启动预热完成，状态报告新投影实现。
- 重启前后骑自行车、户外慢跑、物体参考图、人脸参考图查询：数量和前十排名/时间/框一致，分数绝对容差 1e-6 内一致。
- 44 个视频、146 个片段、正式项目文件和人物登记文件指纹保持一致。

证据：`linear-production-20261028/{before,after,verification,encoder-verification}.json`。真实编码验证可运行 `verify_qwen_linear_production.py`；此检查不写入正式索引。

性能依据仍是 [完整对照实验](QWEN_LINEAR.md) 的编码耗时减少 41.03%、含解码减少 36.70%；本次为接入验证，没有重复测量完整上传耗时。浮点差异及已有 9 个查询前五变化、2 组人物场景定位变化仍适用。
