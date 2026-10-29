# 正式接入独立运行时 OWLv2 SDPA
Status: ready-for-agent
Type: task
Resolution: completed

用户授权先提交推送当前版本，再切换正式入库到已验证的 SDPA 方案。切换前提交 `fd7a8ba` 已推送到 `origin/codex/video-delete` 并核实远端 SHA。

保留当前解码、快速 CPU 预处理、CoreML 人脸、Qwen 和查询配置；仅将 OWLv2 入库推理移到固定依赖的独立运行时。使用共享内存、串行模型复用和明确回退开关。验证进程退出、超时/失败传播、查询隔离，以及正式入口完整入库后再重启服务。历史索引不重算。

## Comments

- 2026-10-28：已完成并重启正式服务。`scripts/owl_sdpa.py` 提供入库进程专用适配，默认 `VIDEO_OWL_SDPA=1`；`VIDEO_OWL_SDPA=0` 仅关闭 SDPA，保留现有快速预处理、实际 FP16、CoreML 和进程复用。`VIDEO_FRAME_ACCELERATION=0` 仍关闭全部帧加速。
- `setup_owl_sdpa.sh` 创建独立的 `data/model-cache/owl-sdpa/runtime/`，固定 Torch 2.14.1、torchvision 0.29.1、Transformers 5.21.0 及完整必要依赖；pip check 通过。主环境 Torch/torchaudio 2.8.0、Transformers 4.57.6 未升级。生产模块不依赖 experiments。只用已缓存原模型权重，不启用 compile。
- 子进程固定 batch=1、960px、FP16 和官方 SDPA；检查运行时及注意力实现，错误通过专用管道传回，不静默回退或发布部分索引。共享 mmap 随生命周期释放，帧进程拥有独立进程组，停止时回收模型后端。测试中发现并修复 macOS 已退出进程组可能返回 EPERM 的清理竞态。
- 28 项相关单元测试通过，覆盖协议复用、初始化失败、崩溃、超时、清理、配置/回退、查询隔离、帧进程生命周期和子进程回收、入库管理。
- 正式 FrameWorker 入口在独立库副本执行同一 35.44 秒视频完整入库两次：首次 55.14s、复用 39.02s，两次使用同一 PID，第二次 created=2/reused=2。OWL 10678、人脸 77、Qwen 8 条；物体和人脸向量/元数据、Qwen 向量及 4 个文字查询前十结果与上一轮冻结 SDPA 输出完全一致。此为部署验证，不是新的基线配对性能实验，不能据此重算提速比例。
- 正式服务重启后 PID 44877，日志确认 owl_sdpa=True，服务 ready；原有 44 视频、146 Qwen 片段、原生 WISE 存储和人脸库未改变。重启前后同一文字查询、独立背包参考图查询前十结果逐项一致。无重建旧索引；实验后端与共享缓冲已清理。
- 证据：`experiments/indexing-optimization/owl-sdpa-production-20261028/summary.json`、`service.json`、`upload/result.json`；复核脚本 `experiments/indexing-optimization/verify_production_owl_sdpa.py`。README 已记录安装、回退和验证边界。
