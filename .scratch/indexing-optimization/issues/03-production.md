Status: ready-for-agent
Type: task
Resolution: completed

# 正式接入入库加速

用户已明确授权接入正式环境。将通过实测的 OWLv2 fast CPU 处理器、MPS 实际 FP16 权重和 InsightFace CoreML 放进专用提取子进程；网页上传、命令行入库共同调用，在线查询配置保持原样。沿用现有 Qwen 原生 WISE 暂存与写入流程。

## 验证与交付

- 加速模块与查询进程隔离，静态模型副本不修改原权重，缓存可校验，提供显式回退开关。
- 回归测试与真实副本上传验证，日志证明实际 dtype、处理器和 CoreML provider。
- 确认没有活动上传后重启正式服务，检查查询、Qwen 状态和正式数据不变。
- 用户已授权部署，无需再次确认；不额外向正式库添加测试视频。

## Comments

- 已接入独立正式入口，网页上传和命令行入库默认启用，查询端不安装适配；保留当前 Qwen 原生存储与暂存流程。
- 72 项测试通过；真实副本上传约 60.89 秒完成，日志确认 fast/FP16/CoreML，新增 8 个 Qwen 片段及三种索引校验通过。
- 无活动上传时重启服务，PID 17369，已核实 VIDEO_FRAME_ACCELERATION=1。42 视频/138 Qwen 片段不变，重启前后四个查询及正式指纹一致。验证详见 `experiments/indexing-optimization/PRODUCTION.md`。
