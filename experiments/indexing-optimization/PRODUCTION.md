# 入库加速正式启用

2026-10-28（America/Los_Angeles），按用户授权完成接入并重启服务。运行进程的 `VIDEO_FRAME_ACCELERATION=1` 已核实，新上传视频将走正式加速入口；不需要重新建立现有索引。

## 生效范围

- `scripts/ingestion.py` 与 `scripts/index.sh` 调用 `scripts/extract_frames.py`。
- 提取子进程安装 OWLv2 快速 CPU 预处理、MPS 实际 FP16 权重、InsightFace CoreML MLProgram。适配不依赖实验目录，不改 WISE 上游文件或已下载的原模型。
- 在线查询不安装这些适配，继续使用原处理器、精度及人脸后端。
- Qwen 保留当前原生 WISE 新索引流程，包括帧提取期间的 store 暂存/恢复。
- 模型副本与编译缓存位于 `data/model-cache/frame-acceleration/`，进行源/副本哈希及权重完整性校验，缓存创建和会话初始化有进程间锁。后端初始化失败会使入库明确失败，不静默宣称加速成功。

## 验证

`python -m pytest tests -q`：**72 项通过**。新增测试覆盖导入不污染查询类、回退开关、加速初始化失败不执行基线、CoreML 缺失和静态缓存权重/源文件保护及损坏检测。

使用正式入口和当前 `IngestionManager.process_job` 在独立项目根目录实际处理一条 35.44 秒视频，约 **60.89 秒**完成。该单次时间仅证明正式实现运行成功，不与此前不同时间的基线组成新的加速承诺。

- 日志确认 `Owlv2ImageProcessorFast device=cpu`、`weights=float16 device=mps`。
- detection、recognition、genderage 三个模型的会话均加载 CoreMLExecutionProvider，支持的 CPU 回退仍保留；不声称全图或全部计算使用 ANE。
- 新增 8 个 Qwen 片段、77 条人脸向量、10,653 个 OWLv2 区域；副本由 42/138 增至 43 视频/146 Qwen 片段，三种索引加载及数据库计数校验通过，没有 action-index。
- 暂存恢复正常，测试副本已清理，测试视频没有进入正式库。

停止前确认无活动上传；服务 PID 从 67991 更新到 17369，Qwen 预热后 ready。重启前后文字查询“骑自行车”“户外慢跑”、背包物体图片、人脸图片查询返回条目数及前 10 名的媒体/时间/属性一致，分数绝对容差为 1e-6。页面返回 200。

正式库保持 **42 视频、138 个 Qwen 片段**；活动指针、完整项目文件、正式源码/配置及登记库在重启前后的指纹完全一致。此次接入未重算历史视频。

## 运维

后续上传默认生效。回退新任务可先停止服务，再执行 `VIDEO_FRAME_ACCELERATION=0 ./scripts/start.sh`；命令行可执行 `VIDEO_FRAME_ACCELERATION=0 ./scripts/index.sh`（先停服务）。取消该环境变量并重启恢复加速。此开关不会修改已有索引。首次编译缓存可能增加初始化时间。

证据：[真实入口验证](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/indexing-optimization/production-entrypoint-20261028/result.json)、[重启查询核验](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/indexing-optimization/deployment-20261028/verification.json)、[生效记录](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/indexing-optimization/deployment-20261028/activation.json)、[重启前](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/indexing-optimization/deployment-20261028/before.json)、[重启后](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/indexing-optimization/deployment-20261028/after.json)。
