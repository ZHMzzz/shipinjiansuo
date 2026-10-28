# PPE 预训练权重对照：测量方法审查

本文件为只读审查结论；未运行模型、停止进程或更改正式服务。日期按本项目会话：2026-10-27。

## 本轮基线与并发干扰

- 正式服务 PID 75954 正常监听 9670；`/mvp-api/actions/status` 为 ready、41 视频、135 片段，Qwen 已加载。
- `/mvp-api/jobs` 返回 28 条历史任务，没有未完成任务。
- 本轮观察到的活动指针是 `projects/demo -> releases/515b4148aa28447984a9ed7ebd76f210/demo`。不能沿用旧 8B 实验中的 release 路径或校验值。
- 审查时另有 `experiments/indexing-optimization/bench.py run --name fast-r1 --owl fast --face cpu` 在运行，父 PID 84089、子 PID 84090。该脚本第 157 行将默认特征模型设备设为 MPS；`--processor-device cpu` 只指定预处理，并不代表推理不占 Apple GPU。
- 审查采样时系统可用内存约 5.96 GiB、已用 swap 约 20.53 GiB。它们是整个系统的状态，不能归因于尚未开始的 PPE 模型。
- 公平耗时对照应等其他 GPU 实验结束；如果运行期间发生新的并发任务，保留进程采样并明确标识受干扰的轮次。不要擅自终止其他任务或正式服务。

## 固定输入和判定

1. 在看预测前冻结视频来源、SHA256、取样时间、帧图 SHA256 和人工标签。两模型读取完全相同的 BGR uint8 帧，不使用文件名、预期标签或人工裁剪作为模型输入。
2. 建议固定 `imgsz=640, rect=False, batch=1, conf=0.25, iou=0.7, half=False, device='mps'`，记录实际输入 shape。`imgsz=640` 配合默认 `rect=True` 仍可能生成不同高宽的矩形张量，不等于固定 640×640。
3. 仅比较头部佩戴检测。工地背景、是否正在施工、同人跨帧关联并不由单帧 detector 自动解决；本轮如未运行 tracker，不应声称已验证 ByteTrack 或持续事件。
4. 以实际 checkpoint 的 `model.names` 固定每个模型的 helmet、no-helmet、person 类别映射，不按两个模型的类 ID 数字直接比较。
5. 同一帧可能同时有戴帽和裸头者；不能将二者当互斥分类。无检测是 missed/unknown，不能自动作为未戴或全员戴帽。
6. 如果只有帧级/片段级标签而没有逐目标框标注，应报告“裸头帧检出/漏检/误报”和逐片段结果，不报告检测 mAP 或逐人准确率。unknown 标签单列，不强行纳入正负准确率。
7. 每模型使用相同固定阈值属于开箱基线。可展示阈值敏感性作为探索性附录，但不能在本测试集调出最优阈值后把其成绩称为独立测试效果。

## 耗时口径

- 下载、包安装和首次网络获取分别记录，全部排除出推理速度。
- `YOLO(path)` 可能只完成 CPU checkpoint 反序列化，首次 `predict()` 还包含设备迁移、模型融合和 predictor 初始化。需分别命名“checkpoint 加载”和“首次预测/预热”，或将设备迁移纳入完整加载区间，不能把首次预测全部称为纯推理。
- 每模型独立进程、顺序运行，固定 CPU 线程数和 MPS 设置；关闭输出保存、绘图和逐帧日志。先预热，再对相同帧完整执行 3 轮热测，并保留各轮总时长，不仅给平均值。
- MPS 异步执行：每次外部计时前后调用 `torch.mps.synchronize()`；实际推理时间应包含预处理、前向、NMS 和必要的结果转 CPU，另行说明是否包含读取 PNG。
- 视频解码/抽帧计时、PNG 读取计时、模型调用计时分别记录；若模型使用已提取帧，不能把模型调用 FPS 称为完整视频处理或在线检索 FPS。
- 建议报告每帧延迟中位数/P95、各轮总推理时长与帧数。只需使用同一个输入协议，不必为每个低影响脚本函数额外编写单元测试。
- 测试轮之间不要每帧执行 `empty_cache()`，这会改变正常缓存行为；不同模型通过退出进程隔离 allocator。

## 内存口径

项目 `scripts/resource_metrics.py` 的 50 ms 采样器可复用，但其类注释写“whole server process”，实际 `psutil.Process()` 是当前调用进程。PPE 报告应准确写当前实验进程。

- 分阶段保留加载前、加载后、预热后、热推理过程的 RSS、MPS allocated、MPS driver memory；同时给绝对峰值与基线，不能只报权重文件大小。
- RSS、MPS allocated、MPS driver memory 存在包含或重叠，绝不能相加称总内存；driver memory 也包含缓存/驱动分配，不是独立显存。
- 50 ms 采样可能漏掉短暂峰值。持续轮询加每帧完成后的同步采样，并将结果明确称“采样峰值”。
- system available 和 swap 为系统级；整个系统 swap 的总量或增量都不能直接当本模型占用。
- 避免长期保留 GPU Results 对象；及时转换为 CPU 普通数值。相同输入帧缓存若常驻，应对两模型保持一致并说明是否包含在 RSS 内。

## 正式检索不变的验证

开始前和结束后分别记录：

1. `os.readlink('projects/demo')` 和解析后绝对路径。
2. 活动 release 中 `action-index/{index.faiss,vectors.npy,metadata.json}`。
3. 活动 release 下 OWLv2、InsightFace 的 `store/**/index/*.faiss` 和 `store/**/features/*.faiss` 文件集合及 SHA256。
4. 活动 release 的 `metadata/internal.db` 与存在的 sidecar；若读取数据库，用只读 URI，避免产生写入。缩略图数据库若需要声称全部索引资产不变，也纳入快照。
5. 正式 `scripts/`、`ui/`、`wise/src/`、`wise_config.yaml`、依赖锁文件的 SHA256；最少包含 server、action_search、library_search、semantic_filter 和 Qwen 编码器。额外实验文件不能混入“生产改动”统计。
6. 主 `.venv` 的包名称/版本清单，以及服务 PID/创建时间、状态计数。新依赖应安装在实验独立目录，勿在主 `.venv` 执行升级。

快照中应包含路径集合差异，不能只比较原文件内容而漏掉新增/删除。SHA256 扫描和大量文件读取放在计时轮次之外。若其他工作同时改动生产文件，保留变化证据并说明来源未确定，不应自行回退或宣称所有正式内容完全未变。
