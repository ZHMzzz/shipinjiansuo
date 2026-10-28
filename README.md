# 寻帧：基于 WISE 2 的本机视频检索研究 MVP

这是可阅读、可修改的源码工程。WISE 负责视频解码、特征提取、向量索引和检索接口；本项目增加中文演示界面、姓名登记与查询、视频上传与后台入库、可播放的视频库，以及 Apple Silicon 环境适配。模型权重使用现成预训练模型，没有训练新模型。

当前自然语言检索（场景与动作）使用 Qwen3-VL-Embedding 2B；物体图片已按用户选择恢复 **WISE2 原生 OWLv2 链路**，人脸使用 InsightFace。SigLIP 停用，页面隐藏物体框。当前实现与验证见 [恢复 WISE2 物体检索](docs/恢复WISE2物体检索.md)，此前 Qwen 替换与图片修复文档保留阶段记录。

## 启动与演示

在此文件所在目录执行：

```sh
./scripts/start.sh
```

浏览器打开 **http://127.0.0.1:9670/mvp/**。原版 WISE 界面为 http://127.0.0.1:9670/demo/，API 文档为 http://127.0.0.1:9670/docs。服务仅监听本机。服务启动时加载物体、人脸和 Qwen 模型，并完成一次 Qwen 文字编码预热后才接收请求；首次查询无需再等待模型加载。启动期间页面暂不可用，进度见 `logs/server.log`，出现 `Application startup complete` 表示就绪。预热失败会终止启动，请按日志处理后重新启动。刷新网页不会重新加载模型，重启后端才会。

1. **默认视频库**：A 版缩略图列表，每页 50 条；点击卡片播放。顶部的“视频名称”按文件名包含关系筛选。
2. **自然语言**：描述场景或动作，例如“锅里正在炒蔬菜”“有人翻越围栏”。Qwen 片段索引先按视频保留分数最高的匹配片段，再按分数排序；每个视频只显示一条，点击从最佳片段起点播放。当前返回最多 100 个视频，每页 50 条，不提供多片段切换开关。
3. **以图搜图**：选择物体参考照片，通过 WISE2 原生 OWLv2 检索，按原生排名显示每个视频的最佳命中，隐藏物体框。自然语言与图片互斥，已删除 Qwen 图文联合检索；两者同时提交会被 API 拒绝。截图溯源功能仍已移除。
4. **人名**：输入已登记姓名，可单独查询，也可与自然语言或物体图片配合。人脸门槛默认 0.45。人物与场景在包含人脸时刻的片段内排序；人物与物体要求物体命中后 4 秒窗口内检测到此人物，不保证同一帧。
5. **新增视频**：图库标题旁点击“新增视频”，在弹窗中选择或拖入本地视频。显示真实传输进度、后台处理任务及失败重试。完成后自动刷新图库；已有查询结果保留，可重新查询纳入新视频。单个文件最大 1 GB，相同内容不会重复入库。关闭弹窗不停止传输，刷新或关闭网页会中断未完成的传输；上传成功后的后台处理须保持电脑与服务运行。
6. **人脸维护**：顶部按钮打开弹窗，左侧输入姓名、上传清晰单人照片，右侧显示已登记头像和姓名，支持确认删除。姓名同步到查询提示。多人或无人脸照片会拒绝登记。
7. **删除视频**：卡片底部点击“删除”，确认后删除整段视频及其所有命中片段。同步清理物体、人脸和 Qwen 索引、特征分片、缩略图、对应历史项目记录及托管视频文件；人物登记保持不变。处理中的上传与删除互斥，失败时可重试。外部导入源文件不在本应用托管目录内时保留。
8. **实验记录**：http://127.0.0.1:9670/mvp/actions.html 保留历史实验记录，不代表当前在线页面。

文件名条件先限定检索候选；语义相关性仍来自模型，不使用文件名替代语义编码。空条件返回完整视频库，查询中的分页只针对本次返回结果。

停止：`./scripts/stop.sh`。重新建索引：`./scripts/index.sh`，请先停止服务。新增视频优先使用网页上传。实现逻辑、文件位置及验证记录见 [视频上传与视频库](docs/视频上传与视频库.md)。

## 素材与结果

- 初始 14 段短视频，缩放到最长边 640、H.264、去掉音轨，保留原片段时间轴。逐项来源见 `data/manifest.json`，预览见 `data/contact-sheet.jpg`。网页上传的视频另存于 `data/uploads/`，保留可用音轨，当前总数以视频库为准。
- 12 段来自 [WISE 官方 Kinetics-6 测试包](https://thor.robots.ox.ac.uk/wise/assets/test/Kinetics-6.tar.gz)。原视频标识保留在文件名中。
- 背包视频来自 Pexels：[室内背包行走](https://www.pexels.com/video/a-man-walking-while-carrying-a-backpack-6326811/)、[街头背包片段](https://www.pexels.com/video/3551155/)。
- 独立查询图来自 [Unsplash：棕色皮背包](https://unsplash.com/photos/brown-leather-backpack-on-white-wall-OFHkPCkhYEY)。
- 查询图片位于 `data/queries/`，初始视频索引读取 `data/videos/`，网页上传读取 `data/uploads/`。检索不使用文件名标签。
- 人脸登记采用库内人物的另一时间帧，仅验证登记、相似度检索、定位链路；不代表跨摄像头、跨日期泛化评测。

这是小规模演示，不证明 Top-5 ≥85%、人物 Precision ≥90%、10GB 下 <3 秒或 7×24 稳定性。相似度不是正确概率。抽帧间隔 0.5 秒也意味着短暂事件可能漏掉；同一场景中相近帧可能存在时间歧义。中文检索使用预训练 Qwen 多帧编码器，8 秒窗口、4 秒步长，不提供帧级动作起止判定；也不能保证复合事件或否定条件识别准确。没有训练或微调模型。

## 源码学习顺序

先沿一条完整的数据链路读，随后再研究模型细节：

```text
视频 → 每秒2帧 → InsightFace → 人脸 FAISS + SQLite
视频 → 每秒2帧 → OWLv2 → 物体区域 FAISS + SQLite
物体图片 → WISE OWLv2 原生查询 ──────────┘
姓名 → 本地人物登记库 → 人脸向量 ────────┘
视频 → 8秒窗口 / 4秒步长 / 16帧 → Qwen FP16 → 共享片段 FAISS
自然语言 → Qwen 文字查询编码 ─────────┘
         → 视频ID、片段时间范围 → 从片段起点播放
人物＋场景 → 人脸门槛 → 包含命中时刻的 Qwen 片段 → 融合排序
```

| 阅读入口 | 需要理解的问题 |
|---|---|
| `scripts/index.sh` | 提取特征和建索引是两个独立阶段 |
| `wise/src/wise/_main/extract_features.py` | 如何遍历媒体、抽帧、写特征与元数据 |
| `wise/src/wise/dataloader/streamreader.py` | 解码、采样率与时间戳如何对应 |
| `wise/src/wise/feature/feature_extractor_factory.py` | 模型ID怎样映射到具体编码器 |
| `wise/src/wise/feature/qwen3vl.py` | 当前文字与视频片段编码，以及历史多模态编码实验 |
| `scripts/qwen_project.py` | 校验保留数据、只移除 SigLIP 资产并保留 OWLv2 |
| `wise/src/wise/feature/insightface.py` | 检测、对齐、身份特征，以及多人帧的多条向量 |
| `wise/src/wise/api/services/search/local.py` | FAISS 内积检索怎样回查视频元数据 |
| `wise/src/wise/api/standalone/search.py` | 查询解析、候选结果、时间窗合并与返回结构 |
| `scripts/server.py` | 姓名→人脸向量 SQLite 映射、图库与查询 API；复用 WISE 搜索服务 |
| `ui/catalog.js` | 图库分页、互斥输入、查询结果与定位播放 |
| `scripts/ingestion.py` | 上传去重、持久任务队列、转码、增量提取、重建索引及版本发布 |
| `ui/maintenance.js` | 上传进度、任务轮询、人脸登记与删除 |
| `scripts/video_deletion.py` | 视频身份校验、副本清理、历史索引清理与中断恢复 |
| `scripts/library_search.py` | 自然语言、原生物体和人物检索路由，文件名候选限定 |
| `scripts/restore_owl_project.py` | 用 WISE 合并旧物体特征、补提取缺失视频并原子恢复 |
| `scripts/combined_search.py` | 人物身份门槛、同视频片段约束与 WISE RRF 排名融合 |
| `scripts/action_search.py` | 滑动窗口、多帧抽样、增量共享片段索引、文字查询与片段排序 |
| `scripts/benchmark_actions.py` | Qwen 视频片段编码、FAISS 查询与 CPU/Metal 内存观测 |

中文检索按相似度展示不同命中片段；物体、人脸界面按模型原始排名保留每个视频的最强命中，不根据预期答案重排。不同模型的分数尺度不同，不能跨模式比较。图库不叠加检测框；人脸阈值默认 0.45，仅是可调演示值，需要独立正负样本标定。

截图定位有独立的二阶段流程：从 Qwen 全部片段中按每个视频的最高相似度取最多 5 个候选视频，按约 10Hz 解码，计算缩小灰度图的归一化相关性，再按实际画面分数排序。这比单用语义向量更适合重复动作的具体时刻定位。每个候选最多扫描前 60 秒，适用于当前短片 MVP；尚不支持长视频完整精排、强裁剪和镜像变化。截图分数是相关性，仍不是正确概率。扩展长视频时应在候选时间窗内扫描。

## 本机环境与改动

- Apple M5，32GB 统一内存；Python 3.11；PyTorch 2.8 / Torchaudio 2.8；FFmpeg 6.1.2。
- 当前 Qwen3-VL-Embedding 2B 使用 MPS / FP16；InsightFace `buffalo_l` 使用 CPU。OWLv2 使用 MPS，已恢复正式检索与上传特征提取；SigLIP 只保留历史源码。
- Qwen 动作模型使用 MPS、FP16、16 帧、总像素预算 1,048,576；配置集中在 `scripts/action_search.py` 的 `SETTINGS`，复用 WISE 编码器。Qwen 缓存恢复方式见 [在线动作检索](docs/在线动作检索.md)。
- `wise_config.yaml` 禁用模型编译，避免上游默认 CUDA/编译设置不适合此 Mac。
- `.media-runtime/` 独立安装 FFmpeg 6；`DYLD_FALLBACK_LIBRARY_PATH` 让 Torchaudio 能找到它，同时避免替换 OpenCV 自带 FFmpeg/libiconv。
- `scripts/fix_openmp.py` 将本虚拟环境中 FAISS、scikit-learn 的 OpenMP 链接到 PyTorch 同一份运行时，原文件保留为 `.original`；没有使用忽略重复运行时错误的开关。`OMP_NUM_THREADS=1` 控制 CPU 并发，Apple GPU 仍启用。重装这些包后需要重新检查这一适配。
- 历史兼容补丁：`mlfoundation_openclip.py` 将 MPS 上的多帧图像批次拆成单帧调用，绕过 PyTorch 2.8 在 M5 上 SigLIP 注意力池化的原生缓冲区错误；向量及其原始时间戳顺序不变。
- 历史兼容补丁：`transformers_owlv2.py` 同样限制 MPS 物体提取为单帧批次，避免 960px / 3600 token 的八帧注意力计算占满统一内存。
- 修复上游开发版两处缩略图问题：提取时使用 `MediaChunkType.THUMBNAILS` 枚举键；API 获取缩略图时使用 `media_id`，而不是向量 `id`。`scripts/backfill_thumbnails.py` 用于给已建索引补缩略图；正常新建索引已直接修复。
- 历史 WISE 接口兼容：外部图片查询也应用编码器的查询侧向量转换，修复 OWLv2 把入库特征直接用于查询导致分数饱和的问题；内部与外部图片使用相同转换，CLIP/人脸默认转换不改变向量。
- WISE 的 `requirements.txt` 在 macOS 使用 `onnxruntime`，固定 InsightFace 0.7.3 与 Transformers 4.x，移除 Triton 不需要的安装选项。
- SQLAlchemy 限定 2.0 系列，避免 2.1 对上游已编译 SQL 执行方式的不兼容。
- Python 依赖锁定：`requirements-macos.lock.txt`；解码环境锁定：`media-runtime-osx-arm64.lock.txt`；环境变量集中于 `scripts/env.sh`。
- WISE 上游：[GitHub 源码](https://github.com/ox-vgg/wise)，基准提交 `fcfa443fbb46eb361bb19151339338616838a5b5`，版本 2.1.0+dev。在 `wise/` 执行 `git diff` 可查看上游修改。本项目新增功能在根目录独立保存，容易区分。

模型下载缓存通常位于 `~/.cache/huggingface/`、`~/.insightface/`。演示启动默认离线读取已下载模型；删除缓存后先执行 `source scripts/env.sh` 和 `HF_HUB_OFFLINE=0 python scripts/warm_models.py` 重新下载。人名与人脸向量保存在 `data/people.sqlite`，删除登记可通过页面完成。

本机重建 Python 环境可执行 `./scripts/setup.sh`；脚本依赖已有的 Homebrew Python 3.11、Node 22、libmagic 和此目录中的 micromamba。它不是跨操作系统的一键安装器。

## GitHub 仓库与首次恢复

仓库包含根目录应用、已修改的完整 `wise/` 源码、测试、文档、依赖锁文件、演示视频及素材来源。`wise/` 作为普通源码目录提交，克隆后无需另外初始化子模块；原始开发目录仍保留 WISE 上游 Git 历史，但新克隆的仓库只包含本工程的历史。

虚拟环境、模型缓存、生成索引、日志、网页上传的视频和本机人物登记数据库不随源码上传；重复的视频 ZIP 包及原始下载缓存也不提交。已有实验报告和处理后的演示素材保留在仓库中。

在满足上述 Apple Silicon 环境前提后，首次克隆可按以下顺序恢复：

```sh
git clone https://github.com/ZHMzzz/shipinjiansuo.git
cd shipinjiansuo
./scripts/setup.sh
./scripts/index.sh
./scripts/start.sh
```

首次恢复会下载模型并重新建立索引；人物需要在页面重新登记，之前网页上传的视频需要重新上传。Qwen 片段索引在首次启动后自动补建，完整覆盖后发布；缓存准备另见 [在线动作检索](docs/在线动作检索.md)。

WISE 代码为 Apache-2.0；InsightFace 代码与预训练模型许可不同，当前按非商业研究使用预训练模型。素材权利归各自作者，来源保留以便课题引用。

图片入口仅保留物体图片检索，移除记录见 [ADR 0004](docs/adr/0004-remove-screenshot-tracing.md)。

中文语义与动作已统一为“中文检索”，负向描述和重排功能已移除，见 [ADR 0005](docs/adr/0005-simplify-semantic-search.md)。

A 版正式页面接入及移除图文联合查询的规格与验证见 [.scratch/video-library-page/spec.md](.scratch/video-library-page/spec.md)。原型保留在 `codex/prototype-video-library` 分支。
