# 寻帧：基于 WISE 2 的本机视频检索研究 MVP

这是可阅读、可修改的源码工程。WISE 负责视频解码、特征提取、向量索引和检索接口；本项目增加中文演示界面、姓名登记与查询、视频上传与后台入库、可播放的视频库，以及 Apple Silicon 环境适配。模型权重使用现成预训练模型，没有训练新模型。

当前自然语言检索（场景与动作）使用 Qwen3-VL-Embedding 2B；物体图片使用 **WISE2 原生 OWLv2 链路**，人脸使用 InsightFace。SigLIP 停用，页面隐藏物体框。本文描述当前行为；模型替换、截图和早期动作实验文档保留历史记录，不作为安装步骤。

## 首次安装

当前安装脚本面向 **Apple Silicon Mac（原生 arm64 终端）**，本机验证配置为 M5 / 32 GB 内存。Windows、Linux、Intel Mac 尚未适配。需要先安装 Xcode Command Line Tools、Homebrew、Python 3.11、Node 22 和 libmagic；完整命令、验收与排错见 [首次安装与运行指南](docs/首次安装与运行.md)。

前置工具安装完成后：

```sh
git clone --depth 1 https://github.com/ZHMzzz/shipinjiansuo.git
cd shipinjiansuo
./scripts/setup.sh --check
./scripts/setup.sh
./scripts/index.sh
./scripts/start.sh
```

`setup.sh` 下载依赖与模型并构建前端；`index.sh` 对仓库自带的 14 个视频建立 OWLv2 与人脸索引；首次启动后后台补建 Qwen 片段索引。网页可打开不代表片段补建已完成，验收方式见指南。模型、索引、原机上传视频与人物登记不随 Git 下载。

## 日常启动与演示

已完成首次安装和建库后，在此文件所在目录执行：

```sh
./scripts/start.sh
```

浏览器打开 **http://127.0.0.1:9670/mvp/**。原版 WISE 界面为 http://127.0.0.1:9670/demo/。`/docs` 是上游 API 文档入口，但全局 OpenAPI Schema 存在已知兼容问题，不用它判断服务是否就绪。服务仅监听本机。启动时加载物体、人脸和 Qwen 模型，并完成一次 Qwen 文字编码预热后才接收请求；首次查询无需再等待模型加载。启动期间页面暂不可用，进度见 `logs/server.log`，出现 `Application startup complete` 表示 HTTP 服务就绪，片段补建仍可能进行中。预热失败会终止启动，请按日志处理后重新启动。刷新网页不会重新加载模型，重启后端才会。

索引发布成功后自动保留当前版本和上一发布版本，回收更早的已登记版本。入库、补建、删除和迁移共用此规则；清理失败记入日志，后台每 30 秒及重启后重试，不撤销已成功的入库。新版本构建期间仍需额外容纳一份完整索引，视频和运行环境不受版本回收影响。旧安装的未登记目录需先核实并通过 `scripts.release_retention.adopt_history` 登记，未知目录及未发布草稿不会自动删除；详见 [版本保留策略](docs/adr/0008-retain-two-published-releases.md)。

1. **默认视频库**：A 版缩略图列表，每页 50 条；点击卡片播放。顶部的“视频名称”按文件名包含关系筛选。
2. **自然语言**：描述场景或动作，例如“锅里正在炒蔬菜”“有人翻越围栏”。通过 WISE 原生 Qwen 片段检索与时间窗合并，再按视频保留最先返回的命中；每个视频只显示一条，从代表命中时间播放。纯文字和图文查询不再使用历史弱匹配过滤，分数低也可能返回结果；最多召回 100 条片段候选，按视频去重后通常少于 100 个视频，每页 50 条。人物＋场景仍有人脸门槛及场景分过滤；[相关度过滤实验](experiments/relevance-filter/README.md) 仅作历史记录。
3. **以图搜图**：选择物体参考照片，通过 WISE2 原生 OWLv2 检索，按原生排名显示每个视频的最佳命中，隐藏物体框。同时输入自然语言与参考图片时，由 Qwen 联合编码检索已有片段索引；纯图片继续走 OWLv2。截图溯源功能仍已移除。
4. **人名**：从下拉框选择已登记姓名，可单独查询，也可与自然语言或物体图片配合。人脸门槛默认 0.45。人物与场景在包含人脸时刻的片段内排序；人物与物体要求物体命中后 4 秒窗口内检测到此人物，不保证同一帧。
5. **新增视频**：图库标题旁点击“新增视频”，在弹窗中多选或拖入多个本地视频，按顺序传输。显示真实传输进度、后台处理任务及失败重试。完成后自动刷新图库；已有查询结果保留，可重新查询纳入新视频。单个文件最大 1 GB，相同内容不会重复入库。关闭弹窗不停止传输，刷新或关闭网页会中断未完成的传输；上传成功后的后台处理须保持电脑与服务运行。
6. **人脸维护**：顶部按钮打开弹窗，输入姓名，可一次选择多张清晰单人照片；输入已有姓名或点击人物旁的“添加照片”会追加，保留原登记。每人最多 10 张，每张最大 10 MB；一次提交全部检查通过后保存，重复上传同一文件自动跳过。右侧显示照片数量，“查看照片”可逐张删除；最后一张须通过“删除人物”移除。已有单张登记自动迁移，无需重新登记或重建视频索引。多人或无人脸照片会拒绝登记。
7. **删除视频**：卡片底部点击“删除”，确认后删除整段视频及其所有命中片段。同步清理物体、人脸和 Qwen 索引、特征分片、缩略图、对应历史项目记录及托管视频文件；人物登记保持不变。处理中的上传与删除互斥，失败时可重试。外部导入源文件不在本应用托管目录内时保留。
8. **实验记录**：文字报告与实验源码保留在 `experiments/`；逐次结果、截图和实验视频不随源码交付。`/mvp/actions.html` 仅在本机已有对应实验数据时可回放，不影响主页面检索。数据恢复方式见 [实验目录说明](experiments/README.md)。

文件名条件先限定检索候选；语义相关性仍来自模型，不使用文件名替代语义编码。空条件返回完整视频库，查询中的分页只针对本次返回结果。

纯图片及人物＋图片检索先由 WISE OWLv2 返回最多 500 条区域候选，再按视频保留最高分命中，最后按 API 的 `limit` 截取视频（默认及上限 100）。这两个数量独立；该路径直接调用 SearchService，原生 WISE HTTP 接口的 `max_search_results` 不控制这项内部候选预算。扩大候选不会改变相似度排序或保证颜色匹配，也不保证找全第 500 条以后的相关区域。

主页面播放上传视频时优先读取原文件，加载或解码失败后自动切换已有的 H.264 压缩版，并保留命中或当前播放时间。浏览器限制自动播放时可点击播放按钮，不会因此切换清晰度。列表分辨率、大小显示原文件信息；无原片的历史视频继续使用现有文件。检索抽帧、特征、时间范围和缩略图仍来自压缩版，不重建索引。原片分辨率首次读取后按文件状态缓存，旋转视频按显示方向报告宽高；无法读取分辨率时显示未知。

停止：`./scripts/stop.sh`。重新建索引：`./scripts/index.sh`，请先停止服务。新增视频优先使用网页上传。实现逻辑、文件位置及验证记录见 [视频上传与视频库](docs/视频上传与视频库.md)。

Apple Silicon 入库默认使用 OWLv2 快速 CPU 预处理、MPS FP16 权重和 InsightFace CoreML，网页上传与 `index.sh` 共用 `scripts/extract_frames.py`。这些适配只在提取子进程启用，查询端保持原处理器/精度；Qwen 继续写入原生 WISE。固定形状模型副本与编译缓存位于 `data/model-cache/frame-acceleration/`，不改下载的原模型。首次编译可能较慢；后端不可用时任务明确失败，已有视频库继续可用。

OWLv2 入库默认启用官方 SDPA，模型推理独立运行于 Torch 2.14.1 / Transformers 5.21.0，解码和查询继续使用原环境。先执行 `./scripts/setup_owl_sdpa.sh` 创建 `data/model-cache/owl-sdpa/runtime/`（完整 `setup.sh` 已包含此步骤）；依赖固定在 `requirements-owl-sdpa.lock.txt`，不安装 torchaudio。模型仍为原权重、FP16、960px、batch=1，不启用 `torch.compile`。输入与输出通过可复用 mmap 传递，推理子进程随帧提取进程复用和退出；版本不符、缺失运行时、崩溃或推理超时均明确失败，不静默回退。

仅回退 SDPA 时，停止服务后执行 `VIDEO_OWL_SDPA=0 ./scripts/start.sh`，保留快速预处理、FP16、CoreML 和模型复用；CLI 同样支持该开关。`VIDEO_OWL_SDPA_PYTHON` 可指定符合固定版本要求的独立 Python 可执行文件。上述开关只影响后续入库，旧索引不重算。35.44 秒视频的隔离测试中，模型复用后完整入库均值减少约 25%，首次加载帧模型时约 5.8%；9 个参考图中有一个查询第五名发生变化，并非逐位无损，详见 [完整 SDPA 回归](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/indexing-optimization/owl-sdpa-full-20261028/summary.json)。

Qwen 新增视频及缺失片段补建默认采用同权重线性图像块投影：仅在 `ActionEngine.encode_segment` 内将全块 Conv3d 等价改写为 Linear，结束或异常后恢复原方法；文字查询保持原路径。批量仍为 1，FP16、16 帧及像素预算不变，不自动重编已有视频。当前模型结构不符合适配要求时明确报错。此优化有浮点差异，完整对照的 Qwen 编码耗时减少约 41%，不代表完整上传提速；测试边界见 [Qwen 线性投影实验](experiments/indexing-optimization/QWEN_LINEAR.md)。

如需回退后续入库任务，先停止服务，再以 `VIDEO_FRAME_ACCELERATION=0 ./scripts/start.sh` 启动；命令行可用 `VIDEO_FRAME_ACCELERATION=0 ./scripts/index.sh`。取消该环境变量并重启即可恢复默认加速。此开关不会重算已有索引。

网页上传默认复用独立的串行帧提取进程及 OWLv2/InsightFace 模型，项目、数据加载器和写入对象仍按任务创建。首次任务需要加载模型，连续任务可省去重复启动和加载时间。进程空闲 600 秒后退出以释放模型内存；失败、崩溃或超时会销毁进程，失败任务仍通过网页“重试”从新临时项目开始，下一条任务自动启动新进程。仅成功校验的临时项目可以发布，Qwen 和查询端配置不变。

可在启动服务时设置 `VIDEO_FRAME_WORKER_REUSE=0` 回退为每条视频新建提取进程，保留既有帧加速；`VIDEO_FRAME_WORKER_IDLE_SECONDS`、`VIDEO_FRAME_WORKER_STARTUP_SECONDS`、`VIDEO_FRAME_WORKER_TIMEOUT_SECONDS` 分别控制空闲回收、进程启动和单次提取超时，默认 600、120、1800 秒，必须为有限正数。长视频如需超过 30 分钟的帧提取时间，可调高最后一项。修改配置、代码或模型文件后须重启服务。`index.sh` 仍为单次命令，不保留后台进程。

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
姓名 → 本地人物登记库 → 多张人脸向量分别检索、同一命中取最高分 ─┘
视频 → 8秒窗口 / 4秒步长 / 16帧 → Qwen FP16 → 共享片段 FAISS
自然语言 → Qwen 文字查询编码 ─────────┘
参考图片＋自然语言 → Qwen 联合编码 ────┘
         → 视频ID、片段时间范围 → 从片段起点播放
人物＋场景 → 人脸门槛 → 包含命中时刻的 Qwen 片段 → 融合排序
```

| 阅读入口 | 需要理解的问题 |
|---|---|
| `scripts/index.sh` | 提取特征和建索引是两个独立阶段 |
| `scripts/extract_frames.py` / `scripts/frame_acceleration.py` | 仅在入库子进程启用加速，隔离在线查询模型 |
| `scripts/frame_worker.py` | 网页入库的模型复用、超时、失败重建及空闲回收 |
| `scripts/owl_sdpa.py` / `scripts/setup_owl_sdpa.sh` | 独立 OWLv2 SDPA 运行时、共享内存通信和固定依赖安装 |
| `wise/src/wise/_main/extract_features.py` | 如何遍历媒体、抽帧、写特征与元数据 |
| `wise/src/wise/dataloader/streamreader.py` | 解码、采样率与时间戳如何对应 |
| `wise/src/wise/feature/feature_extractor_factory.py` | 模型ID怎样映射到具体编码器 |
| `wise/src/wise/feature/qwen3vl.py` | 当前文字与视频片段编码，以及历史多模态编码实验 |
| `scripts/qwen_project.py` | 校验保留数据、只移除 SigLIP 资产并保留 OWLv2 |
| `wise/src/wise/feature/insightface.py` | 检测、对齐、身份特征，以及多人帧的多条向量 |
| `wise/src/wise/api/services/search/local.py` | FAISS 内积检索怎样回查视频元数据 |
| `wise/src/wise/api/standalone/search.py` | 查询解析、候选结果、时间窗合并与返回结构 |
| `scripts/server.py` | 姓名→人脸向量 SQLite 映射、图库与查询 API；复用 WISE 搜索服务 |
| `ui/catalog.js` | 图库分页、图文输入、查询结果与定位播放 |
| `scripts/ingestion.py` | 上传去重、持久任务队列、转码、增量提取、重建索引及版本发布 |
| `ui/maintenance.js` | 上传进度、任务轮询、人脸登记与删除 |
| `scripts/people_store.py` / `scripts/people_api.py` | 多照片登记、旧数据迁移、批量校验与照片维护 |
| `scripts/face_search.py` | 多张登记照分别查询，同一个索引人脸取最高分，复用原有人脸门槛 |
| `scripts/video_deletion.py` | 视频身份校验、副本清理、历史索引清理与中断恢复 |
| `scripts/library_search.py` | 自然语言、原生物体和人物检索路由，文件名候选限定 |
| `scripts/restore_owl_project.py` | 用 WISE 合并旧物体特征、补提取缺失视频并原子恢复 |
| `scripts/combined_search.py` | 人物身份门槛、同视频片段约束与 WISE RRF 排名融合 |
| `scripts/action_search.py` | 滑动窗口、多帧抽样、增量共享片段索引、文字查询与片段排序 |
| `scripts/benchmark_actions.py` | Qwen 视频片段编码、FAISS 查询与 CPU/Metal 内存观测 |

主页面按视频去重展示查询结果，不根据预期答案重排。不同模型的分数尺度不同，不能跨模式比较。图库不叠加检测框；人脸门槛默认 0.45，主页面不提供门槛控件，API 可传入该参数，需要独立正负样本标定。

截图溯源、负向描述及重排均已移除；参考图片用于物体或图文联合检索，不提供来源识别与精细截图时间定位。

## 本机环境与改动

- Apple M5，32GB 统一内存；Python 3.11；PyTorch 2.8 / Torchaudio 2.8；FFmpeg 6.1.2。
- 当前 Qwen3-VL-Embedding 2B 使用 MPS / FP16；InsightFace `buffalo_l` 使用 CPU。OWLv2 使用 MPS，已恢复正式检索与上传特征提取；SigLIP 只保留历史源码。
- Qwen 动作模型使用 MPS、FP16、16 帧、总像素预算 1,048,576；配置集中在 `scripts/action_search.py` 的 `SETTINGS`，复用 WISE 编码器。模型缓存恢复见 [首次安装与运行](docs/首次安装与运行.md)。
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
- WISE 上游：[GitHub 源码](https://github.com/ox-vgg/wise)，基准提交 `fcfa443fbb46eb361bb19151339338616838a5b5`，版本 2.1.0+dev。克隆本仓库后 `wise/` 是普通源码目录，`git diff` 显示相对本仓库的未提交改动，不是与上游的对比。

模型下载缓存通常位于 `~/.cache/huggingface/`、`~/.insightface/`。演示启动默认离线读取已下载模型；删除缓存后先执行 `source scripts/env.sh` 和 `HF_HUB_OFFLINE=0 python scripts/warm_models.py` 重新下载。人名与人脸向量保存在 `data/people.sqlite`，删除登记可通过页面完成。`person_photos` 保存全部登记照；原 `people` 表保留首张照片供历史工具兼容，正式姓名及组合检索使用全部照片。多照片按同一视频人脸的最高匹配分合并，不平均向量；默认门槛仍为 0.45，照片增多后的漏检与误认需独立评测。

环境准备使用 `./scripts/setup.sh`；`--check` 只检查系统前置工具。重装前先停止服务，完整操作见 [首次安装与运行](docs/首次安装与运行.md)。仓库已附带 Apple Silicon 版 `tools/bin/micromamba`，无需额外安装。

## GitHub 仓库与首次恢复

仓库包含根目录应用、已修改的完整 `wise/` 源码、测试、文档、依赖锁文件、演示视频及素材来源。`wise/` 作为普通源码目录提交，克隆后无需另外初始化子模块；原始开发目录仍保留 WISE 上游 Git 历史，但新克隆的仓库只包含本工程的历史。

虚拟环境、模型缓存、生成索引、日志、网页上传的视频和本机人物登记数据库不随源码上传。两个额外视频集合「演示视频_摔倒火情抽烟」「真实监控演示视频_摔倒火情抽烟」、临时工作记录、截图及实验生成的数据也不再跟踪；本机已有文件保留。首次运行所需的 `data/videos/`、`data/queries/`、安装工具、测试源码及文字报告继续交付。

首次运行推荐上方 `--depth 1` 浅克隆，或下载最新源码 ZIP，避免拉取历史大文件。普通完整克隆仍包含旧提交中的实验产物；这次清理不改写历史。需要完整开发历史时可执行 `git fetch --unshallow`。实验报告中的历史证据链接指向清理前固定版本，见 [实验目录说明](experiments/README.md)。

首次安装命令见本文顶部和 [完整指南](docs/首次安装与运行.md)。先完成建库再启动服务；人物需要在页面重新登记，之前网页上传的视频需要重新上传。不要把旧机器生成的索引直接搬到不同绝对路径后当成新安装使用。

WISE 代码为 Apache-2.0；InsightFace 代码与预训练模型许可不同，当前按非商业研究使用预训练模型。素材权利归各自作者，来源保留以便课题引用。

图片入口仅保留物体图片检索，移除记录见 [ADR 0004](docs/adr/0004-remove-screenshot-tracing.md)。

中文语义与动作已统一为“中文检索”，负向描述和重排功能已移除，见 [ADR 0005](docs/adr/0005-simplify-semantic-search.md)。

A 版正式页面接入及当时移除图文联合查询的历史规格与验证见 [.scratch/video-library-page/spec.md](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/.scratch/video-library-page/spec.md)。原型保留在 `codex/prototype-video-library` 分支。图文联合检索现已恢复：参考图与自然语言联合编码进入 Qwen 片段检索，纯图片仍走 OWLv2；详见本文功能说明及 [运行指南](docs/首次安装与运行.md)。
