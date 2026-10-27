# 寻帧：基于 WISE 2 的本机视频检索研究 MVP

这是可阅读、可修改的源码工程。WISE 负责视频解码、特征提取、向量索引和检索接口；本项目增加中文演示界面、姓名登记与查询、视频上传与后台入库、可播放的视频库，以及 Apple Silicon 环境适配。模型权重使用现成预训练模型，没有训练新模型。

原有四类检索、组合查询和在线动作检索均已完成本机验证。初版记录见 [验证报告](docs/验证报告.md)，新增动作检索见 [在线动作检索](docs/在线动作检索.md)。

## 启动与演示

在此文件所在目录执行：

```sh
./scripts/start.sh
```

浏览器打开 **http://127.0.0.1:9670/mvp/**。原版 WISE 界面为 http://127.0.0.1:9670/demo/，API 文档为 http://127.0.0.1:9670/docs。服务仅监听本机。首次启动需要加载模型，进度见 `logs/server.log`。

1. **中文语义**：输入“一个人背着背包走路”“一个人在跑步”“锅里正在炒蔬菜”，开始检索，点击结果播放。
2. **截图溯源**：点“使用 2.25 秒演示截图”，检索跑步原片段。先用 SigLIP2 召回候选，再做画面比对细化时间。也可上传自己的截图。
3. **物体图片**：点“使用独立背包照片”，检索视频中相似的背包区域。该图没有收入视频索引；这个功能检索外观或语义相似物体，不能证明是同一个实物。
4. **人物姓名**：点“载入演示人物”，再按“演示人物A”检索。可以展开登记区上传单人照片、输入姓名，完成登记后检索。多人照片与未检出人脸的照片会拒绝登记，避免静默选错人。演示人物A仅是别名，不推断真实姓名。

5. **视频库与上传**：点击左侧「05 视频库与上传」，或直接打开 http://127.0.0.1:9670/mvp/#library。查看全部已入库视频、按名称筛选、点击卡片从头播放。选择本机视频后点击「上传并处理」，页面显示上传进度和后台处理阶段，完成后自动参与检索，无需重启。单个文件最大 1GB；任务串行处理，关闭网页后仍在后台继续，但电脑与服务须保持运行。失败任务显示原因并可重试，内容相同的文件不会重复入库。
6. **图文组合**：点击「06 图文组合」，使用背包演示图，文字分别输入「在街道上」「在室内」，可比较两段背包视频的排名变化。
7. **负向查询**：语义、物体、图文和人物模式都有「降低这些内容的排名」。这是向量排序条件，不是保证排除的规则过滤。
8. **人物＋场景**：「04 人物姓名」载入演示人物A，场景输入「坐在汽车里」。先应用人脸门槛，再对同一帧场景排序；显示独立的人脸相似度与融合排序分。
9. **在线动作检索**：点击「07 动作片段」，输入任意中文描述，例如「有人在仓库里突然失去平衡倒在地上」。Qwen 2B 实时编码文字，查询已入库视频的多帧片段向量；卡片显示起止时间，点击从片段起点播放。同一视频可返回多个片段。旧视频自动补建，普通上传自动增加动作索引。首次查询可能需要加载模型。实现与边界见 [在线动作检索](docs/在线动作检索.md)。
10. **实验记录**：http://127.0.0.1:9670/mvp/actions.html 保留前期两种配置的预设结果、原始排名、耗时和视频；它与当前在线动作查询是不同入口。详见 [组合查询与动作试验](docs/组合查询与动作试验.md)。

停止：`./scripts/stop.sh`。重新建索引：`./scripts/index.sh`，请先停止服务。新增视频优先使用网页上传。实现逻辑、文件位置及验证记录见 [视频上传与视频库](docs/视频上传与视频库.md)。

## 素材与结果

- 初始 14 段短视频，缩放到最长边 640、H.264、去掉音轨，保留原片段时间轴。逐项来源见 `data/manifest.json`，预览见 `data/contact-sheet.jpg`。网页上传的视频另存于 `data/uploads/`，保留可用音轨，当前总数以视频库为准。
- 12 段来自 [WISE 官方 Kinetics-6 测试包](https://thor.robots.ox.ac.uk/wise/assets/test/Kinetics-6.tar.gz)。原视频标识保留在文件名中。
- 背包视频来自 Pexels：[室内背包行走](https://www.pexels.com/video/a-man-walking-while-carrying-a-backpack-6326811/)、[街头背包片段](https://www.pexels.com/video/3551155/)。
- 独立查询图来自 [Unsplash：棕色皮背包](https://unsplash.com/photos/brown-leather-backpack-on-white-wall-OFHkPCkhYEY)。
- 查询图片位于 `data/queries/`，视频索引只读取 `data/videos/`。检索不使用文件名标签。
- 人脸登记采用库内人物的另一时间帧，仅验证登记、相似度检索、定位链路；不代表跨摄像头、跨日期泛化评测。

这是小规模演示，不证明 Top-5 ≥85%、人物 Precision ≥90%、10GB 下 <3 秒或 7×24 稳定性。相似度不是正确概率。抽帧间隔 0.5 秒也意味着短暂事件可能漏掉；同一场景中相近帧可能存在时间歧义。动作入口使用预训练 Qwen 多帧编码器，8 秒窗口、4 秒步长，不提供帧级动作起止判定；也不能保证复合事件或否定条件识别准确。没有训练或微调模型。

## 源码学习顺序

先沿一条完整的数据链路读，随后再研究模型细节：

```text
视频 → WISE 解码 / 每秒2帧 → 三路编码器 → FAISS + SQLite
中文 → SigLIP2 文本向量 ────────────────┘
截图 → SigLIP2 图像向量 ────────────────┘
物体图片 → OWLv2 区域特征 ──────────────┘
姓名 → 本地人物库 → InsightFace 向量 ────┘
视频 → 8秒窗口/4秒步长/16帧 → Qwen FP16 → 独立动作 FAISS
任意中文动作描述 → 同一 Qwen 文本编码 ──────────┘
         → 相似度排序 → 视频ID、时间范围、目标框 → 从命中时刻播放
```

| 阅读入口 | 需要理解的问题 |
|---|---|
| `scripts/index.sh` | 提取特征和建索引是两个独立阶段 |
| `wise/src/wise/_main/extract_features.py` | 如何遍历媒体、抽帧、写特征与元数据 |
| `wise/src/wise/dataloader/streamreader.py` | 解码、采样率与时间戳如何对应 |
| `wise/src/wise/feature/feature_extractor_factory.py` | 模型ID怎样映射到具体编码器 |
| `wise/src/wise/feature/mlfoundation_openclip.py` | 中文与整帧图像如何变成可比较向量 |
| `wise/src/wise/feature/transformers_owlv2.py` | 整帧与目标区域的差别、检测框与区域特征 |
| `wise/src/wise/feature/insightface.py` | 检测、对齐、身份特征，以及多人帧的多条向量 |
| `wise/src/wise/api/services/search/local.py` | FAISS 内积检索怎样回查视频元数据 |
| `wise/src/wise/api/standalone/search.py` | 查询解析、候选结果、时间窗合并与返回结构 |
| `scripts/server.py` | 我们增加的姓名→人脸向量 SQLite 映射，以及截图候选的时间细化；复用 WISE 搜索服务 |
| `ui/app.js` | 检索请求、帧与片段结果呈现、起止时间与视频跳转播放 |
| `scripts/ingestion.py` | 上传去重、持久任务队列、转码、增量提取、重建索引及版本发布 |
| `ui/library.js` | 上传进度、任务轮询、视频库列表、名称筛选与从头播放 |
| `scripts/combined_search.py` | 人物身份门槛、同帧语义约束与 WISE RRF 排名融合 |
| `scripts/action_search.py` | 滑动窗口、多帧抽样、增量动作索引、任意中文编码与片段排序 |
| `scripts/benchmark_actions.py` | Qwen 视频片段编码、FAISS 查询与 CPU/Metal 内存观测 |

语义、物体、人脸界面按模型原始排名保留每个视频的最强命中，不根据预期答案重排。不同模型的分数尺度不同，不能跨模式比较。物体与人脸显示返回的目标框；人脸阈值默认 0.45，仅是可调演示值，需要独立正负样本标定。

截图定位有独立的二阶段流程：从 WISE 的前 100 条帧结果取最多 5 个候选视频，按约 10Hz 解码，计算缩小灰度图的归一化相关性，再按实际画面分数排序。这比单用语义向量更适合重复动作的具体时刻定位。每个候选最多扫描前 60 秒，适用于当前短片 MVP；尚不支持长视频完整精排、强裁剪和镜像变化。截图分数是相关性，仍不是正确概率。扩展长视频时应在候选时间窗内扫描。

## 本机环境与改动

- Apple M5，32GB 统一内存；Python 3.11；PyTorch 2.8 / Torchaudio 2.8；FFmpeg 6.1.2。
- SigLIP2 `ViT-B-16-SigLIP2-256/webli` 与 OWLv2 `google/owlv2-base-patch16-ensemble` 使用 MPS；InsightFace `buffalo_l` 使用 CPU。
- Qwen 动作模型使用 MPS、FP16、16 帧、总像素预算 1,048,576；配置集中在 `scripts/action_search.py` 的 `SETTINGS`，复用 WISE 编码器。Qwen 缓存恢复方式见 [在线动作检索](docs/在线动作检索.md)。
- `wise_config.yaml` 禁用模型编译，避免上游默认 CUDA/编译设置不适合此 Mac。
- `.media-runtime/` 独立安装 FFmpeg 6；`DYLD_FALLBACK_LIBRARY_PATH` 让 Torchaudio 能找到它，同时避免替换 OpenCV 自带 FFmpeg/libiconv。
- `scripts/fix_openmp.py` 将本虚拟环境中 FAISS、scikit-learn 的 OpenMP 链接到 PyTorch 同一份运行时，原文件保留为 `.original`；没有使用忽略重复运行时错误的开关。`OMP_NUM_THREADS=1` 控制 CPU 并发，Apple GPU 仍启用。重装这些包后需要重新检查这一适配。
- `mlfoundation_openclip.py` 将 MPS 上的多帧图像批次拆成单帧调用，绕过 PyTorch 2.8 在 M5 上 SigLIP 注意力池化的原生缓冲区错误；向量及其原始时间戳顺序不变。
- `transformers_owlv2.py` 同样限制 MPS 物体提取为单帧批次，避免 960px / 3600 token 的八帧注意力计算占满统一内存。
- 修复上游开发版两处缩略图问题：提取时使用 `MediaChunkType.THUMBNAILS` 枚举键；API 获取缩略图时使用 `media_id`，而不是向量 `id`。`scripts/backfill_thumbnails.py` 用于给已建索引补缩略图；正常新建索引已直接修复。
- 外部图片查询也应用编码器的查询侧向量转换，修复 OWLv2 把入库特征直接用于查询导致分数饱和的问题；内部与外部图片使用相同转换，CLIP/人脸默认转换不改变向量。
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

首次恢复会下载模型并重新建立索引；人物需要在页面重新登记，之前网页上传的视频需要重新上传。在线动作模型的缓存准备另见 [在线动作检索](docs/在线动作检索.md)。

WISE 代码为 Apache-2.0；InsightFace 代码与预训练模型许可不同，当前按非商业研究使用预训练模型。素材权利归各自作者，来源保留以便课题引用。
