# InsightFace CoreML 隔离验证

本轮结果支持继续测试 CoreML：三个样本中的人脸处理明显加快，身份向量与 CPU 很接近，且 profiling 确认三个启用模型实际执行了 CoreML kernel。它还不足以证明完整上传提速比例或所有人脸数据完全一致。未修改正式 WISE、Python 环境或原模型文件。

## 配置与证据

- 基线：[cpu-smoke.json](cpu-smoke.json)；候选：[coreml-smoke.json](coreml-smoke.json)；执行诊断：[coreml-profile.json](coreml-profile.json)。
- 环境：InsightFace 0.7.3、ONNX Runtime 1.31.1。原 `buffalo_l` 权重、640×640 检测、0.5 检测阈值、detection/recognition/genderage 均保留。
- CPU 为显式 CPUExecutionProvider；候选为 CoreMLExecutionProvider + CPUExecutionProvider，使用 MLProgram、MLComputeUnits=ALL、RequireStaticInputShapes=1 及实验目录内编译缓存。
- 两边均使用 ORT 默认线程与图优化设置，保留旧版加载五个 ONNX 后筛除两个人脸关键点模型的行为。候选的有效三个模型只在实验缓存固定输入形状；initializer 字节哈希与原文件一致。
- CPU 使用原动态形状模型，CoreML 使用固定形状副本，因此这次比较的是“CoreML + 固定形状 + 缓存”组合，不是单独隔离执行后端变量。CPU 固定形状对照尚未测量。
- 正式计时未开启 profiling；单独诊断轮启用了 profiling。图片预先解码，报告中的推理时间包含 FaceAnalysis 的检测、裁剪/对齐、识别、属性预测与后处理，不含视频解码、模型加载和索引写入。

## 时间

每张图重复 5 次，首轮单独记录，后 4 次取中位数。单位为毫秒。

| 图片 | 人脸数 | CPU 温热中位数 | CoreML 第一次 session | CoreML 第二次 session |
|---|---:|---:|---:|---:|
| coughing-6XvsLPDioVA | 1 | 69.36 | 15.57 | 29.57 |
| shouting-84cxLPOEnGg | 1 | 65.28 | 15.05 | 13.59 |
| whistling-LqQYSUb_TJI | 2 | 107.66 | 21.15 | 19.34 |

CoreML 第一次 session 初始化为 1.695 秒，第一张图第一次推理为 2.616 秒；第二次 session 初始化为 0.694 秒，第一张图第一次推理为 0.044 秒。CPU 初始化为 0.159 秒，首次推理为 0.106 秒。不能把 CoreML 首次准备成本隐藏在温热时延里；第二次 session 的第一张图也有明显波动。

准备阶段仅生成静态 ONNX 副本，未创建 ORT session 或运行 CoreML。未保存首次运行前“编译缓存目录为空”的清单，也未清除 macOS 自身缓存，所以这里的“第一次 session”仅指本次进程的首次会话，不能宣称是完全清空所有缓存后的冷启动。两次会话复用同一实验缓存。预先生成静态 ONNX 文件的一次性耗时不计入以上 session 初始化时间；缓存文件哈希检查的耗时计入其中，并单独记录在 static_preparation_s。

## 实际执行与精度

诊断轮覆盖三张图，检测四张脸。统计对象是优化/融合后的执行 kernel，不是原始 ONNX 节点数。

| 模型 | CoreML 调用数 / 不同 kernel 数 | CPU 调用数 / 不同 kernel 数 | CPU kernel 累计时间 |
|---|---:|---:|---:|
| det_10g | 3 / 1 | 0 / 0 | 0 |
| genderage | 8 / 2 | 8 / 2 | 25 微秒 |
| w600k_r50 | 8 / 2 | 8 / 2 | 53 微秒 |

两个未启用的关键点模型只有初始化记录，没有执行 kernel。CoreML provider 名称出现在配置中不是执行证明；上表来自实际 profile 事件，才构成执行证明。此轮未采集硬件 compute plan，不能据此认定用了 ANE，也不能称全部节点由 CoreML 执行。

实际缓存的五个 `compiled_model.mlmodelc/model.mil` 文件中，检测、识别、属性分区均标注 `fp32`，未出现 `fp16` 类型；ORT 返回的 AllowLowPrecisionAccumulationOnGPU=0。实验没有主动进行 FP16 转换，也不能把 `coremltools.convert()` 的默认 FP16 策略套用到这个 ORT 转换路径。权重一致及 FP32 类型并不保证与 CPU 逐位相同。

缓存核查位置位于 [CoreML 缓存](cache/face/compiled/coreml/)：检测模型分区 fp32 标记 366 处，识别两个分区 640/6 处，属性两个分区 195/13 处，五个分区 fp16 标记均为零。这是类型核查，不是硬件指令追踪。

## 一致性边界

- 四张脸全部按框 IoU 匹配，无新增或缺失；最低框 IoU 约 0.99999926，最低 embedding cosine 约 0.99999985。
- 与五个既有登记向量进行 20 次 0.45 人脸门槛判定，结果均未改变。但仅 1 个正例，最近相似度距离门槛仍约 0.288，未覆盖临界门槛样本。
- 性别字段均一致；一张脸的年龄由 CPU 35 变为 CoreML 34。因此“全部人脸元数据完全一致”不成立。
- 该脚本只保存每张图首次推理的结果用于比较；重复推理用于计时，尚未断言各次输出完全一致。三张图、四张脸不足以判断复杂遮挡、小脸、多人、边界检测置信度等场景的召回变化。
- 后续全流程结果应独立汇总。本次微基准不能线性外推为整个上传流程的加速倍数。

官方依据：[ONNX Runtime CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)、[ORT CoreML 转换实现](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/coreml/builders/model_builder.cc)、[Apple typed execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution-example.html)。
