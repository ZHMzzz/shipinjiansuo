# 两个已训练 PPE 模型的本机对照

**本批素材中 YOLOv8s 的裸头检出更多、戴帽误报更少，适合作为后续实验基线；但它仍漏掉 29/45 个明确裸头帧，两者都不宜直接接入正式检索作硬过滤。** 结论只适用于本轮固定阈值与素材，没有训练或调参。

检测框实图见 [三列对照](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/ppe-pretrained/comparison.jpg)，具体误报和漏检位置见 [视觉复核](visual-review.md)。

本轮只做离线检测，没有训练或微调；两个权重在独立进程中顺序运行。所有识别指标仅使用原主测试第一轮，三轮用于重复性；如果 timing-clean 完成，性能表优先使用逐帧排除已检测并发的补测，识别结果不替换。

两套权重均已在本机 MPS 跑通。在本批确定标签帧上，YOLOv8 的误报更少（3 对 13）、裸头检出更多（16 对 9），但其裸头帧 Recall 仍只有 35.6%。这支持把 YOLOv8 保留为后续对照候选，尚不足以直接作为正式检索的唯一自动过滤条件。

| 权重 | 实际结构 | 参数量 | 文件大小 MB | 固定来源 |
|---|---|---:|---:|---|
| YOLO11 PPE | yolo11n.yaml | 2,590,620 | 5.48 | [melihuzunoglu/ppe-detection](https://huggingface.co/melihuzunoglu/ppe-detection/tree/aed7807c358836eef5a6f67af27dd970ade6fb18) |
| YOLOv8 PPE | yolov8s.pt | 11,142,953 | 22.54 | [killuminati1/construction-ppe-yolov8](https://huggingface.co/killuminati1/construction-ppe-yolov8/tree/06ce70134a78f31881e2f9a223719d60a0a9a6a9) |

运行环境：Apple M5 / 32 GiB，macOS-26.5.1-arm64-arm-64bit，Python 3.11.15；torch 2.8.0、torchvision 0.23.0、ultralytics 8.4.31、numpy 1.26.4、psutil 7.2.2；PyTorch CPU 线程数 1，MPS fallback 环境值 1。实际模型输入设备已核对为 MPS，但这不是逐算子 GPU 跟踪。

## 输入与判定口径

共 12 个测试片段/图片、238 帧。帧标签为：helmet_only 75、bare_present 45、unknown 105、no_person 13。

两模型读取相同的原尺寸 PNG，按 2 fps 取样；实际网络输入均已核对为 1×3×640×640、float32、MPS。固定 batch=1、conf=0.25、iou=0.7、rect=False，无测试集阈值调优。

“检出”表示画面内存在至少一个 no-helmet 类检测框。没有逐目标框标注，以下不是检测 mAP、逐人准确率或完整语义检索效果；是否正在施工也未评测。多人状态、遮挡和无人片尾详见 [标签说明](label-review.md)。

## 确定标签帧的裸头检出

| 模型 | TP / FN（裸头帧） | FP / TN（戴帽帧） | 帧级 Precision | 帧级 Recall | 帧级误报率 | TN 中无头部状态检测 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11 PPE | 9 / 36 | 13 / 62 | 40.9% | 20.0% | 17.3% | 5 |
| YOLOv8 PPE | 16 / 29 | 3 / 72 | 84.2% | 35.6% | 4.0% | 1 |

TN 只表示这批已人工标为戴帽的帧没有触发 no-helmet 类，可能包含完全漏检；不能把 TN 或没有检测框解释成模型确认全员戴帽。相邻帧高度相关，两个仓库片段同源，指标不能外推为新摄像头准确率。

## 不确定与无人画面

| 模型 | 标签 | 帧数 | 检出未戴帽 | 检出安全帽 | 无头部状态检测 | 完全无检测 |
|---|---|---:|---:|---:|---:|---:|
| YOLO11 PPE | unknown | 105 | 39 | 69 | 3 | 0 |
| YOLO11 PPE | no_person | 13 | 0 | 0 | 13 | 13 |
| YOLOv8 PPE | unknown | 105 | 3 | 100 | 5 | 0 |
| YOLOv8 PPE | no_person | 13 | 0 | 0 | 13 | 13 |

unknown 不参与二分类分母：其中的未戴帽预测未经人工确定，不能算真阳性。无人画面单列为误触发检查。检测器本身不输出可靠的“头部不可见”类别。

## 逐片段结果

“分数范围”取每帧最高 no-helmet 检测分数；不是正确概率。“无状态”表示 helmet 和 no-helmet 都未检出。

| 片段 | 人工片段判断 | 帧数 | YOLO11 检出未戴帽 / 无状态 | YOLO11 分数范围 | YOLOv8 检出未戴帽 / 无状态 | YOLOv8 分数范围 |
|---|---|---:|---:|---|---:|---|
| fixture_helmet_indoor | no | 17 | 0 / 0 | — | 1 / 0 | 0.420–0.420 |
| fixture_helmet_outdoor | no | 16 | 11 / 0 | 0.336–0.710 | 0 / 0 | — |
| fixture_helmet_squat | no | 16 | 0 / 0 | — | 1 / 0 | 0.488–0.488 |
| fixture_worker_barehead | yes | 5 | 1 / 4 | 0.536–0.536 | 5 / 0 | 0.382–0.617 |
| fixture_fall | yes | 17 | 4 / 13 | 0.389–0.881 | 4 / 13 | 0.278–0.467 |
| library21 | no | 16 | 0 / 3 | — | 1 / 1 | 0.266–0.266 |
| hat_before | no | 8 | 2 / 2 | 0.288–0.602 | 0 / 0 | — |
| hat_transition | yes | 8 | 1 / 4 | 0.318–0.318 | 3 / 0 | 0.485–0.649 |
| hat_after | yes | 12 | 3 / 8 | 0.663–0.695 | 2 / 1 | 0.284–0.415 |
| library30 | yes | 68 | 1 / 6 | 0.344–0.344 | 3 / 6 | 0.267–0.378 |
| library41 | unknown | 54 | 38 / 8 | 0.254–0.867 | 2 / 10 | 0.254–0.363 |
| head_out_of_frame | unknown | 1 | 0 / 1 | — | 0 / 1 | — |

每帧标签、是否检出安全帽/人员、no-helmet 最高分、时间戳和错误类型见 [frame-results.csv](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/ppe-pretrained/frame-results.csv)。框坐标及所有类别的分数保留在两个原始结果 JSON；以上是原始检测存在性统计，未强制关联到同一人，未运行 ByteTrack。

## 耗时

| 模型 | 权重下载 s | checkpoint 加载 s | 首次预测 s | 额外 5 次预热 s | 热调用中位 ms | P95 ms | 模型调用 FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11 PPE | 3.34 | 0.05 | 0.34 | 0.06 | 14.68 | 27.96 | 61.65 |
| YOLOv8 PPE | 5.70 | 0.05 | 0.30 | 0.07 | 17.26 | 33.06 | 50.00 |

| 模型 | 轮次 | 调用总时长 s | 整轮墙钟 s | 调用数 |
|---|---:|---:|---:|---:|
| YOLO11 PPE | 1 | 4.37 | 9.77 | 238 |
| YOLO11 PPE | 2 | 3.49 | 9.97 | 238 |
| YOLO11 PPE | 3 | 3.72 | 139.39 | 238 |
| YOLOv8 PPE | 1 | 5.86 | 90.49 | 238 |
| YOLOv8 PPE | 2 | 4.47 | 11.14 | 238 |
| YOLOv8 PPE | 3 | 3.96 | 10.18 | 238 |

性能数据来源：YOLO11 PPE：`yolo11-ppe-timing-clean.json`；YOLOv8 PPE：`yolov8-ppe-timing-clean.json`。

计时前后同步 MPS；单次调用包含预处理、前向、后处理及结果转 CPU，排除 PNG 读取、视频解码、绘图、网络下载。整轮墙钟还包括资源/进程采样与日志，门控补测还包括帧间等待和被弃用的推理；首次预测包含设备迁移、predictor 初始化与融合等成本。它是新进程内的首次调用，系统和 Metal 编译缓存可能已热，不能当作清空缓存后的冷启动。模型调用 FPS 不是完整视频入库或在线检索 FPS。

一次性准备帧共 11.35 秒，包含视频打开、抽帧读取、PNG 编码写入和 SHA256，不能称纯解码耗时。每模型预载 PNG 并校验哈希分别为 YOLO11 PPE 0.71 秒、YOLOv8 PPE 0.75 秒。

## 内存

| 模型 | 加载前 RSS→峰值 GiB | Metal allocated 峰值 GiB | Metal driver 峰值 GiB | 系统可用最低 GiB | 系统 swap 起始→结束 GiB |
|---|---:|---:|---:|---:|---|
| YOLO11 PPE | 0.406→0.942 | 0.043 | 0.109 | 9.15 | 24.45→24.23 |
| YOLOv8 PPE | 0.493→0.862 | 0.085 | 0.147 | 4.56 | 24.23→25.14 |

每 50 ms 采样并在热测每帧结束后补采；峰值可能漏掉短暂尖峰。RSS 是当前实验进程，含已预载的相同输入帧；RSS、Metal allocated、Metal driver 口径重叠，不能相加。系统可用内存和 swap 包含正式服务及其他应用，不能当作本模型占用。门控补测的资源采样也覆盖帧间等待和被弃用的推理，不能称为完全隔离 GPU 环境的内存实测。加载前/后和逐轮资源明细保留在原始结果及 summary.json。

## 重复性和运行干扰

- YOLO11 PPE：三轮分类状态一致=True；类别/分数 4 位小数、坐标 3 位小数一致=True。
- YOLOv8 PPE：三轮分类状态一致=True；类别/分数 4 位小数、坐标 3 位小数一致=True。

原主测试并发记录：YOLO11 PPE 共 37 次检查观察到其他实验进程；YOLOv8 PPE 共 16 次检查观察到其他实验进程。原主测试时延保留在 summary.json 的 original_performance，不用其受干扰时延直接作公平比较。

| 门控补测模型 | 开始前等待 s | 帧间等待/门控 s | 弃用并发推理次数 / s | 与主测分类一致 |
|---|---:|---:|---:|---|
| YOLO11 PPE | 47.94 | 138.99 | 2 / 0.04 | True |
| YOLOv8 PPE | 0.01 | 88.94 | 4 / 0.09 | True |

门控补测每次预测前等待其他实验进程退出，预测后若发现新的实验进程，则丢弃该次计时并重做；表中热调用只保留通过两端检查的结果，等待时间不计入调用 FPS。片段边界看到其他进程不等于该帧被计入了并发推理。加载和预热没有逐次门控，其时长只作为该轮观察值。

进程检查并非连续 GPU 活动追踪，不能排除检查间隙内启动并结束的短任务、桌面应用负载或等待后的缓存变化。正式检索服务始终运行。

## 正式检索与复现

前后核对 157 个正式文件哈希、活动发布指针、服务 PID 和索引状态均未改变；服务 PID 75954，41 视频、135 片段。

离线复算：`python3 experiments/ppe-pretrained/evaluate.py`。它只读取已有结果和冻结标签，不加载模型、不访问正式数据库。

两模型的仓库 revision、下载 SHA256、文件大小和路径见 [models.json](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/ppe-pretrained/models.json)；固定标签哈希见 [labels-freeze.json](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/ppe-pretrained/labels-freeze.json)。本轮数据与模型训练来源可能存在重叠，未作训练集排重，不能作为独立泛化评测。
