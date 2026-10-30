# 已训练安全帽检测模型离线对照

结果见 [REPORT.md](REPORT.md)，检测框对照见 [comparison.jpg](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/ppe-pretrained/comparison.jpg) 和 [visual-review.md](visual-review.md)。本实验没有训练模型，没有接入正式检索，也没有运行 ByteTrack。

两个固定版本权重在 `models/`，下载地址、revision、文件大小、SHA256 和下载耗时记录在 [models.json](https://github.com/ZHMzzz/shipinjiansuo/blob/f5bc7e04e9203ab06b4def75b7038199215ee6ff/experiments/ppe-pretrained/models.json)。目录内 `.deps/` 是额外依赖，正式 `.venv` 的包版本不变。

## 复现

在项目根目录执行：

```sh
source scripts/env.sh
python -m pip install --target experiments/ppe-pretrained/.deps --no-deps -r experiments/ppe-pretrained/requirements-experiment.txt
python experiments/ppe-pretrained/download_models.py
```

数据来自本机现有视频，使用原尺寸画面、2 fps、真实采样帧号，先视觉标注再冻结哈希。当前机器已有输入 PNG。若重新准备输入，先审阅 `prepare_dataset.py` 和 `labels-freeze.json`，不要覆盖已冻结结果来调标签。

单模型重测需提供新的 tag，避免覆盖原结果：

```sh
python experiments/ppe-pretrained/benchmark.py yolo11-ppe --tag repeat-1 --isolate
python experiments/ppe-pretrained/benchmark.py yolov8-ppe --tag repeat-1 --isolate
```

`--isolate` 在每次计时前后检查其他实验进程；遇到其他实验则等待，预测期间出现并发则舍弃该次计时并重做。等待时间和舍弃记录单列；两模型应顺序执行。系统其他应用仍可能影响延迟，此方法不等于独占整台机器。

无需重新推理即可复算正式对照与示例图：

```sh
python experiments/ppe-pretrained/evaluate.py
python experiments/ppe-pretrained/render_review.py
```

## 结果口径

- `*-mps-fp32.json` 是首轮完整预测，保留三轮原始检测框。识别统计只取第一轮，不把重复测量增加为样本量。
- `*-timing-clean.json` 是补充的门控性能测量。与主结果分类一致性另行校验；不覆盖首轮结果。
- 两模型实际输入相同：batch 1，640×640，MPS，FP32，conf 0.25，IoU 0.7，rect=False。
- 每次计时包含预处理、前向、后处理和检测结果转 CPU；不含文件读取、视频解码、绘图或进程等待。加载、进程首次预测、额外预热单独记录；它们不是清空系统缓存后的冷启动。
- RSS 包含 Python、运行时和输入帧缓存。Metal allocated/driver 是重叠口径，不能与 RSS 相加。50 ms 轮询加每帧完成后采样，报告的是采样峰值。补测资源采样包含帧间等待，系统 swap 和可用内存不属于单模型占用。
- 只有帧级标签，没有逐人框真值；不能由这些结果报告目标检测 mAP 或逐人准确率。原素材的字幕、水印、内嵌检测框仍在，不能外推为干净监控视频的效果。

生产状态指纹、活动指针、服务 PID 及 Python 包版本的前后对照保留在 `production-*.json` 和 `validation.json`。
