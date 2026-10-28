# 现成安全帽检测权重离线对照

用户要求下载 YOLO11 PPE 与 YOLOv8 Construction PPE 两个已训练模型，使用现有安全帽相关视频比较识别效果、耗时、内存；暂不改正式检索。

实验位置：`experiments/ppe-pretrained/`。依赖使用该目录内的独立安装目标，复用现有只读 PyTorch/MPS 运行时，不修改正式环境包。

固定模型 revision 与 SHA256。相同原始帧、2 fps、640 输入尺寸、batch 1、conf 0.25、IoU 0.7、MPS FP32。先冻结视觉标签，再运行预测；两个模型分进程顺序运行，避开其他 GPU 实验。计时区分下载、加载、首次调用、预热、三轮热推理和文件解码。记录 50 ms RSS、Metal allocated/driver、系统 available/swap；重叠口径不相加。

检查视频/帧级未戴帽检出、戴帽误报、不可见状态，不声称无逐人框标注支持的 mAP；保存原始框和可视化复核。完成后比较生产源码、配置、活动指针、索引哈希、服务 PID 与库数量。
