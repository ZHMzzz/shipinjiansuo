# Qwen 同权重线性投影接入正式入库
Status: ready-for-agent
Type: task

用户要求先提交推送当前代码，再采用已测试的线性投影方案。优化前版本 cd8db34 已推送 origin/codex/video-delete。

视频片段编码期间复用原 Conv3d 权重和偏置执行 Linear；结束或异常后恢复原 forward。保持批量 1、FP16、16 帧、像素预算及窗口设置。文字查询不进入适配。新增入库及缺失片段补建生效，不主动重编已有库。

验证：投影数值和参数保持、异常恢复、结构保护；正式 encode_segment 与实验候选向量一致；服务重启后查询及索引完整性检查。耗时引用已有完整对照，不把局部提速当作整次上传提速。

Resolution: completed

## Comments

2026-10-28：已接入并重启本地正式服务，状态接口报告 `linear_same_weights_v1`。全套 78 项测试通过；真实视频 1/19/41 的三个片段与此前 Linear 实验向量逐元素一致，编码后原 forward 恢复、重新编码的文字查询向量一致。重启前后两组文字、物体图、人脸图查询前十排名/片段/框保持、分数容差 1e-6 内一致；44 视频、146 片段及人物登记文件指纹不变。证据在 `experiments/indexing-optimization/linear-production-20261028/`。未重复上传测试视频或主动重编已有库。
