Status: ready-for-agent

# 接入 A 版视频删除

- [x] 索引副本删除与校验、历史副本和托管文件清理。
- [x] 串行写入、媒体身份校验、失败与重启恢复。
- [x] 删除确认和卡片同行对齐、列表与任务同步。
- [x] 单元测试、临时视频真实删除和浏览器验证。

## Comments

原型主来源：`codex/prototype-video-delete` / `3dede80`，用户已选择 A 并指出按钮与左侧信息没有水平对齐。


完成：45 项项目测试通过。覆盖副本清理、其他视频向量保留、发布失败不变更、中断后继续清理、空库索引加载、写入互斥、旧确认与复用 ID 隔离。浏览器确认桌面和 390px 移动端按钮与左侧信息中心差均为 0。

真实回归：新上传 `删除功能回归_临时.mp4`，通过按钮删除（150 条 WISE 向量、1 个 Qwen 片段），确认源文件/转码文件和任务记录清理、播放 404；相同文件再次上传成功并再次删除。此前 40 个视频、275865 条 WISE 向量 ID、Qwen 向量矩阵以及 5 位登记人物均保持不变。用户新增的 `1246.mp4` 包含在基准内，未删除。

记录：`logs/delete-verification.json`、`.playwright-mcp/delete-production-alignment.png`、`.playwright-mcp/delete-production-confirm.png`、`.playwright-mcp/delete-production-mobile.png`。原型不合入正式分支；采用其 A 版设计重新实现。
