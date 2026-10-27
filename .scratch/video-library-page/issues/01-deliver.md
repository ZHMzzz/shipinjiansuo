Status: ready-for-agent

# 替换为 A 版正式视频图库

原型主来源：`codex/prototype-video-library` / `694d18d`。用户已确认 A 版及两个弹窗。

- [x] 正式图库、分页、查询与定位播放。
- [x] 文件名候选限定及多条件组合。
- [x] 上传进度、入库刷新和失败重试。
- [x] 人脸头像列表、登记、删除和人名同步。
- [x] 自动化、真实接口与浏览器验证。

## Comments

按已确认布局重新实现正式逻辑，不保留原型切换器或模拟提交。


实现完成，已替换 `/mvp/` 正式入口。移除自然语言＋图片组合，文字走 Qwen、物体图片走原生 OWLv2，人物与文件名保留组合筛选。

验证：38 项项目测试通过；现有动作定位与人物场景回归通过；两张用户物体图片仍正确视频 Top 1；浏览器验证分页、文件名、16 秒播放、输入互斥、503 反馈、重置后过期结果保护及移动端无横向溢出。新增并删除唯一测试人物，原有 5 位人物保留。上传验证真实接口和已有文件去重，未新增媒体；失败重试到完成刷新使用浏览器拦截模拟，未冒充新视频全链路实测。现有入库流水线未改动。

记录：`logs/library-verification.json`、`logs/combined-verification.json`、`.playwright-mcp/production-people.png`、`.playwright-mcp/production-upload.png`、`.playwright-mcp/production-mobile.png`。
