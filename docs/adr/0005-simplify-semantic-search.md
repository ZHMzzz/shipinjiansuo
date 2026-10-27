# 合并中文检索并移除负向描述与重排

用户明确要求合并中文语义和动作片段，并移除负向描述及重排功能。

页面只保留“中文检索”文字入口，场景和动作走 `/mvp-api/search`，复用同一个 Qwen3-VL-Embedding 2B 查询编码和片段索引。默认返回最多 20 个命中片段，同一视频可以出现多段，过滤高度重叠的候选，从片段起点播放。仍是最长 8 秒、4 秒步长、16 帧的原片段索引，不重新建库。图文组合也使用该接口，但继续按视频取最佳片段。

删除独立 `/mvp-api/actions/search`，保留 `/mvp-api/actions/status` 和 retry 作为共享片段索引状态与补建接口。历史 score_kind 的 action 值继续表示多片段结果，不代表独立的动作模型。

所有 MVP 页面移除负向输入；自定义搜索引擎删除负向向量相减和相关 API 参数，人物场景查询只使用正向场景。WISE2 原生服务源码保持原样，MVP 不再向其传递负向查询。

移除重排模型实现、加载/切换钩子、页面比较控件、资源采样展示及专属运行/下载脚本。既有实验数据和文档保留为历史记录；其他离线基准需要的资源采样提取为独立 `resource_metrics.py`。不删除共享模型缓存，不影响 OWLv2、InsightFace 和 Qwen embedding。

## 验证

31 项项目测试通过。场景、动作定位、图文组合、人物场景门槛、人脸及两张物体图片实时回归通过。浏览器确认只有“中文检索”文字入口，无负向或重排控件；只向 `/mvp-api/search` 传 query。摔倒命中 16–24 秒并从 16 秒播放。重启前后发布指针和索引数量不变（39 视频、121 片段），原动作搜索路由返回 404。

尝试额外检查全局 OpenAPI 时，现有 WISE 模型中的 numpy.ndarray 不能生成 JSON Schema，`/openapi.json` 返回 500；因此参数移除通过源码签名和实际浏览器请求验证，未声称全局 OpenAPI 检查通过。本次未改动 WISE 的模型 Schema。

实时验证记录：`logs/semantic-unified-standard.log`、`logs/semantic-unified-actions.log`、`logs/semantic-unified-combined.log` 和 `logs/owl-restored-user-images.json`。
