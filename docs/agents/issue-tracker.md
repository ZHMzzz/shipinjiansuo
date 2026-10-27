# Issue tracker: Local Markdown

需求规格和任务以 Markdown 文件保存在 `.scratch/` 下。

## 文件约定

- 每个功能一个目录：`.scratch/<feature-slug>/`。
- 需求规格：`.scratch/<feature-slug>/spec.md`。
- 每个任务一个文件：`.scratch/<feature-slug>/issues/<NN>-<slug>.md`，
  从 `01` 开始编号。
- 任务文件顶部使用 `Status:` 记录 triage 状态，取值见 `triage-labels.md`。
- 评论和讨论追加到文件底部的 `## Comments` 下。

## 发布与读取

技能要求“发布到任务跟踪系统”时，在上述路径创建对应文件及必要目录。
技能要求“读取相关任务”时，读取用户指定路径；仅提供编号且存在多个匹配时，
先确认所属功能。

## Wayfinder 工作流

- 工作地图：`.scratch/<effort>/map.md`，记录 Notes、Decisions-so-far 和 Fog。
- 子任务：`.scratch/<effort>/issues/<NN>-<slug>.md`。
- `Type:` 取值为 research、prototype、grilling 或 task。
- 此工作流的 `Status:` 单独使用 open、claimed、resolved。
- `Blocked by: NN, NN` 记录依赖；所有依赖 resolved 后解除阻塞。
- 按编号选择首个 open、无阻塞且尚未领取的任务。
- 开始前保存 `Status: claimed`。
- 完成后在 `## Answer` 下追加结论，保存 `Status: resolved`，
  并在地图的 Decisions-so-far 中追加摘要及链接。
