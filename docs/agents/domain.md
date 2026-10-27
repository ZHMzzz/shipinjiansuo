# Domain Docs

## 布局

采用 single-context：
- 根目录 `GLOSSARY.md`：项目领域术语。
- `docs/adr/`：项目架构决策记录。

## 阅读规则

探索代码前，阅读已有的 `GLOSSARY.md` 和与当前工作相关的 ADR。
文档缺失时直接继续，无需提示或提前创建。
术语或决策实际确定后，由 domain-modeling 技能按需创建文档。

## 术语与决策

任务标题、方案、假设和测试名称使用词汇表中的术语。
遇到未定义的概念，先核实是否确有需要；真实缺口留给 domain-modeling 处理。
输出与现有 ADR 冲突时，明确指出对应 ADR，并解释重新讨论的理由。
