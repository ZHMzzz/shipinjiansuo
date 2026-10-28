"""Render the audited comparison and per-query evidence without changing labels."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
def load(name):return json.loads((HERE/name).read_text())
def pct(v):return f'{v*100:.1f}%'
def gib(v):return f'{v/2**30:.2f} GiB'
def sec(v):return f'{v:.3f} 秒'

def main():
    s=load('summary.json');a=s['2b-fp16'];b=s['8b-q8'];ma=a['quality']['main']['segment'];mb=b['quality']['main']['segment']
    pa=a['performance'];pb=b['performance'];ra=a['resources'];rb=b['resources']
    wins=[c for c in s['comparisons'] if c['id'][0] in 'BH' and c.get('delta_hit1')==1]
    losses=[c for c in s['comparisons'] if c['id'][0] in 'BH' and c.get('delta_hit1')==-1]
    cases={c['id']:c for c in s['comparisons']}
    lines=['# 全库 2B FP16 与 8B Q8_0 对比','',
        '本次覆盖全部 **40 个视频、129 个现有片段、68 条中文查询**。先核实并冻结标注，再顺序运行两模型。每模型编码全部 129 段，并执行 3 轮、共 204 次未缓存查询；质量评分只使用第一轮。',
        '',f'56 条主查询中，2B 首位严格命中 **{round(ma["hit@1"]*56)}/56**，8B Q8 首位严格命中 **{round(mb["hit@1"]*56)}/56**。8B 新增首位命中 {len(wins)} 条，丢失 {len(losses)} 条，净变化 {(mb["hit@1"]-ma["hit@1"])*100:+.1f} 个百分点。',
        '', '**结论：8B 在这批素材上有局部改善，但收益集中，暂不足以支持默认替换 2B。** 新增的两条首位命中 B36/H14 都是同一个“夜间室外黑裙女子”场景的不同说法，不能视为两个独立场景的提升。8B 更好地区分了室外活动现场与黑色背景棚拍；其代价是全库编码约慢 70%、Metal 分配峰值约增加 82%。本轮热查询中位数接近。现有模型与索引继续保留。',
        '', '现有模型权重、索引、配置、40 个源视频和活动项目指针保持不变；没有切换在线模型。工作区同期有一处 `scripts/action_search.py` 的结果去重改动，非本实验写入，详见末尾审计。',
        '', '## 检索效果','',
        '严格命中要求返回的**片段内**有完整满足查询的画面证据。只找到同一视频但时间不对、只见动作后的状态，均不算严格命中。分级 nDCG 使用完整相关 2、部分相关 1、不相关 0。',
        '', '| 56 条主查询，片段级 | 2B FP16 | 8B Q8_0 |','| --- | ---: | ---: |']
    for k,label in [('hit@1','Hit@1'),('hit@3','Hit@3'),('hit@5','Hit@5'),('recall@5','Recall@5，按查询平均'),('recall@10','Recall@10，按查询平均'),('mrr','MRR'),('ndcg@5','nDCG@5')]:
        fmt=pct if k not in ['mrr','ndcg@5'] else lambda x:f'{x:.3f}'
        lines.append(f'| {label} | {fmt(ma[k])} | {fmt(mb[k])} |')
    lines+=['','| 分组，严格 Hit@1 | 2B FP16 | 8B Q8_0 |','| --- | ---: | ---: |']
    for group,label in [('basic','基础覆盖，40 条'),('hard','混淆条件，16 条'),('paraphrase','口语改写，8 条')]:
        aa=a['quality'][group]['segment'];bb=b['quality'][group]['segment'];n=aa['queries']
        lines.append(f'| {label} | {round(aa["hit@1"]*n)}/{n}（{pct(aa["hit@1"])}） | {round(bb["hit@1"]*n)}/{n}（{pct(bb["hit@1"])}） |')
    lines+=['','| 56 条主查询的补充口径 | 2B FP16 | 8B Q8_0 |','| --- | ---: | ---: |']
    for level,label in [('video','视频级 Hit@1（忽略返回时间是否正确）'),('source_group','同源归组 Hit@1（忽略时间）'),('source_group','同源归组 Recall@5'),('production_overlap_filter','原有重叠过滤后的片段 Hit@1')]:
        key='recall@5' if 'Recall' in label else 'hit@1'
        lines.append(f'| {label} | {pct(a["quality"]["main"][level][key])} | {pct(b["quality"]["main"][level][key])} |')
    lines+=['', '同源归组包括车内人物、墨镜女子、幼儿短片，以及串接视频中的背包原片。视频级和同源归组结果按该视频/来源是否存在正确片段计分，因此不能代替片段级时间定位成绩。8 条口语改写单列，避免重复意图抬高主表；主表自身也含相关场景，不能当作 56 个统计独立意图。',
        '', '## 哪些查询发生变化','', '| 查询 | 2B 首个完整相关排名 | 8B 首个完整相关排名 |','| --- | ---: | ---: |']
    changed=wins+losses
    if not changed:lines.append('| 无首位严格命中变化 | — | — |')
    for c in changed:lines.append(f'| {c["id"]} {c["query"]} | {c["2b"]["first_full_match_rank"]} | {c["8b"]["first_full_match_rank"]} |')
    lines+=['', '两模型仍未首位命中的 B23“白色上衣男子坐在桌前面对镜头讲话”，都把办公室自拍男子的片段排在首位；正确的桌前讲话视频由 2B 的第 4 位升至 8B 的第 2 位。安全帽摘除、工地跌倒、人/狗动作区分、餐桌看手机等重点查询，两者本轮都首位命中；这不能证明 8B 普遍解决了更难的动作或否定条件问题。Recall@10 从 99.3% 略降至 99.1%，也并非所有指标都改善。']
    lines+=['','关键难例（含没有改变首位命中但时间排名变化的查询）：','','| 查询 | 2B 首位片段 / 等级 | 8B 首位片段 / 等级 |','| --- | --- | --- |']
    for qid in ['H01','H02','H03','H05','H06','H07','H08','H09','B18','B32','B35','B40']:
        c=cases[qid]
        def desc(q):
            r=q['top5'][0];return f'{r["file"]} {r["ts"]:g}–{r["te"]:.2f}s / {r["grade"]}'
        lines.append(f'| {qid} {c["query"]} | {desc(c["2b"])} | {desc(c["8b"])} |')
    lines+=['','四条无答案查询均保留在三轮性能测试中，但不计入 Hit/Recall/MRR。当前检索器仍会返回候选，未新增拒答阈值；原始相似度不是概率，也不跨模型直接比较分值。详见 [逐查询 Top 5](QUERY_RESULTS.md)。',
        '', '## 速度','', '| 指标 | 2B FP16 | 8B Q8_0 |','| --- | ---: | ---: |']
    speed=[('观测到的模型加载时间',pa['model_load_seconds'],pb['model_load_seconds']),
        ('全 129 段编码耗时之和',pa['all_segments_encode_seconds'],pb['all_segments_encode_seconds']),
        ('全 129 段循环墙钟时间',pa['all_segments_wall_seconds'],pb['all_segments_wall_seconds']),
        ('片段编码中位数，去除首段',pa['warm_segment_encoding']['median'],pb['warm_segment_encoding']['median']),
        ('片段编码 P95，去除首段',pa['warm_segment_encoding']['p95'],pb['warm_segment_encoding']['p95']),
        ('热查询中位数，后两轮共 136 次',pa['warm_query_with_search']['median'],pb['warm_query_with_search']['median']),
        ('热查询 P95，后两轮共 136 次',pa['warm_query_with_search']['p95'],pb['warm_query_with_search']['p95'])]
    for title,x,y in speed:lines.append(f'| {title} | {sec(x)} | {sec(y)} |')
    lines.append(f'| FAISS 查询中位数 | {pa["search_seconds"]["median"]*1000:.3f} 毫秒 | {pb["search_seconds"]["median"]*1000:.3f} 毫秒 |')
    lines+=['',f'本轮热查询中位数之比（8B/2B）为 **{pb["warm_query_with_search"]["median"]/pa["warm_query_with_search"]["median"]:.2f}×**，全库编码耗时之比为 **{pb["all_segments_encode_seconds"]/pa["all_segments_encode_seconds"]:.2f}×**。',
        '', '片段编码时间包括画面预处理和模型推理，8B 还包括本机进程间传输；不含原视频解码。解码及压缩帧准备在计时前完成。循环墙钟时间另含读取压缩帧和实验日志写入；两者都不是完整上传入库耗时。热查询包含重新编码文字与 FAISS 检索，没有应用层查询缓存，不含页面、网络、结果元数据回查。加载时间受文件缓存和运行时差异影响，不能解释为受控冷启动成绩。',
        '', '后两轮查询中位数分别为：2B 约 198/160ms，8B 约 207/194ms。8B 本轮 P95 较低，但机器有交换内存和同期后台工作，不能据此宣称其尾延迟稳定优于 2B。与上一轮小样本的速度比不同，尤其 2B 本轮时延更高；不要把任何一轮的倍数当成固定性能关系。',
        '', '## 内存','', '| 阶段与指标 | 2B FP16 | 8B Q8_0 |','| --- | ---: | ---: |']
    for phase,title in [('video','视频编码'),('query','查询')]:
        for key,label in [('peak_metal_bytes','Metal 分配峰值'),('peak_rss_bytes','进程树 RSS 峰值'),('peak_footprint_bytes','macOS physical footprint 峰值')]:
            lines.append(f'| {title}：{label} | {gib(ra["phases"][phase][key])} | {gib(rb["phases"][phase][key])} |')
    lines+=['', 'Metal 指标：2B 为 PyTorch MPS driver 分配，8B 为 Metal currentAllocatedSize，50ms 采样；RSS/physical footprint 为工作进程及子进程，100ms 采样。查询阶段保留刚完成视频编码后的运行时缓存。不同指标不相加。GGUF 权重使用 mmap，physical footprint 的低值不能解释为 8B 总体更省内存。',
        '',f'系统在测试前已有明显交换内存占用：2B 开始 {gib(ra["initial_swap_used_bytes"])}、8B 开始 {gib(rb["initial_swap_used_bytes"])}；测试中最低系统可用内存分别为 {gib(ra["minimum_system_available_bytes"])}、{gib(rb["minimum_system_available_bytes"])}。这是有其他应用运行的共享机器，不是受控空载环境，速度比值仅代表本轮观测。',
        '', '## 标注、输入和审计','',
        '- 标注由 Codex 单人视觉核查完成，非两名独立人工标注员复核。全库使用带时间的 1fps 序列图，短动作补看 4fps 序列和局部细节，关键窗口检查真实 16 帧输入；未逐帧浏览所有原视频。标注细节见 [ANNOTATIONS.md](ANNOTATIONS.md)。',
        '- 68 条中有 6 条运行前修正：门岗桌面假设、雨天过街假设，以及背景骑行不确定的无答案题。原文、固定文本及理由均保留在 `query-amendments.json`。B16 的正例从装修环境纠正为真正仓库画面。',
        '- 2B 直接调用现有生产提取器（FP16/MPS），8B 使用上一轮固定的 Q8_0 主干和视觉组件（llama.cpp/Metal）。比较对象是两种部署组合，不可单独归因参数量或量化；没有 8B FP16 对照。',
        '- 全部 129 段使用生产采样公式、同一组 16 帧、8 秒窗口/4 秒步长、同一图像预算和系统模板。129 段的文本 token、时间标记、EOS、视觉 token 数及总 token 数全部一致；RGB uint8 转换最大误差为 0。',
        f'- 2B 实验向量与现有索引逐行余弦最低 {a["validation"]["production_vector_cosine"]["min"]:.8f}，中位数 {a["validation"]["production_vector_cosine"]["median"]:.8f}。2B 三轮完整排名一致：{a["validation"]["rankings_identical_across_rounds"]}；8B：{b["validation"]["rankings_identical_across_rounds"]}。',
        f'- 审计检查 {s["protected_files"]} 个权重、索引、配置及相关代码文件，并再次核对 40 个源视频哈希。模型/索引/配置/源视频和活动指针未变。唯一记录的受检代码变化是工作区并发修改的 `scripts/action_search.py`（增加按视频分组），其差异不涉及采样与编码，保存在 `concurrent-action-search-change.diff`。本实验没有修改任何生产代码。',
        '- 查询来自本库可见内容，且部分视频含字幕和演示检测框，模型可能利用文字；这些结果不能直接推广为无字幕、陌生场景或真实监控泛化能力。',
        '', '## 文件与复现','',
        '- [summary.json](summary.json)：全部指标、逐查询差异及验证结果。',
        '- [query-comparison.csv](query-comparison.csv)：68 条查询的首个完整相关排名与 nDCG。',
        '- [top5-results.csv](top5-results.csv) / [QUERY_RESULTS.md](QUERY_RESULTS.md)：两模型逐查询 Top 5、时间范围与相关性。',
        '- [labels.csv](labels.csv) / [labels-freeze.json](labels-freeze.json)：8,772 条判断及运行前固定哈希。',
        '- `2b-fp16.json`、`8b-q8.json`、`*-resources.json`：原始耗时、排名、分阶段资源采样。',
        '- [production-before.json](production-before.json) / [production-after.json](production-after.json)：前后审计。',
        '', '模型与运行时复用 `../qwen8b-int8/` 中的固定版本；不安装新依赖、不下载新权重。实验生成的 FAISS 索引和向量仅保存在本目录，从未连接到生产入口。首次执行顺序如下；已有结果目录应保留，复测使用新目录并重新固定快照，避免覆盖历史结果。',
        '', '```sh','source scripts/env.sh','python experiments/qwen8b-full-library/prepare.py',
        '# 视觉核查完成后固定标注；不能跳过核查直接把候选列表当答案。',
        'python experiments/qwen8b-full-library/freeze_labels.py',
        'python experiments/qwen8b-full-library/benchmark.py --backend 2b-fp16 --rounds 3',
        'python experiments/qwen8b-full-library/benchmark.py --align-frames',
        'python experiments/qwen8b-full-library/benchmark.py --backend 8b-q8 --rounds 3',
        'python experiments/qwen8b-full-library/evaluate.py',
        'python experiments/qwen8b-full-library/write_report.py','```','']
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    detail=['# 68 条查询的逐条结果','','相关性：2 = 完整满足，1 = 部分相关，0 = 不相关。相似度只在各自模型内用于排序。','']
    for c in s['comparisons']:
        detail += [f'## {c["id"]} {c["query"]}','','| 模型 | 排名 | 视频 | 时间（秒） | 相关性 | 相似度 |','| --- | ---: | --- | --- | ---: | ---: |']
        for name,key in [('2B','2b'),('8B Q8','8b')]:
            for i,r in enumerate(c[key]['top5'],1):detail.append(f'| {name} | {i} | {r["file"]} | {r["ts"]:g}–{r["te"]:.2f} | {r["grade"]} | {r["score"]:.4f} |')
        detail.append('')
    (HERE/'QUERY_RESULTS.md').write_text('\n'.join(detail))
    print('Wrote REPORT.md and QUERY_RESULTS.md')

if __name__=='__main__':main()
