"""Summarize completed PPE runs without importing torch or changing frozen labels.

Binary scores concern the existence of any no-helmet detection in a frame, not
object localization, person association, construction activity, or detector mAP.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
MODELS = ['yolo11-ppe', 'yolov8-ppe']
TITLES = {'yolo11-ppe': 'YOLO11 PPE', 'yolov8-ppe': 'YOLOv8 PPE'}


def read(name):
    return json.loads((HERE / name).read_text())


def sha(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def confidence(values):
    return {'count': len(values), 'min': min(values), 'max': max(values),
            'median': statistics.median(values)} if values else None


def signature(records):
    return [[(d['class_id'], round(d['confidence'], 4),
              [round(x, 3) for x in d['xyxy']]) for d in record['detections']]
            for record in records]


def classify(record, mapping):
    detections = record['detections']
    groups = {kind: [d for d in detections if d['class_id'] in ids]
              for kind, ids in mapping.items()}
    no_helmet = bool(groups['no_helmet'])
    helmet = bool(groups['helmet'])
    state = ('no_helmet_detected' if no_helmet else
             'helmet_detected_no_nohelmet' if helmet else
             'no_head_state_detection')
    return {'no_helmet_detected': no_helmet, 'helmet_detected': helmet,
            'person_detected': bool(groups['person']), 'prediction_state': state,
            'no_detection_at_all': not detections,
            'no_helmet_count': len(groups['no_helmet']), 'helmet_count': len(groups['helmet']),
            'person_count': len(groups['person']),
            'no_helmet_max_confidence': max((d['confidence'] for d in groups['no_helmet']), default=None),
            'helmet_max_confidence': max((d['confidence'] for d in groups['helmet']), default=None)}


def validate_run(run, expected_ids, dataset_hash, labels_hash):
    if 'all_phases_resources' not in run or len(run.get('rounds', [])) != 3:
        raise ValueError('Benchmark is incomplete; wait for all_phases_resources and three rounds')
    assert run['dataset_sha256'] == dataset_hash
    assert run['labels_sha256'] == labels_hash
    assert run['frame_count'] == len(expected_ids)
    for rnd in run['rounds']:
        assert [r['frame_id'] for r in rnd['records']] == expected_ids
        assert all(math.isfinite(r['predict_seconds']) and r['predict_seconds'] > 0
                   for r in rnd['records'])
    expected_options = dict(device='mps', imgsz=640, rect=False, batch=1,
                            conf=0.25, iou=0.7, half=False)
    assert all(run['options'][key] == value for key, value in expected_options.items())
    signatures = run['observed_model_inputs']
    shape_ok = bool(signatures) and all(s[0] == [1, 3, 640, 640] and s[1] == 'torch.float32'
                                      and s[2].startswith('mps') for s in signatures)
    assert shape_ok, signatures
    return shape_ok


def evaluate_model(run, dataset, labels):
    label_map = {r['frame_id']: r for r in labels['frames']}
    case_map = {c['id']: c for c in dataset['cases']}
    frames = []
    confusion = collections.Counter({'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0})
    for record in run['rounds'][0]['records']:
        label = label_map[record['frame_id']]
        prediction = classify(record, run['class_mapping'])
        if label['presence'] == 'no_person':
            outcome = 'excluded_no_person'
        elif label['expected_no_helmet'] == 'unknown':
            outcome = 'excluded_unknown'
        elif label['expected_no_helmet'] == 'yes':
            outcome = 'tp' if prediction['no_helmet_detected'] else 'fn'
        elif label['expected_no_helmet'] == 'no':
            outcome = 'fp' if prediction['no_helmet_detected'] else 'tn'
        else:
            raise ValueError(label)
        if outcome in confusion:
            confusion[outcome] += 1
        frames.append({'model_id': run['model']['id'], 'frame_id': record['frame_id'],
            'case_id': record['case_id'], 'timestamp': record['timestamp'],
            'ground_truth_presence': label['presence'],
            'expected_no_helmet': label['expected_no_helmet'], 'outcome': outcome,
            **prediction, 'first_round_predict_ms': record['predict_seconds'] * 1000})
    c = dict(confusion)
    c.update({'positive_frames': c['tp'] + c['fn'], 'negative_frames': c['fp'] + c['tn'],
        'precision': ratio(c['tp'], c['tp'] + c['fp']),
        'recall': ratio(c['tp'], c['tp'] + c['fn']),
        'false_positive_rate': ratio(c['fp'], c['fp'] + c['tn']),
        'tn_without_head_state_detection': sum(f['outcome'] == 'tn' and f['prediction_state'] == 'no_head_state_detection' for f in frames),
        'fn_without_head_state_detection': sum(f['outcome'] == 'fn' and f['prediction_state'] == 'no_head_state_detection' for f in frames)})
    excluded = {}
    for kind in ['unknown', 'no_person']:
        selected = [f for f in frames if f['ground_truth_presence'] == kind]
        excluded[kind] = {'frames': len(selected), 'no_helmet_detected': sum(f['no_helmet_detected'] for f in selected),
            'helmet_detected': sum(f['helmet_detected'] for f in selected),
            'no_head_state_detection': sum(f['prediction_state'] == 'no_head_state_detection' for f in selected),
            'no_detection_at_all': sum(f['no_detection_at_all'] for f in selected)}
    cases = []
    for case_id, case in case_map.items():
        rows = [f for f in frames if f['case_id'] == case_id]
        detected = [f for f in rows if f['no_helmet_detected']]
        cases.append({'id': case_id, 'group': case['group'], 'frames': len(rows),
            'expected_no_helmet': case['expected_no_helmet'], 'note': case['note'],
            'ground_truth_counts': dict(collections.Counter(f['ground_truth_presence'] for f in rows)),
            'outcome_counts': dict(collections.Counter(f['outcome'] for f in rows)),
            'no_helmet_detected_frames': len(detected), 'helmet_detected_frames': sum(f['helmet_detected'] for f in rows),
            'no_head_state_detection_frames': sum(f['prediction_state'] == 'no_head_state_detection' for f in rows),
            'no_detection_at_all_frames': sum(f['no_detection_at_all'] for f in rows),
            'no_helmet_frame_max_confidence': confidence([f['no_helmet_max_confidence'] for f in detected]),
            'no_helmet_detected_timestamps': [f['timestamp'] for f in detected]})
    first_sig = signature(run['rounds'][0]['records'])
    repeat_same = all(signature(r['records']) == first_sig for r in run['rounds'][1:])
    repeat_states_same = all([classify(x, run['class_mapping'])['prediction_state'] for x in r['records']]
        == [f['prediction_state'] for f in frames] for r in run['rounds'][1:])
    assert repeat_same == run['repeated_outputs_identical_4decimals']
    latencies = [r['predict_seconds'] for rnd in run['rounds'] for r in rnd['records']]
    concurrent = ([{'stage': 'before', 'experiments': run['other_experiments_before']}]
        if run['other_experiments_before'] else [])
    concurrent += [x for x in run['concurrent_process_checks'] if x['experiments']]
    if run['other_experiments_after']:
        concurrent.append({'stage': 'after', 'experiments': run['other_experiments_after']})
    resources = run['all_phases_resources']
    summary = {'model': run['model'], 'architecture': run['architecture'], 'names': run['names'],
        'model_parameters': run['model_parameters'], 'class_mapping': run['class_mapping'],
        'confusion': c, 'excluded_labels': excluded, 'cases': cases,
        'repeat_detections_equal_4decimals_xyxy_3decimals': repeat_same,
        'repeat_prediction_states_identical': repeat_states_same,
        'timing': {'checkpoint_load_seconds': run['checkpoint_load_seconds'],
            'first_predict_seconds': run['first_predict_seconds'],
            'additional_warmup_calls': run['additional_warmup_calls'],
            'additional_warmup_seconds': run['additional_warmup_seconds'],
            'png_read_and_hash_seconds': run['png_read_and_hash_seconds'],
            'predict_call_count': len(latencies), 'median_ms': statistics.median(latencies) * 1000,
            'p95_ms': percentile(latencies, .95) * 1000,
            'total_predict_seconds': sum(latencies), 'model_call_fps': len(latencies) / sum(latencies),
            'rounds': [{'number': r['number'], 'wall_seconds': r['wall_seconds'],
                        'predict_stats': r['predict_stats']} for r in run['rounds']]},
        'resources': resources, 'hot_round_resources': [r['resources'] for r in run['rounds']],
        'observed_model_inputs': run['observed_model_inputs'], 'environment': run['environment'],
        'concurrent_experiments': concurrent, 'performance_comparison_contaminated': bool(concurrent)}
    return summary, frames


def select_performance(model, main_run, dataset, labels, expected_ids, dataset_hash, labels_hash):
    """Keep recognition from the original run; optionally use boundary-gated timing."""
    model['recognition_source'] = f"{main_run['model']['id']}-mps-fp32.json"
    model['performance_source'] = model['recognition_source']
    model['original_performance'] = {
        'timing': model['timing'], 'resources': model['resources'],
        'concurrent_experiments': model['concurrent_experiments'],
        'performance_comparison_contaminated': model['performance_comparison_contaminated']}
    model['isolate_each_prediction'] = False
    model['clean_run_pending'] = True
    path = HERE / f"{main_run['model']['id']}-timing-clean.json"
    if not path.exists():
        return
    clean = json.loads(path.read_text())
    if 'all_phases_resources' not in clean:
        return
    validate_run(clean, expected_ids, dataset_hash, labels_hash)
    assert clean.get('isolate_each_prediction') is True
    assert main_run['options'] == clean['options'] and main_run['model']['sha256'] == clean['model']['sha256']
    selected, selected_frames = evaluate_model(clean, dataset, labels)
    for key in ['timing', 'resources', 'hot_round_resources', 'environment', 'observed_model_inputs', 'concurrent_experiments']:
        model[key] = selected[key]
    model['performance_source'] = path.name
    model['isolate_each_prediction'] = True
    model['clean_run_pending'] = False
    # Case-boundary observations can occur outside admitted timed calls. The
    # benchmark retries calls whose post-prediction check finds another process.
    model['performance_comparison_contaminated'] = False
    model['performance_protocol'] = {
        'description': 'Each accepted prediction had no other experiment detected immediately before and after; transient between-check contention cannot be excluded.',
        'initial_wait_seconds': clean['initial_wait_seconds'],
        'inter_frame_wait_seconds': clean['inter_frame_wait_seconds'],
        'discarded_contended_prediction_count': len(clean['discarded_contended_predictions']),
        'discarded_contended_prediction_seconds': sum(r['seconds'] for r in clean['discarded_contended_predictions']),
        'discarded_contended_predictions': clean['discarded_contended_predictions'],
        'memory_and_round_wall_include_waits_and_discarded_calls': True,
        'cold_load_and_warmup_not_individually_gated': True}
    model['clean_repeat_prediction_states_identical'] = selected['repeat_prediction_states_identical']
    model['clean_vs_original_prediction_states_identical'] = (
        [classify(r, main_run['class_mapping'])['prediction_state'] for r in main_run['rounds'][0]['records']]
        == [f['prediction_state'] for f in selected_frames])


def production_comparison():
    if not (HERE / 'production-after.json').exists():
        return {'status': 'pending_after_snapshot'}
    before, after = read('production-before.json'), read('production-after.json')
    keys = set(before['files']) | set(after['files'])
    changed = sorted(k for k in keys if before['files'].get(k) != after['files'].get(k))
    same_pointer = before['pointer'] == after['pointer'] and before['project'] == after['project']
    same_pid = before['server_pids'] == after['server_pids']
    count_keys = ['indexed_videos', 'indexed_segments', 'status']
    same_status = all(before['actions/status'][k] == after['actions/status'][k] for k in count_keys)
    return {'status': 'unchanged' if same_pointer and same_pid and same_status and not changed else 'changed',
        'files_checked': len(keys), 'changed_files': changed, 'pointer_unchanged': same_pointer,
        'server_pid_unchanged': same_pid, 'index_status_unchanged': same_status,
        'server_pids_after': after['server_pids'], 'index_status_after': {k: after['actions/status'][k] for k in count_keys}}


def percent(value):
    return '—' if value is None else f'{100 * value:.1f}%'


def gi(value):
    return value / 2**30


def write_report(summary):
    models = summary['models']
    lines = ['# 两个已训练 PPE 模型的本机对照', '',
        '本轮只做离线检测，没有训练或微调；两个权重在独立进程中顺序运行。所有识别指标仅使用原主测试第一轮，三轮用于重复性；如果 timing-clean 完成，性能表优先使用逐帧排除已检测并发的补测，识别结果不替换。', '',
        f"两套权重均已在本机 MPS 跑通。在本批确定标签帧上，YOLOv8 的误报更少（{models[1]['confusion']['fp']} 对 {models[0]['confusion']['fp']}）、裸头检出更多（{models[1]['confusion']['tp']} 对 {models[0]['confusion']['tp']}），但其裸头帧 Recall 仍只有 {percent(models[1]['confusion']['recall'])}。这支持把 YOLOv8 保留为后续对照候选，尚不足以直接作为正式检索的唯一自动过滤条件。", '',
        '| 权重 | 实际结构 | 参数量 | 文件大小 MB | 固定来源 |',
        '|---|---|---:|---:|---|',
        *[f"| {TITLES[m['model']['id']]} | {m['architecture']} | {m['model_parameters']:,} | {m['model']['bytes'] / 1e6:.2f} | [{m['model']['repo']}](https://huggingface.co/{m['model']['repo']}/tree/{m['model']['revision']}) |" for m in models], '',
        '运行环境：Apple M5 / 32 GiB，' + models[0]['environment']['platform'] +
        '，Python ' + models[0]['environment']['python'].split()[0] + '；' +
        '、'.join(f'{k} {v}' for k, v in models[0]['environment']['packages'].items()) +
        f"；PyTorch CPU 线程数 {models[0]['environment']['torch_threads']}，MPS fallback 环境值 {models[0]['environment']['mps_fallback_env']}。实际模型输入设备已核对为 MPS，但这不是逐算子 GPU 跟踪。", '',
        '## 输入与判定口径', '',
        f"共 {summary['dataset']['cases']} 个测试片段/图片、{summary['dataset']['frames']} 帧。帧标签为：" +
        '、'.join(f'{k} {v}' for k, v in summary['dataset']['frame_state_counts'].items()) + '。', '',
        '两模型读取相同的原尺寸 PNG，按 2 fps 取样；实际网络输入均已核对为 1×3×640×640、float32、MPS。固定 batch=1、conf=0.25、iou=0.7、rect=False，无测试集阈值调优。', '',
        '“检出”表示画面内存在至少一个 no-helmet 类检测框。没有逐目标框标注，以下不是检测 mAP、逐人准确率或完整语义检索效果；是否正在施工也未评测。多人状态、遮挡和无人片尾详见 [标签说明](label-review.md)。', '',
        '## 确定标签帧的裸头检出', '',
        '| 模型 | TP / FN（裸头帧） | FP / TN（戴帽帧） | 帧级 Precision | 帧级 Recall | 帧级误报率 | TN 中无头部状态检测 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in models:
        c = m['confusion']
        lines.append(f"| {TITLES[m['model']['id']]} | {c['tp']} / {c['fn']} | {c['fp']} / {c['tn']} | {percent(c['precision'])} | {percent(c['recall'])} | {percent(c['false_positive_rate'])} | {c['tn_without_head_state_detection']} |")
    lines += ['', 'TN 只表示这批已人工标为戴帽的帧没有触发 no-helmet 类，可能包含完全漏检；不能把 TN 或没有检测框解释成模型确认全员戴帽。相邻帧高度相关，两个仓库片段同源，指标不能外推为新摄像头准确率。', '',
        '## 不确定与无人画面', '',
        '| 模型 | 标签 | 帧数 | 检出未戴帽 | 检出安全帽 | 无头部状态检测 | 完全无检测 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for m in models:
        for key, row in m['excluded_labels'].items():
            lines.append(f"| {TITLES[m['model']['id']]} | {key} | {row['frames']} | {row['no_helmet_detected']} | {row['helmet_detected']} | {row['no_head_state_detection']} | {row['no_detection_at_all']} |")
    lines += ['', 'unknown 不参与二分类分母：其中的未戴帽预测未经人工确定，不能算真阳性。无人画面单列为误触发检查。检测器本身不输出可靠的“头部不可见”类别。', '', '## 逐片段结果', '',
        '“分数范围”取每帧最高 no-helmet 检测分数；不是正确概率。“无状态”表示 helmet 和 no-helmet 都未检出。', '',
        '| 片段 | 人工片段判断 | 帧数 | YOLO11 检出未戴帽 / 无状态 | YOLO11 分数范围 | YOLOv8 检出未戴帽 / 无状态 | YOLOv8 分数范围 |',
        '|---|---|---:|---:|---|---:|---|']
    for a, b in zip(models[0]['cases'], models[1]['cases']):
        assert a['id'] == b['id']
        def score_range(c):
            v = c['no_helmet_frame_max_confidence']
            return '—' if v is None else f"{v['min']:.3f}–{v['max']:.3f}"
        lines.append(f"| {a['id']} | {a['expected_no_helmet']} | {a['frames']} | {a['no_helmet_detected_frames']} / {a['no_head_state_detection_frames']} | {score_range(a)} | {b['no_helmet_detected_frames']} / {b['no_head_state_detection_frames']} | {score_range(b)} |")
    lines += ['', '每帧标签、是否检出安全帽/人员、no-helmet 最高分、时间戳和错误类型见 [frame-results.csv](frame-results.csv)。框坐标及所有类别的分数保留在两个原始结果 JSON；以上是原始检测存在性统计，未强制关联到同一人，未运行 ByteTrack。', '', '## 耗时', '',
        '| 模型 | 权重下载 s | checkpoint 加载 s | 首次预测 s | 额外 5 次预热 s | 热调用中位 ms | P95 ms | 模型调用 FPS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in models:
        t = m['timing']
        lines.append(f"| {TITLES[m['model']['id']]} | {m['model']['download_seconds']:.2f} | {t['checkpoint_load_seconds']:.2f} | {t['first_predict_seconds']:.2f} | {t['additional_warmup_seconds']:.2f} | {t['median_ms']:.2f} | {t['p95_ms']:.2f} | {t['model_call_fps']:.2f} |")
    lines += ['', '| 模型 | 轮次 | 调用总时长 s | 整轮墙钟 s | 调用数 |', '|---|---:|---:|---:|---:|']
    for m in models:
        for r in m['timing']['rounds']:
            lines.append(f"| {TITLES[m['model']['id']]} | {r['number']} | {r['predict_stats']['total_seconds']:.2f} | {r['wall_seconds']:.2f} | {r['predict_stats']['count']} |")
    lines += ['', '性能数据来源：' + '；'.join(f"{TITLES[m['model']['id']]}：`{m['performance_source']}`" for m in models) + '。', '',
        '计时前后同步 MPS；单次调用包含预处理、前向、后处理及结果转 CPU，排除 PNG 读取、视频解码、绘图、网络下载。整轮墙钟还包括资源/进程采样与日志，门控补测还包括帧间等待和被弃用的推理；首次预测包含设备迁移、predictor 初始化与融合等成本。模型调用 FPS 不是完整视频入库或在线检索 FPS。', '',
        f"一次性准备帧共 {summary['dataset']['decode_and_write_seconds']:.2f} 秒，包含视频打开、抽帧读取、PNG 编码写入和 SHA256，不能称纯解码耗时。每模型预载 PNG 并校验哈希分别为 " + '、'.join(f"{TITLES[m['model']['id']]} {m['timing']['png_read_and_hash_seconds']:.2f} 秒" for m in models) + '。', '', '## 内存', '',
        '| 模型 | 加载前 RSS→峰值 GiB | Metal allocated 峰值 GiB | Metal driver 峰值 GiB | 系统可用最低 GiB | 系统 swap 起始→结束 GiB |',
        '|---|---:|---:|---:|---:|---|']
    for m in models:
        r = m['resources']; p = r['sampled_peak']
        lines.append(f"| {TITLES[m['model']['id']]} | {gi(r['before']['rss_bytes']):.3f}→{gi(p['rss_bytes']):.3f} | {gi(p['mps_allocated_bytes']):.3f} | {gi(p['mps_driver_bytes']):.3f} | {gi(r['min_system_available_bytes']):.2f} | {gi(r['before']['system_swap_used_bytes']):.2f}→{gi(r['after']['system_swap_used_bytes']):.2f} |")
    lines += ['', '每 50 ms 采样并在热测每帧结束后补采；峰值可能漏掉短暂尖峰。RSS 是当前实验进程，含已预载的相同输入帧；RSS、Metal allocated、Metal driver 口径重叠，不能相加。系统可用内存和 swap 包含正式服务及其他应用，不能当作本模型占用。门控补测的资源采样也覆盖帧间等待和被弃用的推理，不能称为完全隔离 GPU 环境的内存实测。加载前/后和逐轮资源明细保留在原始结果及 summary.json。', '', '## 重复性和运行干扰', '']
    for m in models:
        lines.append(f"- {TITLES[m['model']['id']]}：三轮分类状态一致={m['repeat_prediction_states_identical']}；类别/分数 4 位小数、坐标 3 位小数一致={m['repeat_detections_equal_4decimals_xyxy_3decimals']}。")
    lines += ['', '原主测试并发记录：' + '；'.join(
        f"{TITLES[m['model']['id']]} 共 {len(m['original_performance']['concurrent_experiments'])} 次检查观察到其他实验进程" for m in models) + '。原主测试时延保留在 summary.json 的 original_performance，不用其受干扰时延直接作公平比较。']
    if any(m['performance_comparison_contaminated'] for m in models):
        lines += ['', '**至少一个模型仍使用受干扰的主测试时延，当前性能表不能作公平模型速度比较；等待门控补测完成后重算。**']
    if any(m['isolate_each_prediction'] for m in models):
        lines += ['', '| 门控补测模型 | 开始前等待 s | 帧间等待/门控 s | 弃用并发推理次数 / s | 与主测分类一致 |', '|---|---:|---:|---:|---|']
        for m in models:
            if m['isolate_each_prediction']:
                p = m['performance_protocol']
                lines.append(f"| {TITLES[m['model']['id']]} | {p['initial_wait_seconds']:.2f} | {p['inter_frame_wait_seconds']:.2f} | {p['discarded_contended_prediction_count']} / {p['discarded_contended_prediction_seconds']:.2f} | {m['clean_vs_original_prediction_states_identical']} |")
        lines += ['', '门控补测每次预测前等待其他实验进程退出，预测后若发现新的实验进程，则丢弃该次计时并重做；表中热调用只保留通过两端检查的结果，等待时间不计入调用 FPS。片段边界看到其他进程不等于该帧被计入了并发推理。加载和预热没有逐次门控，其时长只作为该轮观察值。', '',
            '进程检查并非连续 GPU 活动追踪，不能排除检查间隙内启动并结束的短任务、桌面应用负载或等待后的缓存变化。正式检索服务始终运行。']
    lines += ['', '## 正式检索与复现', '']
    prod = summary['production']
    if prod['status'] == 'pending_after_snapshot':
        lines.append('结束后生产快照尚未生成；本报告暂不声称已完成前后不变验证。生成 production-after.json 后重新执行 evaluate.py。')
    elif prod['status'] == 'unchanged':
        lines.append(f"前后核对 {prod['files_checked']} 个正式文件哈希、活动发布指针、服务 PID 和索引状态均未改变；服务 PID {', '.join(prod['server_pids_after'])}，{prod['index_status_after']['indexed_videos']} 视频、{prod['index_status_after']['indexed_segments']} 片段。")
    else:
        lines += ['生产快照存在差异，不能宣称全部未变；详见 summary.json 的 production。', '```json', json.dumps(prod, ensure_ascii=False, indent=2), '```']
    lines += ['', '离线复算：`python3 experiments/ppe-pretrained/evaluate.py`。它只读取已有结果和冻结标签，不加载模型、不访问正式数据库。', '',
        '两模型的仓库 revision、下载 SHA256、文件大小和路径见 [models.json](models.json)；固定标签哈希见 [labels-freeze.json](labels-freeze.json)。本轮数据与模型训练来源可能存在重叠，未作训练集排重，不能作为独立泛化评测。', '']
    (HERE / 'REPORT.md').write_text('\n'.join(lines))


def main():
    dataset, labels, frozen = read('dataset.json'), read('labels.json'), read('labels-freeze.json')
    dataset_hash, labels_hash = sha('dataset.json'), sha('labels.json')
    assert dataset_hash == frozen['dataset_sha256'] and labels_hash == frozen['labels_sha256']
    expected_ids = [f['id'] for c in dataset['cases'] for f in c['frames']]
    assert len(expected_ids) == len(set(expected_ids))
    assert set(expected_ids) == {r['frame_id'] for r in labels['frames']}
    assert len(labels['frames']) == len(expected_ids)
    runs = []
    for model_id in MODELS:
        path = HERE / f'{model_id}-mps-fp32.json'
        if not path.exists():
            raise SystemExit(f'Not ready: {path.name} is missing')
        run = json.loads(path.read_text())
        validate_run(run, expected_ids, dataset_hash, labels_hash)
        runs.append(run)
    assert runs[0]['options'] == runs[1]['options']
    models, frame_rows = [], []
    for run in runs:
        model, frames = evaluate_model(run, dataset, labels)
        select_performance(model, run, dataset, labels, expected_ids, dataset_hash, labels_hash)
        models.append(model); frame_rows.extend(frames)
    summary = {'schema': 1, 'scope': 'fixed-threshold frame-level no-helmet detection, not object mAP or construction-query accuracy',
        'dataset': {'sha256': dataset_hash, 'labels_sha256': labels_hash, 'cases': len(dataset['cases']),
            'frames': len(expected_ids), 'frame_state_counts': dict(collections.Counter(x['presence'] for x in labels['frames'])),
            'decode_and_write_seconds': sum(c['decode_and_write_seconds'] for c in dataset['cases']),
            'sampling_fps': dataset['sampling_fps']}, 'options': runs[0]['options'],
        'models': models, 'production': production_comparison()}
    (HERE / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (HERE / 'frame-results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
        writer.writeheader(); writer.writerows(frame_rows)
    write_report(summary)
    print(json.dumps({'models': [{'id': m['model']['id'], 'confusion': m['confusion'],
        'median_ms': m['timing']['median_ms'], 'p95_ms': m['timing']['p95_ms'],
        'performance_source': m['performance_source'],
        'concurrent_check_count': len(m['concurrent_experiments'])} for m in models],
        'production': summary['production']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
