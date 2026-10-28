"""Standalone local text benchmark. No production modules or requests."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import psutil
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

DIR = Path(__file__).resolve().parent
ROOT = DIR.parent.parent

def fingerprint(project):
    return {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((project / 'action-index').iterdir()) if p.is_file()}

def listener():
    return subprocess.run(['lsof', '-nP', '-iTCP:9670', '-sTCP:LISTEN', '-t'], capture_output=True, text=True).stdout.strip()

def validate(text):
    try:
        data = json.loads(text)
        assert isinstance(data, dict) and set(data) == {'groups', 'uncertain'}
        assert isinstance(data['groups'], list)
        assert isinstance(data['uncertain'], list) and all(isinstance(s, str) for s in data['uncertain'])
        for g in data['groups']:
            assert isinstance(g, dict) and set(g) == {'must', 'must_not', 'prefer', 'relations'}
            assert all(isinstance(v, list) and all(isinstance(s, str) for s in v) for v in g.values())
        return data, True
    except (ValueError, AssertionError, TypeError):
        return None, False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', default='design.json')
    parser.add_argument('--output', default='results.json')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='float16')
    args = parser.parse_args()
    output = DIR / args.output
    if output.exists():
        raise SystemExit('Refusing to overwrite results')
    design_path = DIR / args.design
    design = json.loads(design_path.read_text())
    info = json.loads((DIR / 'model.json').read_text())
    project = (ROOT / 'projects/demo').resolve()
    report = {'model': info, 'dtype': args.dtype, 'design_file': args.design, 'design_sha256': hashlib.sha256(design_path.read_bytes()).hexdigest(),
              'environment': {'platform': platform.platform(), 'torch': torch.__version__, 'transformers': transformers.__version__,
                              'ram_bytes': psutil.virtual_memory().total, 'available_bytes_before': psutil.virtual_memory().available},
              'project': str(project), 'index_before': fingerprint(project), 'server_pid_before': listener(), 'results': []}
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.mps.set_per_process_memory_fraction(.45)
    t = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(info['path'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(info['path'], dtype=getattr(torch, args.dtype), device_map='mps', attn_implementation='sdpa', local_files_only=True).eval()
    torch.mps.synchronize()
    report['model_load_seconds'] = time.perf_counter() - t
    print('LOADED', round(report['model_load_seconds'], 3), flush=True)
    schedule = [('warmup', {'id': 'warmup', 'query': design['warmup'], 'kind': 'warmup'})]
    schedule += [('main', c) for c in design['cases']]
    by_id = {c['id']: c for c in design['cases']}
    for r in range(1, 3):
        schedule += [(f'repeat{r}', by_id[i]) for i in design['repeats']]
    for phase, case in schedule:
        started = time.perf_counter()
        messages = [{'role': 'system', 'content': design['system_prompt']}, {'role': 'user', 'content': case['query']}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors='pt').to('mps')
        torch.mps.synchronize()
        prepared = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=640)
        torch.mps.synchronize()
        finished = time.perf_counter()
        ids = generated[0, inputs['input_ids'].shape[1]:]
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        parsed, valid = validate(raw)
        row = {'id': case['id'], 'kind': case['kind'], 'phase': phase, 'query': case['query'], 'raw': raw, 'parsed': parsed,
               'schema_valid': valid, 'truncated': len(ids) >= 640, 'input_tokens': int(inputs['input_ids'].shape[1]),
               'output_tokens': len(ids), 'preprocess_seconds': prepared - started, 'generation_seconds': finished - prepared,
               'total_seconds': time.perf_counter() - started, 'rss_bytes': psutil.Process().memory_info().rss,
               'mps_driver_bytes': torch.mps.driver_allocated_memory(), 'swap_bytes': psutil.swap_memory().used}
        report['results'].append(row)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(phase, case['id'], 'schema', valid, 'tokens', len(ids), 'seconds', round(row['total_seconds'], 3), flush=True)
    report['index_after'] = fingerprint(project)
    report['pointer_unchanged'] = project == (ROOT / 'projects/demo').resolve()
    report['server_pid_after'] = listener()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    assert report['index_before'] == report['index_after'] and report['pointer_unchanged']

if __name__ == '__main__':
    main()
