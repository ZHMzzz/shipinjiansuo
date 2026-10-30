"""Keep the active release and its actual predecessor, under the writer lock.

History lives outside demo/ so copying a project cannot make a draft look
published. Only releases explicitly recorded in the active history are removed.
"""
import json
import logging
import os
from pathlib import Path
import shutil

HISTORY = 'publication.json'


def release_path(root, name):
    if not isinstance(name, str) or not name or name in ('.', '..') or Path(name).name != name:
        raise ValueError('索引版本名称不合法')
    base = Path(root).resolve() / 'projects/releases'
    path = base / name
    if base.is_symlink() or path.is_symlink() or path.resolve() != path:
        raise ValueError('索引版本路径不合法')
    return path


def managed_release(root, project):
    project = Path(project).absolute()
    release = release_path(root, project.parent.name)
    if project != release / 'demo' or project.is_symlink() or not project.is_dir():
        raise ValueError('仅允许发布托管索引版本')
    return release


def read_history(release):
    path = release / HISTORY
    if not path.exists():
        return {'previous': None, 'obsolete': []}
    record = json.loads(path.read_text())
    if record.get('release') != release.name or not isinstance(record.get('obsolete'), list):
        raise ValueError('索引发布记录不合法')
    return record


def write_history(release, previous, obsolete):
    temporary = release / (HISTORY + '.tmp')
    temporary.write_text(json.dumps({'release': release.name, 'previous': previous,
                                    'obsolete': obsolete}, indent=2) + '\n')
    os.replace(temporary, release / HISTORY)


def adopt_history(root, previous, older=()):
    """One-time adoption of explicitly reviewed legacy releases; writer stopped."""
    active = managed_release(root, (Path(root) / 'projects/demo').resolve(strict=True))
    if (active / HISTORY).exists():
        raise ValueError('当前版本已有发布记录')
    prior = managed_release(root, previous)
    obsolete = [managed_release(root, p).name for p in older]
    if active == prior or {active.name, prior.name}.intersection(obsolete):
        raise ValueError('保留版本不能进入清理列表')
    write_history(active, prior.name, obsolete)


def publish_pointer(root, stage):
    """Record history before the atomic pointer swap; do not clean while loading."""
    root = Path(root).resolve()
    pointer = root / 'projects/demo'
    current = managed_release(root, pointer.resolve(strict=True))
    release = managed_release(root, stage)
    if current == release:
        return
    history = read_history(current)
    obsolete = list(history['obsolete'])
    if history['previous']:
        obsolete.append(history['previous'])
    obsolete = list(dict.fromkeys(n for n in obsolete if n not in (current.name, release.name)))
    for name in obsolete:
        release_path(root, name)
    write_history(release, current.name, obsolete)
    temporary = root / 'projects/demo.next'
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(release / 'demo', pointer.parent))
    os.replace(temporary, pointer)


def prune_releases(root):
    """Best-effort, retryable cleanup. Caller holds the exclusive writer lock."""
    removed = []
    try:
        root = Path(root).resolve()
        pointer = root / 'projects/demo'
        active = managed_release(root, pointer.resolve(strict=True))
        record = read_history(active)
        if not record['obsolete']:
            return removed
        deletion = active / 'demo/deletion.json'
        if deletion.exists() and not json.loads(deletion.read_text())['cleanup_complete']:
            return removed
        previous = release_path(root, record['previous'])
        if previous == active or not (previous / 'demo').is_dir():
            raise ValueError('缺少可保留的上一发布版本，暂缓清理')
        candidates = [release_path(root, name) for name in record['obsolete']]
        if any(p in (active, previous) for p in candidates):
            raise ValueError('清理列表包含保留版本')
        for path in candidates:
            if pointer.resolve(strict=True) != active / 'demo':
                raise RuntimeError('活动版本已变化，停止清理')
            try:
                if path.exists():
                    shutil.rmtree(path)
                removed.append(path.name)
            except OSError:
                logging.exception('旧索引版本清理失败，将重试：%s', path.name)
        write_history(active, previous.name, [n for n in record['obsolete'] if n not in removed])
        if removed:
            logging.info('已清理 %d 个旧索引版本，保留 %s 和 %s', len(removed), active.name, previous.name)
    except Exception:
        logging.exception('索引版本清理未完成，将重试；当前版本保持不变')
    return removed
