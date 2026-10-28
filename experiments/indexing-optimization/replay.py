"""Replay upload stages in disposable full copies, without publishing anything.

Run only when the experiment owner schedules the GPU, with scripts/env.sh
sourced. A suite warms one Qwen ActionEngine before timing any replay. Each
trial starts from the same frozen production project and adds one media record
for a retranscoded upload. Existing copies of that content remain in the base
project: this measures today's library size, not its historical upload state.

The extraction subprocess delegates to bench.child, including its common timing
probes. Those nested probe times must not be added to subprocess wall time.
SHA checks and output cleanup are reported separately from pipeline wall time.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OWL = "transformers/owlv2/google/owlv2-base-patch16-ensemble"
FACE = "deepinsight/insightface/buffalo_l/_unknown"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def project_hashes(project):
    return {str(path.relative_to(project)): sha(path)
            for path in sorted(project.rglob("*"))
            if path.is_file() and path.suffix != ".log"}


def rodb(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def child_environment():
    return {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "FEATURE_EXTRACTOR_CONFIG": json.dumps({
            "default": {"device": "mps", "compile": False, "warmup": True},
            FACE: {"device": "cpu", "compile": False, "warmup": True},
        }),
    }


def run_command(command, log_path, *, timeout=2400):
    with Path(log_path).open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=child_environment(),
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")


def source_record(media_id, design):
    media = next(item for item in design["media"] if item["id"] == media_id)
    with rodb(ROOT / "data/ingestion.sqlite") as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT id, filename, suffix, sha256, bytes, duration FROM jobs "
            "WHERE media_id=? AND status='completed' ORDER BY updated DESC LIMIT 1",
            (media_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"No completed original upload for media {media_id}")
    job = dict(row)
    source = ROOT / "data/uploads" / job["id"] / ("source" + job["suffix"])
    if sha(source) != job["sha256"] or source.stat().st_size != job["bytes"]:
        raise RuntimeError(f"Original upload changed for media {media_id}")
    prepared = Path(media["absolute_path"])
    if sha(prepared) != media["sha256"]:
        raise RuntimeError(f"Frozen prepared video changed for media {media_id}")
    stem = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", Path(job["filename"]).stem)[:90] or "video"
    if stem + ".mp4" != media["path"]:
        raise RuntimeError("Original upload filename does not align with the frozen media")
    return {"media": media, "job": job, "source": str(source), "prepared_name": stem + ".mp4"}


def prepare_runtime():
    """Load classes once, as they already are in the production server."""
    import numpy as np
    import torch
    from action_search import ActionEngine, read_snapshot, windows
    from qwen_project import validate_project
    from wise.config import APIConfig
    from wise.wise_project import WiseProject
    from wise.api.services.project import LocalWiseProjectService
    from wise.api.services.search import LocalSearchService
    from resource_metrics import ResourceSampler
    return locals()


def validate_and_load(project, runtime):
    """Run the publisher's checks/load without importing server or swapping it."""
    runtime["validate_project"](project, require_owl=True)
    snapshot = runtime["read_snapshot"](project, complete=True)
    wise_project = runtime["WiseProject"](project, read_only=True)
    try:
        config = runtime["APIConfig"](command="serve", project_dir=project)
        service = runtime["LocalWiseProjectService"](wise_project, config)
        # LocalSearchService.__init__ only stores this handle; loading indexes
        # and featured IDs does not call an embedding model.
        search = runtime["LocalSearchService"](service, None)
        counts = {}
        with rodb(project / "metadata/internal.db") as db:
            for modality, indexes in search.search_indices.items():
                for model, index in indexes.items():
                    expected = db.execute(
                        "SELECT count(*) FROM vectors WHERE feature_extractor_id=? AND modality=?",
                        (model, modality.name),
                    ).fetchone()[0]
                    if index.index.ntotal != expected:
                        raise RuntimeError(f"Loaded index count mismatch: {model}")
                    counts[f"{modality.name}:{model}"] = expected
        info = service.info()
        return {"index_counts": counts, "media_count": info.num_media_files,
                "qwen_segments": len(snapshot[0]["segments"])}
    finally:
        wise_project.db_engine.dispose()
        wise_project.thumbsdb_engine.dispose()


def safe_cleanup(work, work_root):
    """Remove only the unique direct child this suite created."""
    if work.is_symlink() or work.parent.resolve() != work_root.resolve() or not work.name.startswith("replay-"):
        raise RuntimeError(f"Refusing unsafe experiment cleanup: {work}")
    shutil.rmtree(work)


def replay_one(source, config, trial_dir, suite, base, base_hashes, actions, runtime,
               *, keep_project=False):
    trial_dir.mkdir()
    work_root = suite / "work"
    work_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="replay-", dir=work_root))
    project = work / "project"
    prepared = work / "prepared"
    prepared.mkdir()
    video = prepared / source["prepared_name"]
    metrics_path = trial_dir / "timing.json"
    row = {"status": "running", "config": config,
           "original_media_id": source["media"]["id"], "source": source,
           "base_project": str(base), "temporary_project": str(project),
           "stages": {}, "excluded_stages": [], "started_epoch": time.time(),
           "scope": "probe-to-index-load; no HTTP upload, queue wait, or publication"}
    dump(metrics_path, row)
    resources = runtime["ResourceSampler"]()
    resource_sampling_started = False
    started = time.perf_counter()

    def measure(name, operation, *, excluded=False):
        before = time.perf_counter()
        try:
            return operation()
        finally:
            row["stages"][name] = time.perf_counter() - before
            if excluded:
                row["excluded_stages"].append(name)
            dump(metrics_path, row)

    try:
        resources.__enter__()
        resource_sampling_started = True
        def probe():
            result = subprocess.run(
                [str(ROOT / ".media-runtime/bin/ffprobe"), "-v", "error", "-show_format",
                 "-show_streams", "-of", "json", source["source"]],
                capture_output=True, text=True, timeout=30, check=True,
            )
            meta = json.loads(result.stdout)
            duration = float(meta.get("format", {}).get("duration", 0))
            if not math.isfinite(duration) or duration <= 0 or not any(
                stream.get("codec_type") == "video" for stream in meta.get("streams", [])
            ):
                raise RuntimeError("Original upload has no valid video duration")
            return meta
        row["source_probe"] = measure("probe", probe)
        command = [str(ROOT / ".media-runtime/bin/ffmpeg"), "-y", "-v", "error", "-nostdin",
                   "-i", source["source"], "-map", "0:v:0", "-map", "0:a:0?", "-vf",
                   "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2",
                   "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-movflags", "+faststart", "-progress", "pipe:1", str(video)]
        measure("transcode", lambda: run_command(command, trial_dir / "transcode.log"))
        def verify_transcode():
            actual = sha(video)
            row["transcoded_sha256"] = actual
            row["matches_frozen_prepared_sha256"] = actual == source["media"]["sha256"]
            if not row["matches_frozen_prepared_sha256"]:
                raise RuntimeError("Retranscoded pixels/container differ from the frozen prepared video")
        measure("transcoded_sha256_check", verify_transcode, excluded=True)
        if (ROOT / "projects/demo").resolve() != base:
            raise RuntimeError("Active production project changed during the experiment")
        copy_bytes = sum(path.stat().st_size for path in base.rglob("*") if path.is_file())
        if shutil.disk_usage(work).free < copy_bytes + 3 * 1024**3:
            raise RuntimeError("Insufficient disk for an isolated full project copy")
        row["base_copy_bytes"] = copy_bytes
        measure("full_project_copy", lambda: shutil.copytree(base, project))
        def verify_copy():
            copied = project_hashes(project)
            if copied != base_hashes:
                raise RuntimeError("Full-copy hashes do not match the frozen production project")
            row["full_copy_file_count"] = len(copied)
            row["full_copy_hashes_match"] = True
        measure("full_copy_sha256_check", verify_copy, excluded=True)
        measure("pre_extraction_validation", lambda: runtime["validate_project"](project, require_owl=True))
        command = [sys.executable, "-u", str(OUT / "bench.py"), "child",
                   "--inputs", str(prepared), "--project", str(project),
                   "--metrics", str(trial_dir / "extract-metrics.json"),
                   "--owl", config["owl"], "--face", config["face"],
                   "--processor-device", config["processor_device"]]
        measure("wise_extract_subprocess", lambda: run_command(command, trial_dir / "extract.log"))
        measure("wise_create_index_subprocess", lambda: run_command(
            [sys.executable, "-u", "-m", "wise", "create-index", "--project-dir", str(project), "--overwrite"],
            trial_dir / "index.log",
        ))

        def validate_new_media():
            with rodb(project / "metadata/internal.db") as db:
                db.row_factory = sqlite3.Row
                matches = db.execute(
                    "SELECT m.* FROM media m JOIN source_collections s ON s.id=m.source_collection_id "
                    "WHERE s.location=?", (str(prepared),),
                ).fetchall()
                if len(matches) != 1 or matches[0]["path"] != video.name:
                    raise RuntimeError("Replay did not add exactly the intended media")
                media = dict(matches[0])
                original = db.execute("SELECT * FROM media WHERE id=?", (source["media"]["id"],)).fetchone()
                if original is None or media["id"] == original["id"]:
                    raise RuntimeError("Replay did not create a new media identity")
                for field in ("checksum", "width", "height", "num_frames"):
                    if media[field] != original[field]:
                        raise RuntimeError(f"Replay media {field} differs from the frozen media")
                if abs(media["duration"] - source["media"]["duration"]) > 0.05:
                    raise RuntimeError("Replay media duration differs from the frozen prepared video")
                counts = dict(db.execute(
                    "SELECT feature_extractor_id,count(*) FROM vectors WHERE media_id=? GROUP BY 1",
                    (media["id"],),
                ))
                if set(counts) - {OWL, FACE} or not counts.get(OWL):
                    raise RuntimeError("Unexpected replay model set or missing OWLv2 regions")
            with rodb(project / "thumbs.db") as db:
                thumbs = db.execute("SELECT count(*) FROM thumbnails WHERE media_id=?", (media["id"],)).fetchone()[0]
                if not thumbs:
                    raise RuntimeError("Replay has no thumbnails")
            return {"media": media, "vector_counts": counts, "thumbnails": thumbs}
        row["new_media"] = measure("new_media_validation", validate_new_media)
        catalog, records, vectors, missing = measure("qwen_prepare", lambda: actions.prepare(project))
        new_id = row["new_media"]["media"]["id"]
        if [media["id"] for media in missing] != [new_id]:
            raise RuntimeError("Qwen preparation did not isolate exactly the newly added video")
        old_records = list(records)
        old_vectors = runtime["np"].asarray(vectors).copy()
        row["reused_qwen_segments"] = len(old_records)

        def encode_new():
            for media in missing:
                for start, end in runtime["windows"](float(media["duration"])):
                    segment = {"media_id": media["id"], "ts": start, "te": end}
                    vector = actions.encode_segment(media, segment)
                    records.append(segment)
                    vectors.append(vector)
            runtime["torch"].mps.synchronize()
        measure("qwen_encode_new_segments", encode_new)
        measure("qwen_finish", lambda: actions.finish(project, catalog, records, vectors))
        row["loaded"] = measure("validate_and_load", lambda: validate_and_load(project, runtime))

        def verify_reuse():
            metadata, matrix, _ = runtime["read_snapshot"](project, complete=True)
            old_positions = [i for i, segment in enumerate(metadata["segments"]) if segment["media_id"] != new_id]
            if [metadata["segments"][i] for i in old_positions] != old_records:
                raise RuntimeError("Replay changed existing Qwen segment metadata")
            if not runtime["np"].array_equal(matrix[old_positions], old_vectors):
                raise RuntimeError("Replay changed existing Qwen vectors")
            new_positions = [i for i, segment in enumerate(metadata["segments"]) if segment["media_id"] == new_id]
            row["new_qwen_segments"] = len(new_positions)
            row["new_qwen_vectors_sha256"] = hashlib.sha256(matrix[new_positions].tobytes()).hexdigest()
            row["old_qwen_vectors_exactly_unchanged"] = True
            for relative, expected in base_hashes.items():
                if relative.startswith("store/") and "/features/" in relative:
                    if sha(project / relative) != expected:
                        raise RuntimeError(f"Replay modified an existing feature shard: {relative}")
            row["old_wise_feature_shards_exactly_unchanged"] = True
        measure("qwen_reuse_verification", verify_reuse, excluded=True)
        row["status"] = "completed"
    except BaseException as error:
        row["status"] = "failed"
        row["error"] = repr(error)
        raise
    finally:
        row["elapsed_wall_seconds"] = time.perf_counter() - started
        if resource_sampling_started:
            resources.__exit__(*sys.exc_info())
            row["resources"] = resources.result
        row["resource_scope_note"] = (
            "Replay-parent RSS/MPS includes resident Qwen; extraction child has its own "
            "resources in extract-metrics.json. System availability/swap include the "
            "online server and other processes. Do not add overlapping measurements."
        )
        row["excluded_verification_seconds"] = sum(row["stages"][name] for name in row["excluded_stages"])
        row["adjusted_wall_seconds"] = row["elapsed_wall_seconds"] - row["excluded_verification_seconds"]
        row["wall_definition"] = (
            "elapsed_wall_seconds covers probe through completed index load and SHA checks. "
            "adjusted_wall_seconds subtracts only listed excluded_stages. "
            "Qwen cold loading, runtime imports, and post-trial cleanup are outside both."
        )
        row["finished_epoch"] = time.time()
        dump(metrics_path, row)
        cleanup_started = time.perf_counter()
        if not keep_project:
            safe_cleanup(work, work_root)
        row["temporary_project_retained"] = keep_project
        row["cleanup_seconds"] = time.perf_counter() - cleanup_started
        dump(metrics_path, row)
        gc.collect()
    return row


def run_suite(name, configurations, *, media_ids=(40, 41), repeats=1, keep_projects=False,
              alternate_order=True):
    """Run configs sequentially, reversing their order on even repeats by default."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
        raise ValueError("Invalid suite name")
    if repeats < 1 or not media_ids or any(media_id not in (40, 41) for media_id in media_ids):
        raise ValueError("Specify positive repeats and replay media IDs 40 and/or 41")
    configurations = [dict(config) for config in configurations]
    if not configurations:
        raise ValueError("At least one replay configuration is required")
    for config in configurations:
        config.setdefault("face", "cpu")
        config.setdefault("processor_device", "cpu")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", config["name"]):
            raise ValueError("Invalid configuration name")
    if len({config["name"] for config in configurations}) != len(configurations):
        raise ValueError("Configuration names must be unique")
    design_path = OUT / "design.json"
    design = json.loads(design_path.read_text())
    base = (ROOT / "projects/demo").resolve(strict=True)
    if str(base) != design["production_before"]["project"]:
        raise RuntimeError("Active project differs from the frozen design")
    base_hashes = project_hashes(base)
    if base_hashes != design["production_before"]["project_files"]:
        raise RuntimeError("Production project files differ from the frozen design")
    sources = [source_record(media_id, design) for media_id in media_ids]
    suite = OUT / "replays" / name
    suite.mkdir(parents=True, exist_ok=False)
    dump(suite / "base-project-sha256.json", base_hashes)
    report = {"name": name, "design_sha256": sha(design_path), "base_project": str(base),
              "configurations": configurations, "repeats": repeats, "sources": sources,
              "alternate_configuration_order": alternate_order,
              "face_cache": str(OUT / "cache/face"),
              "status": "running", "trials": [],
              "note": "Cold Qwen load is separate; every trial copies the same complete base and adds one media."}
    dump(suite / "suite.json", report)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    started = time.perf_counter()
    runtime = prepare_runtime()
    report["runtime_import_seconds"] = time.perf_counter() - started
    actions = runtime["ActionEngine"]()
    try:
        started = time.perf_counter()
        warmup_resources = runtime["ResourceSampler"]()
        try:
            with warmup_resources:
                actions.query_vector("视频中的场景和动作")
                runtime["torch"].mps.synchronize()
        finally:
            if hasattr(warmup_resources, "result"):
                report["qwen_warmup_resources"] = warmup_resources.result
        report["qwen_load_and_text_warmup_seconds"] = time.perf_counter() - started
        report["qwen_model_load_seconds"] = actions.last_load_seconds
        dump(suite / "suite.json", report)
        for repeat in range(1, repeats + 1):
            ordered_configs = configurations[::-1] if alternate_order and repeat % 2 == 0 else configurations
            for source in sources:
                for config in ordered_configs:
                    trial = f"r{repeat}-media{source['media']['id']}-{config['name']}"
                    row = replay_one(source, config, suite / trial, suite, base, base_hashes,
                                     actions, runtime, keep_project=keep_projects)
                    report["trials"].append({"trial": trial, "status": row["status"],
                                             "elapsed_wall_seconds": row["elapsed_wall_seconds"],
                                             "adjusted_wall_seconds": row["adjusted_wall_seconds"]})
                    dump(suite / "suite.json", report)
                    print(trial, "adjusted_wall_seconds", round(row["adjusted_wall_seconds"], 3), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        actions.unload()
        report["production_pointer_unchanged"] = (ROOT / "projects/demo").resolve() == base
        report["production_project_hashes_unchanged"] = project_hashes(base) == base_hashes
        if not report["production_pointer_unchanged"] or not report["production_project_hashes_unchanged"]:
            report["status"] = "invalid-production-changed"
        dump(suite / "suite.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--name", required=True)
    run.add_argument("--configs", required=True, help="JSON file containing an ordered list of configurations")
    run.add_argument("--media-ids", type=int, nargs="+", default=[40, 41])
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--keep-projects", action="store_true")
    run.add_argument("--fixed-order", action="store_true", help="Do not reverse configuration order on even repeats")
    args = parser.parse_args()
    run_suite(args.name, json.loads(Path(args.configs).read_text()), media_ids=args.media_ids,
              repeats=args.repeats, keep_projects=args.keep_projects,
              alternate_order=not args.fixed_order)


if __name__ == "__main__":
    main()
