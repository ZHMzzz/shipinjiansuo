"""Process-local InsightFace runtime variants; never edits production models.

Install before constructing WISE's InsightFace extractor. The 0.7.3 factory is
patched only for files in the configured buffalo_l directory. Existing sessions
are unaffected. CoreML options follow the ONNX Runtime CoreML EP documentation:
https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import weakref

EXPERIMENT = Path(__file__).resolve().parent
DEFAULT_MODELS = Path.home() / '.insightface/models/buffalo_l'
SHAPES = {
    'det_10g.onnx': (1, 3, 640, 640),
    'w600k_r50.onnx': (1, 3, 112, 112),
    'genderage.onnx': (1, 3, 96, 96),
}
VARIANTS = ('cpu', 'coreml', 'coreml_ane', 'coreml_gpu')
COMPUTE_UNITS = {'coreml': 'ALL', 'coreml_ane': 'CPUAndNeuralEngine',
                 'coreml_gpu': 'CPUAndGPU'}
_active = None


def experiment_path(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(EXPERIMENT):
        raise ValueError(f'Experiment output must remain under {EXPERIMENT}')
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def input_contract(model):
    return [{'name': item.name, 'shape': [dim.dim_value if dim.HasField('dim_value')
             else dim.dim_param for dim in item.type.tensor_type.shape.dim]}
            for item in model.graph.input]


def inspect_models(source_dir=DEFAULT_MODELS):
    """Read protobufs and hashes only. Does not instantiate any ORT session."""
    import onnx
    records = []
    for name, shape in SHAPES.items():
        path = Path(source_dir).expanduser().resolve() / name
        model = onnx.load(path, load_external_data=False)
        records.append({'name': name, 'path': str(path), 'sha256': sha256(path),
                        'inputs': input_contract(model), 'static_shape': list(shape),
                        'nodes': len(model.graph.node), 'outputs': len(model.graph.output)})
    return records


def _initializer_digest(model):
    digest = hashlib.sha256()
    for tensor in model.graph.initializer:
        digest.update(tensor.SerializeToString(deterministic=True))
    return digest.hexdigest()


def static_model(source, cache_dir):
    """Fix batch/spatial input dimensions in a new, content-addressed file."""
    import onnx
    source = Path(source).resolve()
    source_sha = sha256(source)
    shape = SHAPES[source.name]
    identity = hashlib.sha256((source_sha + repr(shape) + 'fixed-input-v1').encode()).hexdigest()
    folder = experiment_path(cache_dir) / 'onnx' / identity
    target, manifest = folder / source.name, folder / 'manifest.json'
    if target.exists() and manifest.exists():
        record = json.loads(manifest.read_text())
        if record['source_sha256'] == source_sha and record['model_sha256'] == sha256(target):
            return target, record
        raise RuntimeError(f'Static model cache was modified: {folder}')
    model = onnx.load(source)
    original_weights = _initializer_digest(model)
    if len(model.graph.input) != 1:
        raise ValueError(f'Expected one input in {source}')
    dimensions = model.graph.input[0].type.tensor_type.shape.dim
    if len(dimensions) != len(shape):
        raise ValueError(f'Unexpected input rank in {source}')
    for dim, size in zip(dimensions, shape):
        dim.ClearField('dim_param')
        dim.dim_value = size
    # ORT also performs shape inference; this resolves intermediate shapes early.
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    existing = {entry.key: entry.value for entry in model.metadata_props}
    existing['CACHE_KEY'] = identity[:63]
    onnx.helper.set_model_props(model, existing)
    onnx.checker.check_model(model)
    if original_weights != _initializer_digest(model):
        raise RuntimeError('Shape specialization unexpectedly changed model weights')
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / (source.name + '.tmp')
    onnx.save(model, temporary)
    temporary.replace(target)
    record = {'source_path': str(source), 'source_sha256': source_sha,
              'model_path': str(target), 'model_sha256': sha256(target),
              'initializer_sha256': original_weights, 'inputs': input_contract(model),
              'weights_unchanged': True, 'cache_key': identity[:63]}
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    return target, record


def summarize_profile(path):
    """Summarize executed kernels, not counts of the original ONNX graph nodes."""
    events = json.loads(Path(path).read_text())
    groups = defaultdict(lambda: {'calls': 0, 'duration_us': 0, 'nodes': set(), 'ops': set()})
    for event in events:
        args = event.get('args', {})
        provider = args.get('provider')
        if not provider or event.get('cat') != 'Node':
            continue
        item = groups[provider]
        item['calls'] += 1
        item['duration_us'] += event.get('dur', 0)
        item['nodes'].add(event.get('name', ''))
        item['ops'].add(args.get('op_name', ''))
    result = {name: {**item, 'nodes': sorted(item['nodes']), 'ops': sorted(item['ops']),
                     'unique_executed_kernels': len(item['nodes'])}
              for name, item in groups.items()}
    return {'path': str(path), 'providers': result,
            'coreml_executed': result.get('CoreMLExecutionProvider', {}).get('calls', 0) > 0,
            'cpu_fallback_executed': result.get('CPUExecutionProvider', {}).get('calls', 0) > 0,
            'count_semantics': 'Executed optimized/fused kernels, not original ONNX node coverage. '
                               'CoreML EP events do not identify ANE versus GPU execution.'}


class VariantInstallation:
    def __init__(self, name, source_dir, cache_dir, profile_dir, quiet, profile_compute_plan):
        import onnxruntime as ort
        from insightface.model_zoo import model_zoo
        self.name, self.source_dir = name, Path(source_dir).expanduser().resolve()
        self.cache_dir = experiment_path(cache_dir)
        self.profile_dir = experiment_path(profile_dir) if profile_dir else None
        self.records, self._sessions = [], []
        self._module, self._original = model_zoo, model_zoo.PickableInferenceSession
        self._installed = None
        self._profiling_finished = False
        if name != 'cpu' and 'CoreMLExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('This ONNX Runtime does not expose CoreMLExecutionProvider')
        self.available_providers = ort.get_available_providers()
        self.stamp = str(time.time_ns())
        controller, original = self, self._original

        class ExperimentSession(original):
            def __init__(self, model_path, **kwargs):
                source = Path(model_path).resolve()
                if source.parent != controller.source_dir:
                    super().__init__(model_path, **kwargs)
                    return
                active = source.name in SHAPES
                coreml = name != 'cpu' and active
                record = {'source': str(source), 'active_task_model': active,
                          'variant': name, 'profile_prefix': None}
                if coreml:
                    prepared_at = time.perf_counter()
                    model_path, provenance = static_model(source, controller.cache_dir)
                    record.update(static_model=provenance,
                                  static_preparation_s=time.perf_counter() - prepared_at)
                    compiled = controller.cache_dir / 'compiled' / name
                    compiled.mkdir(parents=True, exist_ok=True)
                    options = {'ModelFormat': 'MLProgram', 'MLComputeUnits': COMPUTE_UNITS[name],
                               'RequireStaticInputShapes': '1', 'EnableOnSubgraphs': '0',
                               'ModelCacheDirectory': str(compiled)}
                    if profile_compute_plan:
                        options['ProfileComputePlan'] = '1'
                    kwargs['providers'] = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
                    kwargs['provider_options'] = [options, {}]
                else:
                    # Retain 0.7.3's initialization of subsequently discarded landmarks.
                    kwargs['providers'], kwargs['provider_options'] = ['CPUExecutionProvider'], [{}]
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 3 if quiet else 0
                if controller.profile_dir:
                    controller.profile_dir.mkdir(parents=True, exist_ok=True)
                    prefix = controller.profile_dir / f'{name}-{controller.stamp}-{len(controller.records)}-{source.stem}'
                    session_options.enable_profiling = True
                    session_options.profile_file_prefix = str(prefix)
                    record['profile_prefix'] = str(prefix)
                kwargs['sess_options'] = session_options
                started = time.perf_counter()
                super().__init__(str(model_path), **kwargs)
                record.update(session_creation_s=time.perf_counter() - started,
                              providers=self.get_providers(), options=self.get_provider_options(),
                              input_shapes=[item.shape for item in self.get_inputs()])
                controller.records.append(record)
                controller._sessions.append(weakref.ref(self))
                # Do not silently accept complete fallback after CoreML initialization fails.
                if coreml and 'CoreMLExecutionProvider' not in self.get_providers():
                    raise RuntimeError(f'CoreML session initialization fell back entirely: {source.name}')

        self._installed = ExperimentSession
        model_zoo.PickableInferenceSession = ExperimentSession

    def report(self, finish_profiling=False):
        if finish_profiling and not self._profiling_finished:
            for reference in self._sessions:
                session = reference()
                if session is not None and self.profile_dir:
                    session.end_profiling()
            self._profiling_finished = True
        profiles = []
        if self.profile_dir and self._profiling_finished:
            for record in self.records:
                for path in self.profile_dir.glob(Path(record['profile_prefix']).name + '*.json'):
                    profiles.append({'model': Path(record['source']).name,
                                     'active_task_model': record['active_task_model'],
                                     **summarize_profile(path)})
        return {'variant': self.name, 'available_providers': self.available_providers,
                'sessions': self.records, 'profiles': profiles,
                'profiles_enabled': self.profile_dir is not None,
                'original_models_write_policy': 'read_only',
                'note': 'No production source or package was edited. Inspect profiles for actual '
                        'CoreML execution; provider registration alone is not proof of offload.'}

    def uninstall(self):
        global _active
        if self._module.PickableInferenceSession is self._installed:
            self._module.PickableInferenceSession = self._original
        if _active is self:
            _active = None


def install_variant(name, *, source_dir=DEFAULT_MODELS, cache_dir=None,
                    profile_dir=None, quiet=True, profile_compute_plan=False):
    """Return a controller; call report(finish_profiling=True), then uninstall().

    Existing WISE model objects must be discarded before switching variants.
    No ORT session is created by this function itself.
    """
    global _active
    if name not in VARIANTS:
        raise ValueError(f'Unknown face variant: {name}; choose {VARIANTS}')
    if _active is not None:
        raise RuntimeError('Uninstall the previous face variant before installing another')
    _active = VariantInstallation(name, source_dir, cache_dir or EXPERIMENT / 'cache/face',
                                  profile_dir, quiet, profile_compute_plan)
    return _active
