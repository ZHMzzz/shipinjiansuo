"""Process-local OWLv2 variants for the indexing experiment.

Call ``install_variant`` before WISE extraction or query encoding. No model is
loaded by importing this module or installing a variant. The normal WISE model,
feature construction, region ordering, threshold, and MPS one-frame limit stay
in use. Do not install these global class patches in the running application or
switch variants while another thread is encoding.

``fp16`` means MPS autocast, not conversion of the stored model weights. Fast
preprocessing is a separate variable: it may change interpolation/rounding and
must be checked against the slow processor before interpreting retrieval scores.
CPU is its default device; MPS preprocessing is an explicit additional trial.
The separate ``*_weights`` variants load genuinely FP16 weights and require a
fresh extractor/process. Restoring patches does not recast an already loaded
FP16 model into a valid FP32 baseline.
"""

from functools import wraps
from typing import Any


VARIANTS = ("baseline", "fast", "fp16", "fast_fp16", "fp16_weights", "fast_fp16_weights")
_ALIASES = {"fast+fp16": "fast_fp16", "mps_fp16": "fp16"}
_originals: dict[str, Any] | None = None
_active = {
    "name": "baseline",
    "fast_processor": False,
    "processor_device": None,
    "autocast_device": None,
    "autocast_dtype": None,
    "weights_dtype": "unchanged",
}


def active_variant() -> dict[str, Any]:
    """Return JSON-serializable settings without loading a model/processor."""
    return dict(_active)


def _load_patch_targets() -> dict[str, Any]:
    global _originals
    if _originals is None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2ImageProcessorFast, Owlv2Processor
        from wise.feature.transformers_owlv2 import (
            TransformersOWLv2FeatureExtractor,
            TransformersOWLv2Model,
        )

        _originals = {
            "torch": torch,
            "processor_class": Owlv2Processor,
            "torch_model_class": Owlv2ForObjectDetection,
            "fast_class": Owlv2ImageProcessorFast,
            "extractor_class": TransformersOWLv2FeatureExtractor,
            "model_class": TransformersOWLv2Model,
            "processor": TransformersOWLv2FeatureExtractor.__dict__["processor"],
            "preprocess": Owlv2ImageProcessorFast.preprocess,
            "get_image_features": TransformersOWLv2Model.get_image_features,
            "get_text_features": TransformersOWLv2Model.get_text_features,
            "load_torch_model": TransformersOWLv2Model._load_torch_model,
        }
    return _originals


def restore_variant() -> dict[str, Any]:
    """Restore the original class methods, including the original lazy processor.

    Fast processors use a separate instance cache. Thus an extractor that had a
    baseline processor before the experiment keeps that exact object, and one
    created during a fast trial lazily creates its baseline processor afterward.
    Existing model weights are never replaced or cast by this module.
    After a ``*_weights`` trial, discard that extractor before running baseline;
    restoring class methods alone cannot change the already loaded FP16 weights.
    """
    global _active
    if _originals is not None:
        targets = _originals
        targets["extractor_class"].processor = targets["processor"]
        targets["fast_class"].preprocess = targets["preprocess"]
        targets["model_class"].get_image_features = targets["get_image_features"]
        targets["model_class"].get_text_features = targets["get_text_features"]
        targets["model_class"]._load_torch_model = targets["load_torch_model"]
    _active = {
        "name": "baseline",
        "fast_processor": False,
        "processor_device": None,
        "autocast_device": None,
        "autocast_dtype": None,
        "weights_dtype": "unchanged",
    }
    return active_variant()


def install_variant(
    name: str,
    *,
    processor_device: str = "cpu",
) -> dict[str, Any]:
    """Install one of ``VARIANTS`` without loading a model.

    ``fast+fp16`` and ``mps_fp16`` are accepted aliases. Repeated installation
    replaces the previous experiment patches instead of nesting wrappers.
    ``processor_device`` may be ``cpu`` or ``mps`` and affects only fast image
    preprocessing. Loading a fast processor is local-cache-only and fails if
    Transformers silently returns a slow processor.

    Autocast wraps both image and text model calls, so WISE image-query and text
    query vectors use the same numerical regime as the indexed regions. Calling
    an fp16 variant on a non-MPS model fails explicitly rather than silently
    producing an FP32 result. Install/restore do not allocate model weights.

    ``fp16_weights`` and ``fast_fp16_weights`` instead request FP16 weights from
    Hugging Face at load time, without autocast. They check the real parameter
    dtype on first use and reject an already cached FP32 model. Both image and
    text outputs still go through WISE's original FP32 output conversions.
    """
    global _active
    name = _ALIASES.get(name, name)
    if name not in VARIANTS:
        raise ValueError(f"Unknown OWLv2 variant {name!r}; expected one of {VARIANTS}")
    if processor_device not in ("cpu", "mps"):
        raise ValueError("processor_device must be 'cpu' or 'mps'")
    use_fast = name in ("fast", "fast_fp16", "fast_fp16_weights")
    use_fp16 = name in ("fp16", "fast_fp16")
    use_fp16_weights = name in ("fp16_weights", "fast_fp16_weights")
    if not use_fast and processor_device != "cpu":
        raise ValueError("processor_device='mps' requires a fast processor variant")

    # Baseline can be selected without even importing Torch/Transformers.
    if name == "baseline":
        return restore_variant()

    targets = _load_patch_targets()
    restore_variant()
    torch = targets["torch"]

    if use_fast:
        original_preprocess = targets["preprocess"]

        @wraps(original_preprocess)
        def preprocess_on_selected_device(processor, *args, **kwargs):
            supplied_device = kwargs.get("device")
            if supplied_device is not None and str(supplied_device) != processor_device:
                raise ValueError(
                    f"Processor device {supplied_device!s} conflicts with the "
                    f"experiment's {processor_device!r} device"
                )
            kwargs["device"] = processor_device
            return original_preprocess(processor, *args, **kwargs)

        def fast_processor(extractor):
            # A data descriptor shadows, but does not destroy, the production
            # cached_property's 'processor' entry when reusing an extractor.
            cache_key = "_indexing_experiment_fast_processor"
            cached = extractor.__dict__.get(cache_key)
            if cached is None or cached[0] != extractor.model_name:
                processor = targets["processor_class"].from_pretrained(
                    extractor.model_name,
                    use_fast=True,
                    local_files_only=True,
                )
                if not isinstance(processor.image_processor, targets["fast_class"]):
                    raise RuntimeError(
                        "use_fast=True did not load Owlv2ImageProcessorFast; "
                        "refusing to label a slow processor as the fast variant"
                    )
                cached = (extractor.model_name, processor)
                extractor.__dict__[cache_key] = cached
            return cached[1]

        targets["fast_class"].preprocess = preprocess_on_selected_device
        targets["extractor_class"].processor = property(fast_processor)

    if use_fp16:
        def autocast_forward(original):
            @wraps(original)
            def wrapped(model, *args, **kwargs):
                if torch.device(model.DEVICE).type != "mps":
                    raise RuntimeError("The fp16 experiment requires an MPS OWLv2 model")
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    # WISE's original methods return FP32 scores/embeddings/boxes.
                    return original(model, *args, **kwargs)
            return wrapped

        targets["model_class"].get_image_features = autocast_forward(targets["get_image_features"])
        targets["model_class"].get_text_features = autocast_forward(targets["get_text_features"])

    if use_fp16_weights:
        @wraps(targets["load_torch_model"])
        def load_fp16_weights(model):
            kwargs = dict(model.model_kwargs)
            if kwargs.get("quantization_config") is not None:
                raise ValueError("FP16-weights trials cannot be combined with quantization")
            # Do not mutate the caller's config, and avoid competing dtype keys.
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = torch.float16
            return targets["torch_model_class"].from_pretrained(model.model_id, **kwargs)

        def checked_fp16_forward(original):
            @wraps(original)
            def wrapped(model, *args, **kwargs):
                if torch.device(model.DEVICE).type != "mps":
                    raise RuntimeError("The FP16-weights experiment requires an MPS OWLv2 model")
                loaded = model.model
                check_key = "_indexing_experiment_checked_fp16_model"
                if model.__dict__.get(check_key) is not loaded:
                    floating_dtypes = {parameter.dtype for parameter in loaded.parameters()
                                       if parameter.is_floating_point()}
                    if floating_dtypes != {torch.float16}:
                        raise RuntimeError(
                            f"Expected FP16 OWLv2 weights, found {floating_dtypes}; "
                            "create a fresh extractor after installing the weight variant"
                        )
                    model.__dict__[check_key] = loaded
                return original(model, *args, **kwargs)
            return wrapped

        targets["model_class"]._load_torch_model = load_fp16_weights
        targets["model_class"].get_image_features = checked_fp16_forward(targets["get_image_features"])
        targets["model_class"].get_text_features = checked_fp16_forward(targets["get_text_features"])

    _active = {
        "name": name,
        "fast_processor": use_fast,
        "processor_device": processor_device if use_fast else None,
        "autocast_device": "mps" if use_fp16 else None,
        "autocast_dtype": "float16" if use_fp16 else None,
        "weights_dtype": "float16" if use_fp16_weights else "unchanged",
    }
    return active_variant()


def describe_processor(extractor) -> dict[str, Any]:
    """Inspect the active processor/config; this loads no model weights.

    Calling this before timing excludes processor/tokenizer cache loading from
    warm-inference measurements. Keep it inside timing for cold-start trials.
    """
    processor = extractor.processor
    image_processor = processor.image_processor
    return {
        **active_variant(),
        "processor_class": type(processor).__name__,
        "image_processor_class": type(image_processor).__name__,
        "image_processor_config": image_processor.to_dict(),
    }
