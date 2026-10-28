"""Same-weight Qwen3-VL patch projection, scoped to serialized video encoding."""
from contextlib import contextmanager
from types import MethodType

VIDEO_PATCH_PROJECTION = 'linear_same_weights_v1'


@contextmanager
def linear_video_projection(extractor):
    """Restore the original forward even when video inference fails.

    The caller must serialize access to this extractor (server search_gate).
    This implementation is validated for the current 2B model geometry only.
    It preserves parameters/state_dict and does not modify the downloaded model.
    """
    import torch
    import torch.nn.functional as F

    patch = extractor._model.model.visual.patch_embed
    conv = patch.proj
    if not (isinstance(conv, torch.nn.Conv3d)
            and conv.kernel_size == conv.stride == (2, 16, 16)
            and conv.padding == (0, 0, 0) and conv.dilation == (1, 1, 1)
            and conv.groups == 1 and conv.in_channels == patch.in_channels == 3
            and conv.out_channels == patch.embed_dim == 1024
            and patch.temporal_patch_size == 2 and patch.patch_size == 16):
        raise RuntimeError('Qwen 图像块投影结构已变化，需重新验证线性投影适配')

    def forward(self, hidden_states):
        if hidden_states.ndim != 2 or hidden_states.shape[1] != 1536:
            raise RuntimeError('Qwen 图像块输入形状不符合线性投影要求')
        return F.linear(hidden_states.to(dtype=self.proj.weight.dtype),
                        self.proj.weight.reshape(1024, 1536), self.proj.bias)

    had_override = 'forward' in patch.__dict__
    original = patch.__dict__.get('forward')
    patch.forward = MethodType(forward, patch)
    try:
        yield
    finally:
        if had_override:
            patch.forward = original
        else:
            del patch.forward
