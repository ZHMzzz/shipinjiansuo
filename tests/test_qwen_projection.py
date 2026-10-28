from types import SimpleNamespace, MethodType

import pytest
import torch

from scripts.qwen_projection import linear_video_projection


class Patch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels, self.embed_dim = 3, 1024
        self.temporal_patch_size, self.patch_size = 2, 16
        self.proj = torch.nn.Conv3d(3, 1024, (2, 16, 16), stride=(2, 16, 16)).double()

    def forward(self, x):
        return self.proj(x.reshape(-1, 3, 2, 16, 16)).reshape(-1, 1024)


def extractor(patch):
    return SimpleNamespace(_model=SimpleNamespace(model=SimpleNamespace(
        visual=SimpleNamespace(patch_embed=patch))))


def test_same_affine_map_preserves_parameters_and_restores_forward():
    patch = Patch()
    x = torch.randn(3, 1536, dtype=torch.float64)
    expected = patch(x)
    weights = {k: v.clone() for k, v in patch.state_dict().items()}
    parameters = list(patch.parameters())
    with linear_video_projection(extractor(patch)):
        torch.testing.assert_close(patch(x), expected, rtol=1e-12, atol=1e-12)
    assert 'forward' not in patch.__dict__
    assert all(a is b for a, b in zip(parameters, patch.parameters()))
    for key, value in patch.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_failure_restores_existing_instance_override():
    patch = Patch()
    override = MethodType(lambda self, x: x, patch)
    patch.forward = override
    with pytest.raises(ValueError, match='inference failed'):
        with linear_video_projection(extractor(patch)):
            raise ValueError('inference failed')
    assert patch.forward is override


@pytest.mark.parametrize('field,value', [('stride', (1, 16, 16)),
                                        ('padding', (1, 0, 0)), ('groups', 3)])
def test_changed_convolution_geometry_rejected_before_override(field, value):
    patch = Patch()
    setattr(patch.proj, field, value)
    with pytest.raises(RuntimeError, match='结构已变化'):
        with linear_video_projection(extractor(patch)):
            pytest.fail('Unsupported projection was accepted')
    assert 'forward' not in patch.__dict__


def test_invalid_input_restores_forward():
    patch = Patch()
    with pytest.raises(RuntimeError, match='输入形状'):
        with linear_video_projection(extractor(patch)):
            patch(torch.zeros(2, 768))
    assert 'forward' not in patch.__dict__
