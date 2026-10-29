"""Exercise cache correctness against real Transformers position functions on CPU."""
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from qwen_position_cache import PositionCache


def model():
    return Qwen3VLVisionModel(Qwen3VLVisionConfig(depth=1,hidden_size=32,
        intermediate_size=64,num_heads=2,out_hidden_size=32,
        num_position_embeddings=256,deepstack_visual_indexes=[0])).eval()


def test_original_outputs_and_shape_specific_keys():
    visual=model();grids=[torch.tensor([g]) for g in ((2,8,12),(2,12,8),(1,8,12))]
    names=('fast_pos_embed_interpolate','rot_pos_emb')
    with torch.inference_mode():
        expected={(i,n):getattr(visual,n)(g) for i,g in enumerate(grids) for n in names}
        cache=PositionCache(visual)
        for _ in range(2):
            for i,g in enumerate(grids):
                for n in names:torch.testing.assert_close(getattr(visual,n)(g),expected[i,n],rtol=0,atol=0)
        assert cache.hits==6 and cache.misses==6 and len(cache.grids)==3
        cache.close(); assert cache.bytes==0 and not cache.entries


def test_weight_version_invalidates_cache_and_training_bypasses_it():
    visual=model();grid=torch.tensor([[2,8,12]]);original=visual.fast_pos_embed_interpolate
    cache=PositionCache(visual)
    with torch.inference_mode():
        old=visual.fast_pos_embed_interpolate(grid).clone()
        visual.pos_embed.weight.add_(1)
        new=visual.fast_pos_embed_interpolate(grid)
        assert cache.misses==2 and not torch.equal(old,new)
        torch.testing.assert_close(new,original(grid),rtol=0,atol=0)
        visual.train();visual.fast_pos_embed_interpolate(grid)
        assert cache.misses==2
    visual.eval()
    # Grad-enabled inference must not reuse tensors produced under inference_mode.
    assert visual.fast_pos_embed_interpolate(grid).requires_grad
    assert cache.misses==2
    cache.close()


def test_bounded_eviction_and_oversized_bypass():
    visual=model();grid=torch.tensor([[2,8,12]])
    cache=PositionCache(visual,limit=25000)
    with torch.inference_mode():
        visual.fast_pos_embed_interpolate(grid)
        visual.fast_pos_embed_interpolate(torch.tensor([[2,12,8]]))
        assert cache.evictions==1 and cache.bytes<=cache.limit
    cache.close()
    cache=PositionCache(visual,limit=1)
    with torch.inference_mode():visual.rot_pos_emb(grid)
    assert not cache.entries and cache.bytes==0 and cache.misses==1
    cache.close()
