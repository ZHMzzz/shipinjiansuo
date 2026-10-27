"""Download the pinned official reranker into Hugging Face's shared cache."""
import os
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    print(snapshot_download('Qwen/Qwen3-VL-Reranker-2B',
        revision='4bd860ac4f15ad1897a214615cccc700f8f71818',
        allow_patterns=['*.json','*.txt','*.jinja','*.safetensors','README.md'],max_workers=2))
