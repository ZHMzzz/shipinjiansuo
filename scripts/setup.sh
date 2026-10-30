#!/bin/zsh
# Prepare this checkout on Apple Silicon. --check only checks prerequisites.
set -e
cd "${0:A:h:h}"
fail() { print -u2 -- "$1"; exit 1; }
[[ $# -eq 0 || ( $# -eq 1 && "$1" == --check ) ]] || fail '用法：./scripts/setup.sh [--check]'
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || fail '当前安装脚本仅支持 Apple Silicon macOS 的原生 arm64 终端。见 docs/首次安装与运行.md。'
xcode-select -p >/dev/null 2>&1 || fail '缺少 Xcode Command Line Tools：执行 xcode-select --install，安装完成后重试。'
[[ -x /opt/homebrew/opt/python@3.11/bin/python3.11 ]] || fail '缺少 Python 3.11：执行 brew install python@3.11 node@22 libmagic。'
[[ -x /opt/homebrew/opt/node@22/bin/node && -x /opt/homebrew/opt/node@22/bin/npm ]] || fail '缺少 Node 22：执行 brew install node@22。'
[[ -f /opt/homebrew/opt/libmagic/lib/libmagic.dylib ]] || fail '缺少 libmagic：执行 brew install libmagic。'
[[ -x tools/bin/micromamba ]] || fail '缺少 tools/bin/micromamba 或执行权限，请重新克隆完整仓库。'
/opt/homebrew/opt/python@3.11/bin/python3.11 -c 'import platform, sys; assert sys.version_info[:2] == (3, 11) and platform.machine() == "arm64", "需要原生 arm64 Python 3.11"'
print '系统前置检查通过（尚未验证 Python 依赖、模型下载或索引）。'
[[ "${1:-}" == --check ]] && exit 0

trap 'print -u2 "安装未完成，请先处理上方错误，再重新运行 ./scripts/setup.sh。排错见 docs/首次安装与运行.md。"' ZERR
print '[1/6] 安装 Python 依赖'
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos.lock.txt
.venv/bin/python scripts/fix_openmp.py
print '[2/6] 安装 FFmpeg 6 解码环境'
if [[ ! -x .media-runtime/bin/ffmpeg ]]; then
  tools/bin/micromamba create -y -p "$PWD/.media-runtime" -f media-runtime-osx-arm64.lock.txt
fi
source scripts/env.sh
print '[3/6] 检查 MPS、libmagic 和 Torchaudio 解码器'
python - <<'PY'
import torch
import magic
from torchaudio.utils import ffmpeg_utils
assert torch.backends.mps.is_available(), 'MPS 不可用，请确认使用 Apple Silicon 原生环境'
print('libmagic:', magic.from_buffer(b'video-search setup check'))
print('FFmpeg libraries:', ffmpeg_utils.get_versions())
PY
print '[4/6] 下载并预热 InsightFace、OWLv2 和 Qwen 模型'
HF_HUB_OFFLINE=0 python scripts/warm_models.py
print '[5/6] 安装 OWLv2 SDPA 独立运行时'
./scripts/setup_owl_sdpa.sh
print '[6/6] 构建 WISE 前端'
(cd wise/frontend && npm ci && npm run build)
echo '环境准备完成。执行 ./scripts/index.sh 建索引，再 ./scripts/start.sh 启动。'
