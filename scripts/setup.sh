#!/bin/zsh
# Recreate the Python environment in this project (Apple Silicon, Python 3.11).
# Existing tools/bin/micromamba and source checkout are retained.
set -e
cd "${0:A:h:h}"
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos.lock.txt
.venv/bin/python scripts/fix_openmp.py
if [[ ! -x .media-runtime/bin/ffmpeg ]]; then
  tools/bin/micromamba create -y -p "$PWD/.media-runtime" -f media-runtime-osx-arm64.lock.txt
fi
source scripts/env.sh
HF_HUB_OFFLINE=0 python scripts/warm_models.py
(cd wise/frontend && npm ci && npm run build)
echo '环境准备完成。执行 ./scripts/index.sh 建索引，再 ./scripts/start.sh 启动。'
