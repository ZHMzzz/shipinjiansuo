#!/bin/zsh
# Separate inference runtime: never upgrade the decoder or query environment.
set -e
source "${0:A:h}/env.sh"
owl_runtime="$MVP_ROOT/data/model-cache/owl-sdpa/runtime"
python -m venv "$owl_runtime"
"$owl_runtime/bin/python" -m pip install -r requirements-owl-sdpa.lock.txt
"$owl_runtime/bin/python" scripts/owl_sdpa.py --check
echo 'OWLv2 SDPA 独立运行时已就绪。'
