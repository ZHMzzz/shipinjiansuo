#!/bin/zsh
set -e
source "${0:A:h}/env.sh"
mkdir -p projects logs
if [[ -e projects/demo/metadata/internal.db ]]; then
 python scripts/qwen_project.py
 python -c 'from pathlib import Path; from scripts.restore_owl_project import build, publish; r=Path.cwd(); publish(r,build(r))'
fi
wise extract-features data/videos --project-dir projects/demo \
 --video-feature-id transformers/owlv2/google/owlv2-base-patch16-ensemble \
 --video-feature-id deepinsight/insightface/buffalo_l/_unknown --num-workers 0 --yes
wise create-index --project-dir projects/demo --overwrite
