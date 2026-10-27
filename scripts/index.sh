#!/bin/zsh
set -e
source "${0:A:h}/env.sh"
mkdir -p projects logs
if [[ -e projects/demo/metadata/internal.db ]]; then
 python scripts/qwen_project.py
fi
wise extract-features data/videos --project-dir projects/demo \
 --video-feature-id deepinsight/insightface/buffalo_l/_unknown --num-workers 0 --yes
wise create-index --project-dir projects/demo --overwrite
