#!/bin/zsh
set -e
source "${0:A:h}/env.sh"
mkdir -p projects logs
wise extract-features data/videos --project-dir projects/demo \
 --video-feature-id mlfoundations/open_clip/ViT-B-16-SigLIP2-256/webli \
 --video-feature-id transformers/owlv2/google/owlv2-base-patch16-ensemble \
 --video-feature-id deepinsight/insightface/buffalo_l/_unknown --num-workers 0 --yes
wise create-index --project-dir projects/demo --overwrite
