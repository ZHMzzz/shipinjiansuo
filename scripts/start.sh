#!/bin/zsh
set -e
source "${0:A:h}/env.sh"
# All weights are cached by the deployment step. Avoid network HEAD requests
# during demos; override with HF_HUB_OFFLINE=0 if downloading new models.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
python scripts/manage.py start
