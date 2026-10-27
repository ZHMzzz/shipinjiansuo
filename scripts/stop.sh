#!/bin/zsh
set -e
source "${0:A:h}/env.sh"
python scripts/manage.py stop
