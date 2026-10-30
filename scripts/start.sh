#!/bin/zsh
set -e
source "${0:A:h}/env.sh"
if [[ ! -x .venv/bin/python || ! -f wise/frontend/dist/index.html ]]; then
  print -u2 '环境未准备完成，请先执行 ./scripts/setup.sh；见 docs/首次安装与运行.md。'
  exit 1
fi
if [[ ! -f projects/demo/metadata/internal.db ]]; then
  print -u2 '尚未建立初始视频索引，请先执行 ./scripts/index.sh，再启动服务。'
  exit 1
fi
# All weights are cached by the deployment step. Avoid network HEAD requests
# during demos; override with HF_HUB_OFFLINE=0 if downloading new models.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
python scripts/manage.py start
