#!/bin/zsh
# Source from zsh; resolve this file even when sourced by another script.
MVP_ROOT="${${(%):-%x}:A:h:h}"
export PATH="$MVP_ROOT/.venv/bin:$MVP_ROOT/.media-runtime/bin:/opt/homebrew/opt/node@22/bin:$PATH"
# Fallback only: do not override OpenCV's bundled FFmpeg or system libiconv.
export DYLD_FALLBACK_LIBRARY_PATH="$MVP_ROOT/.media-runtime/lib:/usr/local/lib:/usr/lib"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export NO_ALBUMENTATIONS_UPDATE=1
# FAISS and PyTorch ship distinct OpenMP runtimes on macOS. Keep CPU
# OpenMP single-threaded to avoid cross-runtime barrier crashes; MPS stays enabled.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$MVP_ROOT"
