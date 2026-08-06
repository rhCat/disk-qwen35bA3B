#!/usr/bin/env bash
# setup-mlx.sh — create the MLX serving environment inside the repo.
# Wholesome: one command from a fresh clone -> a working mlx_lm.
#
#   ./serving/setup-mlx.sh          # create .venv + install mlx-lm
#   ./serving/setup-mlx.sh --model  # also pull the 4-bit model (~20 GB)
#
# Uses a Homebrew Python 3.13 if present (system 3.9 is too old for
# current mlx-lm); falls back to python3.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# See run-mlx.sh: the agent's PYTHONPATH shadows this venv's numpy.
unset PYTHONPATH

# See run-mlx.sh: an active conda env hijacks subprocess resolution.
if [ -n "${CONDA_DEFAULT_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
  export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -v -E '/(miniconda3|miniforge3|anaconda3|mambaforge|conda)/' \
    | paste -sd: -)"
  unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER \
        CONDA_EXE CONDA_PYTHON_EXE 2>/dev/null || true
fi

PY=""
for cand in /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
            /usr/local/bin/python3.13 python3; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "REFUSE: no python3.12+ found (system 3.9 is too old for mlx-lm)"
  exit 1
fi
echo "using $PY ($($PY --version 2>&1))"

VD=".venv"
BIN_DIR="$REPO_ROOT/$VD/bin"
if [ ! -x "$BIN_DIR/python" ]; then
  "$PY" -m venv "$REPO_ROOT/$VD"
  echo "venv created at $REPO_ROOT/$VD"
fi
"$BIN_DIR/pip" install --quiet --upgrade pip
"$BIN_DIR/pip" install --quiet -r "$REPO_ROOT/serving/requirements.txt"
echo "mlx-lm installed"

if [ "${1:-}" = "--model" ]; then
  echo "pulling mlx-community/Qwen3.5-35B-A3B-4bit (~20.4 GB) ..."
  "$BIN_DIR/python" -m huggingface_hub.commands.huggingface_cli download \
    mlx-community/Qwen3.5-35B-A3B-4bit \
    --local-dir "$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit"
  echo "model cached. free disk:"; df -h / | tail -1
fi
echo "done."
