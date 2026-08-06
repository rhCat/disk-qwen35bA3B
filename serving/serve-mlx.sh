#!/usr/bin/env bash
# serve-mlx.sh — Qwen3.5-35B-A3B via mlx_lm on Apple Silicon.
# Usage: ./serving/serve-mlx.sh {up|down|log|status|test}
# Run 'up' inside tmux (mlx_lm blocks by design); watch with 'log'.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VD=".venv"
BIN_DIR="$REPO_ROOT/$VD/bin"

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

MODEL="${MLX_MODEL:-$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit}"
LOG="$HOME/mlx-qwen35.log"
CMD="${1:-status}"

case "$CMD" in
  up)
    [ -x "$BIN_DIR/mlx_lm" ] || { echo "run setup-mlx.sh first"; exit 1; }
    [ -d "$MODEL" ] || { echo "model missing at $MODEL (setup-mlx.sh --model)"; exit 1; }
    echo "serving $MODEL ..."
    "$BIN_DIR/mlx_lm.server" --model "$MODEL" --port 8089 > "$LOG" 2>&1 &
    echo "started (pid $!) — watch: $0 log";;
  down)
    pkill -f 'mlx_lm.server' 2>/dev/null || true
    echo "stopped";;
  log)
    tail -f "$LOG";;
  status)
    if pgrep -f 'mlx_lm.server' >/dev/null; then
      echo "server: running"
      curl -s -m 3 "http://127.0.0.1:8089/health" 2>/dev/null || echo "health: not answering yet"
    else
      echo "server: not running"
    fi;;
  test)
    curl -s -m 300 "http://127.0.0.1:8089/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"mlx-community/Qwen3.5-35B-A3B-4bit","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":64}';;
  *) echo "usage: $0 {up|down|log|status|test}"; exit 1;;
esac
