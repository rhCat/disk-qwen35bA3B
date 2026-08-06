#!/usr/bin/env bash
# run-mlx.sh — verify the MLX env and/or generate with Qwen3.5-35B-A3B-4bit.
# Usage:
#   ./serving/run-mlx.sh --pull      pull the model (~20.4 GB) then exit
#   ./serving/run-mlx.sh --verify    import check + metal check
#   ./serving/run-mlx.sh "prompt"    one-shot generation (default 64 tokens)
#   ./serving/run-mlx.sh --server    serve on :8089 (blocks; use serve-mlx.sh up)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VD=".venv"
BIN_DIR="$REPO_ROOT/$VD/bin"
PY="$BIN_DIR/python"

MODEL="${MLX_MODEL:-$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit}"
[ -x "$PY" ] || { echo "run setup-mlx.sh first"; exit 1; }

# The Hermes agent exports PYTHONPATH pointing at its own venv's
# site-packages; every child process inherits it and it shadows this
# venv's numpy/mlx (wrong python-version ABI -> import failure).
unset PYTHONPATH

# An active conda env (CONDA_DEFAULT_ENV, e.g. KHorizon) puts conda's
# bin dirs first on PATH and its python/conda shims hijack subprocess
# resolution (git/xet/conda calls from huggingface_hub). Leave conda
# entirely: strip its dirs from PATH and drop its env vars.
if [ -n "${CONDA_DEFAULT_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
  export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -v -E '/(miniconda3|miniforge3|anaconda3|mambaforge|conda)/' \
    | paste -sd: -)"
  unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER \
        CONDA_EXE CONDA_PYTHON_EXE 2>/dev/null || true
fi

case "${1:-}" in
  --diag)
    echo "PYTHONPATH inherited: ${PYTHONPATH:-<unset>}"
    "$PY" - <<'EOF'
import sys
print("--- sys.path entries mentioning hermes/venv ---")
for p in sys.path:
    if "hermes" in p or "venv" in p:
        print(" ", p)
import platform
print("machine:", platform.machine())
try:
    import numpy
    print("numpy:", numpy.__version__, numpy.__file__)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("numpy import FAIL:", e)
EOF
    "$BIN_DIR/pip" list 2>/dev/null | grep -iE 'numpy|mlx' ;;
  --fix-venv)
    "$BIN_DIR/pip" install --force-reinstall --no-cache-dir \
      --only-binary :all: --quiet numpy 2>&1 | tail -2
    "$BIN_DIR/pip" install --quiet -r "$REPO_ROOT/serving/requirements.txt" 2>&1 | tail -1
    "$PY" -c "import numpy, httpx, tqdm; print('numpy', numpy.__version__, '| httpx', httpx.__version__, '| tqdm OK')" ;;
  --pull)
    "$PY" - "$MODEL" <<'EOF'
import os, sys
from huggingface_hub import snapshot_download
dest = sys.argv[1]
print("pulling mlx-community/Qwen3.5-35B-A3B-4bit ->", dest)
p = snapshot_download(
    repo_id="mlx-community/Qwen3.5-35B-A3B-4bit",
    local_dir=dest,
)
print("done:", p)
EOF
    df -h / | tail -1 ;;
  --verify)
    "$PY" - <<'EOF'
import mlx.core as mx
import mlx_lm
print("mlx_lm import OK")
print("metal:", mx.metal.is_available())
print("device:", mx.default_device())
EOF
    ;;
  --server)
    exec "$BIN_DIR/mlx_lm.server" --model "$MODEL" --port 8089 ;;
  *)
    PROMPT="${1:-Say hello in one short sentence.}"
    [ -d "$MODEL" ] || { echo "model missing — run: ./serving/run-mlx.sh --pull"; exit 1; }
    "$PY" - "$MODEL" "$PROMPT" <<'EOF'
import sys, time
from mlx_lm import load, generate
model, tokenizer = load(sys.argv[1])
t0 = time.time()
out = generate(model, tokenizer, prompt=sys.argv[2], max_tokens=64)
dt = time.time() - t0
print("\n--- output ---\n" + out)
print(f"\n--- {dt:.1f}s wall, ~{64/dt:.1f} tok/s (incl. load) ---")
EOF
    ;;
esac
