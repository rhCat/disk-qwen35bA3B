#!/usr/bin/env bash
# run-real.sh -- real-scale loading experiment: engine + real trunk +
# real MLX-4bit pool on THIS Mac. Measures bytes/token + peak RSS with
# the 23 GB graceful-stop gate armed.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH
if [ -n "${CONDA_DEFAULT_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
  export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -v -E '/(miniconda3|miniforge3|anaconda3|mambaforge|conda)/' \
    | paste -sd: -)"
  unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER \
        CONDA_EXE CONDA_PYTHON_EXE 2>/dev/null || true
fi

TRUNK="${1:-/tmp/q35-trunk}"
POOL="${2:-/tmp/q35-pool}"
GEN="${GEN:-4}"
LIMIT="${LIMIT:-23}"
DUMP="${DUMP:-}"

TOK="${TOK:-/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json}"
echo "=== real-scale run: trunk $TRUNK, pool $POOL, gen $GEN, limit ${LIMIT}GB ==="
DUMPARG=()
if [ -n "$DUMP" ]; then DUMPARG=(--dump-state "$DUMP"); fi
"$REPO_ROOT/ds4f" "$TRUNK" \
  --trunk "$TRUNK/trunk.bin" --offsets "$TRUNK/trunk.offsets" \
  --pool "$POOL/pool.bin" \
  --layout-trunk "$TRUNK/trunk.json" \
  --layout-pool "$POOL/manifest.json" \
  --head "$TRUNK/head.json" --embed "$TRUNK/embed.json" \
  --tokenizer "$TOK" --text "${PROMPT:-The capital of France is}" \
  --gen "$GEN" --cache-gb 2 --pin-layers 4 \
  --mem-limit-gb "$LIMIT" ${DUMPARG[@]+"${DUMPARG[@]}"} \
  ${NOSIMD:+--no-simd} \
  > /tmp/q35-run.log 2>&1
rc=$?
echo "exit: $rc"
grep -E 'MEMORY LIMIT|config:|pool:|trunk layout|pool layout|run report|resident|peak|GB/token|bytes|read|tokens/s|WARNING|dropped|maximum|matvec|moe:|^Hello|logits' /tmp/q35-run.log | tail -22
