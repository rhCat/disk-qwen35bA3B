#!/usr/bin/env bash
# run-experiment.sh — the disk/cpu loading experiment on THIS Mac.
# Goal: ~3 GB resident, expert pool streamed from disk, measure the
# actual footprint (invariant #5: measure, don't quote the forecast).
#
#   ./run-experiment.sh          # fixture-sized run + footprint report
#   ./run-experiment.sh --real   # after convert of a real checkpoint
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

unset PYTHONPATH
if [ -n "${CONDA_DEFAULT_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
  export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -v -E '/(miniconda3|miniforge3|anaconda3|mambaforge|conda)/' \
    | paste -sd: -)"
  unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER \
        CONDA_EXE CONDA_PYTHON_EXE 2>/dev/null || true
fi

# A3B-shaped fixture: 40 layers, 256 experts, top-8, hidden 2048,
# moe_inter 512 -- the real Qwen3.5-35B-A3B geometry, synthetic bytes.
FIX="${1:-/tmp/fix35}"
OUT="${2:-/tmp/fix35-ds4f}"
echo "fixture dir: $FIX   out: $OUT"
"$REPO_ROOT/make-fixture" --dir "$FIX" --layers 40 --experts 256 --topk 8 \
  --hidden 2048 --moe-inter 512 --expert-bytes 262144 \
  --trunk-bytes 4194304 --seed 7

echo "=== convert (HF repo -> ds4f trunk + pool) ==="
python3 "$REPO_ROOT/tools/convert-ds4f.py" inspect "$FIX" 2>&1 | \
  grep -E 'routed experts|per-expert|dense per-layer|config mapping|n_layers|n_experts|topk|n_shared' | head -10
python3 "$REPO_ROOT/tools/convert-ds4f.py" convert "$FIX" --out "$OUT" 2>&1 | tail -4

echo
echo "=== run: 3 gen tokens, 2 GB expert cache, 4 pinned layers ==="
echo
/usr/bin/time -l "$REPO_ROOT/ds4f" "$OUT" \
  --trunk "$OUT/trunk.bin" --offsets "$OUT/trunk.offsets" \
  --pool "$OUT/pool.bin" --gen 3 --cache-gb 2 --pin-layers 4 \
  2>&1 | tail -25
