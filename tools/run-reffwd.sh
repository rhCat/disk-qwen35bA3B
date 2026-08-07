#!/usr/bin/env bash
# run-reffwd.sh -- run the mlx-lm faithful Python forward (clean env)
unset PYTHONPATH
if [ -n "${CONDA_DEFAULT_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
  export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -v -E '/(miniconda3|miniforge3|anaconda3|mambaforge|conda)/' \
    | paste -sd: -)"
  unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER \
        CONDA_EXE CONDA_PYTHON_EXE 2>/dev/null || true
fi
exec /Users/ruihe/disk-qwen35bA3B/.venv/bin/python3 \
  /Users/ruihe/disk-qwen35bA3B/tools/dbg-reffwd.py
