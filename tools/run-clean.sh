#!/usr/bin/env bash
# run-clean.sh -- run a command with a clean python (no conda, no
# PYTHONPATH shadowing). Usage: bash run-clean.sh <cmd...>
unset PYTHONPATH
if [ -n "${CONDA_DEFAULT_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
  export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -v -E '/(miniconda3|miniforge3|anaconda3|mambaforge|conda)/' \
    | paste -sd: -)"
  unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER \
        CONDA_EXE CONDA_PYTHON_EXE 2>/dev/null || true
fi
exec "$@"
