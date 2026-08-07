#!/usr/bin/env bash
# py-which.sh -- report which pythons have numpy (guard-safe)
unset PYTHONPATH
for p in /opt/homebrew/bin/python3.12; do
  if [ -x "$p" ]; then
    echo "== $p"
    "$p" -c "import numpy; print('ok', numpy.__version__)" 2>&1 | tail -1
  fi
done
V=/Users/ruihe/disk-qwen35bA3B/.venv
if [ -x "$V/bin/python3" ]; then
  echo "== $V/bin/python3"
  "$V/bin/python3" -c "import numpy; print('ok', numpy.__version__)" 2>&1 | tail -1
fi
