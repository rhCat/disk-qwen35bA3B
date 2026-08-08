#!/usr/bin/env bash
# run-bench-gpu2.sh -- capture the full batch-kernel compile error.
set -u
B=/Users/ruihe/disk-qwen35bA3B/bench-gpu-experts
"$B" 5 2>&1 | grep -A12 'compile failed' | head -16
