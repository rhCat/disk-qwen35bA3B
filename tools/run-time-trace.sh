#!/usr/bin/env bash
# run-time-trace.sh -- per-token timing on the real model
export DS4F_TIME_TOKENS=1
bash /Users/ruihe/disk-qwen35bA3B/tools/run-real.sh /tmp/q35-trunk /tmp/q35-pool 2>&1 | \
  grep -E 'toktime|tokens in'
