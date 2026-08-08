#!/usr/bin/env bash
# rss-gpu2.sh -- RSS via ps sampling on background runs (no binary refs).
set -u
RUN=/Users/ruihe/disk-qwen35bA3B/tools/run-bench-gpu3.sh
meas() {
  local tag="$1"; shift
  bash "$RUN" "$@" > /tmp/rss-$tag.log 2>&1 &
  local pid=$!
  local peak=0
  while kill -0 $pid 2>/dev/null; do
    local kb=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
    [ -n "$kb" ] && [ "$kb" -gt "$peak" ] && peak=$kb
    sleep 0.05
  done
  wait $pid
  echo "$tag peak RSS: $((peak / 1024)) MB"
}
meas cpu --cpu-only 8
meas gpu --gpu-only 8
echo "--- results ---"
grep -E 'CPU|batched|verify' /tmp/rss-cpu.log /tmp/rss-gpu.log 2>/dev/null | grep -vE '==|^$' | head -4
