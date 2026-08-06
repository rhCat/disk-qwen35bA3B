#!/usr/bin/env bash
# dbg-suite-text.sh -- replicate the suite's e2e_text sequence verbatim
# but WITHOUT the 2>/dev/null so the real failure is visible.
cd /Users/ruihe/disk-qwen35bA3B
SYN=$(mktemp -d)
trap 'rm -rf "$SYN"' EXIT
python3 tools/convert-ds4f.py make-synthetic "$SYN/src" >/dev/null 2>&1
mkdir -p "$SYN/wrap"
mv "$SYN/src" "$SYN/wrap/model"
python3 tools/convert-ds4f.py inspect "$SYN/wrap" >/dev/null 2>&1
python3 tools/convert-ds4f.py convert "$SYN/wrap" --out "$SYN/out" >/dev/null 2>&1
python3 tools/convert-ds4f.py quantize "$SYN/wrap" --out "$SYN/q" >/dev/null 2>&1
echo "=== run A ==="
./ds4f "$SYN/q" --trunk "$SYN/out/trunk.bin" \
  --offsets "$SYN/out/trunk.offsets" --pool "$SYN/q/pool-mxfp4.bin" \
  --layout-trunk "$SYN/out/trunk.json" \
  --layout-pool "$SYN/q/pool-mxfp4.json" \
  --head "$SYN/out/head.json" --embed "$SYN/out/embed.json" \
  --prompt-ids 7 --gen 5 --cache-gb 1 \
  --dump-state "$SYN/tdump1.bin" 2>&1 | tail -12
echo "rc=$?"
