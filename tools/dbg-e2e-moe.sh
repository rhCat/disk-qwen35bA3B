#!/usr/bin/env bash
# dbg-e2e-moe.sh -- reproduce the e2e_moe segfault and capture the log.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
export PATH="/Volumes/prod/miniforge3/envs/ca_lpp/bin:$PATH"
SYN=$(mktemp -d)
python3 tools/convert-ds4f.py make-synthetic "$SYN/src" >/dev/null 2>&1
mkdir -p "$SYN/wrap"
mv "$SYN/src" "$SYN/wrap/model"
python3 tools/convert-ds4f.py inspect "$SYN/wrap" >/dev/null 2>&1
python3 tools/convert-ds4f.py convert "$SYN/wrap" --out "$SYN/out" >/dev/null 2>&1
python3 tools/convert-ds4f.py quantize "$SYN/wrap" --out "$SYN/q" >/dev/null 2>&1
bash tools/run-clean.sh ./ds4f "$SYN/q" \
  --trunk "$SYN/out/trunk.bin" --offsets "$SYN/out/trunk.offsets" \
  --pool "$SYN/q/pool-mxfp4.bin" --layout-trunk "$SYN/out/trunk.json" \
  --layout-pool "$SYN/q/pool-mxfp4.json" \
  --gen 3 --cache-gb 1 > "$SYN/moe1.log" 2>&1
echo "rc=$?"
echo "--- last 6 lines ---"
tail -6 "$SYN/moe1.log"
echo "--- q3/lin mentions ---"
grep -iE 'lin|q3' "$SYN/moe1.log" | head -4
echo "SYN=$SYN"
