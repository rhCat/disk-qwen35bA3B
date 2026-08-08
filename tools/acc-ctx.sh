#!/usr/bin/env bash
# acc-ctx.sh -- 500 and 1500 token QA accuracy runs.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
export DS4F_GREEDY=1 DS4F_REP_PENALTY=1.3 DS4F_DEBUG7=1
for spec in "500 /tmp/q35-500-ids.txt" "1500 /tmp/q35-1500-ids.txt"; do
  set -- $spec
  tag="$1"; ids="$2"; log="/tmp/acc-$tag.log"
  bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
    --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
    --layout-trunk /tmp/q35-trunk/trunk.json \
    --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
    --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
    --tokenizer "$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json" \
    --pids-file "$ids" --gen 24 \
    --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
    > "$log" 2>&1
  echo "== $tag rc=$? =="
  python3 tools/acc-decode.py "$log" "$ids" 140
  if grep -q "QX-9911-RED" "$log"; then
    echo "   $tag: QX-9911-RED FOUND"
  else
    echo "   $tag: code not in decoded top1"
  fi
done
