#!/usr/bin/env bash
# bench-gpu-verify.sh -- CPU vs GPU token equality on the 5-token prompt.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
unset PYTHONPATH
export DS4F_GREEDY=1 DS4F_DEBUG7=1
PY="$REPO/.venv/bin/python3"
export HFBASE=/Users/ruihe/.cache/huggingface
run() {
  local mode="$1" log="$2"
  if [ "$mode" = gpu ]; then export DS4F_GPU=1; else unset DS4F_GPU; fi
  bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
    --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
    --layout-trunk /tmp/q35-trunk/trunk.json \
    --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
    --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
    --tokenizer "$HFBASE/mlx-qwen35-a3b-4bit/tokenizer.json" \
    --pids-file /tmp/bench-gpu-ids.txt --gen 8 \
    --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
    > "$log" 2>&1
}
run cpu /tmp/ver-cpu.log
run gpu /tmp/ver-gpu.log
echo "--- top1 decode, gen tokens ---"
for f in /tmp/ver-cpu.log /tmp/ver-gpu.log; do
  echo "$f:"
  grep -oE 'logits: t[0-9]+ .*top5.*' "$f" | tail -8 | grep -oE 't[0-9]+' | tr '\n' ' '
  echo
done
