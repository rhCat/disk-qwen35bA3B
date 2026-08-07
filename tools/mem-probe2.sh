#!/usr/bin/env bash
# mem-probe2.sh -- exact run-real.sh invocation + live RSS sampling.
set -u
TRUNK=/tmp/q35-trunk
POOL=/tmp/q35-pool
REPO=/Users/ruihe/disk-qwen35bA3B
GEN="${GEN:-6}"
TOK="${TOK:-/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json}"
DS4F_GREEDY=1 "$REPO/ds4f" "$TRUNK" \
  --trunk "$TRUNK/trunk.bin" --offsets "$TRUNK/trunk.offsets" \
  --pool "$POOL/pool.bin" \
  --layout-trunk "$TRUNK/trunk.json" \
  --layout-pool "$POOL/manifest.json" \
  --head "$TRUNK/head.json" --embed "$TRUNK/embed.json" \
  --tokenizer "$TOK" --text "The capital of France is" \
  --gen "$GEN" --cache-gb 2 --pin-layers 4 \
  --mem-limit-gb 23 \
  > /tmp/memprobe2.log 2>&1 &
EPID=$!
echo "engine pid $EPID (exact run-real.sh args, gen $GEN)"
for i in $(seq 1 60); do
  kill -0 $EPID 2>/dev/null || break
  ps -o rss= -p $EPID 2>/dev/null | awk -v t="$i" '{printf "t=%02d rss=%.2f GB\n", t, $1/1048576}'
  sleep 1
done
wait $EPID
echo "exit $?"
grep -E 'PEAK RSS|tokens in' /tmp/memprobe2.log | tail -2
