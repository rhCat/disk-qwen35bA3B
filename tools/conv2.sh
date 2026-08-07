#!/usr/bin/env bash
# conv2.sh -- multi-turn conversation on the optimized engine. Each
# turn accumulates the full transcript and re-runs the engine (the
# stateless-server pattern). Replies are read from the engine's STDOUT
# (post-banner, non-debug lines) -- robust, and respects
# DS4F_STRIP_THINK. Usage: conv2.sh "turn1" "turn2" "turn3" ...
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
unset PYTHONPATH
PY="$REPO/.venv/bin/python3"
export DS4F_GREEDY=1 DS4F_REP_PENALTY=1.3 DS4F_STRIP_THINK=1

HFBASE=/Users/ruihe/.cache/huggingface
GEN="${DS4F_CONV_GEN:-128}"

if [ $# -eq 0 ]; then
  echo "usage: conv2.sh 'turn text' ['turn2' ...]" >&2
  exit 2
fi

transcript=""
turn_no=0
for turn in "$@"; do
  turn_no=$((turn_no + 1))
  transcript="${transcript}<|im_start|>user
${turn}<|im_end|>
<|im_start|>assistant
"
  printf '%s' "$transcript" > /tmp/conv2-prompt.txt
  "$PY" - "$HFBASE" <<'PYEOF' > /tmp/conv2-ids.txt 2>/dev/null
import os, sys
from transformers import AutoTokenizer
base = os.path.join(os.environ["HFBASE"], "mlx-qwen35-a3b-4bit")
tok = AutoTokenizer.from_pretrained(base)
ids = tok(open("/tmp/conv2-prompt.txt").read())["input_ids"]
print(",".join(str(i) for i in ids))
PYEOF
  nids=$(wc -c < /tmp/conv2-ids.txt)
  t0=$("$PY" -c 'import time; print(time.time())')
  bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
    --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
    --layout-trunk /tmp/q35-trunk/trunk.json \
    --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
    --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
    --tokenizer "$HFBASE/mlx-qwen35-a3b-4bit/tokenizer.json" \
    --pids-file /tmp/conv2-ids.txt --gen "$GEN" \
    --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
    > /tmp/conv2-turn$turn_no.log 2>&1
  rc=$?
  t1=$("$PY" -c 'import time; print(time.time())')
  dt=$("$PY" -c "print(f'{$t1-$t0:.1f}s')")
  # reply = engine stdout minus banner/debug lines
  "$PY" - "$turn_no" <<'PYEOF' > /tmp/conv2-reply.txt 2>/dev/null
import sys
log = open(f"/tmp/conv2-turn{sys.argv[1]}.log", encoding="utf-8", errors="replace").read()
out = []
for line in log.splitlines():
    s = line.strip()
    if not s: continue
    if s.startswith(("[", "logits:", "moe:", "cache:", "trunk:", "config:",
                     "pool:", "kernels:", "PEAK", "GB read", "---", "EXIT",
                     "ds4f build", "MEMORY", "router:", "hc:", "attn:",
                     "kv scratch", "usage:", "  --", "  <", "MODEL_DIR",
                     "tokenizer", "already", "Already", "Mounting",
                     "mount_smbfs", "Error while")): continue
    out.append(s)
print("\n".join(out))
PYEOF
  reply="$(cat /tmp/conv2-reply.txt)"
  echo "--- turn $turn_no ($dt, ${nids}B prompt) EXIT $rc ---"
  echo "reply: $reply"
  transcript="${transcript}${reply}
"
done
echo "== conversation done =="
