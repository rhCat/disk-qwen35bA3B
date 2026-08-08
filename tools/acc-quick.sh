#!/usr/bin/env bash
# acc-quick.sh -- quick accuracy validation: run QA fixtures, capture
# the decoded gen tokens (DEBUG7 top1), check for expected answers.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
unset PYTHONPATH
export DS4F_GREEDY=1 DS4F_REP_PENALTY=1.3 DS4F_DEBUG7=1

run() {
  local tag="$1" ids="$2" gen="$3" log="$4"
  bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
    --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
    --layout-trunk /tmp/q35-trunk/trunk.json \
    --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
    --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
    --tokenizer "$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json" \
    --pids-file "$ids" --gen "$gen" \
    --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
    > "$log" 2>&1
  local rc=$?
  echo "== $tag rc=$rc =="
}

# decode gen tokens from DEBUG7 top1 lines: "logits: t<N> ... top5 [ ... ]"
decode() {
  python3 - "$1" "$2" <<'PYEOF'
import re, sys
log, npids = sys.argv[1], int(open(sys.argv[2]).read().split(",")[0]) if False else None
ids_path = sys.argv[2]
npids = len(open(ids_path).read().split(","))
toks = []
for line in open(log, encoding="utf-8", errors="replace"):
    m = re.search(r"logits: t(\d+) .*?top5\s+\[\s*\d+ [\d.eE+-]+ (.*?)\]", line)
    if m and int(m.group(1)) >= npids:
        toks.append(m.group(2).strip())
print(" ".join(toks)[:120])
PYEOF
}

# 1) 5-token sanity: "The capital of France is" -> Paris
run "5-token" /tmp/bench-gpu-ids.txt 12 /tmp/acc-5.log
echo -n "  5-token decodes: "; decode /tmp/acc-5.log /tmp/bench-gpu-ids.txt

# 2) 500-token vault QA
run "500-token" /tmp/q35-500-ids.txt 16 /tmp/acc-500.log
echo -n "  500 decodes: "; decode /tmp/acc-500.log /tmp/q35-500-ids.txt
grep -q "QX-9911-RED" /tmp/acc-500.log && echo "  500: QX-9911-RED FOUND" || echo "  500: code not in log"

# 3) 1500-token vault QA
run "1500-token" /tmp/q35-1500-ids.txt 16 /tmp/acc-1500.log
echo -n "  1500 decodes: "; decode /tmp/acc-1500.log /tmp/q35-1500-ids.txt
grep -q "QX-9911-RED" /tmp/acc-1500.log && echo "  1500: QX-9911-RED FOUND" || echo "  1500: code not in log"
