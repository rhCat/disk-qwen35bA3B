#!/usr/bin/env bash
# probe-refrss.sh -- run dbg-reffwd.py in background, sample RSS mid-run.
set -u
unset PYTHONPATH
REPO=/Users/ruihe/disk-qwen35bA3B
"$REPO/.venv/bin/python3" "$REPO/tools/dbg-reffwd.py" > /tmp/reffwd-nc.log 2>&1 &
PID=$!
sleep 25
ps -o rss= -p $PID 2>/dev/null | awk '{printf "ref RSS at 25s: %.2f GB\n", $1/1048576}'
wait $PID
echo "EXIT $?"
grep -E 'ref state rms|t8 L36' /tmp/reffwd-nc.log | tail -2
