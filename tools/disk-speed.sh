#!/usr/bin/env bash
# disk-speed.sh -- measure /tmp read throughput with the same 1.69MB
# x 8 pattern the fetch uses (parallel pread, offset-sorted).
set -u
F=/tmp/q35-pool/pool.bin
echo "pool size: $(stat -f%z "$F")"
python3 - <<'PYEOF'
import os, threading, time
F = "/tmp/q35-pool/pool.bin"
size = os.path.getsize(F)
nbytes = 1769472
fd = os.open(F, os.O_RDONLY)
# 8 parallel 1.69MB reads (like the fetch)
def one(off, out):
    os.pread(fd, nbytes, off)
t0 = time.time()
th = []
offs = [0, size//7, size//3, size//2, size*2//3, size*5//7, size*6//7, size- nbytes]
for o in offs:
    t = threading.Thread(target=one, args=(o, None))
    th.append(t)
for t in th: t.start()
for t in th: t.join()
dt = time.time() - t0
print(f"8 x 1.69MB parallel pread: {dt*1000:.1f} ms = {8*nbytes/dt/1e6:.0f} MB/s")
# sequential big read
t0 = time.time()
os.pread(fd, 64*1024*1024, 0)
dt = time.time() - t0
print(f"64MB sequential pread: {dt*1000:.1f} ms = {64/dt/1e6*1024*1024:.0f} MB/s")
os.close(fd)
PYEOF
