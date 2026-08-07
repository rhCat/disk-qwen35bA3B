#!/usr/bin/env python3
# dbg-nocache.py -- does F_NOCACHE actually prevent page-cache residency?
# Read 1.5 GB of pool.bin with and without F_NOCACHE, measure the
# system's cached-files delta via vm_stat (file-backed pages).
import os, fcntl, time, subprocess, sys

F = '/tmp/q35-pool/pool.bin'

def vmstat_cached():
    out = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        parts = line.split(':')
        if len(parts) == 2 and 'page size' not in parts[0]:
            try:
                d[parts[0].strip()] = int(parts[1].strip().replace('.', ''))
            except ValueError:
                pass
    return d.get('Pages free', 0)

def read_with(flag, label):
    fd = os.open(F, os.O_RDONLY)
    if flag:
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    before = vmstat_cached()
    total = 0
    CHUNK = 1 << 20  # 1 MB
    while total < (1536 << 20):
        data = os.pread(fd, CHUNK, total)
        if not data:
            break
        total += len(data)
        # touch every byte-ish: keep a checksum
        _ = data[0] + data[-1]
    after = vmstat_cached()
    os.close(fd)
    print('%s: read %.1f GB, pages-free delta %d (%.2f GB)' % (
        label, total / 1e9, before - after, (before - after) * 16384 / 1e9))

read_with(False, 'NO F_NOCACHE')
read_with(True, 'F_NOCACHE    ')
