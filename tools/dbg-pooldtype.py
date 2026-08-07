#!/usr/bin/env python3
import json
p = json.load(open('/tmp/q35-pool/manifest.json'))
seen = {}
for t in p['tensors'][:60]:
    key = (t['name'], t.get('dtype'), t.get('fmt'))
    seen.setdefault(key, 0)
    seen[key] += 1
for k, v in sorted(seen.items(), key=str):
    print(k, v)
# expert 0 of layer 0 full record
for t in p['tensors']:
    if t['layer'] == 0 and t['expert'] == 0:
        print({kk: t.get(kk) for kk in ('name', 'dtype', 'fmt', 'shape',
                                        'v_nbytes', 's_nbytes', 'b_nbytes')})
