#!/usr/bin/env python3
import json
m = json.load(open('/tmp/q35-pool/manifest.json'))
t0 = m['tensors'][0]
print('type:', type(t0))
print('keys:', sorted(t0[0].keys()) if isinstance(t0, list) else sorted(t0.keys()))
print('sample:', t0[0] if isinstance(t0, list) else t0)
