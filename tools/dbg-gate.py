#!/usr/bin/env python3
# Check the actual repo: (1) gate weight bits/format, (2) shared expert
# tensors, (3) shared_expert_intermediate_size in config.
import json, os
base = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit'
wm = json.load(open(base + '/model.safetensors.index.json'))['weight_map']
cfg = json.load(open(base + '/config.json'))
tc = cfg.get('text_config', cfg)
print('shared_expert_intermediate_size:', tc.get('shared_expert_intermediate_size'))
print('mlp_only_layers:', tc.get('mlp_only_layers'))
print('num_experts:', tc.get('num_experts'), 'per_tok:', tc.get('num_experts_per_tok'))

# find gate + shared expert + conv tensors in the weight map
gate_k = [k for k in wm if k.endswith('mlp.gate.weight')]
se_k = [k for k in wm if 'shared_expert' in k]
seg_k = [k for k in wm if 'shared_expert_gate' in k]
print('gate tensors:', len(gate_k), gate_k[:1])
print('shared_expert tensors:', len(se_k), se_k[:2])
print('shared_expert_gate:', len(seg_k), seg_k[:1])

# gate weight shape from the shard header
import struct
shard = wm[gate_k[0]]
p = os.path.join(base, shard)
with open(p, 'rb') as f:
    raw = f.read(8)
    hlen = struct.unpack('<Q', raw)[0]
    hdr = json.loads(f.read(hlen))
info = hdr[gate_k[0]]
print('gate weight dtype/shape:', info['dtype'], info['shape'])
gk = [k for k in wm if k.endswith('mlp.gate.scales')]
info2 = hdr[wm[gk[0]] and gk[0]]
print('gate scales dtype/shape:', info2['dtype'], info2['shape'])
