#!/usr/bin/env python3
import json
p = json.load(open('/tmp/q35-pool/manifest.json'))
print('n_experts:', p.get('n_experts'), 'expert_nbytes:', p.get('expert_nbytes'))
t0 = [t for t in p['tensors'] if t['layer'] == 0 and t['expert'] == 0]
for t in t0:
    print(t['name'], t['shape'])
c = json.load(open('/tmp/q35-trunk/config.json'))
print('cfg moe_inter:', c.get('moe_inter'), 'latent:', c.get('latent'),
      'hidden:', c.get('hidden'), 'n_experts:', c.get('n_experts'))
