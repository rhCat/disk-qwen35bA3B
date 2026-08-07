#!/usr/bin/env python3
import json
t = json.load(open('/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json'))
print('keys:', list(t.keys()))
at = t.get('added_tokens', [])
print('added_tokens count:', len(at))
for x in at[:10]:
    print(' ', x['id'], repr(x['content']))
