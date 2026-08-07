#!/usr/bin/env python3
import numpy as np

eng = np.fromfile('/tmp/q35-eng-state.bin', dtype=np.float32)
ref = np.fromfile('/tmp/q35-ref-state.bin', dtype=np.float32)
print('eng rms %.4g  ref rms %.4g  len %d %d' % (
    float(np.sqrt((eng**2).mean())), float(np.sqrt((ref**2).mean())),
    len(eng), len(ref)))
d = eng - ref
print('mean abs diff %.4g  max abs diff %.4g' % (
    float(np.abs(d).mean()), float(np.abs(d).max())))
# cosine similarity
c = float((eng @ ref) / (np.linalg.norm(eng) * np.linalg.norm(ref)))
print('cosine sim: %.6f' % c)
# top indices match?
top_e = np.argsort(np.abs(eng))[-10:][::-1]
top_r = np.argsort(np.abs(ref))[-10:][::-1]
print('eng top10 idx:', top_e.tolist())
print('ref top10 idx:', top_r.tolist())
print('eng top10 vals:', eng[top_e].tolist())
print('ref top10 vals:', ref[top_r].tolist())
