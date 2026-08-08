#!/usr/bin/env python3
"""Split linear_step into proj + lin_body (mechanical, exact).

Extracts the serial body (a/b proj through residual, original lines
505-872) into a new static function lin_body that re-derives its own
indices/geometry from (cfg, tl, L). linear_step keeps the qkv+z
projection and calls lin_body. This is a pure refactor -- the moved
text is byte-identical, so bit-fidelity is preserved by construction.
"""
import re

SRC = "src/attn_qwen.c"
text = open(SRC, encoding="utf-8", errors="replace").read()
lines = text.split("\n")

# locate linear_step body: 1-indexed lines
# find "float a32[32], b32[32];" (body start) and the residual loop end
body_start = None
body_end = None
for i, ln in enumerate(lines):
    if "float a32[32], b32[32];" in ln and body_start is None:
        body_start = i  # 0-indexed
    if "for (int i = 0; i < H; i++) state[i] += o[i];" in ln:
        body_end = i + 1  # include this line
assert body_start is not None and body_end is not None, (body_start, body_end)

body = "\n".join(lines[body_start:body_end])

# build lin_body: re-derive indices + buf layout, then the body text
helper = f"""/* The SERIAL body of the Gated DeltaNet layer: a/b gates ->
 * conv1d -> q/k norm -> delta-rule state update -> readout ->
 * RMSNormGated -> o_proj -> residual. Everything after the qkv+z
 * projections. This part MUST run per token (conv ring + delta state
 * are serial chains); the projections that feed it are batchable
 * (linear_step_chunk, the prefill path). Indices/geometry are
 * re-derived from tl/cfg/L. buf layout (same as linear_step):
 * [xin H][qkv qkv_rows][z z_rows][o o_rows][readout v_heads*vd][qk]. */
static int lin_body(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                    const uint8_t *tr, Ds4fKvCache *kv, int token,
                    float *state, float *qkv, float *z, float *buf) {{
    int pi = tl->q3_pqkv[L], ps = tl->q3_pqkvs[L], pb = tl->q3_pqkvb[L];
    int zi = tl->q3_pz[L], zs = tl->q3_pzs[L], zb = tl->q3_pzb[L];
    int ai = tl->q3_pa[L], as_ = tl->q3_pas[L], ab = tl->q3_pab[L];
    int bi = tl->q3_pb[L], bs_ = tl->q3_pbs[L], bb = tl->q3_pbb[L];
    int ci = tl->q3_conv[L];
    int oi = tl->q3_opa[L], os_ = tl->q3_opas[L], ob = tl->q3_opab[L];
    int ni = tl->q3_lnorm[L];
    int ai_ = tl->q3_a_log[L], di = tl->q3_dt[L];
    int H = cfg->hidden;
    int qkv_rows = (int)tl->t[pi].dims[0];
    int z_rows = (int)tl->t[zi].dims[0];
    int o_rows = (int)tl->t[oi].dims[0];
    int cols = (int)tl->t[pi].dims[1] * 8;      /* decoded 2048 = H */
    int k_heads = 16, v_heads = 32, kd = 128, vd = 128;
    if (cfg->n_heads > 0) k_heads = cfg->n_heads;
    if (cfg->n_kv_heads > 0) v_heads = cfg->n_kv_heads * 16;
    float *xin = buf;
    float *o = z + z_rows;
    float *readout = o + o_rows;
    float *qk = readout + v_heads * vd;
{body}
}}"""

# replace linear_step's body with a call
call = "    return lin_body(cfg, tl, L, tr, kv, token, state, qkv, z, buf);\n"
new_lines = lines[:body_start] + [call.rstrip("\n")] + lines[body_end:]

# insert the helper BEFORE linear_step's definition
# find "static int linear_step(" index in the new text
new_text = "\n".join(new_lines)
pos = new_text.find("static int linear_step(")
assert pos >= 0
new_text = new_text[:pos] + helper + "\n\n" + new_text[pos:]

open(SRC, "w", encoding="utf-8").write(new_text)
print(f"extracted lines {body_start+1}..{body_end} ({body_end-body_start} lines) into lin_body")
print("linear_step tail replaced with the lin_body call")
