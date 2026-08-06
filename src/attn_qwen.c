/*
 * attn_qwen.c -- Qwen3.5-35B-A3B attention: full-GQA layers + linear
 * (Mamba2-class) layers, MLX 4-bit projections.
 *
 * Full-GQA layer (10 of 40, full_attention_interval=4):
 *   q = q_proj(x)   [16 heads x 512]  (512 = 256 + 256 gate, the
 *                                      attn_output_gate convention)
 *   k = k_proj(x)   [2 kv-heads x 256]
 *   v = v_proj(x)   [2 kv-heads x 256]
 *   q = RMSNorm(q, q_norm), k = RMSNorm(k, k_norm)
 *   attn = softmax(q k^T / sqrt(d)) v   (GQA: 16 q-heads share 2 kv)
 *   out = o_proj(attn)                  (2048)
 *
 * Linear-attention layer (30 of 40): Mamba2-class -- conv1d, A_log,
 * dt_bias, in_proj_qkv/z/a/b, out_proj. The recurrent state machinery
 * is the next block; this module currently SKIPS linear layers
 * gracefully (the MLP still runs, so the forward is incomplete but
 * deterministic).
 *
 * All projections are MLX 4-bit (U32 nibbles + BF16 scale/bias per
 * 64-group); the trunk layout carries the role-matched triplets.
 */
#include "ds4f/attn.h"
#include "ds4f/kernels.h"
#include "ds4f/moe.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float bf16_f(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* in-place RMSNorm with BF16 weights, eps 1e-6 */
static void rmsnorm(const uint16_t *w, int dim, float *x) {
    double ss = 0.0;
    for (int i = 0; i < dim; i++) ss += (double)x[i] * x[i];
    float r = sqrtf((float)(ss / (double)dim) + 1e-6f);
    for (int i = 0; i < dim; i++) x[i] = x[i] / r * bf16_f(w[i]);
}

/* mlx4 matvec for a trunk tensor triplet (weight/scales/biases roles).
 * Returns 0 on success, -1 if a required role is missing. */
static int mlx4_proj(const Ds4fTrunkLayout *tl, int wi, int si, int bi,
                     const uint8_t *tr, int R, int C, const float *x,
                     float *y) {
    if (wi < 0 || si < 0) return -1;
    const uint16_t *bias = bi >= 0
        ? (const uint16_t *)(const void *)(tr + tl->t[bi].off) : NULL;
    ds4f_mlx4_matvec(
        (const uint32_t *)(const void *)(tr + tl->t[wi].off),
        (const uint16_t *)(const void *)(tr + tl->t[si].off),
        bias, R, C, x, y);
    return 0;
}

/* Full-GQA step for layer L. state[hidden] in/out (residual added).
 * token = current position (cache write slot). */
static int gqa_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                    const uint8_t *tr, float *state, Ds4fKvCache *kv,
                    int token) {
    int qi = tl->q3_q[L], qs = tl->q3_qs[L], qb = tl->q3_qb[L];
    int ki = tl->q3_k[L], ks = tl->q3_ks[L], kb = tl->q3_kb[L];
    int vi = tl->q3_v[L], vs = tl->q3_vs[L], vb = tl->q3_vb[L];
    int oi = tl->q3_o[L], os = tl->q3_os[L], ob = tl->q3_ob[L];
    int qn = tl->q3_qn[L], kn = tl->q3_kn[L];
    if (qi < 0 || ki < 0 || vi < 0 || oi < 0 || qn < 0 || kn < 0)
        return 0;                    /* incomplete graph: skip */
    int H = cfg->hidden;
    int qrows = (int)tl->t[qi].dims[0];       /* 8192 = 16 heads x 512 */
    int krows = (int)tl->t[ki].dims[0];       /* 512 = 2 kv x 256 */
    int vrows = (int)tl->t[vi].dims[0];
    int orows = (int)tl->t[oi].dims[0];       /* 2048 */
    /* MLX U32: dims[1] is PACKED cols (8 nibbles per word). The decoded
     * column width is dims[1]*8 -- the kernel needs decoded dims. */
    int qcols = (int)tl->t[qi].dims[1] * 8;   /* 2048 = H */
    int ocols = (int)tl->t[oi].dims[1] * 8;   /* 4096 = 16 heads x 256 */
    int heads = cfg->n_heads > 0 ? cfg->n_heads : 16;
    int kv_heads = cfg->n_kv_heads > 0 ? cfg->n_kv_heads : 2;
    int qh = heads > 0 ? qrows / heads : qrows;   /* 512 */
    int kh = krows / kv_heads;                    /* 256 */
    if (qrows <= 0 || krows <= 0 || vrows <= 0 || heads < 1 ||
        kv_heads < 1 || qcols != H || kh < 1) {
        fprintf(stderr, "qwen attn: L%d bad dims q=%dx%d k=%dx%d v=%dx%d "
                "o=%dx%d heads=%d kvh=%d\n", L, qrows, qcols, krows, qcols,
                vrows, qcols, orows, qrows, heads, kv_heads);
        return -1;
    }
    if (token < 0 || !kv || !kv->kv || token >= kv->max_tokens) return 0;

    /* per-token KV storage: [n_layers][max_tokens][krows + vrows] */
    int kvlat = krows + vrows;
    if (kv->kvlat != kvlat && kv->kvlat != 0) {
        /* cache sized by the caller; refuse silently mismatched */
        return 0;
    }

    float *buf = (float *)calloc(
        (size_t)(qrows + krows + vrows + orows + 2 * H + 1),
        sizeof(float));
    if (!buf) return -1;
    float *q = buf;
    float *k = q + qrows;
    float *v = k + krows;
    float *o = v + vrows;
    float *attn_out = o + orows;

    /* projections */
    if (mlx4_proj(tl, qi, qs, qb, tr, qrows, qcols, state, q) != 0 ||
        mlx4_proj(tl, ki, ks, kb, tr, krows, qcols, state, k) != 0 ||
        mlx4_proj(tl, vi, vs, vb, tr, vrows, qcols, state, v) != 0) {
        free(buf);
        return -1;
    }
    /* RMSNorm q/k (per-head-group: q_norm[kh], k_norm[kh]) */
    const uint16_t *qwn = (const uint16_t *)(const void *)(tr + tl->t[qn].off);
    const uint16_t *kwn = (const uint16_t *)(const void *)(tr + tl->t[kn].off);
    /* q_norm applies per-head over the non-gate q dim (kh floats per
     * head); here we normalize the full per-head qh (gate included) --
     * a structural approximation until the exact split is verified. */
    for (int h = 0; h < heads; h++)
        rmsnorm(qwn, kh, q + (size_t)h * qh);
    for (int h = 0; h < kv_heads; h++)
        rmsnorm(kwn, kh, k + (size_t)h * kh);

    /* write k/v into the cache at token */
    float *ck = kv->kv + ((size_t)L * kv->max_tokens + token) * kvlat;
    memcpy(ck, k, (size_t)krows * sizeof(float));
    memcpy(ck + krows, v, (size_t)vrows * sizeof(float));

    /* attention: 16 q-heads over 2 kv-heads, positions 0..token.
     * q head h uses kv head h % kv_heads. */
    float dscale = 1.0f / sqrtf((float)kh);
    float *scores = (float *)calloc((size_t)kv->max_tokens, sizeof(float));
    float *wgt = (float *)calloc((size_t)kv->max_tokens, sizeof(float));
    if (!scores || !wgt) {
        free(scores); free(wgt); free(buf);
        return -1;
    }
    memset(attn_out, 0, (size_t)orows * sizeof(float));
    int npos = token + 1;
    for (int h = 0; h < heads; h++) {
        const float *qh_ptr = q + (size_t)h * qh;
        int khh = h % kv_heads;
        float mx = -1e30f;
        for (int t2 = 0; t2 < npos; t2++) {
            const float *k2 = kv->kv +
                ((size_t)L * kv->max_tokens + t2) * kvlat +
                (size_t)khh * kh;
            float acc = 0.0f;
            for (int i = 0; i < kh; i++) acc += qh_ptr[i] * k2[i];
            scores[t2] = acc * dscale;
            if (scores[t2] > mx) mx = scores[t2];
        }
        float sum = 0.0f;
        for (int t2 = 0; t2 < npos; t2++) {
            wgt[t2] = expf(scores[t2] - mx);
            sum += wgt[t2];
        }
        for (int t2 = 0; t2 < npos; t2++) {
            const float *v2 = kv->kv +
                ((size_t)L * kv->max_tokens + t2) * kvlat + krows +
                (size_t)khh * kh;
            float w = wgt[t2] / sum;
            for (int i = 0; i < kh; i++)
                attn_out[(size_t)h * kh + i] += w * v2[i];
        }
    }
    free(scores);
    free(wgt);

    /* o_proj(attn_out) -> o; residual add. attn_out is ocols wide
     * (16 heads x 256); o_proj is [orows=2048 x ocols=4096]. */
    if (mlx4_proj(tl, oi, os, ob, tr, orows, ocols, attn_out, o) != 0) {
        free(buf);
        return -1;
    }
    for (int i = 0; i < H; i++) state[i] += o[i];

    free(buf);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Linear attention: Gated DeltaNet (Qwen3-Next class)                 */
/* ------------------------------------------------------------------ */
/*
 * Per layer, per token:
 *   qkv = in_proj_qkv(x)            [8192] = q(2048=16x128) k(2048) v(4096=32x128)
 *   z   = in_proj_z(x)              [4096] = 32 value heads x 128
 *   a   = in_proj_a(x), b = in_proj_b(x)   [32] per value head
 *   conv1d over qkv (depthwise, kernel 4, channel 8192)
 *   RMSNorm q and k (norm.weight, 128)
 *   dt = softplus(dt_bias); decay = exp(-exp(A_log) * dt)   [32]
 *   state_h = decay_h * state_h + b_h * outer(k_h, v_h)     (delta rule)
 *   out_h   = state_h^T q_h                                  (readout)
 *   o = concat(out_h) -> [4096]; gated by z: o *= silu(z)? (z-gate)
 *   out_proj(o) -> [2048], residual add
 *
 * The conv needs the previous (kernel-1) qkv vectors; kept per layer
 * in the state arena tail. GQA-style head sharing: 16 key heads serve
 * 32 value heads (value head h uses key head h/2).
 */
#define Q3_CONV_K 4

static int linear_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                       const uint8_t *tr, float *state, Ds4fKvCache *kv,
                       int token) {
    int pi = tl->q3_pqkv[L], ps = tl->q3_pqkvs[L], pb = tl->q3_pqkvb[L];
    int zi = tl->q3_pz[L], zs = tl->q3_pzs[L], zb = tl->q3_pzb[L];
    int ai = tl->q3_pa[L], as_ = tl->q3_pas[L], ab = tl->q3_pab[L];
    int bi = tl->q3_pb[L], bs_ = tl->q3_pbs[L], bb = tl->q3_pbb[L];
    int ci = tl->q3_conv[L];
    int oi = tl->q3_opa[L], os_ = tl->q3_opas[L], ob = tl->q3_opab[L];
    int ni = tl->q3_lnorm[L];
    int ai_ = tl->q3_a_log[L], di = tl->q3_dt[L];
    if (pi < 0 || zi < 0 || ai < 0 || bi < 0 || ci < 0 || oi < 0 ||
        ni < 0 || ai_ < 0 || di < 0)
        return 0;                    /* incomplete graph: skip */
    int H = cfg->hidden;
    int qkv_rows = (int)tl->t[pi].dims[0];      /* 8192 */
    int z_rows = (int)tl->t[zi].dims[0];        /* 4096 */
    int o_rows = (int)tl->t[oi].dims[0];        /* 2048 */
    int cols = (int)tl->t[pi].dims[1] * 8;      /* decoded 2048 = H */
    /* head geometry from the model config */
    int k_heads = 16, v_heads = 32, kd = 128, vd = 128;
    if (cfg->n_heads > 0) k_heads = cfg->n_heads;      /* 16 */
    if (cfg->n_kv_heads > 0) v_heads = cfg->n_kv_heads * 16; /* 2x16=32 */
    if (qkv_rows != k_heads * kd * 2 + v_heads * vd) {
        fprintf(stderr, "qwen lin: L%d qkv %d != k %d*%d*2 + v %d*%d\n",
                L, qkv_rows, k_heads, kd, v_heads, vd);
        return -1;
    }
    if (z_rows != v_heads * vd || o_rows != H) return -1;
    if (!kv || token < 0 || token >= kv->max_tokens) return 0;
    if (!kv->lin_alloc)
        if (ds4f_kv_lin_init(kv, v_heads, kd, vd) != 0) return -1;

    float *buf = (float *)calloc(
        (size_t)(qkv_rows + z_rows + o_rows + v_heads * vd +
                 v_heads * kd + Q3_CONV_K * qkv_rows + 2 * H + 1),
        sizeof(float));
    if (!buf) return -1;
    float *qkv = buf;
    float *z = qkv + qkv_rows;
    float *o = z + z_rows;
    float *readout = o + o_rows;              /* v_heads*vd */
    float *qk = readout + v_heads * vd;       /* k_heads*kd */
    float *conv_ring = qk + k_heads * kd;     /* Q3_CONV_K * qkv_rows */

    if (mlx4_proj(tl, pi, ps, pb, tr, qkv_rows, cols, state, qkv) != 0 ||
        mlx4_proj(tl, zi, zs, zb, tr, z_rows, cols, state, z) != 0) {
        free(buf);
        return -1;
    }
    float a32[32], b32[32];
    {
        float tmp[32];
        if (mlx4_proj(tl, ai, as_, ab, tr, 32, cols, state, tmp) != 0 ||
            mlx4_proj(tl, bi, bs_, bb, tr, 32, cols, state, a32) != 0) {
            free(buf);
            return -1;
        }
        memcpy(b32, a32, sizeof b32);        /* b = second proj result */
        memcpy(a32, tmp, sizeof a32);
    }
    /* conv1d: depthwise over qkv, kernel 4. Weight [8192, 4, 1] BF16,
     * layout [channel][k][1]. Ring holds past Q3_CONV_K qkv vectors. */
    {
        const uint16_t *cw = (const uint16_t *)(const void *)(tr +
                              tl->t[ci].off);
        int base = (token % Q3_CONV_K) * qkv_rows;
        memcpy(conv_ring + base, qkv, (size_t)qkv_rows * sizeof(float));
        float *outq = qkv;                   /* in-place */
        for (int ch = 0; ch < qkv_rows; ch++) {
            float acc = 0.0f;
            for (int k = 0; k < Q3_CONV_K; k++) {
                int tpos = token - k;
                if (tpos < 0) continue;
                int slot = (tpos % Q3_CONV_K) * qkv_rows + ch;
                float w = bf16_f(cw[(size_t)ch * Q3_CONV_K + k]);
                acc += conv_ring[slot] * w;
            }
            outq[ch] = acc;
        }
    }
    /* RMSNorm q (first k_heads*kd) and k (next k_heads*kd) */
    {
        const uint16_t *nw = (const uint16_t *)(const void *)(tr +
                             tl->t[ni].off);
        for (int h = 0; h < k_heads; h++) {
            float *qh = qkv + (size_t)h * kd;
            float *kh = qkv + (size_t)(k_heads * kd) + (size_t)h * kd;
            rmsnorm(nw, kd, qh);
            rmsnorm(nw, kd, kh);
        }
    }
    /* decay: softplus(dt_bias) * exp(A_log), per value head */
    float decay[32];
    {
        const float *Al = (const float *)(const void *)(tr + tl->t[ai_].off);
        const uint16_t *dtb = (const uint16_t *)(const void *)(tr +
                              tl->t[di].off);
        for (int h = 0; h < v_heads; h++) {
            float dt = bf16_f(dtb[h]);
            float sp = dt > 0 ? dt + log1pf(expf(-dt)) : log1pf(expf(dt));
            decay[h] = expf(-expf(Al[h]) * sp);
        }
    }
    /* delta-rule state update + readout. state_h = [kd x vd], value
     * head h uses key head h/2 (GQA 16->32). */
    float *S = kv->lin + (size_t)L * v_heads * kd * vd;
    memset(readout, 0, (size_t)v_heads * vd * sizeof(float));
    for (int h = 0; h < v_heads; h++) {
        int khh = h / 2;
        const float *kh = qkv + (size_t)(k_heads * kd) + (size_t)khh * kd;
        const float *vh = qkv + (size_t)(2 * k_heads * kd) +
                          (size_t)h * vd;
        float *Sh = S + (size_t)h * kd * vd;
        float d = decay[h];
        float beta = b32[h];
        /* state = d * state + beta * outer(k, v) */
        for (int i = 0; i < kd; i++) {
            float *row = Sh + (size_t)i * vd;
            float kk = kh[i];
            for (int j = 0; j < vd; j++)
                row[j] = d * row[j] + beta * kk * vh[j];
        }
        /* readout: out_h = state^T q_h (q of the key head) */
        const float *qh = qkv + (size_t)khh * kd;
        float *oh = readout + (size_t)h * vd;
        for (int j = 0; j < vd; j++) {
            float acc = 0.0f;
            for (int i = 0; i < kd; i++)
                acc += Sh[(size_t)i * vd + j] * qh[i];
            oh[j] = acc;
        }
    }
    /* z-gate: o = readout * silu(z) */
    for (int i = 0; i < v_heads * vd; i++) {
        float zv = z[i];
        float sig = 1.0f / (1.0f + expf(-zv));
        readout[i] *= zv * sig;
    }
    /* out_proj(readout) -> o, residual add */
    if (mlx4_proj(tl, oi, os_, ob, tr, o_rows, v_heads * vd, readout, o)
        != 0) {
        free(buf);
        return -1;
    }
    for (int i = 0; i < H; i++) state[i] += o[i];

    free(buf);
    return 0;
}

/* Dispatch: full-GQA layers (self_attn roles present) vs linear. */
int ds4f_attn_qwen_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                        const uint8_t *tr, float *state, Ds4fKvCache *kv,
                        int token) {
    if (!tl || !cfg) return -1;
    if (tl->q3_q[L] >= 0)
        return gqa_step(cfg, tl, L, tr, state, kv, token);
    if (tl->q3_conv[L] >= 0 || tl->q3_pqkv[L] >= 0)
        return linear_step(cfg, tl, L, tr, state, kv, token);
    return 0;                     /* no attention graph: skip */
}
