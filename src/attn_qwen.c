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

/* Linear-attention step: Mamba2-class. Currently a graceful skip --
 * the recurrent state machinery (conv1d + A_log decay + selective
 * scan) is the next block. */
static int linear_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                       const uint8_t *tr, float *state, Ds4fKvCache *kv,
                       int token) {
    (void)cfg; (void)tl; (void)L; (void)tr; (void)state; (void)kv; (void)token;
    return 0;                     /* skip: MLP still runs this layer */
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
