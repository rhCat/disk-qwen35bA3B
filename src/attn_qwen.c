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
 * token = current position (cache write slot).
 *
 * Reference: modeling_qwen3_5.py Qwen3_5Attention.
 *   q, gate = chunk(q_proj(x) [16 x 512], 2)   -> q[16x256], gate[16x256]
 *   q = q_norm(q); k = k_norm(k_proj(x)); v = v_proj(x)
 *   q, k = RoPE(q, k)          (partial_rotary 0.25 -> 64 dims, mrope)
 *   attn = softmax(q k^T / sqrt(256)) v         (GQA, repeat_kv 8)
 *   out = o_proj(attn * sigmoid(gate))
 */
static int gqa_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                    const uint8_t *tr, float *state, Ds4fKvCache *kv,
                    int token) {
    int qi = tl->q3_q[L], qs = tl->q3_qs[L], qb = tl->q3_qb[L];
    int ki = tl->q3_k[L], ks = tl->q3_ks[L], kb = tl->q3_kb[L];
    int vi = tl->q3_v[L], vs = tl->q3_vs[L], vb = tl->q3_vb[L];
    int oi = tl->q3_o[L], os = tl->q3_os[L], ob = tl->q3_ob[L];
    int qn = tl->q3_qn[L], kn = tl->q3_kn[L];
    int iln = tl->attn_norm[L];
    if (qi < 0 || ki < 0 || vi < 0 || oi < 0 || qn < 0 || kn < 0)
        return 0;                    /* incomplete graph: skip */
    int H = cfg->hidden;
    int qrows = (int)tl->t[qi].dims[0];       /* 8192 = 16 heads x 512 */
    int krows = (int)tl->t[ki].dims[0];       /* 512 = 2 kv x 256 */
    int vrows = (int)tl->t[vi].dims[0];
    int orows = (int)tl->t[oi].dims[0];       /* 2048 */
    /* MLX U32: dims[1] is PACKED cols (8 nibbles per word). */
    int qcols = (int)tl->t[qi].dims[1] * 8;   /* 2048 = H */
    int ocols = (int)tl->t[oi].dims[1] * 8;   /* 4096 = 16 heads x 256 */
    int heads = cfg->n_heads > 0 ? cfg->n_heads : 16;
    int kv_heads = cfg->n_kv_heads > 0 ? cfg->n_kv_heads : 2;
    int qh = heads > 0 ? qrows / heads : qrows;   /* 512 = q(256)+gate(256) */
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
    if (kv->kvlat != kvlat && kv->kvlat != 0) return 0;

    float *buf = (float *)calloc(
        (size_t)(H + qrows + krows + vrows + orows + 2 * H + 1),
        sizeof(float));
    if (!buf) return -1;
    float *xin = buf;               /* input_layernorm(state) */
    float *q = xin + H;             /* qrows: q(16x256) + gate(16x256) */
    float *k = q + qrows;
    float *v = k + krows;
    float *o = v + vrows;
    float *attn_out = o + orows;

    /* reference: x = x + attn(input_layernorm(x)) -- the projections
     * read the NORMED input, the residual adds to the raw state. */
    if (iln >= 0) {
        const uint16_t *nw = (const uint16_t *)(const void *)(tr +
                             tl->t[iln].off);
        memcpy(xin, state, (size_t)H * sizeof(float));
        double ss = 0.0;
        for (int i = 0; i < H; i++) ss += (double)xin[i] * xin[i];
        float r = sqrtf((float)(ss / (double)H) + 1e-6f);
        for (int i = 0; i < H; i++) {
            uint32_t bits = (uint32_t)nw[i] << 16;
            float w;
            memcpy(&w, &bits, 4);
            xin[i] = xin[i] / r * w;
        }
    } else {
        memcpy(xin, state, (size_t)H * sizeof(float));
    }

    /* projections */
    if (mlx4_proj(tl, qi, qs, qb, tr, qrows, qcols, xin, q) != 0 ||
        mlx4_proj(tl, ki, ks, kb, tr, krows, qcols, xin, k) != 0 ||
        mlx4_proj(tl, vi, vs, vb, tr, vrows, qcols, xin, v) != 0) {
        free(buf);
        return -1;
    }
    /* split q | gate: q is the first 256 of each 512-head, gate the
     * rest. q_norm over the full 256-dim head. */
    const uint16_t *qwn = (const uint16_t *)(const void *)(tr + tl->t[qn].off);
    const uint16_t *kwn = (const uint16_t *)(const void *)(tr + tl->t[kn].off);
    float *gate = (float *)calloc((size_t)heads * kh, sizeof(float));
    float *qq = (float *)calloc((size_t)heads * kh, sizeof(float));
    float *kk = (float *)calloc((size_t)heads * kh, sizeof(float));
    if (!gate || !qq || !kk) {
        free(gate); free(qq); free(kk); free(buf);
        return -1;
    }
    for (int h = 0; h < heads; h++) {
        float *hq = q + (size_t)h * qh;
        memcpy(qq + (size_t)h * kh, hq, (size_t)kh * sizeof(float));
        memcpy(gate + (size_t)h * kh, hq + kh, (size_t)kh * sizeof(float));
    }
    /* q_norm / k_norm: RMSNorm per head over the full 256 dims */
    for (int h = 0; h < heads; h++)
        rmsnorm(qwn, kh, qq + (size_t)h * kh);
    for (int h = 0; h < kv_heads; h++)
        rmsnorm(kwn, kh, k + (size_t)h * kh);

    /* RoPE on the first 64 dims (partial_rotary 0.25), interleaved.
     * Plain text decode: position = token (mrope section T only). */
    {
        int rd = (int)(kh * 0.25f);              /* 64 */
        double theta = 10000000.0;
        float *cos_t = (float *)calloc((size_t)rd, sizeof(float));
        float *sin_t = (float *)calloc((size_t)rd, sizeof(float));
        if (!cos_t || !sin_t) {
            free(cos_t); free(sin_t); free(gate); free(qq); free(kk);
            free(buf);
            return -1;
        }
        /* mrope_interleaved: dims 0..rd-1 get rope pairs (d, d+1) --
         * interleaved layout means pair (2i, 2i+1). inv_freq over
         * arange(0, rd, 2)/rd (the transformers rope init). */
        for (int i = 0; i < rd / 2; i++) {
            double inv = 1.0 / pow(theta, (double)(2 * i) / (double)rd);
            double f = (double)token * inv;
            cos_t[i] = (float)cos(f);
            sin_t[i] = (float)sin(f);
        }
        for (int h = 0; h < heads; h++) {
            float *hq = qq + (size_t)h * kh;
            for (int i = 0; i < rd / 2; i++) {
                float x0 = hq[2 * i], x1 = hq[2 * i + 1];
                float c = cos_t[i], s = sin_t[i];
                hq[2 * i] = x0 * c - x1 * s;
                hq[2 * i + 1] = x0 * s + x1 * c;
            }
        }
        for (int h = 0; h < kv_heads; h++) {
            float *hk = k + (size_t)h * kh;
            for (int i = 0; i < rd / 2; i++) {
                float x0 = hk[2 * i], x1 = hk[2 * i + 1];
                float c = cos_t[i], s = sin_t[i];
                hk[2 * i] = x0 * c - x1 * s;
                hk[2 * i + 1] = x0 * s + x1 * c;
            }
        }
        free(cos_t);
        free(sin_t);
    }

    /* write k/v into the cache at token */
    float *ck = kv->kv + ((size_t)L * kv->max_tokens + token) * kvlat;
    memcpy(ck, k, (size_t)krows * sizeof(float));
    memcpy(ck + krows, v, (size_t)vrows * sizeof(float));

    /* attention: 16 q-heads over 2 kv-heads (repeat_kv 8), 0..token */
    float dscale = 1.0f / sqrtf((float)kh);
    float *scores = (float *)calloc((size_t)kv->max_tokens, sizeof(float));
    float *wgt = (float *)calloc((size_t)kv->max_tokens, sizeof(float));
    if (!scores || !wgt) {
        free(scores); free(wgt); free(gate); free(qq); free(kk);
        free(buf);
        return -1;
    }
    memset(attn_out, 0, (size_t)orows * sizeof(float));
    int npos = token + 1;
    for (int h = 0; h < heads; h++) {
        const float *qh_ptr = qq + (size_t)h * kh;
        /* GQA: kv head = h / (heads/kv_heads) -- repeat_kv grouping
         * (q heads 0..7 -> kv head 0, 8..15 -> kv head 1) */
        int khh = h / (heads / kv_heads);
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
        if (getenv("DS4F_DEBUG_ATTN") && h == 0 && L == 3) {
            float wmax = 0.0f;
            int warg = -1;
            for (int t2 = 0; t2 < npos; t2++) {
                wgt[t2] /= sum;
                if (wgt[t2] > wmax) { wmax = wgt[t2]; warg = t2; }
            }
            fprintf(stderr, "[attn] L%d t%d npos=%d topw=%.3f at pos %d "
                    "score[0]=%.3f score[last]=%.3f\n", L, token, npos,
                    wmax, warg, scores[0], scores[npos - 1]);
        } else {
            for (int t2 = 0; t2 < npos; t2++) wgt[t2] /= sum;
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

    /* gate: attn_out *= sigmoid(gate) */
    for (int h = 0; h < heads; h++) {
        const float *gh = gate + (size_t)h * kh;
        float *ao = attn_out + (size_t)h * kh;
        for (int i = 0; i < kh; i++) {
            float gv = gh[i];
            ao[i] *= 1.0f / (1.0f + expf(-gv));
        }
    }
    if (getenv("DS4F_NAN_PROBE") && L == 3) {
        double q2 = 0.0, a2 = 0.0, g2 = 0.0;
        float gmin = 1e30f, gmax = -1e30f;
        for (int i = 0; i < heads * kh; i++) {
            q2 += (double)qq[i] * qq[i];
            a2 += (double)attn_out[i] * attn_out[i];
            if (gate[i] < gmin) gmin = gate[i];
            if (gate[i] > gmax) gmax = gate[i];
        }
        for (int i = 0; i < vrows; i++) g2 += (double)v[i] * v[i];
        fprintf(stderr, "[gqa] L3 q-rms %.6g attn-rms %.6g v-rms %.6g "
                "gate[%.4g, %.4g] heads=%d kh=%d qh=%d qrows=%d\n",
                sqrt(q2 / (heads * kh)), sqrt(a2 / (heads * kh)),
                sqrt(g2 / vrows), gmin, gmax, heads, kh, qh, qrows);
        if (qn >= 0) {
            const uint16_t *qwnp = (const uint16_t *)(const void *)
                (tr + tl->t[qn].off);
            fprintf(stderr, "[gqa] L3 qn[0..3]=%.6g %.6g %.6g %.6g "
                    "gate[0..3]=%.6g %.6g %.6g %.6g\n",
                    bf16_f(qwnp[0]), bf16_f(qwnp[1]), bf16_f(qwnp[2]),
                    bf16_f(qwnp[3]),
                    gate[0], gate[1], gate[2], gate[3]);
        }
        if (getenv("DS4F_DUMP_Z")) {
            FILE *qf = fopen("/tmp/q35-eng-gqaq.bin", "wb");
            if (qf) {
                fwrite(qq, sizeof(float), (size_t)heads * kh, qf);
                fwrite(gate, sizeof(float), (size_t)heads * kh, qf);
                fclose(qf);
            }
            FILE *xf = fopen("/tmp/q35-eng-gqaxin.bin", "wb");
            if (xf) {
                fwrite(xin, sizeof(float), (size_t)H, xf);
                fclose(xf);
            }
            FILE *pf = fopen("/tmp/q35-eng-gqaproj.bin", "wb");
            if (pf) {
                fwrite(q, sizeof(float), (size_t)qrows, pf);
                fclose(pf);
            }
        }
    }
    free(gate);
    free(qq);
    free(kk);

    /* o_proj(attn_out) -> o; residual add. attn_out is ocols wide
     * (16 heads x 256); o_proj is [orows=2048 x ocols=4096]. */
    if (mlx4_proj(tl, oi, os, ob, tr, orows, ocols, attn_out, o) != 0) {
        free(buf);
        return -1;
    }
    if (getenv("DS4F_NAN_PROBE") && L == 3) {
        double o2 = 0.0, a2 = 0.0;
        for (int i = 0; i < H; i++) o2 += (double)o[i] * o[i];
        for (int i = 0; i < ocols; i++)
            a2 += (double)attn_out[i] * attn_out[i];
        fprintf(stderr, "[gqa] L3 o_proj-rms %.6g attn-rms(4096) %.6g "
                "ratio %.3g\n", sqrt(o2 / H), sqrt(a2 / ocols),
                sqrt(o2 / H) / (sqrt(a2 / ocols) + 1e-30f));
    }
    for (int i = 0; i < H; i++) state[i] += o[i];

    free(buf);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Linear attention: Gated DeltaNet (Qwen3-Next class)                 */
/* ------------------------------------------------------------------ */
/*
 * Reference: transformers modeling_qwen3_5.py Qwen3_5GatedDeltaNet.
 * Per layer, per token (decode, seq_len=1):
 *   qkv = in_proj_qkv(x)            [8192] = q(2048=16x128) k(2048) v(4096=32x128)
 *   qkv = silu(conv1d(qkv))         depthwise, kernel 4 (causal, pad 3)
 *   z   = in_proj_z(x)              [4096] = 32 value heads x 128
 *   b   = sigmoid(in_proj_b(x))     [32] beta per value head
 *   a   = in_proj_a(x)              [32]
 *   g   = -exp(A_log) * softplus(a + dt_bias)      [32] log-decay
 *   q,k = repeat_interleave(2)      (16 key heads -> 32 value heads)
 *   q   = l2norm(q)/sqrt(128); k = l2norm(k)
 *   state = state * exp(g)
 *   kv_mem = sum(state * k, dim=k)              (delta-rule correction)
 *   delta  = (v - kv_mem) * beta
 *   state = state + k (x) delta
 *   out   = sum(state * q, dim=k)
 *   out   = RMSNormGated(out, z)   = norm.weight * out * silu(z)
 *   out_proj(out) -> [2048], residual add
 *
 * The recurrent state per value head is [kd x vd] (128x128); 32 heads
 * x 2 MB/layer live in the kv cache's lin arena.
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
    int iln = tl->attn_norm[L];
    if (pi < 0 || zi < 0 || ai < 0 || bi < 0 || ci < 0 || oi < 0 ||
        ni < 0 || ai_ < 0 || di < 0) {
        if (getenv("DS4F_NAN_PROBE") && L < 3)
            fprintf(stderr, "[lin] L%d SKIP pi=%d zi=%d ai=%d bi=%d "
                    "ci=%d oi=%d ni=%d Al=%d dt=%d\n", L, pi, zi, ai, bi,
                    ci, oi, ni, ai_, di);
        return 0;                    /* incomplete graph: skip */
    }
    int H = cfg->hidden;
    int qkv_rows = (int)tl->t[pi].dims[0];      /* 8192 */
    int z_rows = (int)tl->t[zi].dims[0];        /* 4096 */
    int o_rows = (int)tl->t[oi].dims[0];        /* 2048 */
    int cols = (int)tl->t[pi].dims[1] * 8;      /* decoded 2048 = H */
    /* head geometry (from the model config) */
    int k_heads = 16, v_heads = 32, kd = 128, vd = 128;
    if (cfg->n_heads > 0) k_heads = cfg->n_heads;      /* 16 */
    if (cfg->n_kv_heads > 0) v_heads = cfg->n_kv_heads * 16; /* 2x16=32 */
    if (qkv_rows != k_heads * kd * 2 + v_heads * vd) {
        if (getenv("DS4F_NAN_PROBE") && L < 3)
            fprintf(stderr, "[lin] L%d GEOM qkv_rows=%d want=%d\n", L,
                    qkv_rows, k_heads * kd * 2 + v_heads * vd);
        fprintf(stderr, "qwen lin: L%d qkv %d != k %d*%d*2 + v %d*%d\n",
                L, qkv_rows, k_heads, kd, v_heads, vd);
        return -1;
    }
    if (z_rows != v_heads * vd || o_rows != H) {
        if (getenv("DS4F_NAN_PROBE") && L < 3)
            fprintf(stderr, "[lin] L%d GEOM z_rows=%d o_rows=%d\n", L,
                    z_rows, o_rows);
        return -1;
    }
    if (getenv("DS4F_NAN_PROBE") && L < 3)
        fprintf(stderr, "[lin] L%d enter ok\n", L);
    if (!kv || token < 0 || token >= kv->max_tokens) return 0;
    if (!kv->lin_alloc)
        if (ds4f_kv_lin_init(kv, v_heads, kd, vd) != 0) return -1;
    if (!kv->conv_alloc)
        if (ds4f_kv_conv_init(kv, qkv_rows) != 0) return -1;
    /* the conv1d ring lives in the kv cache: it must survive across
     * tokens (the causal conv reads the PREVIOUS qkv vectors) */
    float *conv_ring = kv->conv + (size_t)L * 4 * qkv_rows;

    float *buf = (float *)calloc(
        (size_t)(H + qkv_rows + z_rows + o_rows + v_heads * vd +
                 v_heads * kd + 2 * H + 1),
        sizeof(float));
    if (!buf) return -1;
    float *xin = buf;               /* input_layernorm(state) */
    float *qkv = xin + H;
    float *z = qkv + qkv_rows;
    float *o = z + z_rows;
    float *readout = o + o_rows;              /* v_heads*vd */
    float *qk = readout + v_heads * vd;       /* k_heads*kd (q) + k */

    /* reference: x = x + GatedDeltaNet(input_layernorm(x)) */
    if (iln >= 0) {
        const uint16_t *nw = (const uint16_t *)(const void *)(tr +
                             tl->t[iln].off);
        memcpy(xin, state, (size_t)H * sizeof(float));
        double ss = 0.0;
        for (int i = 0; i < H; i++) ss += (double)xin[i] * xin[i];
        float r = sqrtf((float)(ss / (double)H) + 1e-6f);
        for (int i = 0; i < H; i++) {
            uint32_t bits = (uint32_t)nw[i] << 16;
            float w;
            memcpy(&w, &bits, 4);
            xin[i] = xin[i] / r * w;
        }
    } else {
        memcpy(xin, state, (size_t)H * sizeof(float));
    }

    if (mlx4_proj(tl, pi, ps, pb, tr, qkv_rows, cols, xin, qkv) != 0 ||
        mlx4_proj(tl, zi, zs, zb, tr, z_rows, cols, xin, z) != 0) {
        free(buf);
        return -1;
    }
    float a32[32], b32[32];
    {
        float tmp[32];
        if (mlx4_proj(tl, ai, as_, ab, tr, 32, cols, xin, tmp) != 0 ||
            mlx4_proj(tl, bi, bs_, bb, tr, 32, cols, xin, a32) != 0) {
            free(buf);
            return -1;
        }
        memcpy(b32, a32, sizeof b32);        /* b = second proj result */
        memcpy(a32, tmp, sizeof a32);
    }
    /* causal conv1d (depthwise, kernel 4) with SILU activation. Weight
     * [8192, 4, 1] BF16, layout [channel][k][1]. Ring holds the past
     * Q3_CONV_K qkv vectors. The reference's causal_conv1d_update
     * concatenates [x_{t-3}, x_{t-2}, x_{t-1}, x_t] and convolves with
     * w[0..3] -- so w[0] is the OLDEST tap and w[3] the newest.
     * conv(x) = silu(sum_k w[k]*x[t-3+k]) */
    {
        const uint16_t *cw = (const uint16_t *)(const void *)(tr +
                              tl->t[ci].off);
        int base = (token % Q3_CONV_K) * qkv_rows;
        memcpy(conv_ring + base, qkv, (size_t)qkv_rows * sizeof(float));
        if (getenv("DS4F_NAN_PROBE") && L == 0 && token == 0) {
            FILE *pf = fopen("/tmp/q35-eng-qkvpre.bin", "wb");
            if (pf) {
                fwrite(qkv, sizeof(float), (size_t)qkv_rows, pf);
                fclose(pf);
            }
            fprintf(stderr, "[convw] ci=%d off=%zu cw[0..7]=%.6g %.6g %.6g %.6g "
                    "%.6g %.6g %.6g %.6g\n", ci,
                    (size_t)tl->t[ci].off,
                    bf16_f(cw[0]), bf16_f(cw[1]), bf16_f(cw[2]), bf16_f(cw[3]),
                    bf16_f(cw[4]), bf16_f(cw[5]), bf16_f(cw[6]), bf16_f(cw[7]));
        }
        for (int ch = 0; ch < qkv_rows; ch++) {
            float acc = 0.0f;
            for (int k = 0; k < Q3_CONV_K; k++) {
                int tpos = token - (Q3_CONV_K - 1 - k);   /* oldest..newest */
                if (tpos < 0) continue;
                int slot = (tpos % Q3_CONV_K) * qkv_rows + ch;
                float w = bf16_f(cw[(size_t)ch * Q3_CONV_K + k]);
                acc += conv_ring[slot] * w;
            }
            float sig = 1.0f / (1.0f + expf(-acc));
            qkv[ch] = acc * sig;             /* silu */
        }
    }
    /* q/k normalization, EXACTLY the reference (qwen3_5.py):
     *   inv_scale = kd^-0.5
     *   q = inv_scale^2 * rms_norm(q, None, 1e-6)   (eps on the MEAN)
     *   k = inv_scale   * rms_norm(k, None, 1e-6)
     * rms_norm: x / sqrt(mean(x^2) + 1e-6) -- NOT l2norm on the sum;
     * the eps placement matters for small-norm heads. */
    {
        for (int h = 0; h < k_heads; h++) {
            float *qh = qkv + (size_t)h * kd;
            float *kh = qkv + (size_t)(k_heads * kd) + (size_t)h * kd;
            for (int which = 0; which < 2; which++) {
                float *vec = which ? kh : qh;
                double ss = 0.0;
                for (int i = 0; i < kd; i++)
                    ss += (double)vec[i] * vec[i];
                float rms = sqrtf((float)(ss / (double)kd) + 1e-6f);
                float scale = which ? (1.0f / sqrtf((float)kd))   /* k */
                                     : (1.0f / (float)kd);        /* q */
                for (int i = 0; i < kd; i++)
                    vec[i] = vec[i] / rms * scale;
            }
        }
    }
    /* log-decay per value head: g = -exp(A_log) * softplus(a + dt_bias) */
    float g32[32];
    {
        const float *Al = (const float *)(const void *)(tr + tl->t[ai_].off);
        const uint16_t *dtb = (const uint16_t *)(const void *)(tr +
                              tl->t[di].off);
        for (int h = 0; h < v_heads; h++) {
            float dt = bf16_f(dtb[h]);
            float sp = (a32[h] + dt) > 0
                ? (a32[h] + dt) + log1pf(expf(-(a32[h] + dt)))
                : log1pf(expf(a32[h] + dt));
            g32[h] = -expf(Al[h]) * sp;
        }
        if (getenv("DS4F_NAN_PROBE") && L == 5) {
            fprintf(stderr, "[decay] L%d a[0..3]=[%.4g %.4g %.4g %.4g] "
                    "dt[0..3]=[%.4g %.4g %.4g %.4g] "
                    "g[0..3]=[%.4g %.4g %.4g %.4g] expg=%.4g\n",
                    L, a32[0], a32[1], a32[2], a32[3],
                    bf16_f(dtb[0]), bf16_f(dtb[1]), bf16_f(dtb[2]),
                    bf16_f(dtb[3]),
                    g32[0], g32[1], g32[2], g32[3], expf(g32[0]));
        }
    }
    /* delta-rule state update + readout (the FLA recurrent rule):
     *   state = state * exp(g); kv_mem = sum_k(state * k)
     *   delta = (v - kv_mem) * beta; state += k (x) delta
     *   out = sum_k(state * q)
     * value head h uses key head h/2 (repeat_interleave 2). */
    float *S = kv->lin + (size_t)L * v_heads * kd * vd;
    memset(readout, 0, (size_t)v_heads * vd * sizeof(float));
    if (getenv("DS4F_NAN_PROBE")) {
        double sm = 0.0;
        for (int i = 0; i < kd * vd; i++)
            if (S[i] == S[i]) sm += (double)S[i] * S[i];
        if (L % 8 == 0 && token == 6)
            fprintf(stderr, "[lin] L%d pre-state rms %.6g\n", L, sqrt(sm));
    }
    for (int h = 0; h < v_heads; h++) {
        int khh = h / 2;
        const float *kh = qkv + (size_t)(k_heads * kd) + (size_t)khh * kd;
        const float *qh = qkv + (size_t)khh * kd;
        const float *vh = qkv + (size_t)(2 * k_heads * kd) +
                          (size_t)h * vd;
        float *Sh = S + (size_t)h * kd * vd;
        float beta = 1.0f / (1.0f + expf(-b32[h]));   /* sigmoid */
        float decay = expf(g32[h]);
        /* state *= decay */
        for (int i = 0; i < kd * vd; i++) Sh[i] *= decay;
        /* kv_mem = sum_k(state * k) -> [vd]; delta = (v - kv_mem)*beta */
        float delta[128];
        for (int j = 0; j < vd; j++) {
            float acc = 0.0f;
            for (int i = 0; i < kd; i++)
                acc += Sh[(size_t)i * vd + j] * kh[i];
            delta[j] = (vh[j] - acc) * beta;
        }
        /* state += k (x) delta */
        for (int i = 0; i < kd; i++) {
            float *row = Sh + (size_t)i * vd;
            float kk = kh[i];
            for (int j = 0; j < vd; j++)
                row[j] += kk * delta[j];
        }
        /* readout: out_h = sum_k(state * q) */
        float *oh = readout + (size_t)h * vd;
        for (int j = 0; j < vd; j++) {
            float acc = 0.0f;
            for (int i = 0; i < kd; i++)
                acc += Sh[(size_t)i * vd + j] * qh[i];
            oh[j] = acc;
        }
        if (getenv("DS4F_NAN_PROBE") && L == 0 && token == 6 && h == 0) {
            double sm = 0.0;
            for (int i = 0; i < kd * vd; i++)
                if (Sh[i] == Sh[i]) sm += (double)Sh[i] * Sh[i];
            double k2 = 0.0, v2 = 0.0, d2 = 0.0, q2 = 0.0;
            for (int i = 0; i < kd; i++) {
                k2 += (double)kh[i] * kh[i];
                q2 += (double)qh[i] * qh[i];
            }
            for (int i = 0; i < vd; i++) v2 += (double)vh[i] * vh[i];
            for (int i = 0; i < vd; i++) d2 += (double)delta[i] * delta[i];
            fprintf(stderr, "[delta] L%d h0 decay %.6g beta %.6g k-rms %.6g "
                    "q-rms %.6g v-rms %.6g delta-rms %.6g state-rms %.6g\n",
                    L, decay, beta, sqrt(k2 / kd), sqrt(q2 / kd),
                    sqrt(v2 / vd), sqrt(d2 / vd), sqrt(sm / (kd * vd)));
        }
    }
    /* RMSNormGated: norm.weight * out * silu(z) -- the learned norm is
     * on the OUTPUT, gated by silu(z). The norm is PER-HEAD (each
     * 128-dim value-head slice normalized independently), matching the
     * reference's mean over the last dim of [N, head_v_dim]. */
    {
        const uint16_t *nw = (const uint16_t *)(const void *)(tr +
                             tl->t[ni].off);
        for (int h = 0; h < v_heads; h++) {
            float *oh = readout + (size_t)h * vd;
            float *zh = z + (size_t)h * vd;
            double ss = 0.0;
            for (int i = 0; i < vd; i++)
                ss += (double)oh[i] * oh[i];
            float r = sqrtf((float)(ss / (double)vd) + 1e-6f);
            if (getenv("DS4F_NAN_PROBE") && L == 0 && token == 0 && h == 0)
                fprintf(stderr, "[norm] L0 t0 h0 pre-r=%.6g first=%.6g\n",
                        r, oh[0]);
            if (getenv("DS4F_NAN_PROBE") && L == 0 && token == 0 && h == 0 &&
                getenv("DS4F_DUMP_Z")) {
                FILE *rf = fopen("/tmp/q35-eng-readout.bin", "wb");
                if (rf) {
                    fwrite(readout, sizeof(float),
                           (size_t)v_heads * vd, rf);
                    fclose(rf);
                }
            }
            for (int i = 0; i < vd; i++) {
                float zv = zh[i];
                float sig = 1.0f / (1.0f + expf(-zv));
                oh[i] = oh[i] / r * bf16_f(nw[i]) * (zv * sig);
            }
            if (getenv("DS4F_NAN_PROBE") && L == 0 && token == 0 && h == 0) {
                double hn = 0.0;
                for (int i = 0; i < vd; i++)
                    hn += (double)oh[i] * oh[i];
                fprintf(stderr, "[norm] h0 post-rms %.6g oh[0..3] %.6g %.6g "
                        "%.6g %.6g\n", sqrt(hn / vd), oh[0], oh[1], oh[2],
                        oh[3]);
                fprintf(stderr, "[norm] h0 zh[0..3] %.6g %.6g %.6g %.6g "
                        "zh[64..67] %.6g %.6g %.6g %.6g zh[124..127] %.6g "
                        "%.6g %.6g %.6g\n", zh[0], zh[1], zh[2], zh[3],
                        zh[64], zh[65], zh[66], zh[67],
                        zh[124], zh[125], zh[126], zh[127]);
            }
        }
    }
    /* out_proj(readout) -> o, residual add */
    if (mlx4_proj(tl, oi, os_, ob, tr, o_rows, v_heads * vd, readout, o)
        != 0) {
        free(buf);
        return -1;
    }
    if (getenv("DS4F_NAN_PROBE") && L == 0 &&
        (token == 0 || token == 1)) {
        double r2 = 0.0, z2 = 0.0, o2 = 0.0, q2 = 0.0;
        for (int i = 0; i < v_heads * vd; i++) {
            r2 += (double)readout[i] * readout[i];
            z2 += (double)z[i] * z[i];
        }
        for (int i = 0; i < H; i++) o2 += (double)o[i] * o[i];
        for (int i = 0; i < k_heads * kd; i++) q2 += (double)qkv[i] * qkv[i];
        if (getenv("DS4F_DUMP_Z")) {
            char qfn[128];
            snprintf(qfn, sizeof qfn, "/tmp/q35-eng-qkv-t%d.bin", token);
            FILE *zf = fopen(qfn, "wb");
            if (zf) {
                fwrite(qkv, sizeof(float), (size_t)qkv_rows, zf);
                fclose(zf);
            }
            FILE *bf = fopen("/tmp/q35-eng-ab.bin", "wb");
            if (bf) {
                fwrite(a32, sizeof(float), 32, bf);
                fwrite(b32, sizeof(float), 32, bf);
                fclose(bf);
            }
            if (token == 1) {
                FILE *sf = fopen("/tmp/q35-eng-state-t1.bin", "wb");
                if (sf) {
                    fwrite(state, sizeof(float), (size_t)H, sf);
                    fclose(sf);
                }
            }
        }
        /* per-head r of head 0 */
        double h0 = 0.0;
        for (int i = 0; i < vd; i++) h0 += (double)readout[i] * readout[i];
        fprintf(stderr, "[linout] L%d t%d readout-rms %.6g z-rms %.6g "
                "o-rms %.6g q-pre rms %.6g head0-r %.6g nw0..3 %.4g %.4g %.4g %.4g "
                "z[0..3] %.4g %.4g %.4g %.4g xin[0..3] %.4g %.4g %.4g %.4g\n",
                L, token,
                sqrt(r2 / (v_heads * vd)), sqrt(z2 / (v_heads * vd)),
                sqrt(o2 / H), sqrt(q2 / (k_heads * kd)),
                sqrt(h0 / (double)vd),
                bf16_f(((const uint16_t *)(const void *)(tr + tl->t[ni].off))[0]),
                bf16_f(((const uint16_t *)(const void *)(tr + tl->t[ni].off))[1]),
                bf16_f(((const uint16_t *)(const void *)(tr + tl->t[ni].off))[2]),
                bf16_f(((const uint16_t *)(const void *)(tr + tl->t[ni].off))[3]),
                z[0], z[1], z[2], z[3],
                xin[0], xin[1], xin[2], xin[3]);
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
