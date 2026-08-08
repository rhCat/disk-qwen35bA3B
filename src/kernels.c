/* kernels.c -- scalar mxfp4 reference kernels (portable C99). */
#include "ds4f/kernels.h"
#include "ds4f/simd.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

static int g_simd = 1;
static __thread int g_in_expert = 0;

void ds4f_kernels_set_simd(int on) { g_simd = on ? 1 : 0; }

int ds4f_kernels_simd(void) { return g_simd && ds4f_simd_available(); }

void ds4f_kernels_set_in_expert(int in_expert) { g_in_expert = in_expert ? 1 : 0; }

int ds4f_kernels_in_expert(void) { return g_in_expert; }

float ds4f_e8m0_value(uint8_t b) {
    return ldexpf(1.0f, (int)b - 127);   /* b = 0 -> 2^-127 */
}

static float e2m1_mag(int idx) {
    static const float M[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                               2.0f, 3.0f, 4.0f, 6.0f};
    return M[idx & 7];
}

void ds4f_mxfp4_decode(const uint8_t *vals, const uint8_t *scales,
                       int n, int bsize, float *out) {
    if (ds4f_kernels_simd()) {
        ds4f_simd_mxfp4_decode(vals, scales, n, bsize, out);
        return;
    }
    for (int i = 0; i < n; i++) {
        int nib = (vals[i >> 1] >> ((i & 1) ? 4 : 0)) & 0xF;
        float v = e2m1_mag(nib);
        if (nib & 8) v = -v;
        out[i] = v * ds4f_e8m0_value(scales[i / bsize]);
    }
}

void ds4f_mxfp4_matvec(const uint8_t *vals, const uint8_t *scales,
                       int R, int C, int bsize, const float *x, float *y,
                       float *scratch) {
    if (ds4f_kernels_simd()) {
        ds4f_simd_mxfp4_matvec(vals, scales, R, C, bsize, x, y, scratch);
        return;
    }
    ds4f_mxfp4_decode(vals, scales, R * C, bsize, scratch);
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const float *wr = scratch + (size_t)r * C;
        for (int c = 0; c < C; c++) acc += wr[c] * x[c];
        y[r] = acc;
    }
}

void ds4f_router_scores(const float *W, const float *bias, int E, int H,
                        const float *x, float *scores) {
    for (int e = 0; e < E; e++) {
        float acc = 0.0f;
        const float *wr = W + (size_t)e * H;
        for (int c = 0; c < H; c++) acc += wr[c] * x[c];
        scores[e] = acc + (bias ? bias[e] : 0.0f);
    }
}

void ds4f_f32_matvec(const float *W, int R, int C, const float *x,
                     float *y) {
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const float *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) acc += wr[c] * x[c];
        y[r] = acc;
    }
}

static float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;   /* bf16 = top half of fp32 */
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

void ds4f_bf16_matvec(const uint16_t *W, int R, int C, const float *x,
                      const float *bias, float *y) {
    if (ds4f_kernels_simd()) {
        ds4f_simd_bf16_matvec(W, R, C, x, bias, y);
        return;
    }
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const uint16_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) acc += bf16_to_f32(wr[c]) * x[c];
        y[r] = acc + (bias ? bias[r] : 0.0f);
    }
}

/* ------------------------------------------------------------------ */
/* MLX 4-bit (Qwen3.5-35B-A3B) -- scalar reference                     */
/* ------------------------------------------------------------------ */
/*
 * Packing (verified against the real repo, 2026-08-06):
 *   values  U32, 8 nibbles per word, low nibble first:
 *           elem(k) = (v[k/8] >> (4*(k%8))) & 0xF
 *   scales  BF16, one per 64 elements, row-major group index
 *   biases  BF16, same layout as scales
 *   element = (q - 8) * scale[g] + bias[g], g = k / 64
 */

void ds4f_mlx4_decode(const uint32_t *vals, const uint16_t *scales,
                      const uint16_t *biases, int n, float *out) {
    float lut[16];
    long gcur = -1;
    for (int k = 0; k < n; k++) {
        int g = k / DS4F_MLX4_GROUP;
        if (g != gcur) {
            gcur = g;
            float s = bf16_to_f32(scales[g]);
            float b = biases ? bf16_to_f32(biases[g]) : 0.0f;
            /* MLX affine dequant: w = scale*q + bias (q raw 0..15).
             * NO -8 offset -- that was a systematic per-group error */
            for (int q = 0; q < 16; q++)
                lut[q] = (float)q * s + b;
        }
        int q = (int)((vals[k >> 3] >> (4 * (k & 7))) & 0xFu);
        out[k] = lut[q];
    }
}

/* row-partitioned mlx4 matvec worker (large matrices: expert MLPs,
 * the 248320-row head) */
typedef struct {
    const uint32_t *vals;
    const uint16_t *scales, *biases;
    int R, C, r0, r1;
    const float *x;
    float *y;
} Mlx4RowJob;

static void *mlx4_row_worker(void *arg) {
    Mlx4RowJob *j = (Mlx4RowJob *)arg;
    ds4f_simd_mlx4_matvec(j->vals, j->scales, j->biases, j->R, j->C,
                          j->x, j->y, j->r0, j->r1);
    return NULL;
}

void ds4f_mlx4_matvec(const uint32_t *vals, const uint16_t *scales,
                      const uint16_t *biases, int R, int C,
                      const float *x, float *y) {
    /* SIMD fast path (NEON/AVX2) when the layout permits: C % 8 == 0
     * (word-aligned rows). Large matrices (the expert MLPs and the
     * 248320-row head) row-partition across worker threads. */
    if (ds4f_kernels_simd() && (C % 8) == 0 && (C % DS4F_MLX4_GROUP) == 0) {
        int nth = 1;
        /* Row-split only off the expert path. The 8 expert worker
         * threads already saturate the P-cores (8 on this Mac), so a
         * matvec they call must NOT spawn more threads -- it would
         * oversubscribe and thrash. The main thread (attention/head)
         * has idle cores, so its matvecs split down to R=2048 (the z,
         * o, q/k/v projections in linear/gqa). */
        if (!ds4f_kernels_in_expert() && R >= 2048) {
            nth = 8;
            const char *env = getenv("DS4F_ATTN_THREADS");
            if (env) {
                int v = atoi(env);
                if (v >= 1 && v <= 32) nth = v;
            }
        }
        if (nth > 1 && nth <= 16 && R >= 2048 && !ds4f_kernels_in_expert()) {
            /* stack-local jobs: this function is called concurrently
             * by the 8 expert worker threads, so NO static scratch */
            pthread_t th[16];
            Mlx4RowJob job[16];
            int chunk = (R + nth - 1) / nth;
            int nspawn = 0;
            for (int t = 0; t < nth; t++) {
                int r0 = t * chunk;
                int r1 = (t + 1) * chunk < R ? (t + 1) * chunk : R;
                if (r0 >= R) continue;
                job[t].vals = vals; job[t].scales = scales;
                job[t].biases = biases; job[t].R = R; job[t].C = C;
                job[t].x = x; job[t].y = y;
                job[t].r0 = r0; job[t].r1 = r1;
                pthread_create(&th[t], NULL, mlx4_row_worker, &job[t]);
                nspawn++;
            }
            for (int t = 0; t < nspawn; t++)
                pthread_join(th[t], NULL);
            return;
        }
        ds4f_simd_mlx4_matvec(vals, scales, biases, R, C, x, y, 0, R);
        return;
    }
    /* scalar fallback (LUT per group) */
    float lut[16];
    long gcur = -1;
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const float *xr = x;
        for (int c = 0; c < C; c++) {
            long k = (long)r * C + c;
            long g = k / DS4F_MLX4_GROUP;
            if (g != gcur) {
                gcur = g;
                float s = bf16_to_f32(scales[g]);
                float b = biases ? bf16_to_f32(biases[g]) : 0.0f;
                for (int q = 0; q < 16; q++)
                    lut[q] = (float)q * s + b;
            }
            int q = (int)((vals[k >> 3] >> (4 * (int)(k & 7))) & 0xFu);
            acc += lut[q] * xr[c];
        }
        y[r] = acc;
    }
}

/* fp8_e4m3fn decode table (issue #6); e=0 subnormal-ish, e=15 clamp. */
static float fp8_lut[256];
static int  fp8_lut_ready = 0;

static void fp8_lut_build(void) {
    for (int b = 0; b < 256; b++) {
        int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
        float v;
        if (e == 0)
            v = (float)m * 0.001953125f;         /* m * 2^-9 */
        else if (e == 0xF)
            v = 448.0f;                          /* inf/nan -> E4M3FN max */
        else
            v = (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
        fp8_lut[b] = s ? -v : v;
    }
    fp8_lut_ready = 1;   /* benign race: identical values either way */
}

void ds4f_f8_matvec(const uint8_t *W, const uint8_t *scales,
                    int R, int C, int SR, int SC, const float *x,
                    float *y) {
    /* SIMD when the scale blocks are 16-aligned: SC == 1 (per-row) or
     * the block width C/SC is a multiple of 16. Otherwise scalar. */
    if (ds4f_kernels_simd()) {
        int ssc = SC < 1 ? 1 : SC;
        if (ssc == 1 || (C % ssc == 0 && ((C / ssc) % 16) == 0)) {
            ds4f_simd_f8_matvec(W, scales, R, C, SR, ssc, x, y, 0, R);
            return;
        }
    }
    if (!fp8_lut_ready) fp8_lut_build();
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        int sr = (int)(((int64_t)r * SR) / R);      /* row block */
        const uint8_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);  /* col block */
            float s = scales ? ds4f_e8m0_value(scales[sr * SC + sc])
                             : 1.0f;
            acc += fp8_lut[wr[c]] * s * x[c];
        }
        y[r] = acc;
    }
}

void ds4f_f8_matvec_rows(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y, int r0, int r1) {
    if (!fp8_lut_ready) fp8_lut_build();
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    for (int r = r0; r < r1; r++) {
        float acc = 0.0f;
        int sr = (int)(((int64_t)r * SR) / R);      /* GLOBAL row block */
        const uint8_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? ds4f_e8m0_value(scales[sr * SC + sc])
                             : 1.0f;
            acc += fp8_lut[wr[c]] * s * x[c];
        }
        y[r] = acc;
    }
}

void ds4f_f8_decode_row(const uint8_t *W, const uint8_t *scales,
                        int V, int H, int SR, int SC, int row, float *out) {
    if (!fp8_lut_ready) fp8_lut_build();
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    const uint8_t *wr = W + (size_t)row * H;
    int sr = (int)(((int64_t)row * SR) / V);
    for (int c = 0; c < H; c++) {
        int sc = (int)(((int64_t)c * SC) / H);
        float s = scales ? ds4f_e8m0_value(scales[sr * SC + sc]) : 1.0f;
        out[c] = fp8_lut[wr[c]] * s;
    }
}

float ds4f_f8_value(uint8_t b) {
    if (!fp8_lut_ready) fp8_lut_build();
    return fp8_lut[b];
}

void ds4f_i8_matvec(const uint8_t *W, const uint8_t *scales,
                    int R, int C, int SR, int SC, const float *x,
                    float *y) {
    if (ds4f_kernels_simd()) {
        int ssc = SC < 1 ? 1 : SC;
        if (ssc == 1 || (C % ssc == 0 && ((C / ssc) % 16) == 0)) {
            ds4f_simd_i8_matvec(W, scales, R, C, SR, ssc, x, y);
            return;
        }
    }
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        int sr = (int)(((int64_t)r * SR) / R);
        const uint8_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? ds4f_e8m0_value(scales[sr * SC + sc])
                             : 1.0f;
            acc += (float)(int8_t)wr[c] * s * x[c];
        }
        y[r] = acc;
    }
}

float ds4f_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; exp--; }
            man &= 0x3FFu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

void ds4f_f16_matvec(const uint16_t *W, int R, int C, const float *x,
                     float *y) {
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const uint16_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) acc += ds4f_f16_to_f32(wr[c]) * x[c];
        y[r] = acc;
    }
}

/* ---- combined two-matvec row split (qkv + z, one spawn) ---------- */
typedef struct {
    const uint32_t *v1; const uint16_t *s1, *b1; int R1;
    const uint32_t *v2; const uint16_t *s2, *b2; int R2;
    int C;
    const float *x;
    float *y1, *y2;
    int r0, r1;          /* combined row range [r0, r1) over R1+R2 */
} Mlx4RowJob2;

static void *mlx4_row_worker2(void *arg) {
    Mlx4RowJob2 *j = (Mlx4RowJob2 *)arg;
    /* combined row space: [0,R1) -> matvec 1, [R1, R1+R2) -> matvec 2 */
    if (j->r1 <= j->R1) {
        ds4f_simd_mlx4_matvec(j->v1, j->s1, j->b1, j->R1, j->C,
                              j->x, j->y1, j->r0, j->r1);
    } else if (j->r0 >= j->R1) {
        ds4f_simd_mlx4_matvec(j->v2, j->s2, j->b2, j->R2, j->C,
                              j->x, j->y2, j->r0 - j->R1, j->r1 - j->R1);
    } else {
        ds4f_simd_mlx4_matvec(j->v1, j->s1, j->b1, j->R1, j->C,
                              j->x, j->y1, j->r0, j->R1);
        ds4f_simd_mlx4_matvec(j->v2, j->s2, j->b2, j->R2, j->C,
                              j->x, j->y2, 0, j->r1 - j->R1);
    }
    return NULL;
}

/* Two independent MLX4 matvecs with the same x, computed in ONE
 * 8-way row split over the combined row space (R1+R2). The linear
 * attention qkv (8192 rows) and z (4096) projections both read xin
 * and write disjoint outputs -- running them sequentially cost two
 * spawn/join cycles per layer; this overlaps them. Same per-row
 * math as ds4f_mlx4_matvec (bit-identical rows), just partitioned
 * together. Returns 0 if both computed, -1 on fallback. */
int ds4f_mlx4_matvec2(const uint32_t *v1, const uint16_t *s1,
                      const uint16_t *b1, int R1,
                      const uint32_t *v2, const uint16_t *s2,
                      const uint16_t *b2, int R2, int C,
                      const float *x, float *y1, float *y2) {
    if (!ds4f_kernels_simd() || (C % 8) != 0 ||
        (C % DS4F_MLX4_GROUP) != 0 || R1 < 1 || R2 < 1)
        return -1;
    if (ds4f_kernels_in_expert()) return -1;   /* expert threads: no spawn */
    int nth = 8;
    const char *env = getenv("DS4F_ATTN_THREADS");
    if (env) {
        int v = atoi(env);
        if (v >= 1 && v <= 32) nth = v;
    }
    if (nth < 2) {
        ds4f_simd_mlx4_matvec(v1, s1, b1, R1, C, x, y1, 0, R1);
        ds4f_simd_mlx4_matvec(v2, s2, b2, R2, C, x, y2, 0, R2);
        return 0;
    }
    int total = R1 + R2;
    int chunk = (total + nth - 1) / nth;
    pthread_t th[32];
    Mlx4RowJob2 job[32];
    int nspawn = 0;
    for (int t = 0; t < nth; t++) {
        int r0 = t * chunk;
        int r1 = (t + 1) * chunk < total ? (t + 1) * chunk : total;
        if (r0 >= total) continue;
        job[t].v1 = v1; job[t].s1 = s1; job[t].b1 = b1; job[t].R1 = R1;
        job[t].v2 = v2; job[t].s2 = s2; job[t].b2 = b2; job[t].R2 = R2;
        job[t].C = C; job[t].x = x; job[t].y1 = y1; job[t].y2 = y2;
        job[t].r0 = r0; job[t].r1 = r1;
        pthread_create(&th[t], NULL, mlx4_row_worker2, &job[t]);
        nspawn++;
    }
    for (int t = 0; t < nspawn; t++)
        pthread_join(th[t], NULL);
    return 0;
}

/* ---- batched prefill matvec (M1 of prefill-batch) ---------------- */
/* Y[B][R] = W[R x C] * X, where X is COLUMN-MAJOR [C][B] (tokens
 * contiguous per column -- the GEMM layout, so the inner loop reads
 * a contiguous B-float block). The 4-bit dequant is done ONCE per
 * weight row and reused across the B token vectors -- the prefill
 * win. Bit-identical per (row, token) to ds4f_mlx4_matvec (same
 * accumulation order, same FMA), so e2e traces hold. */
typedef struct {
    const uint32_t *vals; const uint16_t *scales, *biases;
    int R, C, B;
    const float *xs;   /* B x C, row-major per token */
    float *ys;         /* B x R, row-major per token */
    int r0, r1;        /* row range [r0, r1) of W */
} Mlx4BatchJob;

static void *mlx4_batch_worker(void *arg) {
    Mlx4BatchJob *j = (Mlx4BatchJob *)arg;
    const int C = j->C;
    int B = j->B;
    /* per-token accumulators in registers: ys[t*R+r] scattered writes
     * (32KB stride) were the pathology -- ~1GB of scattered stores per
     * matvec. Accumulate in a local array, write ys once per row. */
    float acc[512];
    if (B > 512) B = 512;
    for (int r = j->r0; r < j->r1; r++) {
        /* decode this row ONCE: 16-entry LUT per 64-group. The group
         * index is ABSOLUTE across the R*C matrix -- (r*C+c)/64 --
         * exactly like ds4f_simd_mlx4_matvec (simd.c:158). For rows
         * with C%64==0 this is r*(C/64) + c/64; using per-row c/64
         * reads ROW 0's scales for every row (a 2x z-rms divergence
         * that compounding through the delta-rule state completely
         * derailed generation). */
        float lut[16];
        int gcur = -1;
        long rbase = (long)r * C;
        for (int t = 0; t < B; t++) acc[t] = 0.0f;
        for (int c = 0; c < C; c++) {
            int g = (int)((rbase + c) / DS4F_MLX4_GROUP);
            if (g != gcur) {
                gcur = g;
                uint32_t sb = (uint32_t)j->scales[g] << 16;
                uint32_t bb = j->biases ? (uint32_t)j->biases[g] << 16 : 0;
                float s, b;
                memcpy(&s, &sb, 4);
                memcpy(&b, &bb, 4);
                for (int q = 0; q < 16; q++) lut[q] = (float)q * s + b;
            }
            int q = (int)((j->vals[(size_t)r * (C / 8) + (c >> 3)] >>
                           (4 * (c & 7))) & 0xFu);
            float w = lut[q];
            /* xs is [C][B]: tokens contiguous per column */
            const float *xcol = j->xs + (size_t)c * B;
#ifdef __aarch64__
            if (ds4f_kernels_simd()) {
                /* NEON: 4 tokens in parallel. Each acc[t] keeps the
                 * SAME c-order accumulation as the scalar loop and
                 * vmlaq matches -O2 fp-contract, so per-token results
                 * are bit-identical to the scalar reference. */
                int t = 0;
                for (; t + 4 <= B; t += 4) {
                    float32x4_t av = vld1q_f32(acc + t);
                    float32x4_t xv = vld1q_f32(xcol + t);
                    vst1q_f32(acc + t, vmlaq_n_f32(av, xv, w));
                }
                for (; t < B; t++) acc[t] += w * xcol[t];
            } else
#endif
            for (int t = 0; t < B; t++)
                acc[t] += w * xcol[t];
        }
        for (int t = 0; t < B; t++)
            j->ys[(size_t)t * j->R + r] = acc[t];
    }
    return NULL;
}

int ds4f_mlx4_matvec_batch(const uint32_t *vals, const uint16_t *scales,
                           const uint16_t *biases, int R, int C, int B,
                           const float *xs, float *ys) {
    if (!ds4f_kernels_simd() || (C % 8) != 0 || R < 1 || B < 1)
        return -1;
    if (ds4f_kernels_in_expert()) return -1;
    /* zero the outputs (the worker ACCUMULATES into ys) */
    memset(ys, 0, (size_t)B * R * sizeof(float));
    int nth = 8;
    const char *env = getenv("DS4F_ATTN_THREADS");
    if (env) {
        int v = atoi(env);
        if (v >= 1 && v <= 32) nth = v;
    }
    if (nth < 2 || R < nth) {
        Mlx4BatchJob j;
        j.vals = vals; j.scales = scales; j.biases = biases;
        j.R = R; j.C = C; j.B = B; j.xs = xs; j.ys = ys;
        j.r0 = 0; j.r1 = R;
        mlx4_batch_worker(&j);
        return 0;
    }
    int chunk = (R + nth - 1) / nth;
    pthread_t th[32];
    Mlx4BatchJob job[32];
    int nspawn = 0;
    for (int t = 0; t < nth; t++) {
        int r0 = t * chunk;
        int r1 = (t + 1) * chunk < R ? (t + 1) * chunk : R;
        if (r0 >= R) continue;
        job[t].vals = vals; job[t].scales = scales; job[t].biases = biases;
        job[t].R = R; job[t].C = C; job[t].B = B;
        job[t].xs = xs; job[t].ys = ys;
        job[t].r0 = r0; job[t].r1 = r1;
        pthread_create(&th[t], NULL, mlx4_batch_worker, &job[t]);
        nspawn++;
    }
    for (int t = 0; t < nspawn; t++)
        pthread_join(th[t], NULL);
    return 0;
}
