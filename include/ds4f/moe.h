/*
 * moe.h -- trunk/pool tensor layouts + the real MoE compute step
 * (issue #2, milestone step 3).
 *
 * The layouts come from the converter (trunk.json) and the quantizer
 * (pool-mxfp4.json). The MoE step runs REAL math: router matvec on the
 * resident fp32 gate matrix, top-k selection, and the mxfp4 expert
 * chain w1 -> w2 -> w3 against the pool bytes the cache fetched.
 */
#ifndef DS4F_MOE_H
#define DS4F_MOE_H

#include "ds4f/ds4f.h"

#include <stdint.h>

#define DS4F_MAX_LAYERS 256
#define DS4F_MAX_TENSORS_PER_EXPERT 8

typedef struct Ds4fMoETensor {
    long dims[4];
    int  rank;
    long rel_v;         /* values offset, relative to slot start */
    long rel_s;         /* block scales offset, relative to slot */
    long rel_b;         /* biases offset (mlx4), relative to slot; -1 none */
    long v_nbytes, s_nbytes;
    int  bsize;         /* 16 or 32 (mxfp4) */
    int  fmt;           /* 0 = mxfp4, 1 = mlx4 (U32 nibbles + BF16) */
} Ds4fMoETensor;

typedef struct Ds4fExpertLayout {
    int n;
    int chain;          /* 0 = DS-V4 sequential w1->w2->w3,
                           1 = Qwen3 parallel gate||up -> silu -> down */
    Ds4fMoETensor t[DS4F_MAX_TENSORS_PER_EXPERT];
} Ds4fExpertLayout;

typedef struct Ds4fPoolLayout {
    int      n_layers, n_experts;
    int64_t  expert_nbytes;
    int64_t  max_rc;            /* largest R*C over all expert tensors */
    Ds4fExpertLayout *exp;      /* [n_layers*n_experts] */
} Ds4fPoolLayout;

/* Load pool-mxfp4.json (as written by tools/convert-ds4f.py quantize). */
int ds4f_pool_layout_load(Ds4fPoolLayout *pl, const char *path,
                          const Ds4fCfg *cfg);

typedef struct Ds4fTrunkTensor {
    char name[96];
    int  dtype;                 /* 0=F32, 1=I8, 2=F8_E4M3, 4=BF16, 3=other */
    long dims[4];
    int  rank;
    long off;                   /* relative to layer payload start */
    long nbytes;
} Ds4fTrunkTensor;

typedef struct Ds4fTrunkLayout {
    int   n_layers;
    Ds4fTrunkTensor *t;         /* flat, layer-major */
    int  *t_off;                /* [n_layers+1] index into t */
    int   gate[DS4F_MAX_LAYERS]; /* tensor idx or -1 */
    int   gate_bias[DS4F_MAX_LAYERS];
    int   down[DS4F_MAX_LAYERS];
    int   up[DS4F_MAX_LAYERS];
    /* MLA attention roles (issue #6); _s = the E8M0 scale sibling */
    int   attn_qn[DS4F_MAX_LAYERS];
    int   attn_kvn[DS4F_MAX_LAYERS];
    int   attn_wqa[DS4F_MAX_LAYERS],  attn_wqa_s[DS4F_MAX_LAYERS];
    int   attn_wqb[DS4F_MAX_LAYERS],  attn_wqb_s[DS4F_MAX_LAYERS];
    int   attn_wkv[DS4F_MAX_LAYERS],  attn_wkv_s[DS4F_MAX_LAYERS];
    int   attn_woa[DS4F_MAX_LAYERS],  attn_woa_s[DS4F_MAX_LAYERS];
    int   attn_wob[DS4F_MAX_LAYERS],  attn_wob_s[DS4F_MAX_LAYERS];
    int   attn_woc[DS4F_MAX_LAYERS],  attn_woc_s[DS4F_MAX_LAYERS];
    int   attn_sink[DS4F_MAX_LAYERS];
    int   attn_norm[DS4F_MAX_LAYERS]; /* input_layernorm (BF16) */
    int   ffn_norm[DS4F_MAX_LAYERS]; /* post_attention_layernorm */
    /* mHC (issue #6 step 6): per layer, three tensors per connection:
     * fn = the W projections (cols: pre, post, res), base = the S
     * biases (same order), scale = the alpha gating scalars
     * (alpha_pre, alpha_post, alpha_res -- NVIDIA bridge mapping).
     * n_hc = 1: the residual transform B is constrained to 1. */
    int   hc_attn_fn[DS4F_MAX_LAYERS], hc_attn_base[DS4F_MAX_LAYERS],
          hc_attn_scale[DS4F_MAX_LAYERS];
    int   hc_ffn_fn[DS4F_MAX_LAYERS],  hc_ffn_base[DS4F_MAX_LAYERS],
          hc_ffn_scale[DS4F_MAX_LAYERS];
    /* global HC head (learned output contraction over the streams) */
    int   hc_head_fn, hc_head_base, hc_head_scale;
    /* Qwen3.5 attention: GQA (self_attn q/k/v/o) + linear_attn
     * (conv1d, A_log, dt_bias, in_proj_qkv/z/a/b, out_proj, norm).
     * Each projection has weight + scales + biases triplet. */
    int q3_q[DS4F_MAX_LAYERS], q3_qs[DS4F_MAX_LAYERS], q3_qb[DS4F_MAX_LAYERS];
    int q3_k[DS4F_MAX_LAYERS], q3_ks[DS4F_MAX_LAYERS], q3_kb[DS4F_MAX_LAYERS];
    int q3_v[DS4F_MAX_LAYERS], q3_vs[DS4F_MAX_LAYERS], q3_vb[DS4F_MAX_LAYERS];
    int q3_o[DS4F_MAX_LAYERS], q3_os[DS4F_MAX_LAYERS], q3_ob[DS4F_MAX_LAYERS];
    int q3_qn[DS4F_MAX_LAYERS], q3_kn[DS4F_MAX_LAYERS];
    int q3_conv[DS4F_MAX_LAYERS], q3_a_log[DS4F_MAX_LAYERS];
    int q3_dt[DS4F_MAX_LAYERS];
    int q3_pqkv[DS4F_MAX_LAYERS], q3_pqkvs[DS4F_MAX_LAYERS],
        q3_pqkvb[DS4F_MAX_LAYERS];
    int q3_pz[DS4F_MAX_LAYERS], q3_pzs[DS4F_MAX_LAYERS],
        q3_pzb[DS4F_MAX_LAYERS];
    int q3_pa[DS4F_MAX_LAYERS], q3_pas[DS4F_MAX_LAYERS],
        q3_pab[DS4F_MAX_LAYERS];
    int q3_pb[DS4F_MAX_LAYERS], q3_pbs[DS4F_MAX_LAYERS],
        q3_pbb[DS4F_MAX_LAYERS];
    int q3_opa[DS4F_MAX_LAYERS], q3_opas[DS4F_MAX_LAYERS],
        q3_opab[DS4F_MAX_LAYERS];
    int q3_lnorm[DS4F_MAX_LAYERS];
    /* Qwen3.5 shared expert: dense every-token MLP added after the
     * routed experts -- y += sigmoid(shared_expert_gate(x)) *
     * shared_expert(x). Triplets like the routed experts. */
    int se_g[DS4F_MAX_LAYERS], se_gs[DS4F_MAX_LAYERS], se_gb[DS4F_MAX_LAYERS];
    int se_u[DS4F_MAX_LAYERS], se_us[DS4F_MAX_LAYERS], se_ub[DS4F_MAX_LAYERS];
    int se_d[DS4F_MAX_LAYERS], se_ds[DS4F_MAX_LAYERS], se_db[DS4F_MAX_LAYERS];
    int se_r[DS4F_MAX_LAYERS], se_rs[DS4F_MAX_LAYERS], se_rb[DS4F_MAX_LAYERS];
    int   final_norm;            /* final norm before lm_head (-1 = none) */
    int   kvlat;                /* wkv output dim (0 = no attention) */
} Ds4fTrunkLayout;

/* Load trunk.json (as written by tools/convert-ds4f.py convert). */
int ds4f_trunk_layout_load(Ds4fTrunkLayout *tl, const char *path);

/* mHC (DeepSeek-V4 paper eq. 1-8, n_hc general):
 *   X_{l+1} = B·X + C·F(A·X)     X in R^{n_hc x H}, vec(X) = state
 *   xhat = RMSNorm(vec(X))
 *   A = sigmoid(alpha_pre * (xhat . W_pre) + S_pre)          [n_hc]
 *   C = 2*sigmoid(alpha_post * (xhat . W_post) + S_post)     [n_hc]
 *   B = Sinkhorn(exp(alpha_res * Mat(xhat . W_res) + S_res)) [n_hc x n_hc]
 * fn = [(n_hc*(2+n_hc)) x (n_hc*H)]: rows [0,n_hc) W_pre,
 * [n_hc,2n_hc) W_post, [2n_hc,..) W_res (the checkpoint stores the
 * projections transposed). base = [n_hc*(2+n_hc)] (S in the same row
 * order), scale = [3] (alpha_pre, alpha_post, alpha_res).
 * Returns n_hc (>0) with A/C/B set; 0 when absent; -1 on mismatch. */
int ds4f_hc_params(const Ds4fTrunkLayout *tl, int fn_i, int base_i,
                   int sc_i, const uint8_t *tr, int H,
                   const float *state, int *n_hc_out,
                   float *A, float *C, float *B);

/* Combine the n_hc residual streams with A into the layer input
 * x_in[i] = sum_j A[j] * state[j*H + i]. Needs A from hc_params;
 * x_in must hold H floats. */
void ds4f_hc_combine(int n_hc, int H, const float *A, const float *state,
                     float *x_in);

/* Top-k over scores: descending, tie-break by expert index (earlier
 * expert wins ties -- deterministic). */
void ds4f_topk(const float *scores, int E, int k, int *idx, float *w);

/* Real MoE step for layer L. state[hidden] in/out. tr = trunk layer
 * payload; es[j] = cache slot payload for sel[j] (topk, already
 * fetched). scratch holds max_rc floats; job_scratch[k] for k in
 * [0, topk) holds another max_rc floats each, allocated ONCE by the
 * caller (the parallel expert chains must not malloc per call --
 * fresh mmap'd scratch page-faults ~16 MB x topk x layers). Counters
 * accumulate. */
int ds4f_moe_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                  const uint8_t *tr, const Ds4fPoolLayout *pl,
                  const uint8_t *const *es, const int *sel, const float *wsel,
                  float *state, float *scratch, long scratch_n,
                  float *const *job_scratch,
                  int64_t *n_matvec, int64_t *n_decode);

/* M2 batched moe for the chunked prefill: B tokens x topk selections
 * grouped by expert, one batched matvec per distinct expert (dequant
 * amortized over the group), combine in selection order. xins is
 * [H][B] column-major; es/sel/wsel are [B][topk]; states[B][H]
 * in/out (residual add). Returns -1 when the layer needs a path the
 * batch can't do (mHC / up tensor) -- caller falls back to serial
 * ds4f_moe_step per token. */
int ds4f_moe_step_batch(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                        const uint8_t *tr, const Ds4fPoolLayout *pl,
                        const float *xins, int B, int topk,
                        const int *sel, const float *wsel,
                        const uint8_t *const *es,
                        float *const *states,
                        int64_t *n_matvec, int64_t *n_decode);

#endif /* DS4F_MOE_H */
