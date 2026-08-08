/*
 * attn.h -- MLA attention step + KV cache (issue #6, step 2).
 *
 * The real V4-Flash attention is MLA (multi-head latent attention):
 *   ql = wq_a . x            (q latent)
 *   qn = RMSNorm(ql, q_norm)
 *   q  = wq_b . qn           (up-project to full width)
 *   kv = RMSNorm(wkv . x, kv_norm)      (kv latent, cached per token)
 *   k  = kv[:kvlat/2], v = kv[kvlat/2:]
 *   scores = q . k / sqrt(d) + sink[pos] (sink positions always attend)
 *   out    = softmax(scores) . v
 *   state += wo_c . wo_b . wo_a . out
 *
 * The KV cache is per-layer (each layer computes its own kv latent);
 * MLA's shared-cache memory optimization is the roadmap refinement.
 * All tensors come from the trunk layout (F8 + E8M0 group scales,
 * BF16 norms, F32 sink). Attention is skipped for layers whose graph
 * is incomplete, so the engine degrades gracefully.
 */
#ifndef DS4F_ATTN_H
#define DS4F_ATTN_H

#include "ds4f/ds4f.h"
#include "ds4f/moe.h"

typedef struct Ds4fKvCache {
    float *kv;              /* [n_layers * max_tokens * kvlat] (MLA/GQA) */
    int    n_layers, kvlat, max_tokens;
    float *lin;             /* linear-attn recurrent state
                              [n_layers][v_heads][kd][vd] (Gated DeltaNet) */
    int    lin_vh, lin_kd, lin_vd, lin_alloc;
    float *conv;            /* linear-attn conv1d ring, PERSISTENT across
                              tokens: [n_layers][Q3_CONV_K][qkv_rows] */
    int    conv_rows, conv_alloc;
    float *scratch;         /* per-token work arena (GQA + linear attn),
                              allocated once at init: covers the worst
                              per-layer buffer need. Removes ~1.1M
                              malloc/free per 20K-token run (macOS keeps
                              freed large blocks at the zone high-water,
                              ~2.9 GB resident). */
    long   scratch_n;       /* floats */
} Ds4fKvCache;

int  ds4f_kv_init(Ds4fKvCache *c, int n_layers, int kvlat, int max_tokens);
void ds4f_kv_free(Ds4fKvCache *c);
/* Allocate the per-token attention work arena (floats). Call after
 * kv_init, before the token loop; frees with ds4f_kv_free. */
int  ds4f_kv_scratch_init(Ds4fKvCache *c, long floats);
/* Allocate the linear-attention state arena (v_heads x kd x vd per
 * layer). Returns 0 on success, -1 on failure. */
int  ds4f_kv_lin_init(Ds4fKvCache *c, int v_heads, int kd, int vd);
/* Allocate the linear-attn conv1d ring (4 x qkv_rows per layer). */
int  ds4f_kv_conv_init(Ds4fKvCache *c, int qkv_rows);

/* Attention for layer L. tr = trunk layer payload; state[hidden] in/out
 * (residual added). token = current token index (cache position).
 * Returns 0 (ok, possibly skipped), -1 on layout/config error. */
int ds4f_attn_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                   const uint8_t *tr, float *state, Ds4fKvCache *kv,
                   int token);

/* Dispatch: full-GQA layers (self_attn roles present) vs linear. */
int ds4f_attn_qwen_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                        const uint8_t *tr, float *state, Ds4fKvCache *kv,
                        int token);
/* Chunked prefill (M1): batch the qkv+z projections of one linear
 * layer over B prompt tokens, serial delta rule per token. states
 * is an array of B pointers to [H] token states; t0 = first token
 * index (kv cache addressing). Bit-identical to B serial calls. */
int ds4f_attn_linear_chunk(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl,
                           int L, const uint8_t *tr, float *const *states,
                           int t0, int B, Ds4fKvCache *kv);
/* Chunked GQA (M3-lite): batch the q/k/v + o_proj projections of one
 * full-GQA layer over B prompt tokens; the softmax attention body
 * stays serial per token (reads the growing K cache). Same body code
 * as gqa_step and the same bit-identical batch kernel as M1. */
int ds4f_attn_gqa_chunk(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl,
                        int L, const uint8_t *tr, float *const *states,
                        int t0, int B, Ds4fKvCache *kv);

#endif /* DS4F_ATTN_H */
