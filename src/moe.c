/* moe.c -- layout loaders + real MoE compute step. */
#include "ds4f/moe.h"
#include "ds4f/kernels.h"
#include "json.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, long *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz <= 0 || sz > (1L << 30)) { fclose(f); return NULL; }
    rewind(f);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    buf[sz] = 0;
    *len_out = sz;
    return buf;
}

static int _moeacc0_dumped = 0;
static int _exp106_dumped = 0;
static int _exp106_out_dumped = 0;
static pthread_mutex_t _expdump_mu = PTHREAD_MUTEX_INITIALIZER;

static int dtype_of(const char *s, size_t n) {
    if (n == 3 && !memcmp(s, "F32", 3)) return 0;
    if (n == 2 && !memcmp(s, "I8", 2)) return 1;
    if (n == 7 && !memcmp(s, "F8_E4M3", 7)) return 2;
    if (n == 4 && !memcmp(s, "BF16", 4)) return 4;
    if (n == 3 && !memcmp(s, "U32", 3)) return 5;   /* MLX 4-bit packed */
    return 3;
}

/* name ends with suffix (NUL-terminated C strings) */
static int name_ends(const char *name, const char *suffix) {
    size_t nl = strlen(name), sl = strlen(suffix);
    return nl >= sl && !memcmp(name + nl - sl, suffix, sl);
}

int ds4f_pool_layout_load(Ds4fPoolLayout *pl, const char *path,
                          const Ds4fCfg *cfg) {
    long len;
    char *buf = read_file(path, &len);
    if (!buf) {
        fprintf(stderr, "pool layout: cannot read %s\n", path);
        return -1;
    }
    JDoc *doc = json_parse(buf, (size_t)len);
    if (!doc) {
        fprintf(stderr, "pool layout: %s not parseable\n", path);
        free(buf);
        return -1;
    }
    const JEntry *nl = json_get(doc->root, doc->nroot, "n_layers");
    const JEntry *ne = json_get(doc->root, doc->nroot, "n_experts");
    const JEntry *eb = json_get(doc->root, doc->nroot, "expert_nbytes");
    const JEntry *ts = json_get(doc->root, doc->nroot, "tensors");
    if (!nl || !ne || !eb || !ts || ts->type != 3) {
        fprintf(stderr, "pool layout: %s missing keys\n", path);
        json_free(doc); free(buf);
        return -1;
    }
    memset(pl, 0, sizeof *pl);
    pl->n_layers = (int)nl->inum;
    pl->n_experts = (int)ne->inum;
    pl->expert_nbytes = eb->inum;
    if (pl->n_layers != cfg->n_layers || pl->n_experts != cfg->n_experts) {
        fprintf(stderr, "pool layout: %dx%d vs config %dx%d\n",
                pl->n_layers, pl->n_experts, cfg->n_layers, cfg->n_experts);
        json_free(doc); free(buf);
        return -1;
    }
    int nt = ts->nchild;
    pl->exp = (Ds4fExpertLayout *)calloc(
        (size_t)(pl->n_layers * pl->n_experts), sizeof(Ds4fExpertLayout));
    if (!pl->exp) { json_free(doc); free(buf); return -1; }
    for (int i = 0; i < nt; i++) {
        const JEntry *e = &ts->child[i];
        const JEntry *lay = json_get(e->child, e->nchild, "layer");
        const JEntry *exp = json_get(e->child, e->nchild, "expert");
        const JEntry *shp = json_get(e->child, e->nchild, "shape");
        const JEntry *vo  = json_get(e->child, e->nchild, "v_off");
        const JEntry *so  = json_get(e->child, e->nchild, "s_off");
        const JEntry *vn  = json_get(e->child, e->nchild, "v_nbytes");
        const JEntry *sn  = json_get(e->child, e->nchild, "s_nbytes");
        if (!lay || !exp || !shp || !vo || !so || !vn || !sn) continue;
        int L = (int)lay->inum, X = (int)exp->inum;
        if (L < 0 || L >= pl->n_layers || X < 0 || X >= pl->n_experts)
            continue;
        Ds4fExpertLayout *el = &pl->exp[(size_t)L * pl->n_experts + X];
        if (el->n >= DS4F_MAX_TENSORS_PER_EXPERT) continue;
        /* Qwen3 chain marker: set once from the first tensor entry */
        if (el->n == 0) {
            const JEntry *ch = json_get(e->child, e->nchild, "chain");
            el->chain = (ch && ch->type == 1 && ch->str &&
                         ch->str_end - ch->str == 5 &&
                         !memcmp(ch->str, "qwen3", 5)) ? 1 : 0;
        }
        Ds4fMoETensor *t = &el->t[el->n++];
        t->rank = shp->nchild;
        if (t->rank > 4) t->rank = 4;
        long rc = 1;
        for (int d = 0; d < t->rank; d++) {
            t->dims[d] = (long)shp->child[d].inum;
            rc *= t->dims[d];
        }
        if (rc > pl->max_rc) pl->max_rc = rc;
        /* v_off is absolute in the pool file; slot base is arithmetic.
         * Compute in int64 throughout: pool offsets reach ~68 GB and any
         * 32-bit step wraps rel_v negative -> out-of-bounds reads. */
        int64_t slot_off = 24 +
            (int64_t)L * pl->n_experts * pl->expert_nbytes +
            (int64_t)X * pl->expert_nbytes;
        t->rel_v = (long)(vo->inum - slot_off);
        t->rel_s = (long)(so->inum - slot_off);
        t->v_nbytes = vn->inum;
        t->s_nbytes = sn->inum;
        t->bsize = 32;             /* mxfp4-pool-v1 output format */
        t->fmt = 0;
        t->rel_b = -1;
        {
            /* optional mlx4 fields: fmt=1 selects the MLX 4-bit kernel,
             * b_off/b_nbytes carry the per-group BF16 biases */
            const JEntry *fm = json_get(e->child, e->nchild, "fmt");
            if (fm && fm->type == 0 && fm->inum == 1) t->fmt = 1;
            const JEntry *bo = json_get(e->child, e->nchild, "b_off");
            const JEntry *bn = json_get(e->child, e->nchild, "b_nbytes");
            if (bo && bn && bo->type == 0 && bn->type == 0) {
                long rel_b = (long)(bo->inum - slot_off);
                if (rel_b >= 0 && rel_b + bn->inum <= pl->expert_nbytes) {
                    t->rel_b = rel_b;
                }
            }
        }
        if (t->rel_v < 0 || t->rel_s < 0 ||
            t->rel_v + t->v_nbytes > pl->expert_nbytes ||
            t->rel_s + t->s_nbytes > pl->expert_nbytes) {
            fprintf(stderr,
                "pool layout: tensor L%d E%d out of slot bounds "
                "(rel_v=%ld rel_s=%ld, slot=%lld)\n",
                L, X, t->rel_v, t->rel_s, (long long)pl->expert_nbytes);
            json_free(doc); free(buf);
            return -1;
        }
    }
    json_free(doc);
    free(buf);
    return 0;
}

int ds4f_trunk_layout_load(Ds4fTrunkLayout *tl, const char *path) {
    long len;
    char *buf = read_file(path, &len);
    if (!buf) {
        fprintf(stderr, "trunk layout: cannot read %s\n", path);
        return -1;
    }
    JDoc *doc = json_parse(buf, (size_t)len);
    if (!doc) {
        fprintf(stderr, "trunk layout: %s not parseable\n", path);
        free(buf);
        return -1;
    }
    const JEntry *nl = json_get(doc->root, doc->nroot, "n_layers");
    const JEntry *ls = json_get(doc->root, doc->nroot, "layers");
    if (!nl || !ls || ls->type != 3) {
        fprintf(stderr, "trunk layout: %s missing keys\n", path);
        json_free(doc); free(buf);
        return -1;
    }
    memset(tl, 0, sizeof *tl);
    tl->n_layers = (int)nl->inum;
    if (tl->n_layers < 1 || tl->n_layers > DS4F_MAX_LAYERS) {
        fprintf(stderr, "trunk layout: n_layers %d out of range\n",
                tl->n_layers);
        json_free(doc); free(buf);
        return -1;
    }
    for (int L = 0; L < DS4F_MAX_LAYERS; L++) {
        tl->gate[L] = tl->down[L] = tl->up[L] = -1;
        tl->gate_bias[L] = -1;
        tl->se_g[L] = tl->se_gs[L] = tl->se_gb[L] = -1;
        tl->se_u[L] = tl->se_us[L] = tl->se_ub[L] = -1;
        tl->se_d[L] = tl->se_ds[L] = tl->se_db[L] = -1;
        tl->se_r[L] = tl->se_rs[L] = tl->se_rb[L] = -1;
        tl->attn_qn[L] = tl->attn_kvn[L] = -1;
        tl->attn_wqa[L] = tl->attn_wqa_s[L] = -1;
        tl->attn_wqb[L] = tl->attn_wqb_s[L] = -1;
        tl->attn_wkv[L] = tl->attn_wkv_s[L] = -1;
        tl->attn_woa[L] = tl->attn_woa_s[L] = -1;
        tl->attn_wob[L] = tl->attn_wob_s[L] = -1;
        tl->attn_woc[L] = tl->attn_woc_s[L] = -1;
        tl->attn_sink[L] = -1;
        tl->attn_norm[L] = -1;
        tl->ffn_norm[L] = -1;
        tl->hc_attn_fn[L] = tl->hc_attn_base[L] = tl->hc_attn_scale[L] = -1;
        tl->hc_ffn_fn[L] = tl->hc_ffn_base[L] = tl->hc_ffn_scale[L] = -1;
        /* Qwen3.5 attention roles: GQA (self_attn) + linear (linear_attn) */
        tl->q3_q[L] = tl->q3_qs[L] = tl->q3_qb[L] = -1;
        tl->q3_k[L] = tl->q3_ks[L] = tl->q3_kb[L] = -1;
        tl->q3_v[L] = tl->q3_vs[L] = tl->q3_vb[L] = -1;
        tl->q3_o[L] = tl->q3_os[L] = tl->q3_ob[L] = -1;
        tl->q3_qn[L] = tl->q3_kn[L] = -1;
        tl->q3_conv[L] = tl->q3_a_log[L] = tl->q3_dt[L] = -1;
        tl->q3_pqkv[L] = tl->q3_pqkvs[L] = tl->q3_pqkvb[L] = -1;
        tl->q3_pz[L] = tl->q3_pzs[L] = tl->q3_pzb[L] = -1;
        tl->q3_pa[L] = tl->q3_pas[L] = tl->q3_pab[L] = -1;
        tl->q3_pb[L] = tl->q3_pbs[L] = tl->q3_pbb[L] = -1;
        tl->q3_opa[L] = tl->q3_opas[L] = tl->q3_opab[L] = -1;
        tl->q3_lnorm[L] = -1;
    }
    tl->final_norm = -1;
    tl->hc_head_fn = tl->hc_head_base = tl->hc_head_scale = -1;
    tl->kvlat = 0;

    int total = 0;
    for (int i = 0; i < ls->nchild; i++) {
        const JEntry *ly = &ls->child[i];
        if (ly->type != 2 || !ly->child || ly->nchild < 1) {
            fprintf(stderr, "trunk layout: layer entry %d malformed\n", i);
            json_free(doc); free(buf);
            return -1;
        }
        const JEntry *tss = json_get(ly->child, ly->nchild, "tensors");
        if (tss && tss->type == 3) total += tss->nchild;
    }
    fprintf(stderr, "trunk layout: %d layer entries, %d tensors total\n",
            ls->nchild, total);
    tl->t = (Ds4fTrunkTensor *)calloc((size_t)total,
                                      sizeof(Ds4fTrunkTensor));
    tl->t_off = (int *)calloc((size_t)(tl->n_layers + 1), sizeof(int));
    if (!tl->t || !tl->t_off) {
        json_free(doc); free(buf);
        return -1;
    }
    int k = 0;
    for (int L = 0; L < tl->n_layers && L < DS4F_MAX_LAYERS; L++) {
        tl->t_off[L] = k;
        const JEntry *ly = NULL;
        for (int i = 0; i < ls->nchild; i++) {
            const JEntry *cand = &ls->child[i];
            const JEntry *lid = json_get(cand->child, cand->nchild, "layer");
            if (lid && lid->inum == L) { ly = cand; break; }
        }
        if (!ly) continue;
        const JEntry *tss = json_get(ly->child, ly->nchild, "tensors");
        if (!tss || tss->type != 3) continue;
        if (!tss->child && tss->nchild > 0) {
            fprintf(stderr, "trunk layout: layer %d tensors array dangles\n", L);
            json_free(doc); free(buf);
            return -1;
        }
        for (int i = 0; i < tss->nchild; i++) {
            const JEntry *e = &tss->child[i];
            if (k >= total) {
                fprintf(stderr, "trunk layout: tensor overflow at layer %d "
                        "entry %d (k=%d total=%d)\n", L, i, k, total);
                json_free(doc); free(buf);
                return -1;
            }
            if (e->type != 2 || !e->child) {
                fprintf(stderr, "trunk layout: layer %d tensor %d malformed\n",
                        L, i);
                json_free(doc); free(buf);
                return -1;
            }
            Ds4fTrunkTensor *tt = &tl->t[k++];
            const JEntry *nm = json_get(e->child, e->nchild, "n");
            const JEntry *dt = json_get(e->child, e->nchild, "dtype");
            const JEntry *shp = json_get(e->child, e->nchild, "shape");
            const JEntry *of = json_get(e->child, e->nchild, "off");
            const JEntry *nb = json_get(e->child, e->nchild, "nbytes");
            tt->name[0] = 0;
            if (nm && nm->type == 1) {
                size_t nn = (size_t)(nm->str_end - nm->str);
                if (nn > 95) nn = 95;
                memcpy(tt->name, nm->str, nn);
                tt->name[nn] = 0;
            }
            if (dt && dt->type == 1)
                tt->dtype = dtype_of(dt->str,
                                     (size_t)(dt->str_end - dt->str));
            if (shp) {
                tt->rank = shp->nchild;
                if (tt->rank > 4) tt->rank = 4;
                for (int d2 = 0; d2 < tt->rank; d2++)
                    tt->dims[d2] = (long)shp->child[d2].inum;
            }
            if (of) tt->off = of->inum;
            if (nb) tt->nbytes = nb->inum;
            /* roles: exact ffn leaf names only. The gate weight is BF16
             * on the real checkpoint with an F32 bias; down/up are the
             * latent projections (fp32 when present). Match the leaf
             * exactly so attn.*.wgate.weight can never impersonate the
             * ffn router. */
            if (tt->name[0]) {
                /* Qwen3.5 roles: mlp.gate.weight = router, self_attn
                 * q/k/v/o = attention, mlp.shared_expert.* = resident
                 * shared MLP. These names are unique (no other tensor
                 * ends with .mlp.gate.weight), so exact-leaf matching
                 * stays unambiguous. */
                if (name_ends(tt->name, ".mlp.gate.weight") &&
                    (tt->dtype == 0 || tt->dtype == 4 || tt->dtype == 5)) {
                    if (tl->gate[L] < 0) tl->gate[L] = k - 1;
                } else if (name_ends(tt->name, ".mlp.gate.biases") &&
                           tt->dtype == 0) {
                    if (tl->gate_bias[L] < 0) tl->gate_bias[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.gate_proj.weight")) {
                    if (tl->se_g[L] < 0) tl->se_g[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.gate_proj.scales")) {
                    if (tl->se_gs[L] < 0) tl->se_gs[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.gate_proj.biases")) {
                    if (tl->se_gb[L] < 0) tl->se_gb[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.up_proj.weight")) {
                    if (tl->se_u[L] < 0) tl->se_u[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.up_proj.scales")) {
                    if (tl->se_us[L] < 0) tl->se_us[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.up_proj.biases")) {
                    if (tl->se_ub[L] < 0) tl->se_ub[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.down_proj.weight")) {
                    if (tl->se_d[L] < 0) tl->se_d[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.down_proj.scales")) {
                    if (tl->se_ds[L] < 0) tl->se_ds[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert.down_proj.biases")) {
                    if (tl->se_db[L] < 0) tl->se_db[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert_gate.weight")) {
                    if (tl->se_r[L] < 0) tl->se_r[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert_gate.scales")) {
                    if (tl->se_rs[L] < 0) tl->se_rs[L] = k - 1;
                } else if (name_ends(tt->name,
                                     ".mlp.shared_expert_gate.biases")) {
                    if (tl->se_rb[L] < 0) tl->se_rb[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn.gate.weight") &&
                    (tt->dtype == 0 || tt->dtype == 4)) {
                    if (tl->gate[L] < 0) tl->gate[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn.gate.bias") &&
                           tt->dtype == 0) {
                    if (tl->gate_bias[L] < 0) tl->gate_bias[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn.down.weight") &&
                           tt->dtype == 0) {
                    if (tl->down[L] < 0) tl->down[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn.up.weight") &&
                           tt->dtype == 0) {
                    if (tl->up[L] < 0) tl->up[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.q_norm.weight")) {
                    if (tl->attn_qn[L] < 0) tl->attn_qn[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.kv_norm.weight")) {
                    if (tl->attn_kvn[L] < 0) tl->attn_kvn[L] = k - 1;
                } else if (name_ends(tt->name, ".attn_norm.weight") ||
                           name_ends(tt->name, ".input_layernorm.weight")) {
                    if (tl->attn_norm[L] < 0) tl->attn_norm[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn_norm.weight") ||
                           name_ends(tt->name,
                                     ".post_attention_layernorm.weight")) {
                    if (tl->ffn_norm[L] < 0) tl->ffn_norm[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wq_a.weight")) {
                    if (tl->attn_wqa[L] < 0) tl->attn_wqa[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wq_a.scale")) {
                    if (tl->attn_wqa_s[L] < 0) tl->attn_wqa_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wq_b.weight")) {
                    if (tl->attn_wqb[L] < 0) tl->attn_wqb[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wq_b.scale")) {
                    if (tl->attn_wqb_s[L] < 0) tl->attn_wqb_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wkv.weight")) {
                    if (tl->attn_wkv[L] < 0) tl->attn_wkv[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wkv.scale")) {
                    if (tl->attn_wkv_s[L] < 0) tl->attn_wkv_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_a.weight")) {
                    if (tl->attn_woa[L] < 0) tl->attn_woa[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_a.scale")) {
                    if (tl->attn_woa_s[L] < 0) tl->attn_woa_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_b.weight")) {
                    if (tl->attn_wob[L] < 0) tl->attn_wob[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_b.scale")) {
                    if (tl->attn_wob_s[L] < 0) tl->attn_wob_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_c.weight")) {
                    if (tl->attn_woc[L] < 0) tl->attn_woc[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_c.scale")) {
                    if (tl->attn_woc_s[L] < 0) tl->attn_woc_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.attn_sink") &&
                           tt->dtype == 0) {
                    if (tl->attn_sink[L] < 0) tl->attn_sink[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_attn_fn")) {
                    if (tl->hc_attn_fn[L] < 0) tl->hc_attn_fn[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_attn_base")) {
                    if (tl->hc_attn_base[L] < 0) tl->hc_attn_base[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_attn_scale")) {
                    if (tl->hc_attn_scale[L] < 0)
                        tl->hc_attn_scale[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_ffn_fn")) {
                    if (tl->hc_ffn_fn[L] < 0) tl->hc_ffn_fn[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_ffn_base")) {
                    if (tl->hc_ffn_base[L] < 0) tl->hc_ffn_base[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_ffn_scale")) {
                    if (tl->hc_ffn_scale[L] < 0)
                        tl->hc_ffn_scale[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_head_fn")) {
                    if (tl->hc_head_fn < 0) tl->hc_head_fn = k - 1;
                } else if (name_ends(tt->name, ".hc_head_base")) {
                    if (tl->hc_head_base < 0) tl->hc_head_base = k - 1;
                } else if (name_ends(tt->name, ".hc_head_scale")) {
                    if (tl->hc_head_scale < 0) tl->hc_head_scale = k - 1;
                } else if (name_ends(tt->name, ".self_attn.q_proj.weight")) {
                    if (tl->q3_q[L] < 0) tl->q3_q[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.q_proj.scales")) {
                    if (tl->q3_qs[L] < 0) tl->q3_qs[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.q_proj.biases")) {
                    if (tl->q3_qb[L] < 0) tl->q3_qb[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.k_proj.weight")) {
                    if (tl->q3_k[L] < 0) tl->q3_k[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.k_proj.scales")) {
                    if (tl->q3_ks[L] < 0) tl->q3_ks[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.k_proj.biases")) {
                    if (tl->q3_kb[L] < 0) tl->q3_kb[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.v_proj.weight")) {
                    if (tl->q3_v[L] < 0) tl->q3_v[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.v_proj.scales")) {
                    if (tl->q3_vs[L] < 0) tl->q3_vs[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.v_proj.biases")) {
                    if (tl->q3_vb[L] < 0) tl->q3_vb[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.o_proj.weight")) {
                    if (tl->q3_o[L] < 0) tl->q3_o[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.o_proj.scales")) {
                    if (tl->q3_os[L] < 0) tl->q3_os[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.o_proj.biases")) {
                    if (tl->q3_ob[L] < 0) tl->q3_ob[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.q_norm.weight")) {
                    if (tl->q3_qn[L] < 0) tl->q3_qn[L] = k - 1;
                } else if (name_ends(tt->name, ".self_attn.k_norm.weight")) {
                    if (tl->q3_kn[L] < 0) tl->q3_kn[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.conv1d.weight")) {
                    if (tl->q3_conv[L] < 0) tl->q3_conv[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.A_log")) {
                    if (tl->q3_a_log[L] < 0) tl->q3_a_log[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.dt_bias")) {
                    if (tl->q3_dt[L] < 0) tl->q3_dt[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_qkv.weight")) {
                    if (tl->q3_pqkv[L] < 0) tl->q3_pqkv[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_qkv.scales")) {
                    if (tl->q3_pqkvs[L] < 0) tl->q3_pqkvs[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_qkv.biases")) {
                    if (tl->q3_pqkvb[L] < 0) tl->q3_pqkvb[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_z.weight")) {
                    if (tl->q3_pz[L] < 0) tl->q3_pz[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_z.scales")) {
                    if (tl->q3_pzs[L] < 0) tl->q3_pzs[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_z.biases")) {
                    if (tl->q3_pzb[L] < 0) tl->q3_pzb[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_a.weight")) {
                    if (tl->q3_pa[L] < 0) tl->q3_pa[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_a.scales")) {
                    if (tl->q3_pas[L] < 0) tl->q3_pas[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_a.biases")) {
                    if (tl->q3_pab[L] < 0) tl->q3_pab[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_b.weight")) {
                    if (tl->q3_pb[L] < 0) tl->q3_pb[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_b.scales")) {
                    if (tl->q3_pbs[L] < 0) tl->q3_pbs[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.in_proj_b.biases")) {
                    if (tl->q3_pbb[L] < 0) tl->q3_pbb[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.out_proj.weight")) {
                    if (tl->q3_opa[L] < 0) tl->q3_opa[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.out_proj.scales")) {
                    if (tl->q3_opas[L] < 0) tl->q3_opas[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.out_proj.biases")) {
                    if (tl->q3_opab[L] < 0) tl->q3_opab[L] = k - 1;
                } else if (name_ends(tt->name, ".linear_attn.norm.weight")) {
                    if (tl->q3_lnorm[L] < 0) tl->q3_lnorm[L] = k - 1;
                } else if (name_ends(tt->name, ".model.norm.weight")) {
                    /* final norm before lm_head (synthetic layer) */
                    if (tl->final_norm < 0) tl->final_norm = k - 1;
                }
            }
        }
    }
    tl->t_off[tl->n_layers] = k;
    json_free(doc);
    free(buf);
    /* Typed reads (F32/BF16) require aligned offsets: misaligned ones
     * are UB that clang -O2 exploits (widened loads past the buffer,
     * nondeterministic garbage). The converter pads to 8 B; refuse
     * layouts that violate it (re-convert with the current tool). */
    for (int i = 0; i < k; i++) {
        Ds4fTrunkTensor *tt = &tl->t[i];
        if (tt->dtype == 0 && (tt->off & 3)) {
            fprintf(stderr,
                    "trunk layout: F32 tensor %s at misaligned off %ld "
                    "(re-convert: aligned packer required)\n",
                    tt->name, tt->off);
            return -1;
        }
        if (tt->dtype == 4 && (tt->off & 1)) {
            fprintf(stderr,
                    "trunk layout: BF16 tensor %s at misaligned off %ld "
                    "(re-convert: aligned packer required)\n",
                    tt->name, tt->off);
            return -1;
        }
    }
    /* the KV latent width: from the first layer's wkv tensor (derived
     * AFTER the role matching above; main sizes the KV cache with it) */
    tl->kvlat = 0;
    for (int L = 0; L < tl->n_layers && tl->kvlat < 1; L++)
        if (tl->attn_wkv[L] >= 0 && tl->t[tl->attn_wkv[L]].rank == 2)
            tl->kvlat = (int)tl->t[tl->attn_wkv[L]].dims[0];
    return 0;
}

void ds4f_topk(const float *scores, int E, int k, int *idx, float *w) {
    if (k > E) k = E;
    for (int j = 0; j < k; j++) { idx[j] = -1; w[j] = 0.0f; }
    for (int e = 0; e < E; e++) {
        float s = scores[e];
        int j = k - 1;
        while (j >= 0 && (idx[j] < 0 || s > w[j])) j--;
        if (j < k - 1) {
            for (int q = k - 2; q > j; q--) {
                idx[q + 1] = idx[q];
                w[q + 1] = w[q];
            }
            idx[j + 1] = e;
            w[j + 1] = s;
        }
    }
    /* Qwen3.5 SwitchMLP routing: softmax over ALL expert scores, then
     * renormalize the selected top-k weights to sum 1 (the reference
     * does routing_weights = softmax(gate_logits); topk; /= sum).
     * The selected w[] currently holds the raw top-k logits. */
    double mx = -1e30, sw = 0.0;
    for (int e = 0; e < E; e++)
        if ((double)scores[e] > mx) mx = (double)scores[e];
    for (int e = 0; e < E; e++)
        sw += exp((double)scores[e] - mx);
    double ss = 0.0;
    for (int j = 0; j < k && idx[j] >= 0; j++) {
        w[j] = (float)(exp((double)w[j] - mx) / sw);
        ss += (double)w[j];
    }
    if (ss > 0.0)
        for (int j = 0; j < k && idx[j] >= 0; j++) w[j] = (float)(w[j] / ss);
}

/* Parallel expert chain job (issue #5): one per topk expert, run on its
 * own thread once the slot is resident. Combine stays in selection
 * order in the caller, so results are bit-identical to the serial path. */
typedef struct {
    const Ds4fExpertLayout *el;
    const Ds4fPoolLayout *pool;
    const uint8_t *slot;
    const float *latent;
    float *out;               /* chain result, Lat floats */
    float *scratch;           /* max_rc floats, private */
    int Lat, D;
    long scratch_n;
    int64_t n_matvec, n_decode;
    int fail;
} ExpJob;

static void *exp_run(void *arg) {
    ExpJob *j = (ExpJob *)arg;
    /* calloc: the chain tail (clen < Lat) is stale after the last
     * matvec; zero-init makes the combine deterministic regardless of
     * heap layout (uninit tails caused run-to-run token variation on
     * macOS, issue #6 step 5). */
    float *cur = (float *)calloc((size_t)j->D, sizeof(float));
    float *tmp = (float *)calloc((size_t)j->D, sizeof(float));
    if (!cur || !tmp) { free(cur); free(tmp); j->fail = 1; return NULL; }
    memcpy(cur, j->latent, (size_t)j->Lat * sizeof(float));
    long clen = j->Lat;
    if (j->el->chain == 1) {
        /* Qwen3 parallel expert: silu(gate(x)) * up(x) -> down.
         * t[0]=gate_proj, t[1]=up_proj, t[2]=down_proj (manifest order).
         * gate/up: [moe_inter x H]; down: [H x moe_inter]. The chain
         * input is Lat floats (the residual/MLP input), H wide. */
        const Ds4fMoETensor *g = &j->el->t[0];
        const Ds4fMoETensor *u = &j->el->t[1];
        const Ds4fMoETensor *d = &j->el->t[2];
        if (g->rank == 2 && u->rank == 2 && d->rank == 2 &&
            g->dims[1] == j->Lat && u->dims[1] == j->Lat &&
            g->dims[0] == u->dims[0] && g->dims[0] <= j->D &&
            d->dims[1] == g->dims[0] && d->dims[0] == j->Lat) {
            long M = g->dims[0];
            float *gx = tmp;            /* gate(x), M floats */
            float *ux = tmp + M;        /* up(x), M floats -- SEPARATE
                                         * from cur: the matvec reads
                                         * x=cur while writing y, so y
                                         * must not alias the input
                                         * (row-over-row corruption). */
            float *chain = (float *)calloc((size_t)j->D, sizeof(float));
            if (!chain) { free(cur); free(tmp); j->fail = 1; return NULL; }
            if (g->fmt == 1) {
                const uint16_t *gb = g->rel_b >= 0
                    ? (const uint16_t *)(const void *)(j->slot + g->rel_b)
                    : NULL;
                const uint16_t *ub = u->rel_b >= 0
                    ? (const uint16_t *)(const void *)(j->slot + u->rel_b)
                    : NULL;
                ds4f_mlx4_matvec(
                    (const uint32_t *)(const void *)(j->slot + g->rel_v),
                    (const uint16_t *)(const void *)(j->slot + g->rel_s),
                    gb, (int)M, (int)j->Lat, cur, gx);
                ds4f_mlx4_matvec(
                    (const uint32_t *)(const void *)(j->slot + u->rel_v),
                    (const uint16_t *)(const void *)(j->slot + u->rel_s),
                    ub, (int)M, (int)j->Lat, cur, ux);
            } else {
                ds4f_mxfp4_matvec(j->slot + g->rel_v, j->slot + g->rel_s,
                                  (int)M, (int)j->Lat, g->bsize, cur, gx,
                                  j->scratch);
                ds4f_mxfp4_matvec(j->slot + u->rel_v, j->slot + u->rel_s,
                                  (int)M, (int)j->Lat, u->bsize, cur, ux,
                                  j->scratch);
            }
            j->n_matvec += 2;
            j->n_decode += 2 * M * j->Lat;
            /* silu(gate(x)) * up(x) -> chain, M floats */
            for (long i = 0; i < M; i++) {
                float s = gx[i];
                float sig = 1.0f / (1.0f + expf(-s));
                chain[i] = s * sig * ux[i];
            }
            if (getenv("DS4F_NAN_PROBE")) {
                pthread_mutex_lock(&_expdump_mu);
                if (!_exp106_dumped) {
                    long eidx = j->el - j->pool->exp;
                    int eid = (int)(eidx % j->pool->n_experts);
                FILE *cf = fopen("/tmp/q35-eng-chain.bin", "wb");
                if (cf) {
                    fwrite(&eid, sizeof(int), 1, cf);
                    fwrite(chain, sizeof(float), (size_t)M, cf);
                    fclose(cf);
                }
                FILE *gf = fopen("/tmp/q35-eng-gateup.bin", "wb");
                if (gf) {
                    fwrite(&eid, sizeof(int), 1, gf);
                    fwrite(gx, sizeof(float), (size_t)M, gf);
                    fwrite(ux, sizeof(float), (size_t)M, gf);
                    fclose(gf);
                }
                fprintf(stderr, "[expdump] expert %d\n", eid);
                {
                    FILE *lf = fopen("/tmp/q35-eng-latent.bin", "wb");
                    if (lf) {
                        fwrite(j->latent, sizeof(float),
                               (size_t)j->Lat, lf);
                        fclose(lf);
                    }
                }
                {
                    /* first gate weight words + scales for offset check */
                    const uint32_t *w0 = (const uint32_t *)(const void *)
                        (j->slot + g->rel_v);
                    const uint16_t *s0 = (const uint16_t *)(const void *)
                        (j->slot + g->rel_s);
                    const uint16_t *b0 = g->rel_b >= 0
                        ? (const uint16_t *)(const void *)(j->slot + g->rel_b)
                        : NULL;
                    fprintf(stderr, "[expw] rel_v=%ld rel_s=%ld "
                            "dims=[%ld,%ld] Lat=%d "
                            "w0=%08x %08x %08x %08x s0=%04x %04x %04x %04x\n",
                            (long)g->rel_v, (long)g->rel_s,
                            (long)g->dims[0], (long)g->dims[1], j->Lat,
                            w0[0], w0[1], w0[2], w0[3],
                            s0[0], s0[1], s0[2], s0[3]);
                    /* decode row 0 elements 0..15 with the engine's own
                     * pointers: q*s+b, group 0 */
                    for (int c = 0; c < 16; c++) {
                        int q = (int)((w0[c >> 3] >> (4 * (c & 7))) & 0xFu);
                        float s, bb = 0.0f;
                        uint32_t sb = (uint32_t)s0[c / 64] << 16;
                        memcpy(&s, &sb, 4);
                        if (b0) {
                            uint32_t bb2 = (uint32_t)b0[c / 64] << 16;
                            memcpy(&bb, &bb2, 4);
                        }
                        fprintf(stderr, "[expw] row0[%d] q=%d s=%.6g b=%.6g "
                                "W=%.6g\n", c, q, (double)s, (double)bb,
                                (double)((float)q * s + bb));
                    }
                }
                _exp106_dumped = 1;
                }
                pthread_mutex_unlock(&_expdump_mu);
            }
            if (getenv("DS4F_NAN_PROBE") && j->latent &&
                j->latent[0] != j->latent[0]) { /* NaN latent? */
            }
            if (getenv("DS4F_DEBUG_CHAIN")) {
                double g2 = 0.0, u2 = 0.0, c2 = 0.0;
                for (long i = 0; i < M; i++) {
                    g2 += (double)gx[i] * gx[i];
                    u2 += (double)ux[i] * ux[i];
                    c2 += (double)chain[i] * chain[i];
                }
                fprintf(stderr, "[chain] gate-rms %.6g up-rms %.6g "
                        "chain-rms %.6g\n", sqrt(g2 / M), sqrt(u2 / M),
                        sqrt(c2 / M));
            }
            /* down(chain) -> cur (Lat floats) */
            if (d->fmt == 1) {
                const uint16_t *db = d->rel_b >= 0
                    ? (const uint16_t *)(const void *)(j->slot + d->rel_b)
                    : NULL;
                ds4f_mlx4_matvec(
                    (const uint32_t *)(const void *)(j->slot + d->rel_v),
                    (const uint16_t *)(const void *)(j->slot + d->rel_s),
                    db, (int)j->Lat, (int)M, chain, tmp);
            } else {
                ds4f_mxfp4_matvec(j->slot + d->rel_v, j->slot + d->rel_s,
                                  (int)j->Lat, (int)M, d->bsize, chain, tmp,
                                  j->scratch);
            }
            j->n_matvec++;
            j->n_decode += j->Lat * M;
            memcpy(cur, tmp, (size_t)j->Lat * sizeof(float));
            clen = j->Lat;
            free(chain);
            if (getenv("DS4F_NAN_PROBE") && _exp106_dumped &&
                !_exp106_out_dumped) {
                FILE *of = fopen("/tmp/q35-eng-expout.bin", "wb");
                if (of) {
                    fwrite(cur, sizeof(float), (size_t)j->Lat, of);
                    fclose(of);
                }
                _exp106_out_dumped = 1;
            }
        } else {
            /* shape mismatch: fall through to the sequential path so
             * the run still completes (garbage, but not a crash) */
            for (int ti = 0; ti < j->el->n; ti++) {
                const Ds4fMoETensor *t = &j->el->t[ti];
                if (t->rank != 2) continue;
                long R = t->dims[0], C = t->dims[1];
                if (C != clen || R > j->D) continue;
                if (R * C > j->scratch_n) { j->fail = 1; break; }
                if (t->fmt == 1) {
                    const uint16_t *biases = t->rel_b >= 0
                        ? (const uint16_t *)(const void *)(j->slot + t->rel_b)
                        : NULL;
                    ds4f_mlx4_matvec(
                        (const uint32_t *)(const void *)(j->slot + t->rel_v),
                        (const uint16_t *)(const void *)(j->slot + t->rel_s),
                        biases, (int)R, (int)C, cur, tmp);
                } else {
                    ds4f_mxfp4_matvec(j->slot + t->rel_v, j->slot + t->rel_s,
                                      (int)R, (int)C, t->bsize, cur, tmp,
                                      j->scratch);
                }
                j->n_matvec++;
                j->n_decode += R * C;
                memcpy(cur, tmp, (size_t)R * sizeof(float));
                clen = R;
            }
        }
    } else {
    for (int ti = 0; ti < j->el->n; ti++) {
        const Ds4fMoETensor *t = &j->el->t[ti];
        if (t->rank != 2) continue;
        long R = t->dims[0], C = t->dims[1];
        if (C != clen || R > j->D) continue;
        if (R * C > j->scratch_n) { j->fail = 1; break; }
        if (t->fmt == 1) {
            const uint16_t *biases = t->rel_b >= 0
                ? (const uint16_t *)(const void *)(j->slot + t->rel_b)
                : NULL;
            ds4f_mlx4_matvec(
                (const uint32_t *)(const void *)(j->slot + t->rel_v),
                (const uint16_t *)(const void *)(j->slot + t->rel_s),
                biases, (int)R, (int)C, cur, tmp);
        } else {
            ds4f_mxfp4_matvec(j->slot + t->rel_v, j->slot + t->rel_s,
                              (int)R, (int)C, t->bsize, cur, tmp,
                              j->scratch);
        }
        j->n_matvec++;
        j->n_decode += R * C;
        memcpy(cur, tmp, (size_t)R * sizeof(float));
        clen = R;
    }
    }
    long ncopy = j->Lat < clen ? j->Lat : clen;
    memcpy(j->out, cur, (size_t)ncopy * sizeof(float));
    /* the caller combines over all Lat elements: the tail must be
     * zero, not malloc garbage (nondeterministic dumps otherwise) */
    if (ncopy < j->Lat)
        memset(j->out + ncopy, 0,
               (size_t)(j->Lat - ncopy) * sizeof(float));
    free(cur);
    free(tmp);
    return NULL;
}

static float bf16_f(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* read one element of a small F32/BF16 tensor by index */
static float hc_elem(const Ds4fTrunkTensor *t, const uint8_t *tr, long i) {
    const uint8_t *p = tr + t->off;
    if (t->dtype == 4) {
        const uint16_t *b = (const uint16_t *)(const void *)p;
        return bf16_f(b[i]);
    }
    const float *f = (const float *)(const void *)p;
    return f[i];
}

/* Sinkhorn-Knopp: B = doubly stochastic projection of exp(Btilde)
 * (paper eq. 8), 20 row/col normalizations. */
/* in-place RMSNorm with BF16 weights (the attn.c counterpart;
 * applies the checkpoint's layer norms, eps 1e-6) */
static void rmsnorm_moe(const uint16_t *w, int dim, float *x) {
    double ss = 0.0;
    for (int i = 0; i < dim; i++) ss += (double)x[i] * x[i];
    float r = sqrtf((float)(ss / (double)dim) + 1e-6f);
    for (int i = 0; i < dim; i++) {
        uint32_t bits = (uint32_t)w[i] << 16;  /* bf16 = top half */
        float bv;
        memcpy(&bv, &bits, 4);
        x[i] = x[i] / r * bv;
    }
}

static void sinkhorn(const float *btilde, int n, float *B) {
    double M[64];
    for (int i = 0; i < n * n; i++) M[i] = exp((double)btilde[i]);
    for (int it = 0; it < 20; it++) {
        for (int r = 0; r < n; r++) {
            double s = 0.0;
            for (int c = 0; c < n; c++) s += M[r * n + c];
            if (s > 0.0)
                for (int c = 0; c < n; c++) M[r * n + c] /= s;
        }
        for (int c = 0; c < n; c++) {
            double s = 0.0;
            for (int r = 0; r < n; r++) s += M[r * n + c];
            if (s > 0.0)
                for (int r = 0; r < n; r++) M[r * n + c] /= s;
        }
    }
    for (int i = 0; i < n * n; i++) B[i] = (float)M[i];
}

/* mHC params (DeepSeek-V4 paper eq. 1/3-8). fn = [(n_hc*(2+n_hc)) x
 * (n_hc*H)]: rows [0,n_hc) W_pre, [n_hc,2n_hc) W_post, then W_res.
 * base = [n_hc*(2+n_hc)] (S in the same row order), scale = [3]
 * (alpha_pre, alpha_post, alpha_res). F32/BF16. */
int ds4f_hc_params(const Ds4fTrunkLayout *tl, int fn_i, int base_i,
                   int sc_i, const uint8_t *tr, int H,
                   const float *state, int *n_hc_out,
                   float *A, float *C, float *B) {
    if (fn_i < 0 || base_i < 0 || sc_i < 0) return 0;
    const Ds4fTrunkTensor *fn = &tl->t[fn_i];
    const Ds4fTrunkTensor *bs = &tl->t[base_i];
    const Ds4fTrunkTensor *al = &tl->t[sc_i];
    long rows = fn->dims[0], cols = fn->dims[1];
    if (fn->rank != 2 || cols % H != 0) {
        fprintf(stderr, "hc: fn shape [%ld x %ld] unsupported "
                        "(want [n*(2+n) x n*%d])\n", rows, cols, H);
        return -1;
    }
    int nhc = (int)(cols / H);
    if (rows != (long)nhc * (2 + nhc)) {
        fprintf(stderr, "hc: fn rows %ld unsupported (n_hc=%d wants %d)\n",
                rows, nhc, nhc * (2 + nhc));
        return -1;
    }
    if ((fn->dtype != 0 && fn->dtype != 4) ||
        (bs->dtype != 0 && bs->dtype != 4) ||
        (al->dtype != 0 && al->dtype != 4)) {
        fprintf(stderr, "hc: dtype unsupported (F32/BF16 only)\n");
        return -1;
    }
    /* xhat = RMSNorm(vec(state)) over the whole n_hc*H stream */
    long total = (long)nhc * H;
    double ss = 0.0;
    for (long i = 0; i < total; i++) ss += (double)state[i] * state[i];
    float r = sqrtf((float)(ss / (double)total) + 1e-6f);

    for (int j = 0; j < nhc; j++) {
        double dp = 0.0, dq = 0.0;
        for (long i = 0; i < total; i++) {
            float xh = state[i] / r;
            dp += (double)xh * hc_elem(fn, tr, (long)j * cols + i);
            dq += (double)xh * hc_elem(fn, tr, (long)(nhc + j) * cols + i);
        }
        float a_pre = hc_elem(al, tr, 0) * (float)dp + hc_elem(bs, tr, j);
        float a_post = hc_elem(al, tr, 1) * (float)dq +
                       hc_elem(bs, tr, nhc + j);
        A[j] = 1.0f / (1.0f + expf(-a_pre));
        C[j] = 2.0f / (1.0f + expf(-a_post));
    }
    float btilde[64];
    for (int r2 = 0; r2 < nhc; r2++) {
        for (int c2 = 0; c2 < nhc; c2++) {
            double d = 0.0;
            for (long i = 0; i < total; i++)
                d += (double)(state[i] / r) * hc_elem(
                    fn, tr, (long)(2 * nhc + r2 * nhc + c2) * cols + i);
            btilde[r2 * nhc + c2] =
                hc_elem(al, tr, 2) * (float)d +
                hc_elem(bs, tr, 2 * nhc + r2 * nhc + c2);
        }
    }
    sinkhorn(btilde, nhc, B);
    *n_hc_out = nhc;
    return nhc;
}

void ds4f_hc_combine(int n_hc, int H, const float *A, const float *state,
                     float *x_in) {
    for (int i = 0; i < H; i++) {
        float s = 0.0f;
        for (int j = 0; j < n_hc; j++) s += A[j] * state[j * H + i];
        x_in[i] = s;
    }
}

int ds4f_moe_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                  const uint8_t *tr, const Ds4fPoolLayout *pl,
                  const uint8_t *const *es, const int *sel, const float *wsel,
                  float *state, float *scratch, long scratch_n,
                  float *const *job_scratch,
                  int64_t *n_matvec, int64_t *n_decode) {
    int H = cfg->hidden, Lat = cfg->latent, M = cfg->moe_inter;
    int D = H > Lat ? H : Lat;
    if (M > D) D = M;
    if (D < 1) return -1;

    /* mHC (issue #6 step 6): F_ffn sees x_in = A·vec(X); the update is
     * new[j*H+i] = sum_k B[j][k]*orig[k*H+i] + C[j]*F[i]. orig holds
     * the pre-ffn streams; the RMS-rescale below is the no-hc
     * fallback. */
    float A[8], C[8], B[64];
    int nhc = 1;
    int hc_ok = tl ? ds4f_hc_params(tl, tl->hc_ffn_fn[L], tl->hc_ffn_base[L],
                                    tl->hc_ffn_scale[L], tr, H, state,
                                    &nhc, A, C, B) : 0;
    if (hc_ok < 0) return -1;
    float *orig = NULL, *xin = NULL;
    if (hc_ok) {
        orig = (float *)malloc((size_t)nhc * H * sizeof(float));
        xin = (float *)malloc((size_t)H * sizeof(float));
        if (!orig || !xin) { free(orig); free(xin); return -1; }
        memcpy(orig, state, (size_t)nhc * H * sizeof(float));
        ds4f_hc_combine(nhc, H, A, state, xin);
        /* the real model's post_attention_layernorm (was never
         * applied -- the raw state fed the router/experts) */
        if (tl->ffn_norm[L] >= 0 && !getenv("DS4F_NO_NORMS"))
            rmsnorm_moe((const uint16_t *)(const void *)(
                            tr + tl->t[tl->ffn_norm[L]].off),
                        H, xin);
    } else {
        /* Qwen3.5 (no mHC): x = x + mlp(post_attention_layernorm(x)).
         * The experts project the NORMED input; the residual adds to
         * the raw state. */
        xin = (float *)malloc((size_t)H * sizeof(float));
        if (!xin) return -1;
        memcpy(xin, state, (size_t)H * sizeof(float));
        if (tl->ffn_norm[L] >= 0 && !getenv("DS4F_NO_NORMS"))
            rmsnorm_moe((const uint16_t *)(const void *)(
                            tr + tl->t[tl->ffn_norm[L]].off),
                        H, xin);
    }

    /* Entry RMS: the F-rescale target (hc) or the RMS-rescale fallback
     * target (no hc). */
    double ss_in = 0.0;
    if (hc_ok) {
        for (int i = 0; i < H; i++)
            ss_in += (double)xin[i] * xin[i];
    } else {
        for (int i = 0; i < H; i++)
            ss_in += (double)state[i] * state[i];
    }
    float rms_in = sqrtf((float)(ss_in / (double)H));

    float *latent = (float *)calloc((size_t)D, sizeof(float));
    float *cur    = (float *)calloc((size_t)D, sizeof(float));
    float *out    = (float *)calloc((size_t)D, sizeof(float));
    float *acc    = (float *)calloc((size_t)D, sizeof(float));
    if (!latent || !cur || !out || !acc) {
        free(latent); free(cur); free(out); free(acc);
        return -1;
    }
    (void)cur;              /* expert chains now run in worker threads */
    (void)scratch;          /* job 0 uses the caller's warm buffer */

    /* latent = W_down * x_in (identity when absent / mismatched) */
    int di = tl ? tl->down[L] : -1, ui = tl ? tl->up[L] : -1;
    int did_ok = 0;
    if (di >= 0 && tl->t[di].dtype == 0 && tl->t[di].rank == 2) {
        long R = tl->t[di].dims[0], C = tl->t[di].dims[1];
        if (C == H && R <= D) {
            ds4f_f32_matvec((const float *)(const void *)(tr + tl->t[di].off),
                            (int)R, (int)C, xin, latent);
            (*n_matvec)++;
            did_ok = 1;
        }
    }
    if (!did_ok) {
        for (int i = 0; i < Lat && i < H; i++) latent[i] = xin[i];
        for (int i = H; i < Lat; i++) latent[i] = 0.0f;
    }

    /* Parallel expert chains (issue #5): each topk expert's w1->w2->w3
     * chain is independent once its slot is resident, so run them on
     * separate threads (up to topk) and combine in j order -- the
     * combine order is unchanged, so results are bit-identical to the
     * serial path. */
    pthread_t th[64];
    ExpJob job[64];
    int njob = 0;
    for (int j = 0; j < cfg->topk; j++) {
        if (!es[j]) continue;
        ExpJob *jb = &job[njob];
        memset(jb, 0, sizeof *jb);
        jb->el = &pl->exp[(size_t)L * pl->n_experts + sel[j]];
        jb->pool = pl;
        jb->slot = es[j];
        jb->latent = latent;
        jb->out = (float *)calloc((size_t)Lat, sizeof(float));
        /* job 0 reuses the caller's warm scratch; the rest come from
         * the caller's pool -- never malloc per call (page faults). */
        jb->scratch = (njob == 0) ? scratch : job_scratch[njob - 1];
        if (!jb->out || !jb->scratch) {
            free(jb->out);           /* scratch is borrowed, never freed */
            free(latent); free(cur); free(out); free(acc);
            return -1;
        }
        jb->Lat = Lat;
        jb->D = D;
        jb->scratch_n = scratch_n;
        if (pthread_create(&th[njob], NULL, exp_run, jb) != 0) {
            free(jb->out); free(jb->scratch);
            free(latent); free(cur); free(out); free(acc);
            return -1;
        }
        njob++;
    }
    for (int j = 0; j < njob; j++)
        pthread_join(th[j], NULL);
    for (int j = 0; j < njob; j++) {
        ExpJob *jb = &job[j];
        if (jb->fail) {
            for (int q = 0; q < njob; q++)
                free(job[q].out);    /* scratch is borrowed */
            free(latent); free(cur); free(out); free(acc);
            return -1;
        }
        *n_matvec += jb->n_matvec;
        *n_decode += jb->n_decode;
    }
    /* combine in selection order: acc[i] += wsel[sel order] * chain[i] */
    {
        int sj = 0;
        for (int j = 0; j < cfg->topk; j++) {
            if (!es[j]) continue;
            ExpJob *jb = &job[sj++];
            if (getenv("DS4F_NAN_PROBE") && L == 0 && !_exp106_dumped) {
                FILE *ef = fopen("/tmp/q35-eng-exp106.bin", "wb");
                if (ef) {
                    fwrite(jb->out, sizeof(float), (size_t)Lat, ef);
                    fclose(ef);
                }
                _exp106_dumped = 1;
            }
            if (getenv("DS4F_DEBUG_CHAIN") && L < 3) {
                double o2 = 0.0;
                for (int i = 0; i < Lat; i++)
                    o2 += (double)jb->out[i] * jb->out[i];
                fprintf(stderr, "[wsel] L%d j%d w=%.6g out-rms %.6g\n",
                        L, j, (double)wsel[j], sqrt(o2 / Lat));
            }
            for (int i = 0; i < Lat; i++)
                acc[i] += wsel[j] * jb->out[i];
        }
    }
    for (int j = 0; j < njob; j++)
        free(job[j].out);            /* scratch is borrowed */

    /* shared expert (Qwen3.5): dense every-token MLP added after the
     * routed experts -- acc += sigmoid(shared_expert_gate(xin)) *
     * down(silu(gate(xin)) * up(xin)). Same mlx4 triplets. */
    if (tl->se_g[L] >= 0 && tl->se_u[L] >= 0 && tl->se_d[L] >= 0 &&
        tl->se_gs[L] >= 0 && tl->se_us[L] >= 0 && tl->se_ds[L] >= 0 &&
        tl->se_r[L] >= 0 && tl->se_rs[L] >= 0 && tl->se_rb[L] >= 0) {
        const Ds4fTrunkTensor *t0 = &tl->t[tl->se_g[L]];
        long M = t0->dims[0];        /* 512 (moe_intermediate_size) */
        float *sg = (float *)malloc((size_t)(M * 3 + H + 1) * sizeof(float));
        if (sg) {
            float *upv = sg + M;
            float *chain = sg + 2 * M;
            float *sout = sg + 3 * M;      /* H floats */
            float sgate[1];
            ds4f_mlx4_matvec((const uint32_t *)(const void *)
                                 (tr + tl->t[tl->se_g[L]].off),
                             (const uint16_t *)(const void *)
                                 (tr + tl->t[tl->se_gs[L]].off),
                             (const uint16_t *)(const void *)
                                 (tr + tl->t[tl->se_gb[L]].off),
                             (int)M, (int)H, xin, sg);
            ds4f_mlx4_matvec((const uint32_t *)(const void *)
                                 (tr + tl->t[tl->se_u[L]].off),
                             (const uint16_t *)(const void *)
                                 (tr + tl->t[tl->se_us[L]].off),
                             (const uint16_t *)(const void *)
                                 (tr + tl->t[tl->se_ub[L]].off),
                             (int)M, (int)H, xin, upv);
            for (long i = 0; i < M; i++) {
                float s = sg[i];
                float sig = 1.0f / (1.0f + expf(-s));
                chain[i] = s * sig * upv[i];     /* silu(gate)*up */
            }
            ds4f_mlx4_matvec((const uint32_t *)(const void *)
                                 (tr + tl->t[tl->se_d[L]].off),
                             (const uint16_t *)(const void *)
                                 (tr + tl->t[tl->se_ds[L]].off),
                             (const uint16_t *)(const void *)
                                 (tr + tl->t[tl->se_db[L]].off),
                             (int)H, (int)M, chain, sout);
            ds4f_mlx4_matvec((const uint32_t *)(const void *)
                                 (tr + tl->t[tl->se_r[L]].off),
                             (const uint16_t *)(const void *)
                                 (tr + tl->t[tl->se_rs[L]].off),
                             (const uint16_t *)(const void *)
                                 (tr + tl->t[tl->se_rb[L]].off),
                             1, (int)H, xin, sgate);
            {
                float sg2 = 1.0f / (1.0f + expf(-sgate[0]));
                for (int i = 0; i < H; i++)
                    acc[i] += sg2 * sout[i];
                if (getenv("DS4F_DEBUG_CHAIN") && L < 3) {
                    double o2 = 0.0;
                    for (int i = 0; i < H; i++)
                        o2 += (double)sout[i] * sout[i];
                    fprintf(stderr, "[shared] L%d gate=%.4f out-rms %.6g\n",
                            L, sg2, sqrt(o2 / (double)H));
                }
            }
            free(sg);
        }
    }

    if (getenv("DS4F_NAN_PROBE") && L < 3) {
        double a2 = 0.0, i2 = 0.0;
        for (int i = 0; i < H; i++) {
            a2 += (double)acc[i] * acc[i];
            i2 += (double)xin[i] * xin[i];
        }
        fprintf(stderr, "[moeacc] L%d xin-rms %.6g acc-rms %.6g "
                "gain %.6g\n", L, sqrt(i2 / H), sqrt(a2 / H),
                sqrt(a2 / H) / (sqrt(i2 / H) + 1e-30f));
        if (L == 0 && !_moeacc0_dumped) {
            FILE *af = fopen("/tmp/q35-eng-moeacc0.bin", "wb");
            if (af) {
                fwrite(acc, sizeof(float), (size_t)H, af);
                fclose(af);
            }
            FILE *xf = fopen("/tmp/q35-eng-xin0.bin", "wb");
            if (xf) {
                fwrite(xin, sizeof(float), (size_t)H, xf);
                fclose(xf);
            }
            _moeacc0_dumped = 1;
        }
    }

    /* state = state + W_up * acc (mHC: streams = B*orig + C*F) */
    int up_ok = 0;
    if (ui >= 0 && tl->t[ui].dtype == 0 && tl->t[ui].rank == 2) {
        long R = tl->t[ui].dims[0], Uc = tl->t[ui].dims[1];
        if (Uc == Lat && R <= D) {
            ds4f_f32_matvec((const float *)(const void *)(tr + tl->t[ui].off),
                            (int)R, (int)Uc, acc, out);
            (*n_matvec)++;
            /* a short up matvec (R < H) leaves out[R..H) untouched:
             * zero it so the update is deterministic (the tail must
             * not be malloc garbage) */
            if (R < H) memset(out + R, 0, (size_t)(H - R) * sizeof(float));
            if (getenv("DS4F_NAN_PROBE") && L < 3) {
                double o2 = 0.0, i2 = 0.0;
                for (int i = 0; i < H; i++) {
                    o2 += (double)out[i] * out[i];
                    i2 += (double)xin[i] * xin[i];
                }
                fprintf(stderr, "[moeout] L%d xin-rms %.6g out-rms %.6g "
                        "gain %.6g\n", L, sqrt(i2 / H), sqrt(o2 / H),
                        sqrt(o2 / H) / (sqrt(i2 / H) + 1e-30f));
            }
            if (hc_ok) {
                /* F-rescale: the approximate expert reads amplify (the
                 * real model bounds F by training). Rescale the up
                 * output to the layer-input RMS so the mHC update is
                 * finite; the real fix is the exact MLA/MoE reads. */
                double s2 = 0.0;
                for (int i = 0; i < H; i++) s2 += (double)out[i] * out[i];
                float rms_f = sqrtf((float)(s2 / (double)H)) + 1e-30f;
                float gain = rms_in / rms_f;
                if (gain > 0.0f && gain < 1e30f)
                    for (int i = 0; i < H; i++) out[i] *= gain;
                /* new[j*H+i] = sum_k B[j][k]*orig[k*H+i] + C[j]*out[i] */
                for (int i = 0; i < H; i++) {
                    float mix[8];
                    for (int j = 0; j < nhc; j++) {
                        float s = 0.0f;
                        for (int k = 0; k < nhc; k++)
                            s += B[j * nhc + k] * orig[k * H + i];
                        if (getenv("DS4F_NO_B_MIX"))
                            s = orig[j * H + i];
                        mix[j] = s + C[j] * out[i];
                    }
                    for (int j = 0; j < nhc; j++) state[j * H + i] = mix[j];
                }
                /* state-rescale to the layer-input RMS (see attn.c);
                 * DS4F_STATE_RMS_TARGET overrides the target (same
                 * dead-state-at-embed-scale issue) */
                {
                    double t2 = 0.0;
                    for (int i = 0; i < nhc * H; i++)
                        t2 += (double)state[i] * state[i];
                    float rms_s =
                        sqrtf((float)(t2 / (double)(nhc * H))) + 1e-30f;
                    float sgain = rms_in / rms_s;
                    const char *tgt = getenv("DS4F_STATE_RMS_TARGET");
                    if (tgt) {
                        float t = (float)atof(tgt);
                        if (t > 0.0f) sgain = t / rms_s;
                    }
                    if (sgain > 0.0f && sgain < 1e30f)
                        for (int i = 0; i < nhc * H; i++)
                            state[i] *= sgain;
                }
            } else {
                for (int i = 0; i < H; i++) state[i] += out[i];
            }
            up_ok = 1;
        } else {
            fprintf(stderr, "moe: up[%d] shape [%ld x %ld] unsupported "
                            "(lat=%d d=%d) -- identity fallback\n",
                    ui, (long)tl->t[ui].dims[0], (long)tl->t[ui].dims[1],
                    Lat, D);
        }
    } else if (ui >= 0) {
        fprintf(stderr, "moe: up[%d] dtype/rank unsupported\n", ui);
    }
    if (!up_ok) {
        if (hc_ok) {
            /* F identity: out = x_in; streams = B*orig + C*x_in */
            for (int i = 0; i < H; i++) {
                float mix[8];
                for (int j = 0; j < nhc; j++) {
                    float s = 0.0f;
                    for (int k = 0; k < nhc; k++)
                        s += B[j * nhc + k] * orig[k * H + i];
                    mix[j] = s + C[j] * xin[i];
                }
                for (int j = 0; j < nhc; j++) state[j * H + i] = mix[j];
            }
            /* state-rescale to the layer-input RMS (same as the up_ok
             * branch; the fallback must bound the state too) */
            {
                double t2 = 0.0;
                for (int i = 0; i < nhc * H; i++)
                    t2 += (double)state[i] * state[i];
                float rms_s =
                    sqrtf((float)(t2 / (double)(nhc * H))) + 1e-30f;
                float sgain = rms_in / rms_s;
                if (sgain > 0.0f && sgain < 1e30f)
                    for (int i = 0; i < nhc * H; i++)
                        state[i] *= sgain;
            }
        } else {
            for (int i = 0; i < H && i < Lat; i++) state[i] += acc[i];
        }
    }

    /* Activation bounding. With mHC tensors the stream update above
     * already is the bounded residual (B doubly stochastic, C <= 2).
     * Without them, the Qwen3.5 path applies the post_attention norm
     * (stable by construction -- the reference needs no rescale). The
     * RMS-rescale was a DS-V4 mHC-era fallback; keep it only when the
     * fixture explicitly asks (DS4F_RMS_RESCALE=1), otherwise the
     * residual is exact. */
    if (!hc_ok && getenv("DS4F_RMS_RESCALE")) {
        double ss = 0.0;
        for (int i = 0; i < H; i++) {
            float v = state[i];
            ss += (double)v * v;
        }
        float rms_out = sqrtf((float)(ss / (double)H)) + 1e-30f;
        float gain = rms_in / rms_out;
        if (getenv("DS4F_NAN_PROBE") && (L < 5 || L == 6 || L == 8 ||
                                         L == 12 || L == 16 || L == 20 ||
                                         L == 24 || L == 27))
            fprintf(stderr, "[moe] L%d hc_ok=%d rms_in %.6g rms_out %.6g "
                    "gain %.6g\n", L, hc_ok, rms_in, rms_out, gain);
        if (gain > 0.0f && gain < 1e30f)
            for (int i = 0; i < H; i++) state[i] *= gain;
    }

    free(orig);
    free(xin);
    free(latent); free(cur); free(out); free(acc);
    return 0;
}
