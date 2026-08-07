# disk-qwen35bA3B

Disk-streaming MoE inference **structure** in portable C99 for
**Qwen3.5-35B-A3B** (35B total, ~3B active per token; 256 experts,
top-8; hybrid linear/full attention). No BLAS, no framework, no GPU,
no dependencies beyond libc + pthreads.

> **HARD RULE: never load the full model into RAM/VRAM.** The
> disk-streaming design is the point — the ~21.7 GB of weights stay on
> disk and only the 8 routed experts per token are fetched. Full-model
> reference runs (e.g. loading the checkpoint wholesale into mlx/PyTorch
> for comparison) are forbidden on development machines: they overload
> memory. Compare against the reference only through the streaming path
> or bit-fidelity probes (see `docs/fidelity-audit.md`).

The systems architecture mirrors the family proven by
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c),
[Colibrì](https://github.com/JustVugg/colibri), and the author's
[ds4f-disk](https://github.com/rhCat/ds4f-disk): a tiny in-RAM pointer
map, a contiguous packed trunk, and policy-controlled expert streaming.
The A3B's 3%-of-params-active routing (8/256 experts per token) is
exactly what the structure was built for.

## Quantized footprint (real HF file sizes, Feb 2026)

| Variant | On disk | vs BF16 | repo |
|---|---|---|---|
| BF16 | ~75 GB | 100% | `Qwen/Qwen3.5-35B-A3B` |
| **FP8** | **~37.6 GB** | **50%** | `Qwen/Qwen3.5-35B-A3B-FP8` |
| **NVFP4** | **~25.4 GB** | **34%** | `txn545/Qwen3.5-35B-A3B-NVFP4` |

FP8 keeps 1.5B params in BF16 (embed/lm_head/conv1d/gates/shared-expert
gate — `modules_to_not_convert`). The engine targets the FP8 pool.

## Architecture (real config, `qwen3_5_moe`)

- 40 layers: **30 linear-attention (SSM-style: conv1d k=4, in_proj_a/b,
  per-head recurrent state) + 10 full GQA** (16 heads, 2 kv, head_dim
  256), `full_attention_interval: 4`
- **256 experts, top-8**, moe_inter 512, `norm_topk_prob`
- **1 shared expert** per layer (moe_inter 512)
- MTP block (1 layer), mrope (partial_rotary 0.25), vocab 248320,
  262144 max pos
- Multimodal wrapper: vision tower (27 layers, hidden 1152) — **text
  subset only**; vision/video tensors are stripped at convert

## Memory profile

| Bucket | Size | Why |
|---|---|---|
| **Resident (RAM)** | ~4 GB | full-attn q/k/v/o (10 layers), shared expert, gates, embed+lm_head, norms |
| **Streamed (disk)** | ~33 GB FP8 pool | 256-experts, 8 touched per token (3.1%) |
| **KV** | full-attn: ~40 KB/token → ~10.7 GB @ 262K; linear-attn: **fixed state, O(1)** | the hybrid's context is nearly free |
| **Per-token stream** | ~0.8–1 GB/token (mxfp4 experts) | 40 layers × 8 experts × 3 proj (2048×512) |

Serving reality: 128 GB CPU box hosts the FP8 pool + fat resident set
with context to spare; on 2× DGX Spark the whole thing is resident with
~90 GB/machine free for KV.

## Layout (target)

```
include/ds4f/ds4f.h     public API + invariants (forked from ds4f-disk)
src/cfg.c               config reader, qwen3_5_moe keys
src/st.c                safetensors index -> pointer map
src/trunk.c             packed trunk: pin + ring + async prefetch
src/cache.c             routed-expert cache: 3-phase fetch
src/router.c            top-8 router (norm_topk_prob)
src/attn.c              full-GQA path (10 layers) + linear-attn
                        (conv1d + recurrent state, 30 layers)
src/moe.c               Qwen3.5 expert chain (gate/up/down, silu,
                        shared expert)
src/kernels.c           scalar mxfp4 / fp8 / bf16 kernels
src/head.c              embed + lm_head
src/tokenizer.c         Qwen3.5 tokenizer (vocab 248320)
tools/convert-ds4f.py   HF checkpoint -> ds4f layout (qwen3_5 arch)
tools/check-download.py shard existence/size/dtype verification
tools/make-fixture.c    synthetic A3B fixture (no real weights)
tests/                  weightless gate suite + fixture e2e
```

## Status

- [x] Config + sizes grounded from HF (base / FP8 / NVFP4)
- [x] `tools/check-download.py` — shard verify before convert
- [ ] Converter arch support: `model.language_model.layers.N.` naming,
      n_shared=1, 256 experts, linear-attn layer tensors, strip vision
- [ ] Full-GQA attention path (attn.c) — 10 layers
- [ ] Linear-attention path (conv1d + recurrent state) — 30 layers
      (the big new piece)
- [ ] Qwen3.5 expert chain (moe.c): gate/up/down, shared expert, top-8
- [ ] Head/embed + tokenizer (vocab 248320)
- [ ] Fixture gate e2e (A3B-shaped synthetic, deterministic)
- [ ] Real checkpoint (FP8) on a 128 GB box

## Build and verify (inherited from ds4f-disk, adapted)

```sh
make            # ds4f, pack-trunk, make-fixture
make test       # gate suite (weightless) + fixture e2e
```

## License

MIT. Model weights are not included and are subject to their own
licenses (Qwen3.5-35B-A3B is Apache-2.0).
