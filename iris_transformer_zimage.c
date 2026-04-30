/*
 * Z-Image S3-DiT Transformer Implementation
 *
 * Implements the Z-Image-Turbo (6B) Scalable Single-Stream DiT architecture.
 *
 * Architecture:
 * - 2 noise_refiner blocks (modulated, image-only self-attention)
 * - 2 context_refiner blocks (unmodulated, text-only self-attention)
 * - 30 main transformer blocks (modulated, full self-attention)
 * - 30 heads, 128 dim per head (3840 hidden)
 * - 3-axis RoPE (32+48+48 = 128 dims, theta=256)
 * - SwiGLU activation (8/3 expansion)
 * - AdaLN modulation: scale + tanh(gate) only (no shift)
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <Accelerate/Accelerate.h>

#ifdef USE_METAL
#include "iris_metal.h"
#endif

/* ========================================================================
 * Constants
 * ======================================================================== */

#define ZI_SEQ_MULTI_OF     32      /* Pad sequences to multiples of 32 */
#define ZI_NORM_EPS         1e-5f   /* RMSNorm epsilon */
#define ZI_MAX_SHARDS       32

/* Cumulative zImage timing counters (defined in iris_sample.c). */
extern double iris_timing_zi_total;
extern double iris_timing_zi_embeddings;
extern double iris_timing_zi_noise_refiner;
extern double iris_timing_zi_context_refiner;
extern double iris_timing_zi_main_blocks;
extern double iris_timing_zi_final;

static inline double zi_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* ========================================================================
 * Data Structures
 * ======================================================================== */

/* Single transformer block weights */
typedef struct {
    /* Attention */
    float *attn_q_weight;       /* [dim, dim] */
    float *attn_k_weight;       /* [dim, dim] */
    float *attn_v_weight;       /* [dim, dim] */
    float *attn_out_weight;     /* [dim, dim] */
    float *attn_norm_q;         /* [n_heads, head_dim] for QK norm */
    float *attn_norm_k;         /* [n_heads, head_dim] */
    float *attn_norm1;          /* [dim] RMSNorm before attention */
    float *attn_norm2;          /* [dim] RMSNorm after attention */

    /* FFN (SwiGLU) */
    float *ffn_w1;              /* [ffn_dim, dim] gate projection */
    float *ffn_w2;              /* [dim, ffn_dim] down projection */
    float *ffn_w3;              /* [ffn_dim, dim] up projection */
    float *ffn_norm1;           /* [dim] RMSNorm before FFN */
    float *ffn_norm2;           /* [dim] RMSNorm after FFN */

    /* Fused CPU weights (concatenated for single-GEMM dispatch) */
    float *attn_qkv_weight_f32; /* [3*dim, dim] fused [q;k;v] — CPU path */
    float *ffn_w13_f32;         /* [2*ffn_dim, dim] fused [w1;w3] — CPU path */

    /* AdaLN modulation (NULL for context_refiner blocks) */
    float *adaln_weight;        /* [4*dim, adaln_dim] */
    float *adaln_bias;          /* [4*dim] */

#ifdef USE_METAL
    /* BF16 weight pointers for GPU path */
    uint16_t *attn_q_weight_bf16;   /* [dim, dim] */
    uint16_t *attn_k_weight_bf16;   /* [dim, dim] */
    uint16_t *attn_v_weight_bf16;   /* [dim, dim] */
    uint16_t *attn_qkv_weight_bf16; /* [3*dim, dim] fused [q;k;v] */
    uint16_t *attn_out_weight_bf16; /* [dim, dim] */
    uint16_t *ffn_w1_bf16;          /* [ffn_dim, dim] */
    uint16_t *ffn_w2_bf16;          /* [dim, ffn_dim] */
    uint16_t *ffn_w3_bf16;          /* [ffn_dim, dim] */
    uint16_t *ffn_w13_weight_bf16;  /* [2*ffn_dim, dim] fused [w1;w3] */
    int bf16_from_mmap;             /* 1 if individual bf16 ptrs are mmap (don't free) */
    int bf16_fused_from_cache;      /* 1 if fused qkv/w13 ptrs are from cache mmap */
#endif
} zi_block_t;

/* Final layer weights */
typedef struct {
    float *adaln_weight;        /* [dim, adaln_dim] */
    float *adaln_bias;          /* [dim] */
    float *norm_weight;         /* NULL (no affine) or [dim] */
    float *linear_weight;       /* [out_ch, dim] */
    float *linear_bias;         /* [out_ch] */
} zi_final_t;

/* Z-Image transformer context */
typedef struct zi_transformer {
    /* Architecture config */
    int dim;                    /* 3840 */
    int n_heads;                /* 30 */
    int head_dim;               /* 128 */
    int n_layers;               /* 30 */
    int n_refiner;              /* 2 */
    int ffn_dim;                /* 8*dim/3 = 10240 */
    int in_channels;            /* 16 */
    int patch_size;             /* 2 */
    int adaln_dim;              /* min(dim, 256) = 256 */
    float rope_theta;           /* 256.0 */
    int axes_dims[3];           /* [32, 48, 48] */
    int axes_lens[3];           /* [1024, 512, 512] */

    /* Embedders */
    float *t_emb_mlp0_weight;   /* [mid_size, 256] */
    float *t_emb_mlp0_bias;     /* [mid_size] */
    float *t_emb_mlp2_weight;   /* [adaln_dim, mid_size] */
    float *t_emb_mlp2_bias;     /* [adaln_dim] */
    int t_emb_mid_size;         /* intermediate timestep MLP size */

    float *cap_emb_norm;        /* [cap_feat_dim] RMSNorm weight */
    float *cap_emb_linear_w;    /* [dim, cap_feat_dim] */
    float *cap_emb_linear_b;    /* [dim] */
    int cap_feat_dim;           /* 2560 */

    float *x_emb_weight;        /* [dim, patch_feat] where patch_feat = ps*ps*in_ch */
    float *x_emb_bias;          /* [dim] */

    float *x_pad_token;         /* [dim] */
    float *cap_pad_token;       /* [dim] */

    /* Transformer blocks */
    zi_block_t *noise_refiner;  /* [n_refiner] */
    zi_block_t *context_refiner;/* [n_refiner] */
    zi_block_t *layers;         /* [n_layers] */

    /* Final layer */
    zi_final_t final_layer;

    /* mmap modes: keep shard files open for zero-copy weight access. */
    int mmap_f32_weights;       /* CPU: direct f32 pointers from mmap */
    int mmap_bf16_weights;      /* GPU: direct bf16 pointers from mmap */
    safetensors_file_t *sf_files[ZI_MAX_SHARDS];
    int num_sf_files;

    /* BF16 cache: pre-converted weights stored on disk for fast reload */
    void *bf16_cache_mmap;      /* mmap'd cache file base pointer */
    size_t bf16_cache_size;     /* mmap'd region size */
    int bf16_fused_from_cache;  /* 1 if fused bf16 ptrs are from cache mmap */

    /* F16 cache: zero-copy Metal buffers via newBufferWithBytesNoCopy */
    void *f16_cache_mmap;       /* mmap'd F16 cache base pointer */
    size_t f16_cache_size;      /* mmap'd region size */
    int f16_from_cache;         /* 1 if BF16 ptrs actually point to F16 data */

    /* Precomputed RoPE frequencies (complex pairs) */
    float *rope_cos[3];         /* [axes_lens[i], axes_dims[i]/2] */
    float *rope_sin[3];         /* [axes_lens[i], axes_dims[i]/2] */

    /* Working memory */
    float *work_x;              /* Main token buffer */
    float *work_tmp;            /* Temporary buffer */
    float *work_qkv;            /* Q, K, V buffers */
    float *work_attn;           /* Attention scores */
    float *work_ffn;            /* FFN intermediate */
    float *work_ffn_fused;      /* [seq, 2*ffn_dim] fused W1/W3 GEMM scratch */
    size_t work_alloc;          /* Total allocated */
    int max_seq;                /* Max sequence length allocated for */

    /* Pre-allocated scratch buffers (reused across blocks/steps) */
    float *mod_scratch;         /* [4 * dim] AdaLN modulation scratch */
    float *final_scale;         /* [dim] final layer scale scratch */
    float *final_normed;        /* [max_seq * dim] final layer normed scratch */
    int final_normed_cap;       /* max seq allocated for final_normed */

    /* Cached position IDs and masks (reused when dimensions unchanged) */
    int cpu_cache_latent_h;     /* latent_h of cached pos/mask arrays */
    int cpu_cache_latent_w;     /* latent_w of cached pos/mask arrays */
    int cpu_cache_cap_seq_len;  /* cap_seq_len of cached pos/mask arrays */
    int *cpu_cache_img_pos;     /* [img_padded * 3] */
    int *cpu_cache_cap_pos;     /* [cap_padded * 3] */
    int *cpu_cache_img_mask;    /* [img_padded] */
    int *cpu_cache_cap_mask;    /* [cap_padded] */
    int *cpu_cache_unified_pos; /* [unified_seq * 3] */
    int *cpu_cache_unified_mask;/* [unified_seq] */
    int cpu_cache_img_padded;   /* cached img_padded value */
    int cpu_cache_cap_padded;   /* cached cap_padded value */
    int cpu_cache_unified_seq;  /* cached unified_seq value */

#ifdef USE_METAL
    int use_gpu;                /* 1 if GPU path available */
    /* Cached preassembled RoPE tables for GPU path (reused across steps) */
    int gpu_rope_img_seq;
    int gpu_rope_cap_seq;
    int gpu_rope_uni_seq;
    int gpu_rope_h_tokens;
    int gpu_rope_w_tokens;
    float *gpu_img_rope_cos;
    float *gpu_img_rope_sin;
    float *gpu_cap_rope_cos;
    float *gpu_cap_rope_sin;
    float *gpu_uni_rope_cos;
    float *gpu_uni_rope_sin;
#endif

    /* Pre-allocated forward buffers (reused across denoising steps).
     * GPU path uses: fwd_img_patches, fwd_cap_normed, fwd_step_mod,
     *   fwd_gpu_final_scale, fwd_gpu_final_shift (zero-filled once),
     *   fwd_gpu_final_scale_param, fwd_final_out.
     * CPU path uses: fwd_img_patches, fwd_cap_normed, fwd_img_emb,
     *   fwd_cap_emb, fwd_cap_padded_feats, fwd_img_out, fwd_final_out. */
    float *fwd_img_patches;         /* GPU: [img_seq * patch_feat]
                                       CPU: [img_padded * patch_feat] */
    float *fwd_cap_normed;          /* GPU: [cap_seq * cap_feat_dim]
                                       CPU: [cap_padded * cap_feat_dim] */
    float *fwd_step_mod;            /* GPU: [n_mod_blocks * 4 * dim] (fixed size) */
    float *fwd_gpu_final_scale;     /* GPU: [dim] */
    float *fwd_gpu_final_shift;     /* GPU: [dim] — zero-filled once, never touched */
    float *fwd_gpu_final_scale_param; /* GPU: [dim] */
    float *fwd_final_out;           /* [img_seq * out_ch] */
    float *fwd_img_emb;             /* CPU: [img_padded * dim] */
    float *fwd_cap_emb;             /* CPU: [cap_padded * dim] */
    float *fwd_cap_padded_feats;    /* CPU: [cap_padded * cap_feat_dim] */
    float *fwd_img_out;             /* CPU: [img_seq * dim] */
    /* Cached sizes for realloc check */
    int fwd_img_seq;                /* last img_seq (GPU) or img_padded (CPU) */
    int fwd_cap_seq;                /* last cap_seq_len (GPU) or cap_padded (CPU) */
} zi_transformer_t;

void iris_transformer_free_zimage(zi_transformer_t *tf);

#ifdef USE_METAL
/* GPU scratch buffers for block forward pass.
 * Pre-allocated once for max sequence length, reused across all blocks. */
typedef struct {
    int seq, dim, ffn_dim;
    iris_gpu_tensor_t norm;     /* [seq, dim] */
    iris_gpu_tensor_t fused;    /* [seq, max(3*dim, 2*ffn_dim)] */
    iris_gpu_tensor_t q;        /* [seq, dim] */
    iris_gpu_tensor_t k;        /* [seq, dim] */
    iris_gpu_tensor_t v;        /* [seq, dim] */
    iris_gpu_tensor_t attn_out; /* [seq, dim] */
    iris_gpu_tensor_t proj;     /* [seq, dim] */
    iris_gpu_tensor_t norm2;    /* [seq, dim] */
    iris_gpu_tensor_t gate_up;  /* [seq, ffn_dim] */
    iris_gpu_tensor_t up;       /* [seq, ffn_dim] */
    iris_gpu_tensor_t down;     /* [seq, dim] */
    float *mod;                     /* [4*dim] CPU modulation scratch */
    float *fused_attn_norm;         /* [dim] CPU fused RMS weight scratch */
    float *fused_ffn_norm;          /* [dim] CPU fused RMS weight scratch */
} zi_gpu_scratch_t;

static void zi_gpu_scratch_free(zi_gpu_scratch_t *s) {
    if (!s) return;
    if (s->norm) iris_gpu_tensor_free(s->norm);
    if (s->fused) iris_gpu_tensor_free(s->fused);
    if (s->q) iris_gpu_tensor_free(s->q);
    if (s->k) iris_gpu_tensor_free(s->k);
    if (s->v) iris_gpu_tensor_free(s->v);
    if (s->attn_out) iris_gpu_tensor_free(s->attn_out);
    if (s->proj) iris_gpu_tensor_free(s->proj);
    if (s->norm2) iris_gpu_tensor_free(s->norm2);
    if (s->gate_up) iris_gpu_tensor_free(s->gate_up);
    if (s->up) iris_gpu_tensor_free(s->up);
    if (s->down) iris_gpu_tensor_free(s->down);
    if (s->mod) free(s->mod);
    if (s->fused_attn_norm) free(s->fused_attn_norm);
    if (s->fused_ffn_norm) free(s->fused_ffn_norm);
    memset(s, 0, sizeof(*s));
}

static int zi_gpu_scratch_init(zi_gpu_scratch_t *s, int seq, int dim, int ffn_dim) {
    memset(s, 0, sizeof(*s));
    s->seq = seq;
    s->dim = dim;
    s->ffn_dim = ffn_dim;
    int fused_dim = 3 * dim;
    if (2 * ffn_dim > fused_dim) fused_dim = 2 * ffn_dim;

    s->norm = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->fused = iris_gpu_tensor_alloc((size_t)seq * fused_dim);
    s->q = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->k = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->v = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->attn_out = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->proj = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->norm2 = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->gate_up = iris_gpu_tensor_alloc((size_t)seq * ffn_dim);
    s->up = iris_gpu_tensor_alloc((size_t)seq * ffn_dim);
    s->down = iris_gpu_tensor_alloc((size_t)seq * dim);

    if (!s->norm || !s->fused || !s->q || !s->k || !s->v || !s->attn_out ||
        !s->proj || !s->norm2 || !s->gate_up || !s->up || !s->down) {
        zi_gpu_scratch_free(s);
        return 0;
    }

    s->mod = (float *)malloc(4 * (size_t)dim * sizeof(float));
    s->fused_attn_norm = (float *)malloc((size_t)dim * sizeof(float));
    s->fused_ffn_norm = (float *)malloc((size_t)dim * sizeof(float));
    if (!s->mod || !s->fused_attn_norm || !s->fused_ffn_norm) {
        zi_gpu_scratch_free(s);
        return 0;
    }

    return 1;
}

static void zi_build_rope_table(float *cos_out, float *sin_out,
                                 const int *pos_ids, int seq,
                                 zi_transformer_t *tf);

static uint16_t *zi_concat_bf16(const uint16_t *a, size_t na,
                                 const uint16_t *b, size_t nb) {
    if (!a || !b || na == 0 || nb == 0) return NULL;
    uint16_t *out = (uint16_t *)malloc((na + nb) * sizeof(uint16_t));
    if (!out) return NULL;
    memcpy(out, a, na * sizeof(uint16_t));
    memcpy(out + na, b, nb * sizeof(uint16_t));
    return out;
}

static uint16_t *zi_concat3_bf16(const uint16_t *a, size_t na,
                                  const uint16_t *b, size_t nb,
                                  const uint16_t *c, size_t nc) {
    if (!a || !b || !c || na == 0 || nb == 0 || nc == 0) return NULL;
    uint16_t *out = (uint16_t *)malloc((na + nb + nc) * sizeof(uint16_t));
    if (!out) return NULL;
    memcpy(out, a, na * sizeof(uint16_t));
    memcpy(out + na, b, nb * sizeof(uint16_t));
    memcpy(out + na + nb, c, nc * sizeof(uint16_t));
    return out;
}

/* GPU linear projection writing into a preallocated f32 output tensor.
 * Tries bf16 weight path first (fast), falls back to f32 weights. The "into"
 * variant avoids allocating a new tensor each call, which matters when
 * running 30+ blocks per step. */
static int zi_gpu_linear_into_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x,
                                   const uint16_t *W_bf16, const float *W_f32,
                                   int seq_len, int in_dim, int out_dim) {
    size_t n = (size_t)seq_len * (size_t)out_dim;

    if (W_bf16) {
        if (iris_gpu_linear_bf16_into(out, x, W_bf16, seq_len, in_dim, out_dim)) {
            return 1;
        }
        iris_gpu_tensor_t tmp_bf16 = iris_gpu_linear_bf16(x, W_bf16, seq_len, in_dim, out_dim);
        if (tmp_bf16) {
            iris_gpu_copy_f32(out, tmp_bf16, n);
            iris_gpu_tensor_free(tmp_bf16);
            return 1;
        }
    }

    if (W_f32) {
        iris_gpu_tensor_t tmp_f32 = iris_gpu_linear(x, W_f32, NULL, seq_len, in_dim, out_dim);
        if (tmp_f32) {
            iris_gpu_copy_f32(out, tmp_f32, n);
            iris_gpu_tensor_free(tmp_f32);
            return 1;
        }
    }

    return 0;
}

/* Self-attention dispatcher for GPU. Tries custom BF16 attention first
 * (bf16 I/O with f32 accumulation — fast and artifact-free), then falls
 * back to pure F32 (flash → fused).
 *
 * Note: MPSGraph SDPA (used by iris_gpu_attention_bf16 and the default
 * path in iris_gpu_attention_fused_bf16) causes grid artifacts at
 * non-square resolutions. iris_gpu_attention_custom_bf16 bypasses SDPA
 * and uses only the custom Metal kernel which is safe. */
static int zi_gpu_attention(iris_gpu_tensor_t out_f32,
                             iris_gpu_tensor_t q_f32, iris_gpu_tensor_t k_f32, iris_gpu_tensor_t v_f32,
                             int seq, int n_heads, int head_dim, float attn_scale,
                             zi_gpu_scratch_t *scratch) {
    (void)scratch;
    /* Try custom BF16 kernel (f32 accum, no MPSGraph SDPA — no grid artifacts) */
    if (iris_gpu_attention_custom_bf16(out_f32, q_f32, k_f32, v_f32,
                                       seq, seq, n_heads, head_dim, attn_scale))
        return 1;
    /* Fall back to F32 flash attention */
    if (iris_gpu_attention_flash(out_f32, q_f32, k_f32, v_f32,
                                 seq, seq, n_heads, head_dim, attn_scale))
        return 1;
    /* Fall back to F32 fused attention */
    return iris_gpu_attention_fused(out_f32, q_f32, k_f32, v_f32,
                                    seq, seq, n_heads, head_dim, attn_scale);
}

static void zi_gpu_rope_cache_clear(zi_transformer_t *tf) {
    free(tf->gpu_img_rope_cos); tf->gpu_img_rope_cos = NULL;
    free(tf->gpu_img_rope_sin); tf->gpu_img_rope_sin = NULL;
    free(tf->gpu_cap_rope_cos); tf->gpu_cap_rope_cos = NULL;
    free(tf->gpu_cap_rope_sin); tf->gpu_cap_rope_sin = NULL;
    free(tf->gpu_uni_rope_cos); tf->gpu_uni_rope_cos = NULL;
    free(tf->gpu_uni_rope_sin); tf->gpu_uni_rope_sin = NULL;
    tf->gpu_rope_img_seq = 0;
    tf->gpu_rope_cap_seq = 0;
    tf->gpu_rope_uni_seq = 0;
    tf->gpu_rope_h_tokens = 0;
    tf->gpu_rope_w_tokens = 0;
}

/* Preassembles and caches RoPE cos/sin tables for the current image geometry
 * (H_tokens, W_tokens, cap_seq_len). The geometry is stable across denoising
 * steps, so this avoids rebuilding tables every transformer call. Invalidated
 * when dimensions change (e.g., different image size). Builds separate tables
 * for noise refiner (image-only), context refiner (caption-only), and main
 * blocks (unified [img, cap] sequence). */
static int zi_gpu_rope_cache_prepare(zi_transformer_t *tf,
                                      int cap_seq_len, int H_tokens, int W_tokens) {
    int img_seq = H_tokens * W_tokens;
    int uni_seq = img_seq + cap_seq_len;

    if (tf->gpu_img_rope_cos &&
        tf->gpu_rope_img_seq == img_seq &&
        tf->gpu_rope_cap_seq == cap_seq_len &&
        tf->gpu_rope_uni_seq == uni_seq &&
        tf->gpu_rope_h_tokens == H_tokens &&
        tf->gpu_rope_w_tokens == W_tokens) {
        return 1;
    }

    zi_gpu_rope_cache_clear(tf);

    int head_dim = tf->head_dim;
    tf->gpu_img_rope_cos = (float *)malloc((size_t)img_seq * head_dim * sizeof(float));
    tf->gpu_img_rope_sin = (float *)malloc((size_t)img_seq * head_dim * sizeof(float));
    tf->gpu_cap_rope_cos = (float *)malloc((size_t)cap_seq_len * head_dim * sizeof(float));
    tf->gpu_cap_rope_sin = (float *)malloc((size_t)cap_seq_len * head_dim * sizeof(float));
    tf->gpu_uni_rope_cos = (float *)malloc((size_t)uni_seq * head_dim * sizeof(float));
    tf->gpu_uni_rope_sin = (float *)malloc((size_t)uni_seq * head_dim * sizeof(float));

    if (!tf->gpu_img_rope_cos || !tf->gpu_img_rope_sin ||
        !tf->gpu_cap_rope_cos || !tf->gpu_cap_rope_sin ||
        !tf->gpu_uni_rope_cos || !tf->gpu_uni_rope_sin) {
        zi_gpu_rope_cache_clear(tf);
        return 0;
    }

    int cap_padded_for_pos = ((cap_seq_len + ZI_SEQ_MULTI_OF - 1) / ZI_SEQ_MULTI_OF)
                              * ZI_SEQ_MULTI_OF;
    int *img_pos = (int *)calloc((size_t)img_seq * 3, sizeof(int));
    int *cap_pos = (int *)calloc((size_t)cap_seq_len * 3, sizeof(int));
    int *uni_pos = (int *)malloc((size_t)uni_seq * 3 * sizeof(int));
    if (!img_pos || !cap_pos || !uni_pos) {
        free(img_pos);
        free(cap_pos);
        free(uni_pos);
        zi_gpu_rope_cache_clear(tf);
        return 0;
    }

    for (int h = 0; h < H_tokens; h++) {
        for (int w = 0; w < W_tokens; w++) {
            int idx = h * W_tokens + w;
            img_pos[idx * 3 + 0] = cap_padded_for_pos + 1;
            img_pos[idx * 3 + 1] = h;
            img_pos[idx * 3 + 2] = w;
        }
    }

    for (int s = 0; s < cap_seq_len; s++) {
        cap_pos[s * 3 + 0] = 1 + s;
        cap_pos[s * 3 + 1] = 0;
        cap_pos[s * 3 + 2] = 0;
    }

    memcpy(uni_pos, img_pos, (size_t)img_seq * 3 * sizeof(int));
    memcpy(uni_pos + (size_t)img_seq * 3, cap_pos, (size_t)cap_seq_len * 3 * sizeof(int));

    zi_build_rope_table(tf->gpu_img_rope_cos, tf->gpu_img_rope_sin, img_pos, img_seq, tf);
    zi_build_rope_table(tf->gpu_cap_rope_cos, tf->gpu_cap_rope_sin, cap_pos, cap_seq_len, tf);
    zi_build_rope_table(tf->gpu_uni_rope_cos, tf->gpu_uni_rope_sin, uni_pos, uni_seq, tf);

    free(img_pos);
    free(cap_pos);
    free(uni_pos);

    tf->gpu_rope_img_seq = img_seq;
    tf->gpu_rope_cap_seq = cap_seq_len;
    tf->gpu_rope_uni_seq = uni_seq;
    tf->gpu_rope_h_tokens = H_tokens;
    tf->gpu_rope_w_tokens = W_tokens;
    return 1;
}

static void iris_warmup_bf16_zimage(zi_transformer_t *tf) {
    if (!tf || !tf->use_gpu) return;
    if (!iris_metal_available()) return;

    size_t attn_elems = (size_t)tf->dim * tf->dim;
    size_t ffn_up_elems = (size_t)tf->ffn_dim * tf->dim;
    size_t ffn_down_elems = (size_t)tf->dim * tf->ffn_dim;

    zi_block_t *groups[3] = { tf->noise_refiner, tf->context_refiner, tf->layers };
    int counts[3] = { tf->n_refiner, tf->n_refiner, tf->n_layers };

    for (int g = 0; g < 3; g++) {
        zi_block_t *blocks = groups[g];
        int n = counts[g];
        if (!blocks) continue;

        for (int i = 0; i < n; i++) {
            zi_block_t *b = &blocks[i];

            if (tf->f16_from_cache) {
                /* F16 cache: register zero-copy Metal buffers (no conversion) */
                if (b->attn_qkv_weight_bf16) iris_metal_register_f16_nocopy(b->attn_qkv_weight_bf16, attn_elems * 3);
                if (b->attn_out_weight_bf16) iris_metal_register_f16_nocopy(b->attn_out_weight_bf16, attn_elems);
                if (b->ffn_w13_weight_bf16) iris_metal_register_f16_nocopy(b->ffn_w13_weight_bf16, ffn_up_elems * 2);
                if (b->ffn_w2_bf16) iris_metal_register_f16_nocopy(b->ffn_w2_bf16, ffn_down_elems);
            } else {
                /* BF16 cache: convert BF16→F16 and cache in Metal buffers */
                if (b->attn_qkv_weight_bf16) iris_metal_warmup_bf16(b->attn_qkv_weight_bf16, attn_elems * 3);
                if (b->attn_out_weight_bf16) iris_metal_warmup_bf16(b->attn_out_weight_bf16, attn_elems);
                if (b->ffn_w13_weight_bf16) iris_metal_warmup_bf16(b->ffn_w13_weight_bf16, ffn_up_elems * 2);
                if (b->ffn_w2_bf16) iris_metal_warmup_bf16(b->ffn_w2_bf16, ffn_down_elems);
            }
        }
    }
}
#endif /* USE_METAL */

/* ========================================================================
 * Forward declarations
 * ======================================================================== */

void iris_transformer_free_zimage(zi_transformer_t *tf);

/* Forward declarations for functions used by GPU path */
static void zi_patchify(float *out, const float *latent,
                         int in_ch, int H, int W, int ps);
static void zi_unpatchify(float *latent, const float *patches,
                            int in_ch, int H, int W, int ps);
static int zi_final_compute_scale(float *scale, const zi_final_t *fl,
                                   const float *t_emb, zi_transformer_t *tf);
static void zi_final_forward(float *out, const float *x, const zi_final_t *fl,
                               const float *t_emb, int seq, zi_transformer_t *tf);
static void zi_rms_norm(float *out, const float *x, const float *weight,
                         int rows, int dim, float eps);

/* ========================================================================
 * Pre-allocated forward buffers
 * ======================================================================== */

/* Ensure all per-step temporary buffers are allocated (or grown) to fit the
 * current image / caption dimensions.  Called once at the top of each forward
 * function.  Buffers whose required size is constant (fwd_step_mod and the
 * three GPU final-layer scalars) are allocated on first call and never
 * reallocated.  fwd_gpu_final_shift is calloc'd once (always zero).
 *
 * img_seq:     number of image tokens   (GPU: unpadded, CPU: padded)
 * cap_seq:     caption sequence length   (GPU: unpadded, CPU: padded)
 * patch_feat:  ps * ps * in_channels
 * out_ch:      ps * ps * in_channels  (same value, distinct semantic)
 *
 * Returns 1 on success, 0 on allocation failure. */
static int zi_ensure_forward_buffers(zi_transformer_t *tf,
                                      int img_seq, int cap_seq,
                                      int patch_feat, int out_ch) {
    int dim = tf->dim;
    int cap_feat_dim = tf->cap_feat_dim;

    /* --- Fixed-size buffers (allocate once) --- */

    /* fwd_step_mod: n_mod_blocks * 4 * dim (only used by GPU path) */
    if (!tf->fwd_step_mod) {
        int n_mod = tf->n_refiner + tf->n_layers;
        if (n_mod > 0) {
            tf->fwd_step_mod = (float *)malloc((size_t)n_mod * 4 * dim * sizeof(float));
            if (!tf->fwd_step_mod) return 0;
        }
    }

    /* GPU final-layer scalars: [dim] each, allocated once */
    if (!tf->fwd_gpu_final_scale) {
        tf->fwd_gpu_final_scale = (float *)malloc(dim * sizeof(float));
        if (!tf->fwd_gpu_final_scale) return 0;
    }
    if (!tf->fwd_gpu_final_shift) {
        tf->fwd_gpu_final_shift = (float *)calloc(dim, sizeof(float));
        if (!tf->fwd_gpu_final_shift) return 0;
    }
    if (!tf->fwd_gpu_final_scale_param) {
        tf->fwd_gpu_final_scale_param = (float *)malloc(dim * sizeof(float));
        if (!tf->fwd_gpu_final_scale_param) return 0;
    }

    /* --- Variable-size buffers (grow if needed) --- */

    if (img_seq > tf->fwd_img_seq) {
        free(tf->fwd_img_patches);
        tf->fwd_img_patches = (float *)malloc((size_t)img_seq * patch_feat * sizeof(float));
        if (!tf->fwd_img_patches) { tf->fwd_img_seq = 0; return 0; }

        free(tf->fwd_final_out);
        tf->fwd_final_out = (float *)malloc((size_t)img_seq * out_ch * sizeof(float));
        if (!tf->fwd_final_out) { tf->fwd_img_seq = 0; return 0; }

        free(tf->fwd_img_emb);
        tf->fwd_img_emb = (float *)malloc((size_t)img_seq * dim * sizeof(float));
        if (!tf->fwd_img_emb) { tf->fwd_img_seq = 0; return 0; }

        free(tf->fwd_img_out);
        tf->fwd_img_out = (float *)malloc((size_t)img_seq * dim * sizeof(float));
        if (!tf->fwd_img_out) { tf->fwd_img_seq = 0; return 0; }

        tf->fwd_img_seq = img_seq;
    }

    if (cap_seq > tf->fwd_cap_seq) {
        free(tf->fwd_cap_normed);
        tf->fwd_cap_normed = (float *)malloc((size_t)cap_seq * cap_feat_dim * sizeof(float));
        if (!tf->fwd_cap_normed) { tf->fwd_cap_seq = 0; return 0; }

        free(tf->fwd_cap_emb);
        tf->fwd_cap_emb = (float *)malloc((size_t)cap_seq * dim * sizeof(float));
        if (!tf->fwd_cap_emb) { tf->fwd_cap_seq = 0; return 0; }

        free(tf->fwd_cap_padded_feats);
        tf->fwd_cap_padded_feats = (float *)malloc((size_t)cap_seq * cap_feat_dim * sizeof(float));
        if (!tf->fwd_cap_padded_feats) { tf->fwd_cap_seq = 0; return 0; }

        tf->fwd_cap_seq = cap_seq;
    }

    return 1;
}

/* ========================================================================
 * Timestep Embedding
 * ======================================================================== */

/* Converts scalar timestep to a 256-dim vector using log-spaced frequencies,
 * the same idea as the original Transformer positional encoding but here it
 * encodes the denoising step. The caller scales the input by 1000 before
 * calling (t * 1000.0f), mapping the [0,1] sigma range to [0,1000]. */
static void zi_sinusoidal_embedding(float *out, float t, int dim) {
    int half = dim / 2;
    float log_max_period = logf(10000.0f);
    for (int i = 0; i < half; i++) {
        float freq = expf(-log_max_period * (float)i / (float)half);
        float angle = t * freq;
        out[i] = cosf(angle);
        out[i + half] = sinf(angle);
    }
}

/* Projects the sinusoidal timestep embedding through an MLP
 * (Linear -> SiLU -> Linear) to produce the adaln_dim-sized conditioning
 * vector. This drives all AdaLN modulation in the transformer -- it is how
 * every block knows which denoising step it is operating on. */
static void zi_timestep_embed(zi_transformer_t *tf, float *out, float t) {
    float sin_emb[256];
    zi_sinusoidal_embedding(sin_emb, t * 1000.0f, 256);

    /* MLP: Linear(256 -> mid) + SiLU + Linear(mid -> adaln_dim) */
    int mid = tf->t_emb_mid_size;
    float *hidden = (float *)malloc(mid * sizeof(float));
    if (!hidden) {
        memset(out, 0, tf->adaln_dim * sizeof(float));
        return;
    }

    /* Linear 0 */
    iris_matmul_t(hidden, sin_emb, tf->t_emb_mlp0_weight, 1, 256, mid);
    for (int i = 0; i < mid; i++) hidden[i] += tf->t_emb_mlp0_bias[i];

    /* SiLU */
    iris_silu(hidden, mid);

    /* Linear 2 */
    iris_matmul_t(out, hidden, tf->t_emb_mlp2_weight, 1, mid, tf->adaln_dim);
    for (int i = 0; i < tf->adaln_dim; i++) out[i] += tf->t_emb_mlp2_bias[i];

    free(hidden);
}

/* ========================================================================
 * RoPE
 * ======================================================================== */

/* Precomputes cos/sin frequency tables for all 3 RoPE axes
 * (T=32 dims, H=48 dims, W=48 dims) up to max_pos=1024 per axis.
 * Uses theta=256.0, much smaller than the usual 10000, giving shorter-range
 * position sensitivity suited to Z-Image's spatial layout. Tables are
 * allocated once at load time and reused across all denoising steps. */
static void zi_precompute_rope(zi_transformer_t *tf) {
    for (int ax = 0; ax < 3; ax++) {
        int d = tf->axes_dims[ax];
        int half_d = d / 2;
        int max_pos = tf->axes_lens[ax];

        tf->rope_cos[ax] = (float *)malloc(max_pos * half_d * sizeof(float));
        tf->rope_sin[ax] = (float *)malloc(max_pos * half_d * sizeof(float));
        if (!tf->rope_cos[ax] || !tf->rope_sin[ax]) {
            fprintf(stderr, "zi_precompute_rope: malloc failed for axis %d\n", ax);
            free(tf->rope_cos[ax]); tf->rope_cos[ax] = NULL;
            free(tf->rope_sin[ax]); tf->rope_sin[ax] = NULL;
            return;
        }

        /* Hoist powf: precompute inverse frequencies once per axis */
        float inv_freq[64];  /* max half_d = 48/2 = 24 */
        if (half_d > 64) {
            fprintf(stderr, "zi_precompute_rope: half_d=%d exceeds buffer\n", half_d);
            return;
        }
        for (int i = 0; i < half_d; i++) {
            inv_freq[i] = 1.0f / powf(tf->rope_theta, (float)(2 * i) / (float)d);
        }

        for (int pos = 0; pos < max_pos; pos++) {
            for (int i = 0; i < half_d; i++) {
                float angle = (float)pos * inv_freq[i];
                tf->rope_cos[ax][pos * half_d + i] = cosf(angle);
                tf->rope_sin[ax][pos * half_d + i] = sinf(angle);
            }
        }
    }
}

/* Applies 3-axis RoPE to Q or K in-place using consecutive-pair rotation:
 * (x0*cos - x1*sin, x1*cos + x0*sin) on elements (d, d+1).
 * Each axis section of head_dim (T=32, H=48, W=48) gets its own position
 * from pos_ids[s,3]. This differs from Flux's split-half convention --
 * Z-Image pairs adjacent elements (d, d+1) rather than (d, d+half). */
static void zi_apply_rope(float *x, const int *pos_ids, int seq, int n_heads,
                           zi_transformer_t *tf) {
    int head_dim = tf->head_dim;
    int offset = 0;

    for (int ax = 0; ax < 3; ax++) {
        int d = tf->axes_dims[ax];
        int half_d = d / 2;

        for (int s = 0; s < seq; s++) {
            int pos = pos_ids[s * 3 + ax];
            if (pos < 0 || pos >= tf->axes_lens[ax]) continue;

            const float *cos_tab = tf->rope_cos[ax] + pos * half_d;
            const float *sin_tab = tf->rope_sin[ax] + pos * half_d;

            for (int h = 0; h < n_heads; h++) {
                float *head = x + (s * n_heads + h) * head_dim + offset;
                for (int i = 0; i < half_d; i++) {
                    float x0 = head[2 * i];
                    float x1 = head[2 * i + 1];
                    float c = cos_tab[i];
                    float sn = sin_tab[i];
                    head[2 * i]     = x0 * c - x1 * sn;
                    head[2 * i + 1] = x1 * c + x0 * sn;
                }
            }
        }
        offset += d;
    }
}

/* ========================================================================
 * Block Forward Pass (BLAS)
 * ======================================================================== */

/* RMSNorm: out = x * weight / sqrt(mean(x^2) + eps) */
static void zi_rms_norm(float *out, const float *x, const float *weight,
                         int rows, int dim, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + r * dim;
        float *or_ = out + r * dim;
        float sum_sq;
        vDSP_svesq(xr, 1, &sum_sq, dim);
        float rms = 1.0f / sqrtf(sum_sq / dim + eps);
        vDSP_vsmul(xr, 1, &rms, or_, 1, dim);
        vDSP_vmul(or_, 1, weight, 1, or_, 1, dim);
    }
}

/* Per-head RMSNorm for QK normalization.
 * x: [seq, n_heads * head_dim], norm_weight: [head_dim] (shared across heads) */
static void zi_qk_norm(float *x, const float *norm_weight, int seq,
                         int n_heads, int head_dim, float eps) {
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < n_heads; h++) {
            float *ptr = x + s * n_heads * head_dim + h * head_dim;
            float sum_sq;
            vDSP_svesq(ptr, 1, &sum_sq, head_dim);
            float rms = 1.0f / sqrtf(sum_sq / head_dim + eps);
            vDSP_vsmul(ptr, 1, &rms, ptr, 1, head_dim);
            vDSP_vmul(ptr, 1, norm_weight, 1, ptr, 1, head_dim);
        }
    }
}

/* Scaled dot-product self-attention on the CPU path.
 * Computes Q@K^T per head, applies padding mask (sets masked positions to
 * -1e9 so softmax zeros them out), then scores@V. The mask distinguishes
 * real tokens from padding in the sequence. This is the slow reference path;
 * the GPU path uses fused SDPA kernels instead. */
static void zi_attention(float *out, const float *x,
                          const zi_block_t *block, const int *pos_ids,
                          const int *mask, int seq,
                          zi_transformer_t *tf) {
    int dim = tf->dim;
    int n_heads = tf->n_heads;
    int head_dim = tf->head_dim;

    float *q = tf->work_qkv;
    float *k = q + (size_t)seq * dim;
    float *v = k + (size_t)seq * dim;

    /* Q, K, V projections — fused single GEMM when available */
    if (block->attn_qkv_weight_f32) {
        /* Single matmul into work_tmp: [seq, dim] @ [3*dim, dim]^T -> [seq, 3*dim]
         * Then scatter rows into separate q, k, v buffers. work_tmp is safe to
         * use here because zi_attention() overwrites it with attn_out later. */
        float *fused_buf = tf->work_tmp;
        iris_matmul_t(fused_buf, x, block->attn_qkv_weight_f32, seq, dim, 3 * dim);
        for (int s = 0; s < seq; s++) {
            const float *row = fused_buf + (size_t)s * 3 * dim;
            memcpy(q + (size_t)s * dim, row,             dim * sizeof(float));
            memcpy(k + (size_t)s * dim, row + dim,       dim * sizeof(float));
            memcpy(v + (size_t)s * dim, row + 2 * dim,   dim * sizeof(float));
        }
    } else {
        iris_matmul_t(q, x, block->attn_q_weight, seq, dim, dim);
        iris_matmul_t(k, x, block->attn_k_weight, seq, dim, dim);
        iris_matmul_t(v, x, block->attn_v_weight, seq, dim, dim);
    }

    /* QK normalization */
    zi_qk_norm(q, block->attn_norm_q, seq, n_heads, head_dim, ZI_NORM_EPS);
    zi_qk_norm(k, block->attn_norm_k, seq, n_heads, head_dim, ZI_NORM_EPS);

    /* Apply RoPE */
    zi_apply_rope(q, pos_ids, seq, n_heads, tf);
    zi_apply_rope(k, pos_ids, seq, n_heads, tf);

    /* Scaled dot-product attention per head */
    float scale = 1.0f / sqrtf((float)head_dim);
    float *attn_out = tf->work_tmp;

    for (int h = 0; h < n_heads; h++) {
        float *scores = tf->work_attn;

        /* Q @ K^T: [seq, head_dim] x [head_dim, seq] -> [seq, seq]
         * Q and K are interleaved across heads with stride = dim = n_heads * head_dim.
         * Use lda/ldb = dim to read only this head's slice without copying. */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq, seq, head_dim,
                    scale, q + h * head_dim, dim, k + h * head_dim, dim,
                    0.0f, scores, seq);

        /* Apply mask: set padding positions to -inf */
        if (mask) {
            for (int i = 0; i < seq; i++) {
                for (int j = 0; j < seq; j++) {
                    if (!mask[j])
                        scores[i * seq + j] = -1e9f;
                }
            }
        }

        /* Softmax */
        iris_softmax(scores, seq, seq);

        /* scores @ V: [seq, seq] x [seq, head_dim] -> [seq, head_dim]
         * V is interleaved with stride = dim; output is also interleaved. */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq, head_dim, seq,
                    1.0f, scores, seq, v + h * head_dim, dim,
                    0.0f, attn_out + h * head_dim, dim);
    }

    /* Output projection */
    iris_matmul_t(out, attn_out, block->attn_out_weight, seq, dim, dim);
}

/* SwiGLU FFN: silu(W1 @ x) * (W3 @ x) then W2 */
static void zi_ffn(float *out, const float *x, const zi_block_t *block,
                    int seq, zi_transformer_t *tf) {
    int dim = tf->dim;
    int ffn_dim = tf->ffn_dim;
    float *gate = tf->work_ffn;
    float *up = gate + seq * ffn_dim;

    /* W1 (gate) and W3 (up) projections — fused single GEMM when available */
    if (block->ffn_w13_f32) {
        /* Single matmul: [seq, dim] @ [2*ffn_dim, dim]^T -> [seq, 2*ffn_dim]
         * Output goes into work_ffn_fused, then scatter into gate and up. */
        float *fused_buf = tf->work_ffn_fused;
        iris_matmul_t(fused_buf, x, block->ffn_w13_f32, seq, dim, 2 * ffn_dim);
        for (int s = 0; s < seq; s++) {
            const float *row = fused_buf + (size_t)s * 2 * ffn_dim;
            memcpy(gate + (size_t)s * ffn_dim, row,           ffn_dim * sizeof(float));
            memcpy(up   + (size_t)s * ffn_dim, row + ffn_dim, ffn_dim * sizeof(float));
        }
    } else {
        iris_matmul_t(gate, x, block->ffn_w1, seq, dim, ffn_dim);
        iris_matmul_t(up, x, block->ffn_w3, seq, dim, ffn_dim);
    }

    /* SiLU(gate) * up */
    int n = seq * ffn_dim;
    iris_silu_mul(gate, up, n);

    /* W2 (down) projection */
    iris_matmul_t(out, gate, block->ffn_w2, seq, ffn_dim, dim);
}

/* One S3-DiT block on CPU. Two modes: modulated (noise_refiner + main layers)
 * applies AdaLN with scale and tanh-gated residuals; unmodulated
 * (context_refiner) is a plain pre-norm attention + FFN block. The modulation
 * uses 4 parameters per block: scale_msa, gate_msa, scale_mlp, gate_mlp.
 * Note: no additive shift in Z-Image's block modulation (unlike Flux's
 * AdaLN which has shift). */
static void zi_block_forward(float *x, const zi_block_t *block,
                              const int *pos_ids, const int *mask,
                              const float *t_emb, int seq,
                              zi_transformer_t *tf) {
    int dim = tf->dim;
    int n = seq * dim;
    float *attn_out = tf->work_tmp;
    float *norm_out = tf->work_tmp + n;
    float *scaled = tf->work_tmp + 2 * n;
    float *ffn_out = tf->work_tmp + 3 * n;

    if (!tf->work_tmp || tf->max_seq < seq) return;

    if (block->adaln_weight) {
        /* Modulated block: extract scale_msa, gate_msa, scale_mlp, gate_mlp.
         * Uses pre-allocated mod_scratch instead of per-block malloc/free. */
        float *mod = tf->mod_scratch;
        if (!mod) return;
        iris_matmul_t(mod, t_emb, block->adaln_weight, 1, tf->adaln_dim, 4 * dim);
        vDSP_vadd(mod, 1, block->adaln_bias, 1, mod, 1, 4 * dim);

        float *scale_msa = mod;
        float *gate_msa  = mod + dim;
        float *scale_mlp = mod + 2 * dim;
        float *gate_mlp  = mod + 3 * dim;

        /* Apply tanh to gates, 1+scale */
        {
            float one = 1.0f;
            int dim_int = dim;
            vDSP_vsadd(scale_msa, 1, &one, scale_msa, 1, dim);
            vvtanhf(gate_msa, gate_msa, &dim_int);
            vDSP_vsadd(scale_mlp, 1, &one, scale_mlp, 1, dim);
            vvtanhf(gate_mlp, gate_mlp, &dim_int);
        }

        /* Attention: h = attention(norm1(x) * scale_msa) */
        zi_rms_norm(norm_out, x, block->attn_norm1, seq, dim, ZI_NORM_EPS);
        for (int s = 0; s < seq; s++)
            vDSP_vmul(norm_out + s * dim, 1, scale_msa, 1, scaled + s * dim, 1, dim);

        zi_attention(attn_out, scaled, block, pos_ids, mask, seq, tf);

        /* x = x + gate_msa * norm2(attn_out) */
        zi_rms_norm(norm_out, attn_out, block->attn_norm2, seq, dim, ZI_NORM_EPS);

        for (int s = 0; s < seq; s++)
            vDSP_vma(gate_msa, 1, norm_out + s * dim, 1, x + s * dim, 1, x + s * dim, 1, dim);

        /* FFN: h = ffn(norm1(x) * scale_mlp) */
        zi_rms_norm(norm_out, x, block->ffn_norm1, seq, dim, ZI_NORM_EPS);
        for (int s = 0; s < seq; s++)
            vDSP_vmul(norm_out + s * dim, 1, scale_mlp, 1, scaled + s * dim, 1, dim);

        zi_ffn(ffn_out, scaled, block, seq, tf);

        /* x = x + gate_mlp * norm2(ffn_out) */
        zi_rms_norm(norm_out, ffn_out, block->ffn_norm2, seq, dim, ZI_NORM_EPS);
        for (int s = 0; s < seq; s++)
            vDSP_vma(gate_mlp, 1, norm_out + s * dim, 1, x + s * dim, 1, x + s * dim, 1, dim);

        /* mod_scratch is reused across blocks — no free needed */
    } else {
        /* Unmodulated block (context_refiner): no scale/gate */

        /* Attention: h = attention(norm1(x)) */
        zi_rms_norm(norm_out, x, block->attn_norm1, seq, dim, ZI_NORM_EPS);

        zi_attention(attn_out, norm_out, block, pos_ids, mask, seq, tf);

        /* x = x + norm2(attn_out) */
        zi_rms_norm(norm_out, attn_out, block->attn_norm2, seq, dim, ZI_NORM_EPS);
        for (int i = 0; i < n; i++) x[i] += norm_out[i];
        /* FFN */
        zi_rms_norm(norm_out, x, block->ffn_norm1, seq, dim, ZI_NORM_EPS);

        zi_ffn(ffn_out, norm_out, block, seq, tf);

        zi_rms_norm(norm_out, ffn_out, block->ffn_norm2, seq, dim, ZI_NORM_EPS);
        for (int i = 0; i < n; i++) x[i] += norm_out[i];
    }
}

/* ========================================================================
 * GPU Forward Pass (Metal)
 * ======================================================================== */

#ifdef USE_METAL

/* Convert f32 array to bf16 (CPU-side, for weight conversion at load time).
 * Uses round-to-nearest-even for best accuracy.
 * Caller owns the returned buffer. */
static uint16_t *zi_f32_to_bf16(const float *src, size_t n) {
    uint16_t *dst = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!dst) return NULL;
    for (size_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &src[i], 4);
        if ((bits & 0x7F800000u) == 0x7F800000u) {
            dst[i] = (uint16_t)(bits >> 16);  // Inf/NaN: truncate, don't round
            continue;
        }
        uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        bits += rounding_bias;
        dst[i] = (uint16_t)(bits >> 16);
    }
    return dst;
}

/* Build a pre-assembled [seq, head_dim] RoPE cos/sin table by merging 3 axes.
 * pos_ids: [seq, 3] with (T, H, W) per position.
 * The table has cos/sin values laid out as consecutive pairs so that
 * iris_gpu_rope_2d (axis_dim=head_dim) applies rotation across the full head. */
static void zi_build_rope_table(float *cos_out, float *sin_out,
                                 const int *pos_ids, int seq,
                                 zi_transformer_t *tf) {
    int head_dim = tf->head_dim;

    for (int s = 0; s < seq; s++) {
        int offset = 0;
        for (int ax = 0; ax < 3; ax++) {
            int d = tf->axes_dims[ax];
            int half_d = d / 2;
            int pos = pos_ids[s * 3 + ax];

            /* Clamp pos to valid range */
            if (pos < 0) pos = 0;
            if (pos >= tf->axes_lens[ax]) pos = tf->axes_lens[ax] - 1;

            const float *ax_cos = tf->rope_cos[ax] + pos * half_d;
            const float *ax_sin = tf->rope_sin[ax] + pos * half_d;

            /* Write as consecutive pairs [cos_0, cos_0, cos_1, cos_1, ...] */
            for (int i = 0; i < half_d; i++) {
                cos_out[s * head_dim + offset + 2 * i]     = ax_cos[i];
                cos_out[s * head_dim + offset + 2 * i + 1] = ax_cos[i];
                sin_out[s * head_dim + offset + 2 * i]     = ax_sin[i];
                sin_out[s * head_dim + offset + 2 * i + 1] = ax_sin[i];
            }
            offset += d;
        }
    }
}

/* GPU-accelerated block forward. Fuses norm weight with modulation scale on
 * CPU (one multiply per dim), then passes the fused weight to GPU RMSNorm to
 * avoid an extra GPU kernel. Uses fused QKV and W1/W3 matmuls when available.
 * Returns 0 on any failure so the caller can fall back to CPU. */
static int zi_block_forward_gpu(iris_gpu_tensor_t hidden_gpu,
                                 const zi_block_t *block,
                                 const float *rope_cos, const float *rope_sin,
                                 const float *t_emb, const float *precomputed_mod, int seq,
                                 zi_transformer_t *tf,
                                 zi_gpu_scratch_t *scratch) {
    if (!iris_metal_available() || !iris_metal_shaders_available()) return 0;
    if (!hidden_gpu || !scratch) return 0;

    int dim = tf->dim;
    int n_heads = tf->n_heads;
    int head_dim = tf->head_dim;
    int ffn_dim = tf->ffn_dim;

    if (block->adaln_weight) {
        /* ---- Modulated block ---- */

        const float *mod = precomputed_mod;
        if (!mod) {
            /* Fallback path: compute modulation on the fly. */
            float *scratch_mod = scratch->mod;
            iris_matmul_t(scratch_mod, t_emb, block->adaln_weight, 1, tf->adaln_dim, 4 * dim);
            vDSP_vadd(scratch_mod, 1, block->adaln_bias, 1, scratch_mod, 1, 4 * dim);

            /* Apply 1+scale to scales, tanh to gates */
            {
                float one = 1.0f;
                int dim_int = dim;
                vDSP_vsadd(scratch_mod, 1, &one, scratch_mod, 1, dim);
                vvtanhf(scratch_mod + dim, scratch_mod + dim, &dim_int);
                vDSP_vsadd(scratch_mod + 2 * dim, 1, &one, scratch_mod + 2 * dim, 1, dim);
                vvtanhf(scratch_mod + 3 * dim, scratch_mod + 3 * dim, &dim_int);
            }
            mod = scratch_mod;
        }

        const float *scale_msa = mod;
        const float *gate_msa  = mod + dim;
        const float *scale_mlp = mod + 2 * dim;
        const float *gate_mlp  = mod + 3 * dim;

        /* CPU: fuse norm_weight * scale into a single weight for RMSNorm */
        float *fused_attn_norm = scratch->fused_attn_norm;
        for (int i = 0; i < dim; i++)
            fused_attn_norm[i] = block->attn_norm1[i] * scale_msa[i];

        /* GPU: RMSNorm with fused weight (= rms_norm(x) * attn_norm1 * scale_msa) */
        iris_gpu_rms_norm_f32(scratch->norm, hidden_gpu, fused_attn_norm,
                               seq, dim, ZI_NORM_EPS);

        /* GPU: Q, K, V projections (fused when available). */
        if (block->attn_qkv_weight_bf16) {
            if (zi_gpu_linear_into_f32(scratch->fused, scratch->norm,
                                       block->attn_qkv_weight_bf16, NULL,
                                       seq, dim, 3 * dim)) {
                iris_gpu_split_qkv_mlp(scratch->fused,
                                       scratch->q, scratch->k, scratch->v,
                                       scratch->gate_up, scratch->up,
                                       seq, dim, 0);
            } else {
                /* Fallback to unfused projections on GPU if fused path fails. */
                if (!zi_gpu_linear_into_f32(scratch->q, scratch->norm,
                                            block->attn_q_weight_bf16, block->attn_q_weight,
                                            seq, dim, dim)) return 0;
                if (!zi_gpu_linear_into_f32(scratch->k, scratch->norm,
                                            block->attn_k_weight_bf16, block->attn_k_weight,
                                            seq, dim, dim)) return 0;
                if (!zi_gpu_linear_into_f32(scratch->v, scratch->norm,
                                            block->attn_v_weight_bf16, block->attn_v_weight,
                                            seq, dim, dim)) return 0;
            }
        } else {
            if (!zi_gpu_linear_into_f32(scratch->q, scratch->norm,
                                        block->attn_q_weight_bf16, block->attn_q_weight,
                                        seq, dim, dim)) return 0;
            if (!zi_gpu_linear_into_f32(scratch->k, scratch->norm,
                                        block->attn_k_weight_bf16, block->attn_k_weight,
                                        seq, dim, dim)) return 0;
            if (!zi_gpu_linear_into_f32(scratch->v, scratch->norm,
                                        block->attn_v_weight_bf16, block->attn_v_weight,
                                        seq, dim, dim)) return 0;
        }

        /* GPU: fused QK normalization + RoPE */
        if (!iris_gpu_qknorm_rope(scratch->q, scratch->k,
                                   block->attn_norm_q, block->attn_norm_k,
                                   rope_cos, rope_sin,
                                   seq, n_heads, head_dim, ZI_NORM_EPS)) {
            iris_gpu_qk_rms_norm(scratch->q, scratch->k,
                                  block->attn_norm_q, block->attn_norm_k,
                                  seq, n_heads, head_dim, ZI_NORM_EPS);
            iris_gpu_rope_single_pair_f32(scratch->q, scratch->k,
                                          rope_cos, rope_sin,
                                          seq, n_heads, head_dim);
        }

        /* GPU: Self-attention */
        float attn_scale = 1.0f / sqrtf((float)head_dim);
        if (!zi_gpu_attention(scratch->attn_out, scratch->q, scratch->k, scratch->v,
                               seq, n_heads, head_dim, attn_scale, scratch)) {
            return 0;
        }

        /* GPU: Output projection */
        if (!zi_gpu_linear_into_f32(scratch->proj, scratch->attn_out,
                                    block->attn_out_weight_bf16, block->attn_out_weight,
                                    seq, dim, dim)) return 0;

        /* GPU: fused attn_norm2 + gated residual: x += gate_msa * norm2(proj) */
        if (!iris_gpu_norm_gated_add(hidden_gpu, scratch->proj,
                                      block->attn_norm2, gate_msa,
                                      seq, dim, ZI_NORM_EPS)) {
            iris_gpu_rms_norm_f32(scratch->norm2, scratch->proj, block->attn_norm2,
                                   seq, dim, ZI_NORM_EPS);
            iris_gpu_gated_add(hidden_gpu, gate_msa, scratch->norm2, seq, dim);
        }

        /* CPU: fuse FFN norm weight * scale_mlp */
        float *fused_ffn_norm = scratch->fused_ffn_norm;
        for (int i = 0; i < dim; i++)
            fused_ffn_norm[i] = block->ffn_norm1[i] * scale_mlp[i];

        /* GPU: FFN input norm with fused weight */
        iris_gpu_rms_norm_f32(scratch->norm, hidden_gpu, fused_ffn_norm,
                               seq, dim, ZI_NORM_EPS);

        /* GPU: SwiGLU FFN (fused w1/w3 when available). */
        if (block->ffn_w13_weight_bf16) {
            if (zi_gpu_linear_into_f32(scratch->fused, scratch->norm,
                                       block->ffn_w13_weight_bf16, NULL,
                                       seq, dim, 2 * ffn_dim)) {
                if (!iris_gpu_split_silu_mul(scratch->gate_up, scratch->fused,
                                             seq, ffn_dim)) {
                    iris_gpu_split_qkv_mlp(scratch->fused,
                                           scratch->q, scratch->k, scratch->v,
                                           scratch->gate_up, scratch->up,
                                           seq, 0, ffn_dim);
                    iris_gpu_silu_mul(scratch->gate_up, scratch->up, seq * ffn_dim);
                }
            } else {
                if (!zi_gpu_linear_into_f32(scratch->gate_up, scratch->norm,
                                            block->ffn_w1_bf16, block->ffn_w1,
                                            seq, dim, ffn_dim)) return 0;
                if (!zi_gpu_linear_into_f32(scratch->up, scratch->norm,
                                            block->ffn_w3_bf16, block->ffn_w3,
                                            seq, dim, ffn_dim)) return 0;
                iris_gpu_silu_mul(scratch->gate_up, scratch->up, seq * ffn_dim);
            }
        } else {
            if (!zi_gpu_linear_into_f32(scratch->gate_up, scratch->norm,
                                        block->ffn_w1_bf16, block->ffn_w1,
                                        seq, dim, ffn_dim)) return 0;
            if (!zi_gpu_linear_into_f32(scratch->up, scratch->norm,
                                        block->ffn_w3_bf16, block->ffn_w3,
                                        seq, dim, ffn_dim)) return 0;
            iris_gpu_silu_mul(scratch->gate_up, scratch->up, seq * ffn_dim);
        }

        /* GPU: FFN down projection */
        if (!zi_gpu_linear_into_f32(scratch->down, scratch->gate_up,
                                    block->ffn_w2_bf16, block->ffn_w2,
                                    seq, ffn_dim, dim)) return 0;

        /* GPU: fused ffn_norm2 + gated residual: x += gate_mlp * norm2(ffn_out) */
        if (!iris_gpu_norm_gated_add(hidden_gpu, scratch->down,
                                      block->ffn_norm2, gate_mlp,
                                      seq, dim, ZI_NORM_EPS)) {
            iris_gpu_rms_norm_f32(scratch->norm2, scratch->down, block->ffn_norm2,
                                   seq, dim, ZI_NORM_EPS);
            iris_gpu_gated_add(hidden_gpu, gate_mlp, scratch->norm2, seq, dim);
        }

    } else {
        /* ---- Unmodulated block (context_refiner) ---- */

        /* GPU: RMSNorm (plain weight, no scale) */
        iris_gpu_rms_norm_f32(scratch->norm, hidden_gpu, block->attn_norm1,
                               seq, dim, ZI_NORM_EPS);

        /* GPU: Q, K, V projections (fused when available). */
        if (block->attn_qkv_weight_bf16) {
            if (zi_gpu_linear_into_f32(scratch->fused, scratch->norm,
                                       block->attn_qkv_weight_bf16, NULL,
                                       seq, dim, 3 * dim)) {
                iris_gpu_split_qkv_mlp(scratch->fused,
                                       scratch->q, scratch->k, scratch->v,
                                       scratch->gate_up, scratch->up,
                                       seq, dim, 0);
            } else {
                if (!zi_gpu_linear_into_f32(scratch->q, scratch->norm,
                                            block->attn_q_weight_bf16, block->attn_q_weight,
                                            seq, dim, dim)) return 0;
                if (!zi_gpu_linear_into_f32(scratch->k, scratch->norm,
                                            block->attn_k_weight_bf16, block->attn_k_weight,
                                            seq, dim, dim)) return 0;
                if (!zi_gpu_linear_into_f32(scratch->v, scratch->norm,
                                            block->attn_v_weight_bf16, block->attn_v_weight,
                                            seq, dim, dim)) return 0;
            }
        } else {
            if (!zi_gpu_linear_into_f32(scratch->q, scratch->norm,
                                        block->attn_q_weight_bf16, block->attn_q_weight,
                                        seq, dim, dim)) return 0;
            if (!zi_gpu_linear_into_f32(scratch->k, scratch->norm,
                                        block->attn_k_weight_bf16, block->attn_k_weight,
                                        seq, dim, dim)) return 0;
            if (!zi_gpu_linear_into_f32(scratch->v, scratch->norm,
                                        block->attn_v_weight_bf16, block->attn_v_weight,
                                        seq, dim, dim)) return 0;
        }

        /* GPU: fused QK normalization + RoPE */
        if (!iris_gpu_qknorm_rope(scratch->q, scratch->k,
                                   block->attn_norm_q, block->attn_norm_k,
                                   rope_cos, rope_sin,
                                   seq, n_heads, head_dim, ZI_NORM_EPS)) {
            iris_gpu_qk_rms_norm(scratch->q, scratch->k,
                                  block->attn_norm_q, block->attn_norm_k,
                                  seq, n_heads, head_dim, ZI_NORM_EPS);
            iris_gpu_rope_single_pair_f32(scratch->q, scratch->k,
                                          rope_cos, rope_sin,
                                          seq, n_heads, head_dim);
        }

        /* GPU: Self-attention */
        float attn_scale = 1.0f / sqrtf((float)head_dim);
        if (!zi_gpu_attention(scratch->attn_out, scratch->q, scratch->k, scratch->v,
                               seq, n_heads, head_dim, attn_scale, scratch)) {
            return 0;
        }

        /* GPU: Output projection */
        if (!zi_gpu_linear_into_f32(scratch->proj, scratch->attn_out,
                                    block->attn_out_weight_bf16, block->attn_out_weight,
                                    seq, dim, dim)) return 0;

        /* GPU: fused norm2 + residual: x += norm2(attn_out) */
        if (!iris_gpu_norm_add(hidden_gpu, scratch->proj, block->attn_norm2,
                                seq, dim, ZI_NORM_EPS)) {
            iris_gpu_rms_norm_f32(scratch->norm2, scratch->proj, block->attn_norm2,
                                   seq, dim, ZI_NORM_EPS);
            iris_gpu_add_f32(hidden_gpu, hidden_gpu, scratch->norm2, seq * dim);
        }

        /* GPU: FFN */
        iris_gpu_rms_norm_f32(scratch->norm, hidden_gpu, block->ffn_norm1,
                               seq, dim, ZI_NORM_EPS);
        if (block->ffn_w13_weight_bf16) {
            if (zi_gpu_linear_into_f32(scratch->fused, scratch->norm,
                                       block->ffn_w13_weight_bf16, NULL,
                                       seq, dim, 2 * ffn_dim)) {
                if (!iris_gpu_split_silu_mul(scratch->gate_up, scratch->fused,
                                             seq, ffn_dim)) {
                    iris_gpu_split_qkv_mlp(scratch->fused,
                                           scratch->q, scratch->k, scratch->v,
                                           scratch->gate_up, scratch->up,
                                           seq, 0, ffn_dim);
                    iris_gpu_silu_mul(scratch->gate_up, scratch->up, seq * ffn_dim);
                }
            } else {
                if (!zi_gpu_linear_into_f32(scratch->gate_up, scratch->norm,
                                            block->ffn_w1_bf16, block->ffn_w1,
                                            seq, dim, ffn_dim)) return 0;
                if (!zi_gpu_linear_into_f32(scratch->up, scratch->norm,
                                            block->ffn_w3_bf16, block->ffn_w3,
                                            seq, dim, ffn_dim)) return 0;
                iris_gpu_silu_mul(scratch->gate_up, scratch->up, seq * ffn_dim);
            }
        } else {
            if (!zi_gpu_linear_into_f32(scratch->gate_up, scratch->norm,
                                        block->ffn_w1_bf16, block->ffn_w1,
                                        seq, dim, ffn_dim)) return 0;
            if (!zi_gpu_linear_into_f32(scratch->up, scratch->norm,
                                        block->ffn_w3_bf16, block->ffn_w3,
                                        seq, dim, ffn_dim)) return 0;
            iris_gpu_silu_mul(scratch->gate_up, scratch->up, seq * ffn_dim);
        }
        if (!zi_gpu_linear_into_f32(scratch->down, scratch->gate_up,
                                    block->ffn_w2_bf16, block->ffn_w2,
                                    seq, ffn_dim, dim)) return 0;

        /* GPU: fused ffn_norm2 + residual: x += norm2(ffn_out) */
        if (!iris_gpu_norm_add(hidden_gpu, scratch->down, block->ffn_norm2,
                                seq, dim, ZI_NORM_EPS)) {
            iris_gpu_rms_norm_f32(scratch->norm2, scratch->down, block->ffn_norm2,
                                   seq, dim, ZI_NORM_EPS);
            iris_gpu_add_f32(hidden_gpu, hidden_gpu, scratch->norm2, seq * dim);
        }
    }

    return 1;
}

/* Precompute modulation for one block:
 * mod_out layout = [scale_msa, gate_msa, scale_mlp, gate_mlp], each dim.
 * Scales are stored as (1 + scale), gates are tanh(gate). */
static int zi_precompute_block_modulation(float *mod_out, const zi_block_t *block,
                                          const float *t_emb, int adaln_dim, int dim) {
    if (!mod_out || !block || !block->adaln_weight || !block->adaln_bias || !t_emb) return 0;

    iris_matmul_t(mod_out, t_emb, block->adaln_weight, 1, adaln_dim, 4 * dim);
    vDSP_vadd(mod_out, 1, block->adaln_bias, 1, mod_out, 1, 4 * dim);
    {
        float one = 1.0f;
        int dim_int = dim;
        vDSP_vsadd(mod_out, 1, &one, mod_out, 1, dim);
        vvtanhf(mod_out + dim, mod_out + dim, &dim_int);
        vDSP_vsadd(mod_out + 2 * dim, 1, &one, mod_out + 2 * dim, 1, dim);
        vvtanhf(mod_out + 3 * dim, mod_out + 3 * dim, &dim_int);
    }

    return 1;
}

/* Full GPU-accelerated Z-Image transformer forward pass. Pipeline:
 * CPU timestep embed + patchify + caption norm -> GPU embedding projections ->
 * GPU noise refiner (2 blocks, image only) -> GPU context refiner (2 blocks,
 * caption only) -> GPU concat [img, cap] -> GPU main blocks (30, unified) ->
 * GPU final layer -> CPU unpatchify. Pre-computes all block modulations once
 * per step. Uses batch mode to submit all GPU work in one command buffer.
 * Returns NULL on failure (caller falls back to CPU). */
static float *zi_transformer_forward_gpu(zi_transformer_t *tf,
                                          const float *latent,
                                          int latent_h, int latent_w,
                                          float timestep,
                                          const float *cap_feats,
                                          int cap_seq_len) {
    int dim = tf->dim;
    int ps = tf->patch_size;
    int in_ch = tf->in_channels;
    int patch_feat = ps * ps * in_ch;

    int H_tokens = latent_h / ps;
    int W_tokens = latent_w / ps;
    int img_seq = H_tokens * W_tokens;
    int refiner_total = tf->n_refiner * 2;

    /* No padding for GPU path — GPU attention handles arbitrary seq lengths */
    int cap_padded = cap_seq_len;
    int unified_seq = img_seq + cap_padded;
    int out_ch = ps * ps * in_ch;
    double t_embed_ms = 0.0, t_noise_ms = 0.0, t_context_ms = 0.0;
    double t_main_ms = 0.0, t_final_ms = 0.0;
    double stage_start = zi_time_ms();

    /* Ensure pre-allocated forward buffers are large enough */
    if (!zi_ensure_forward_buffers(tf, img_seq, cap_seq_len, patch_feat, out_ch))
        return NULL;

    /* === CPU: Timestep embedding === */
    float t_emb[256];
    zi_timestep_embed(tf, t_emb, timestep);

    /* === CPU: Patchify image === */
    float *img_patches = tf->fwd_img_patches;
    zi_patchify(img_patches, latent, in_ch, latent_h, latent_w, ps);

    /* === CPU: Caption RMSNorm === */
    float *cap_normed = tf->fwd_cap_normed;
    zi_rms_norm(cap_normed, cap_feats, tf->cap_emb_norm,
                cap_seq_len, tf->cap_feat_dim, ZI_NORM_EPS);

    /* === Embed image/caption (prefer GPU linear, fall back to CPU) === */
    iris_gpu_tensor_t img_gpu = NULL;
    iris_gpu_tensor_t cap_gpu = NULL;

    /* Image projection on GPU */
    iris_gpu_tensor_t img_patch_gpu = iris_gpu_tensor_create(img_patches, (size_t)img_seq * patch_feat);
    if (img_patch_gpu) {
        img_gpu = iris_gpu_linear(img_patch_gpu, tf->x_emb_weight, tf->x_emb_bias,
                                  img_seq, patch_feat, dim);
        iris_gpu_tensor_free(img_patch_gpu);
    }

    /* Caption projection on GPU */
    iris_gpu_tensor_t cap_norm_gpu = iris_gpu_tensor_create(cap_normed, (size_t)cap_seq_len * tf->cap_feat_dim);
    if (cap_norm_gpu) {
        cap_gpu = iris_gpu_linear(cap_norm_gpu, tf->cap_emb_linear_w, tf->cap_emb_linear_b,
                                  cap_seq_len, tf->cap_feat_dim, dim);
        iris_gpu_tensor_free(cap_norm_gpu);
    }

    /* CPU fallback if either embedding projection failed */
    if (!img_gpu || !cap_gpu) {
        if (img_gpu) {
            iris_gpu_tensor_free(img_gpu);
            img_gpu = NULL;
        }
        if (cap_gpu) {
            iris_gpu_tensor_free(cap_gpu);
            cap_gpu = NULL;
        }

        float *img_emb = (float *)malloc((size_t)img_seq * dim * sizeof(float));
        float *cap_emb = (float *)malloc((size_t)cap_seq_len * dim * sizeof(float));
        if (!img_emb || !cap_emb) {
            free(img_emb);
            free(cap_emb);
            return NULL;
        }

        iris_matmul_t(img_emb, img_patches, tf->x_emb_weight, img_seq, patch_feat, dim);
        for (int s = 0; s < img_seq; s++)
            vDSP_vadd(img_emb + s * dim, 1, tf->x_emb_bias, 1, img_emb + s * dim, 1, dim);

        iris_matmul_t(cap_emb, cap_normed, tf->cap_emb_linear_w,
                      cap_seq_len, tf->cap_feat_dim, dim);
        for (int s = 0; s < cap_seq_len; s++)
            vDSP_vadd(cap_emb + s * dim, 1, tf->cap_emb_linear_b, 1, cap_emb + s * dim, 1, dim);

        img_gpu = iris_gpu_tensor_create(img_emb, (size_t)img_seq * dim);
        cap_gpu = iris_gpu_tensor_create(cap_emb, (size_t)cap_seq_len * dim);
        free(img_emb);
        free(cap_emb);
    }

    /* img_patches and cap_normed are pre-allocated in tf — no free needed */

    /* === CPU: Pre-assemble RoPE tables (cached across steps) === */
    if (!zi_gpu_rope_cache_prepare(tf, cap_seq_len, H_tokens, W_tokens)) {
        if (img_gpu) iris_gpu_tensor_free(img_gpu);
        if (cap_gpu) iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }
    const float *img_rope_cos = tf->gpu_img_rope_cos;
    const float *img_rope_sin = tf->gpu_img_rope_sin;
    const float *cap_rope_cos = tf->gpu_cap_rope_cos;
    const float *cap_rope_sin = tf->gpu_cap_rope_sin;
    const float *uni_rope_cos = tf->gpu_uni_rope_cos;
    const float *uni_rope_sin = tf->gpu_uni_rope_sin;
    t_embed_ms = zi_time_ms() - stage_start;

    /* === GPU: Process embedded tokens === */
    if (!img_gpu || !cap_gpu) {
        if (img_gpu) iris_gpu_tensor_free(img_gpu);
        if (cap_gpu) iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }
    iris_gpu_tensor_set_persistent(img_gpu, 1);
    iris_gpu_tensor_set_persistent(cap_gpu, 1);

    /* Allocate scratch for max sequence length (unified_seq) */
    zi_gpu_scratch_t scratch;
    if (!zi_gpu_scratch_init(&scratch, unified_seq, dim, tf->ffn_dim)) {
        iris_gpu_tensor_free(img_gpu);
        iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }

    /* Precompute modulation once per step for all modulated blocks.
     * Uses pre-allocated tf->fwd_step_mod buffer. */
    int n_mod_blocks = tf->n_refiner + tf->n_layers;
    float *step_mod = tf->fwd_step_mod;  /* pre-allocated in ensure_forward_buffers */
    if (n_mod_blocks > 0 && step_mod) {
        int mod_idx = 0;
        int mod_ok = 1;
        for (int i = 0; i < tf->n_refiner && mod_ok; i++) {
            mod_ok = zi_precompute_block_modulation(
                step_mod + (size_t)mod_idx * 4 * dim,
                &tf->noise_refiner[i], t_emb, tf->adaln_dim, dim);
            mod_idx++;
        }
        for (int i = 0; i < tf->n_layers && mod_ok; i++) {
            mod_ok = zi_precompute_block_modulation(
                step_mod + (size_t)mod_idx * 4 * dim,
                &tf->layers[i], t_emb, tf->adaln_dim, dim);
            mod_idx++;
        }
        if (!mod_ok) {
            step_mod = NULL;
        }
    }

    iris_gpu_batch_begin();

    /* === Noise refiner: 2 modulated blocks on image tokens === */
    int gpu_ok = 1;
    int mod_idx = 0;
    stage_start = zi_time_ms();
    for (int i = 0; i < tf->n_refiner && gpu_ok; i++) {
        const float *block_mod = step_mod ? (step_mod + (size_t)mod_idx * 4 * dim) : NULL;
        mod_idx++;
        gpu_ok = zi_block_forward_gpu(img_gpu, &tf->noise_refiner[i],
                                       img_rope_cos, img_rope_sin,
                                       t_emb, block_mod, img_seq, tf, &scratch);
        if (gpu_ok && iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, i, refiner_total);
    }
    t_noise_ms = zi_time_ms() - stage_start;

    /* === Context refiner: 2 unmodulated blocks on caption tokens === */
    stage_start = zi_time_ms();
    for (int i = 0; i < tf->n_refiner && gpu_ok; i++) {
        gpu_ok = zi_block_forward_gpu(cap_gpu, &tf->context_refiner[i],
                                       cap_rope_cos, cap_rope_sin,
                                       NULL, NULL, cap_seq_len, tf, &scratch);
        if (gpu_ok && iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, tf->n_refiner + i, refiner_total);
    }
    t_context_ms = zi_time_ms() - stage_start;

    if (!gpu_ok) {
        iris_gpu_batch_end();
        zi_gpu_scratch_free(&scratch);
        iris_gpu_tensor_free(img_gpu);
        iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }

    /* === Concatenate: unified = [img, cap] === */
    iris_gpu_tensor_t unified_gpu = iris_gpu_tensor_alloc((size_t)unified_seq * dim);
    if (!unified_gpu) {
        iris_gpu_batch_end();
        zi_gpu_scratch_free(&scratch);
        iris_gpu_tensor_free(img_gpu);
        iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }
    iris_gpu_tensor_set_persistent(unified_gpu, 1);

    /* Copy img then cap into unified entirely on GPU (no CPU sync). */
    size_t img_elems = (size_t)img_seq * dim;
    size_t cap_elems = (size_t)cap_seq_len * dim;
    iris_gpu_copy_region_f32(unified_gpu, 0, img_gpu, 0, img_elems);
    iris_gpu_copy_region_f32(unified_gpu, img_elems, cap_gpu, 0, cap_elems);

    iris_gpu_tensor_free(img_gpu);
    iris_gpu_tensor_free(cap_gpu);

    /* === Main transformer: 30 modulated blocks on unified sequence === */
    stage_start = zi_time_ms();
    for (int i = 0; i < tf->n_layers && gpu_ok; i++) {
        const float *block_mod = step_mod ? (step_mod + (size_t)mod_idx * 4 * dim) : NULL;
        mod_idx++;
        gpu_ok = zi_block_forward_gpu(unified_gpu, &tf->layers[i],
                                       uni_rope_cos, uni_rope_sin,
                                       t_emb, block_mod, unified_seq, tf, &scratch);
        if (gpu_ok && iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->n_layers);
    }
    t_main_ms = zi_time_ms() - stage_start;

    if (!gpu_ok) {
        iris_gpu_batch_end();
        zi_gpu_scratch_free(&scratch);
        iris_gpu_tensor_free(unified_gpu);
        return NULL;
    }

    /* === Final layer on GPU: slice image tokens -> LayerNorm+scale -> Linear === */
    stage_start = zi_time_ms();
    iris_gpu_tensor_t img_hidden_gpu = iris_gpu_tensor_alloc((size_t)img_seq * dim);
    iris_gpu_tensor_t final_norm_gpu = iris_gpu_tensor_alloc((size_t)img_seq * dim);
    iris_gpu_tensor_t final_out_gpu = NULL;

    /* Prepare final AdaLN parameters on CPU using pre-allocated buffers. */
    float *final_scale = tf->fwd_gpu_final_scale;
    float *final_shift = tf->fwd_gpu_final_shift;       /* always zero */
    float *final_scale_param = tf->fwd_gpu_final_scale_param;
    if (!img_hidden_gpu || !final_norm_gpu ||
        !zi_final_compute_scale(final_scale, &tf->final_layer, t_emb, tf)) {
        iris_gpu_batch_end();
        zi_gpu_scratch_free(&scratch);
        iris_gpu_tensor_free(unified_gpu);
        if (img_hidden_gpu) iris_gpu_tensor_free(img_hidden_gpu);
        if (final_norm_gpu) iris_gpu_tensor_free(final_norm_gpu);
        return NULL;
    }
    for (int i = 0; i < dim; i++) final_scale_param[i] = final_scale[i] - 1.0f;

    /* Slice first img_seq tokens from unified hidden. */
    iris_gpu_copy_region_f32(img_hidden_gpu, 0, unified_gpu, 0, (size_t)img_seq * dim);
    iris_gpu_adaln_norm(final_norm_gpu, img_hidden_gpu,
                        final_shift, final_scale_param, img_seq, dim, 1e-6f);
    final_out_gpu = iris_gpu_linear(final_norm_gpu, tf->final_layer.linear_weight,
                                    tf->final_layer.linear_bias,
                                    img_seq, dim, out_ch);
    if (!final_out_gpu) {
        iris_gpu_batch_end();
        zi_gpu_scratch_free(&scratch);
        iris_gpu_tensor_free(unified_gpu);
        iris_gpu_tensor_free(img_hidden_gpu);
        iris_gpu_tensor_free(final_norm_gpu);
        return NULL;
    }

    iris_gpu_batch_end();
    zi_gpu_scratch_free(&scratch);
    iris_gpu_tensor_free(unified_gpu);
    iris_gpu_tensor_free(img_hidden_gpu);
    iris_gpu_tensor_free(final_norm_gpu);

    /* Read back final projected patches into pre-allocated buffer. */
    float *final_out = tf->fwd_final_out;
    float *final_out_data = iris_gpu_tensor_data(final_out_gpu);
    memcpy(final_out, final_out_data, (size_t)img_seq * out_ch * sizeof(float));
    iris_gpu_tensor_free(final_out_gpu);
    if (iris_substep_callback)
        iris_substep_callback(IRIS_SUBSTEP_FINAL_LAYER, 0, 1);

    /* === CPU: Unpatchify (zi_unpatchify fills all elements, malloc is safe) === */
    float *output = (float *)malloc((size_t)in_ch * latent_h * latent_w * sizeof(float));
    if (!output) return NULL;
    zi_unpatchify(output, final_out, in_ch, latent_h, latent_w, ps);
    /* final_out is pre-allocated in tf — no free needed */
    t_final_ms = zi_time_ms() - stage_start;

    /* Accumulate per-step zImage GPU timing. */
    iris_timing_zi_embeddings += t_embed_ms;
    iris_timing_zi_noise_refiner += t_noise_ms;
    iris_timing_zi_context_refiner += t_context_ms;
    iris_timing_zi_main_blocks += t_main_ms;
    iris_timing_zi_final += t_final_ms;
    iris_timing_zi_total += t_embed_ms + t_noise_ms + t_context_ms + t_main_ms + t_final_ms;

    return output;
}

#endif /* USE_METAL */

/* ========================================================================
 * Final Layer
 * ======================================================================== */

/* Final layer AdaLN modulation: scale = 1 + Linear(SiLU(t_emb)) */
static int zi_final_compute_scale(float *scale, const zi_final_t *fl,
                                   const float *t_emb, zi_transformer_t *tf) {
    if (!scale || !fl || !t_emb || !tf) return 0;

    float silu_emb[256];
    memcpy(silu_emb, t_emb, tf->adaln_dim * sizeof(float));
    iris_silu(silu_emb, tf->adaln_dim);

    iris_matmul_t(scale, silu_emb, fl->adaln_weight, 1, tf->adaln_dim, tf->dim);
    for (int i = 0; i < tf->dim; i++) scale[i] = 1.0f + scale[i] + fl->adaln_bias[i];
    return 1;
}

/* Z-Image final layer: LayerNorm (no affine) -> scale by
 * (1 + SiLU(Linear(t_emb))) -> Linear projection to patch channels.
 * Note the SiLU activation in the final layer's modulation -- this differs
 * from the block modulation which has no activation. Output shape is
 * [img_seq, patch_size^2 * in_channels]. */
static void zi_final_forward(float *out, const float *x, const zi_final_t *fl,
                               const float *t_emb, int seq, zi_transformer_t *tf) {
    int dim = tf->dim;
    int out_dim = tf->patch_size * tf->patch_size * tf->in_channels;

    /* Use pre-allocated scale buffer from transformer struct */
    float *scale = tf->final_scale;
    if (!scale || !zi_final_compute_scale(scale, fl, t_emb, tf)) {
        return;
    }

    /* Ensure pre-allocated normed buffer is large enough */
    float *normed;
    if (tf->final_normed && seq <= tf->final_normed_cap) {
        normed = tf->final_normed;
    } else {
        /* Grow the pre-allocated buffer */
        free(tf->final_normed);
        tf->final_normed = (float *)malloc((size_t)seq * dim * sizeof(float));
        tf->final_normed_cap = seq;
        normed = tf->final_normed;
        if (!normed) { tf->final_normed_cap = 0; return; }
    }

    /* LayerNorm (no affine) -> scale */
    for (int s = 0; s < seq; s++) {
        const float *xr = x + s * dim;
        float *nr = normed + s * dim;

        /* Compute mean */
        float mean;
        vDSP_meanv(xr, 1, &mean, dim);

        /* Subtract mean: nr = xr - mean */
        float neg_mean = -mean;
        vDSP_vsadd(xr, 1, &neg_mean, nr, 1, dim);

        /* Compute variance = sum((xr - mean)^2) / dim */
        float sum_sq;
        vDSP_svesq(nr, 1, &sum_sq, dim);
        float var = sum_sq / dim;
        float inv_std = 1.0f / sqrtf(var + 1e-6f); /* Final LayerNorm uses 1e-6 */

        /* nr = nr * inv_std */
        vDSP_vsmul(nr, 1, &inv_std, nr, 1, dim);

        /* nr = nr * scale */
        vDSP_vmul(nr, 1, scale, 1, nr, 1, dim);
    }

    /* Linear projection: dim -> out_dim */
    iris_matmul_t(out, normed, fl->linear_weight, seq, dim, out_dim);
    for (int s = 0; s < seq; s++)
        vDSP_vadd(out + s * out_dim, 1, fl->linear_bias, 1, out + s * out_dim, 1, out_dim);

    /* scale and normed are pre-allocated — no free needed */
}

/* ========================================================================
 * Patchify / Unpatchify
 * ======================================================================== */

/* Converts latent [in_ch, H, W] to patch sequence [n_patches, ps*ps*in_ch].
 * Gathers each ps x ps spatial block into a flat vector, ordering as
 * (ph, pw, channel). This is the inverse of unpatchify and creates the
 * token sequence the transformer operates on. */
static void zi_patchify(float *out, const float *latent,
                         int in_ch, int H, int W, int ps) {
    int H_tokens = H / ps;
    int W_tokens = W / ps;
    int patch_feat = ps * ps * in_ch;

    for (int h = 0; h < H_tokens; h++) {
        for (int w = 0; w < W_tokens; w++) {
            int patch_idx = h * W_tokens + w;
            float *dst = out + patch_idx * patch_feat;
            int di = 0;

            /* Gather patch: iterate (ph, pw, c) */
            for (int ph = 0; ph < ps; ph++) {
                for (int pw = 0; pw < ps; pw++) {
                    for (int c = 0; c < in_ch; c++) {
                        int sy = h * ps + ph;
                        int sx = w * ps + pw;
                        dst[di++] = latent[c * H * W + sy * W + sx];
                    }
                }
            }
        }
    }
}

/* Unpatchify: [n_patches, patch_feat_dim] -> [in_ch, H, W] */
static void zi_unpatchify(float *latent, const float *patches,
                            int in_ch, int H, int W, int ps) {
    int H_tokens = H / ps;
    int W_tokens = W / ps;
    int patch_feat = ps * ps * in_ch;

    for (int h = 0; h < H_tokens; h++) {
        for (int w = 0; w < W_tokens; w++) {
            int patch_idx = h * W_tokens + w;
            const float *src = patches + patch_idx * patch_feat;
            int si = 0;

            for (int ph = 0; ph < ps; ph++) {
                for (int pw = 0; pw < ps; pw++) {
                    for (int c = 0; c < in_ch; c++) {
                        int sy = h * ps + ph;
                        int sx = w * ps + pw;
                        latent[c * H * W + sy * W + sx] = src[si++];
                    }
                }
            }
        }
    }
}

/* ========================================================================
 * Main Forward Pass
 * ======================================================================== */

/* Top-level Z-Image transformer entry point. Tries GPU path first, falls
 * back to CPU on failure. CPU path pads sequences to multiples of 32 and
 * uses padding masks. Pipeline: patchify -> embed image/caption ->
 * noise refiner (image self-attention) -> context refiner (caption
 * self-attention) -> concatenate [image, caption] -> main blocks (full
 * self-attention) -> final layer -> unpatchify. */
float *iris_transformer_forward_zimage(zi_transformer_t *tf,
                                const float *latent,
                                int latent_h, int latent_w,
                                float timestep,
                                const float *cap_feats,
                                int cap_seq_len) {
#ifdef USE_METAL
    /* Try GPU-accelerated path first */
    if (tf->use_gpu) {
        float *result = zi_transformer_forward_gpu(tf, latent, latent_h, latent_w,
                                                    timestep, cap_feats, cap_seq_len);
        if (result) return result;
        /* Fall back to CPU on GPU failure */
        fprintf(stderr, "Z-Image GPU path failed, falling back to CPU\n");
    }
#endif

    int dim = tf->dim;
    int ps = tf->patch_size;
    int in_ch = tf->in_channels;
    int patch_feat = ps * ps * in_ch;  /* 64 */

    int H_tokens = latent_h / ps;
    int W_tokens = latent_w / ps;
    int img_seq = H_tokens * W_tokens;
    int refiner_total = tf->n_refiner * 2;

    /* Pad sequences to multiples of ZI_SEQ_MULTI_OF */
    int img_pad = (ZI_SEQ_MULTI_OF - (img_seq % ZI_SEQ_MULTI_OF)) % ZI_SEQ_MULTI_OF;
    int cap_pad = (ZI_SEQ_MULTI_OF - (cap_seq_len % ZI_SEQ_MULTI_OF)) % ZI_SEQ_MULTI_OF;
    int img_padded = img_seq + img_pad;
    int cap_padded = cap_seq_len + cap_pad;
    int unified_seq = img_padded + cap_padded;

    /* Ensure working memory is sufficient */
    size_t needed = (size_t)unified_seq * dim * 4 +
                    (size_t)unified_seq * dim * 3 +  /* QKV */
                    (size_t)unified_seq * unified_seq + /* attention scores */
                    (size_t)unified_seq * tf->ffn_dim * 2;
    if (needed > tf->work_alloc) {
        free(tf->work_x);
        free(tf->work_tmp);
        free(tf->work_qkv);
        free(tf->work_attn);
        free(tf->work_ffn);
        free(tf->work_ffn_fused);
        tf->work_x = (float *)malloc((size_t)unified_seq * dim * sizeof(float));
        tf->work_tmp = (float *)malloc((size_t)unified_seq * dim * 4 * sizeof(float));
        tf->work_qkv = (float *)malloc((size_t)unified_seq * dim * 3 * sizeof(float));
        tf->work_attn = (float *)malloc((size_t)unified_seq * unified_seq * sizeof(float));
        tf->work_ffn = (float *)malloc((size_t)unified_seq * tf->ffn_dim * 2 * sizeof(float));
        tf->work_ffn_fused = (float *)malloc((size_t)unified_seq * tf->ffn_dim * 2 * sizeof(float));
        if (!tf->work_x || !tf->work_tmp || !tf->work_qkv || !tf->work_attn ||
            !tf->work_ffn || !tf->work_ffn_fused) {
            free(tf->work_x); tf->work_x = NULL;
            free(tf->work_tmp); tf->work_tmp = NULL;
            free(tf->work_qkv); tf->work_qkv = NULL;
            free(tf->work_attn); tf->work_attn = NULL;
            free(tf->work_ffn); tf->work_ffn = NULL;
            free(tf->work_ffn_fused); tf->work_ffn_fused = NULL;
            tf->work_alloc = 0;
            tf->max_seq = 0;
            return NULL;
        }
        tf->work_alloc = needed;
        tf->max_seq = unified_seq;
    }

    int out_ch = ps * ps * in_ch;

    /* Ensure pre-allocated forward buffers are large enough */
    if (!zi_ensure_forward_buffers(tf, img_padded, cap_padded, patch_feat, out_ch))
        return NULL;

    /* 1. Timestep embedding */
    float t_emb[256];
    zi_timestep_embed(tf, t_emb, timestep);

    /* 2. Patchify image -> [img_seq, patch_feat] */
    float *img_patches = tf->fwd_img_patches;
    zi_patchify(img_patches, latent, in_ch, latent_h, latent_w, ps);

    /* Pad image patches (repeat last token) */
    for (int i = img_seq; i < img_padded; i++)
        memcpy(img_patches + i * patch_feat,
               img_patches + (img_seq - 1) * patch_feat,
               patch_feat * sizeof(float));

    /* Embed image: [img_padded, patch_feat] -> [img_padded, dim] */
    float *img_emb = tf->fwd_img_emb;
    iris_matmul_t(img_emb, img_patches, tf->x_emb_weight, img_padded, patch_feat, dim);
    for (int s = 0; s < img_padded; s++)
        vDSP_vadd(img_emb + s * dim, 1, tf->x_emb_bias, 1, img_emb + s * dim, 1, dim);
    /* img_patches is pre-allocated in tf — no free needed */

    /* Apply pad token to image padding positions */
    for (int s = img_seq; s < img_padded; s++)
        memcpy(img_emb + s * dim, tf->x_pad_token, dim * sizeof(float));

    /* 3. Caption embedding: RMSNorm -> Linear */
    float *cap_emb = tf->fwd_cap_emb;
    float *cap_normed = tf->fwd_cap_normed;

    /* Pad caption features (repeat last token) */
    float *cap_padded_feats = tf->fwd_cap_padded_feats;
    memcpy(cap_padded_feats, cap_feats, cap_seq_len * tf->cap_feat_dim * sizeof(float));
    for (int s = cap_seq_len; s < cap_padded; s++)
        memcpy(cap_padded_feats + s * tf->cap_feat_dim,
               cap_feats + (cap_seq_len - 1) * tf->cap_feat_dim,
               tf->cap_feat_dim * sizeof(float));

    zi_rms_norm(cap_normed, cap_padded_feats, tf->cap_emb_norm,
                cap_padded, tf->cap_feat_dim, ZI_NORM_EPS);
    /* cap_padded_feats is pre-allocated in tf — no free needed */

    iris_matmul_t(cap_emb, cap_normed, tf->cap_emb_linear_w,
                  cap_padded, tf->cap_feat_dim, dim);
    for (int s = 0; s < cap_padded; s++)
        vDSP_vadd(cap_emb + s * dim, 1, tf->cap_emb_linear_b, 1, cap_emb + s * dim, 1, dim);
    /* cap_normed is pre-allocated in tf — no free needed */

    /* Apply pad token to caption padding positions */
    for (int s = cap_seq_len; s < cap_padded; s++)
        memcpy(cap_emb + s * dim, tf->cap_pad_token, dim * sizeof(float));

    /* 4. Build position IDs and masks — cached when dimensions unchanged */

    int cache_hit = (tf->cpu_cache_latent_h == latent_h &&
                     tf->cpu_cache_latent_w == latent_w &&
                     tf->cpu_cache_cap_seq_len == cap_seq_len &&
                     tf->cpu_cache_img_padded == img_padded &&
                     tf->cpu_cache_cap_padded == cap_padded &&
                     tf->cpu_cache_img_pos != NULL);

    if (!cache_hit) {
        /* Cache miss: rebuild and store */
        free(tf->cpu_cache_img_pos);
        free(tf->cpu_cache_cap_pos);
        free(tf->cpu_cache_img_mask);
        free(tf->cpu_cache_cap_mask);
        free(tf->cpu_cache_unified_pos);
        free(tf->cpu_cache_unified_mask);
        tf->cpu_cache_img_pos = NULL;
        tf->cpu_cache_cap_pos = NULL;
        tf->cpu_cache_img_mask = NULL;
        tf->cpu_cache_cap_mask = NULL;
        tf->cpu_cache_unified_pos = NULL;
        tf->cpu_cache_unified_mask = NULL;

        /* Image position IDs: (T=cap_padded+1, H=h_idx, W=w_idx) */
        tf->cpu_cache_img_pos = (int *)calloc(img_padded * 3, sizeof(int));
        if (!tf->cpu_cache_img_pos) {
            return NULL;
        }
        for (int h = 0; h < H_tokens; h++) {
            for (int w = 0; w < W_tokens; w++) {
                int idx = h * W_tokens + w;
                tf->cpu_cache_img_pos[idx * 3 + 0] = cap_padded + 1;
                tf->cpu_cache_img_pos[idx * 3 + 1] = h;
                tf->cpu_cache_img_pos[idx * 3 + 2] = w;
            }
        }

        /* Caption position IDs: (T=1+seq_idx, H=0, W=0) */
        tf->cpu_cache_cap_pos = (int *)calloc(cap_padded * 3, sizeof(int));
        if (!tf->cpu_cache_cap_pos) {
            return NULL;
        }
        for (int s = 0; s < cap_padded; s++) {
            tf->cpu_cache_cap_pos[s * 3 + 0] = 1 + s;
            tf->cpu_cache_cap_pos[s * 3 + 1] = 0;
            tf->cpu_cache_cap_pos[s * 3 + 2] = 0;
        }

        /* Image mask */
        tf->cpu_cache_img_mask = (int *)malloc(img_padded * sizeof(int));
        if (!tf->cpu_cache_img_mask) {
            return NULL;
        }
        for (int i = 0; i < img_seq; i++) tf->cpu_cache_img_mask[i] = 1;
        for (int i = img_seq; i < img_padded; i++) tf->cpu_cache_img_mask[i] = 0;

        /* Caption mask */
        tf->cpu_cache_cap_mask = (int *)malloc(cap_padded * sizeof(int));
        if (!tf->cpu_cache_cap_mask) {
            return NULL;
        }
        for (int i = 0; i < cap_seq_len; i++) tf->cpu_cache_cap_mask[i] = 1;
        for (int i = cap_seq_len; i < cap_padded; i++) tf->cpu_cache_cap_mask[i] = 0;

        /* Unified position IDs */
        tf->cpu_cache_unified_pos = (int *)malloc(unified_seq * 3 * sizeof(int));
        if (!tf->cpu_cache_unified_pos) {
            return NULL;
        }
        memcpy(tf->cpu_cache_unified_pos,
               tf->cpu_cache_img_pos, img_padded * 3 * sizeof(int));
        memcpy(tf->cpu_cache_unified_pos + img_padded * 3,
               tf->cpu_cache_cap_pos, cap_padded * 3 * sizeof(int));

        /* Unified mask */
        tf->cpu_cache_unified_mask = (int *)malloc(unified_seq * sizeof(int));
        if (!tf->cpu_cache_unified_mask) {
            return NULL;
        }
        memcpy(tf->cpu_cache_unified_mask,
               tf->cpu_cache_img_mask, img_padded * sizeof(int));
        memcpy(tf->cpu_cache_unified_mask + img_padded,
               tf->cpu_cache_cap_mask, cap_padded * sizeof(int));

        /* Update cache key */
        tf->cpu_cache_latent_h = latent_h;
        tf->cpu_cache_latent_w = latent_w;
        tf->cpu_cache_cap_seq_len = cap_seq_len;
        tf->cpu_cache_img_padded = img_padded;
        tf->cpu_cache_cap_padded = cap_padded;
        tf->cpu_cache_unified_seq = unified_seq;
    }

    int *img_pos = tf->cpu_cache_img_pos;
    int *cap_pos = tf->cpu_cache_cap_pos;
    int *img_mask = tf->cpu_cache_img_mask;
    int *cap_mask = tf->cpu_cache_cap_mask;
    int *unified_pos = tf->cpu_cache_unified_pos;
    int *unified_mask = tf->cpu_cache_unified_mask;

    /* 5. Noise refiner: image-only self-attention with modulation */
    for (int i = 0; i < tf->n_refiner; i++) {
        zi_block_forward(img_emb, &tf->noise_refiner[i], img_pos, img_mask,
                          t_emb, img_padded, tf);
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, i, refiner_total);
    }

    /* 6. Context refiner: caption-only self-attention without modulation */
    for (int i = 0; i < tf->n_refiner; i++) {
        zi_block_forward(cap_emb, &tf->context_refiner[i], cap_pos, cap_mask,
                          NULL, cap_padded, tf);
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, tf->n_refiner + i, refiner_total);
    }

    /* 7. Build unified sequence: [image_tokens, caption_tokens] */
    float *unified = tf->work_x;
    memcpy(unified, img_emb, img_padded * dim * sizeof(float));
    memcpy(unified + img_padded * dim, cap_emb, cap_padded * dim * sizeof(float));
    /* img_emb and cap_emb are pre-allocated in tf — no free needed */

    /* 8. Main transformer layers */
    for (int i = 0; i < tf->n_layers; i++) {
        zi_block_forward(unified, &tf->layers[i], unified_pos, unified_mask,
                          t_emb, unified_seq, tf);
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->n_layers);
    }

    /* Position/mask arrays are cached in transformer struct — no free here */

    /* 9. Final layer: extract image tokens only, then project.
     * img_out and final_out use pre-allocated buffers. */
    float *img_out = tf->fwd_img_out;
    memcpy(img_out, unified, (size_t)img_seq * dim * sizeof(float));

    /* out_ch already declared at top of function */
    float *final_out = tf->fwd_final_out;
    zi_final_forward(final_out, img_out, &tf->final_layer, t_emb, img_seq, tf);
    /* img_out is pre-allocated in tf — no free needed */
    if (iris_substep_callback)
        iris_substep_callback(IRIS_SUBSTEP_FINAL_LAYER, 0, 1);

    /* 10. Unpatchify: [n_patches, 64] -> [16, latent_h, latent_w]
     * zi_unpatchify fills all elements, so malloc is safe (no need for calloc). */
    float *output = (float *)malloc((size_t)in_ch * latent_h * latent_w * sizeof(float));
    if (!output) return NULL;
    zi_unpatchify(output, final_out, in_ch, latent_h, latent_w, ps);
    /* final_out is pre-allocated in tf — no free needed */

    return output;
}

/* ========================================================================
 * Weight Loading (Safetensors)
 * ======================================================================== */

static float *zi_get_tensor(safetensors_file_t **files, int n_files,
                              const char *name, int mmap_f32_weights) {
    for (int f = 0; f < n_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (!t) continue;
        if (mmap_f32_weights) {
            if (t->dtype != DTYPE_F32) {
                fprintf(stderr, "Error: Z-Image tensor '%s' is not F32 in mmap mode\n", name);
                return NULL;
            }
            return (float *)safetensors_data(files[f], t);
        }
        return safetensors_get_f32(files[f], t);
    }
    fprintf(stderr, "Warning: Z-Image tensor '%s' not found\n", name);
    return NULL;
}

static uint16_t *zi_get_tensor_bf16_direct(safetensors_file_t **files, int n_files,
                                            const char *name) {
    for (int f = 0; f < n_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (!t) continue;
        if (t->dtype != DTYPE_BF16) return NULL;
        return (uint16_t *)safetensors_data(files[f], t);
    }
    return NULL;
}

static float *zi_get_tensor_optional(safetensors_file_t **files, int n_files,
                                       const char *name, int mmap_f32_weights) {
    for (int f = 0; f < n_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (!t) continue;
        if (mmap_f32_weights) {
            if (t->dtype != DTYPE_F32) return NULL;
            return (float *)safetensors_data(files[f], t);
        }
        return safetensors_get_f32(files[f], t);
    }
    return NULL;
}

static int zi_all_tensors_f32(safetensors_file_t **files, int n_files) {
    for (int f = 0; f < n_files; f++) {
        safetensors_file_t *sf = files[f];
        if (!sf) return 0;
        for (int i = 0; i < sf->num_tensors; i++) {
            if (sf->tensors[i].dtype != DTYPE_F32) return 0;
        }
    }
    return 1;
}

/* Load only the small F32 norm/adaln weights for a block (used when
 * BF16 fused weights come from cache — norms are always F32). */
static int zi_load_block_norms(zi_block_t *block, safetensors_file_t **files,
                                int n_files, const char *prefix, int has_modulation) {
    char name[256];

    snprintf(name, sizeof(name), "%s.attention.norm_q.weight", prefix);
    block->attn_norm_q = zi_get_tensor(files, n_files, name, 0);
    snprintf(name, sizeof(name), "%s.attention.norm_k.weight", prefix);
    block->attn_norm_k = zi_get_tensor(files, n_files, name, 0);
    snprintf(name, sizeof(name), "%s.attention_norm1.weight", prefix);
    block->attn_norm1 = zi_get_tensor(files, n_files, name, 0);
    snprintf(name, sizeof(name), "%s.attention_norm2.weight", prefix);
    block->attn_norm2 = zi_get_tensor(files, n_files, name, 0);

    snprintf(name, sizeof(name), "%s.ffn_norm1.weight", prefix);
    block->ffn_norm1 = zi_get_tensor(files, n_files, name, 0);
    snprintf(name, sizeof(name), "%s.ffn_norm2.weight", prefix);
    block->ffn_norm2 = zi_get_tensor(files, n_files, name, 0);

    if (has_modulation) {
        snprintf(name, sizeof(name), "%s.adaLN_modulation.0.weight", prefix);
        block->adaln_weight = zi_get_tensor(files, n_files, name, 0);
        snprintf(name, sizeof(name), "%s.adaLN_modulation.0.bias", prefix);
        block->adaln_bias = zi_get_tensor(files, n_files, name, 0);
    }

    if (!block->attn_norm_q || !block->attn_norm_k ||
        !block->attn_norm1 || !block->attn_norm2 ||
        !block->ffn_norm1 || !block->ffn_norm2) {
        return 0;
    }
    if (has_modulation && (!block->adaln_weight || !block->adaln_bias)) {
        return 0;
    }
    return 1;
}

static int zi_load_block(zi_block_t *block, safetensors_file_t **files,
                          int n_files, const char *prefix, int has_modulation,
                          int dim, int ffn_dim, int use_gpu,
                          int mmap_f32_weights, int mmap_bf16, int mmap_gpu_f32) {
    char name[256];

#ifdef USE_METAL
    block->bf16_from_mmap = 0;
    block->bf16_fused_from_cache = 0;
#endif

    /* For GPU mmap with F32 weights, use direct mmap pointers for large tensors
     * to avoid malloc+memcpy. The pointers are only used temporarily as source
     * for F32→BF16 conversion, then NULLed without free. Small tensors (norms,
     * adaln) are still copied since they're tiny and used as-is in F32. */
    int use_mmap_for_large = (mmap_f32_weights || mmap_gpu_f32);


    /* Attention weights */
    snprintf(name, sizeof(name), "%s.attention.to_q.weight", prefix);
    block->attn_q_weight = zi_get_tensor(files, n_files, name, use_mmap_for_large);
    snprintf(name, sizeof(name), "%s.attention.to_k.weight", prefix);
    block->attn_k_weight = zi_get_tensor(files, n_files, name, use_mmap_for_large);
    snprintf(name, sizeof(name), "%s.attention.to_v.weight", prefix);
    block->attn_v_weight = zi_get_tensor(files, n_files, name, use_mmap_for_large);
    snprintf(name, sizeof(name), "%s.attention.to_out.0.weight", prefix);
    block->attn_out_weight = zi_get_tensor(files, n_files, name, use_mmap_for_large);

    /* QK norm — small, always copy */
    snprintf(name, sizeof(name), "%s.attention.norm_q.weight", prefix);
    block->attn_norm_q = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.attention.norm_k.weight", prefix);
    block->attn_norm_k = zi_get_tensor(files, n_files, name, mmap_f32_weights);

    /* Pre/post attention norms — small, always copy */
    snprintf(name, sizeof(name), "%s.attention_norm1.weight", prefix);
    block->attn_norm1 = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.attention_norm2.weight", prefix);
    block->attn_norm2 = zi_get_tensor(files, n_files, name, mmap_f32_weights);

    /* FFN weights */
    snprintf(name, sizeof(name), "%s.feed_forward.w1.weight", prefix);
    block->ffn_w1 = zi_get_tensor(files, n_files, name, use_mmap_for_large);
    snprintf(name, sizeof(name), "%s.feed_forward.w2.weight", prefix);
    block->ffn_w2 = zi_get_tensor(files, n_files, name, use_mmap_for_large);
    snprintf(name, sizeof(name), "%s.feed_forward.w3.weight", prefix);
    block->ffn_w3 = zi_get_tensor(files, n_files, name, use_mmap_for_large);

    /* FFN norms — small, always copy */
    snprintf(name, sizeof(name), "%s.ffn_norm1.weight", prefix);
    block->ffn_norm1 = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.ffn_norm2.weight", prefix);
    block->ffn_norm2 = zi_get_tensor(files, n_files, name, mmap_f32_weights);

    /* AdaLN modulation — small, always copy */
    if (has_modulation) {
        snprintf(name, sizeof(name), "%s.adaLN_modulation.0.weight", prefix);
        block->adaln_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
        snprintf(name, sizeof(name), "%s.adaLN_modulation.0.bias", prefix);
        block->adaln_bias = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    } else {
        block->adaln_weight = NULL;
        block->adaln_bias = NULL;
    }


    if (!block->attn_q_weight || !block->attn_k_weight || !block->attn_v_weight ||
        !block->attn_out_weight || !block->attn_norm_q || !block->attn_norm_k ||
        !block->attn_norm1 || !block->attn_norm2 || !block->ffn_w1 ||
        !block->ffn_w2 || !block->ffn_w3 || !block->ffn_norm1 ||
        !block->ffn_norm2) {
        return 0;
    }
    if (has_modulation && (!block->adaln_weight || !block->adaln_bias)) {
        return 0;
    }

    /* Build fused F32 weight matrices for CPU single-GEMM dispatch.
     * These are only used by the CPU path; the GPU path frees F32 weights
     * and uses its own BF16 fused weights. We build them unconditionally
     * here — if the GPU path runs, it NULLs the F32 originals and these
     * fused copies remain orphaned but harmless (freed in zi_free_block). */
    if (!use_gpu) {
        /* QKV fusion: [3*dim, dim] = [Q; K; V] stacked row-wise */
        size_t attn_elems = (size_t)dim * dim;
        block->attn_qkv_weight_f32 = (float *)malloc(3 * attn_elems * sizeof(float));
        if (block->attn_qkv_weight_f32) {
            memcpy(block->attn_qkv_weight_f32,
                   block->attn_q_weight, attn_elems * sizeof(float));
            memcpy(block->attn_qkv_weight_f32 + attn_elems,
                   block->attn_k_weight, attn_elems * sizeof(float));
            memcpy(block->attn_qkv_weight_f32 + 2 * attn_elems,
                   block->attn_v_weight, attn_elems * sizeof(float));
        }

        /* W1/W3 fusion: [2*ffn_dim, dim] = [W1; W3] stacked row-wise */
        size_t ffn_elems = (size_t)ffn_dim * dim;
        block->ffn_w13_f32 = (float *)malloc(2 * ffn_elems * sizeof(float));
        if (block->ffn_w13_f32) {
            memcpy(block->ffn_w13_f32,
                   block->ffn_w1, ffn_elems * sizeof(float));
            memcpy(block->ffn_w13_f32 + ffn_elems,
                   block->ffn_w3, ffn_elems * sizeof(float));
        }
    } else {
        block->attn_qkv_weight_f32 = NULL;
        block->ffn_w13_f32 = NULL;
    }

#ifdef USE_METAL
    if (use_gpu && mmap_bf16) {
        /* GPU mmap BF16: read BF16 directly from mmap'd safetensors (zero-copy).
         * Only the fused QKV/W13 concatenations need allocation. */
        size_t attn_mat_elems = (size_t)dim * dim;
        size_t ffn_mat_elems = (size_t)ffn_dim * dim;
        block->bf16_from_mmap = 1;

        snprintf(name, sizeof(name), "%s.attention.to_q.weight", prefix);
        block->attn_q_weight_bf16 = zi_get_tensor_bf16_direct(files, n_files, name);
        snprintf(name, sizeof(name), "%s.attention.to_k.weight", prefix);
        block->attn_k_weight_bf16 = zi_get_tensor_bf16_direct(files, n_files, name);
        snprintf(name, sizeof(name), "%s.attention.to_v.weight", prefix);
        block->attn_v_weight_bf16 = zi_get_tensor_bf16_direct(files, n_files, name);
        snprintf(name, sizeof(name), "%s.attention.to_out.0.weight", prefix);
        block->attn_out_weight_bf16 = zi_get_tensor_bf16_direct(files, n_files, name);
        snprintf(name, sizeof(name), "%s.feed_forward.w1.weight", prefix);
        block->ffn_w1_bf16 = zi_get_tensor_bf16_direct(files, n_files, name);
        snprintf(name, sizeof(name), "%s.feed_forward.w2.weight", prefix);
        block->ffn_w2_bf16 = zi_get_tensor_bf16_direct(files, n_files, name);
        snprintf(name, sizeof(name), "%s.feed_forward.w3.weight", prefix);
        block->ffn_w3_bf16 = zi_get_tensor_bf16_direct(files, n_files, name);
        if (!block->attn_q_weight_bf16 || !block->attn_k_weight_bf16 ||
            !block->attn_v_weight_bf16 || !block->attn_out_weight_bf16 ||
            !block->ffn_w1_bf16 || !block->ffn_w2_bf16 || !block->ffn_w3_bf16) {
            return 0;
        }

        block->attn_qkv_weight_bf16 = zi_concat3_bf16(block->attn_q_weight_bf16, attn_mat_elems,
                                                       block->attn_k_weight_bf16, attn_mat_elems,
                                                       block->attn_v_weight_bf16, attn_mat_elems);
        block->ffn_w13_weight_bf16 = zi_concat_bf16(block->ffn_w1_bf16, ffn_mat_elems,
                                                    block->ffn_w3_bf16, ffn_mat_elems);
        /* NULL from concat is OK — forward pass falls back to separate weights */
        if (!block->attn_qkv_weight_bf16)
            fprintf(stderr, "zi_load_block: fused QKV alloc failed (%s), using separate weights\n", prefix);
        if (!block->ffn_w13_weight_bf16)
            fprintf(stderr, "zi_load_block: fused W1W3 alloc failed (%s), using separate weights\n", prefix);

        /* F32 pointers were from BF16→F32 conversion, free them */
        free(block->attn_q_weight); block->attn_q_weight = NULL;
        free(block->attn_k_weight); block->attn_k_weight = NULL;
        free(block->attn_v_weight); block->attn_v_weight = NULL;
        free(block->attn_out_weight); block->attn_out_weight = NULL;
        free(block->ffn_w1); block->ffn_w1 = NULL;
        free(block->ffn_w2); block->ffn_w2 = NULL;
        free(block->ffn_w3); block->ffn_w3 = NULL;
    } else if (use_gpu) {
        /* GPU path: convert F32→BF16. Source may be mmap'd (mmap_gpu_f32) or
         * heap-allocated. zi_f32_to_bf16 only reads, doesn't care about source. */
        size_t attn_mat_elems = (size_t)dim * dim;
        size_t ffn_mat_elems = (size_t)ffn_dim * dim;

        block->attn_q_weight_bf16 = zi_f32_to_bf16(block->attn_q_weight, (size_t)dim * dim);
        block->attn_k_weight_bf16 = zi_f32_to_bf16(block->attn_k_weight, (size_t)dim * dim);
        block->attn_v_weight_bf16 = zi_f32_to_bf16(block->attn_v_weight, (size_t)dim * dim);
        block->attn_out_weight_bf16 = zi_f32_to_bf16(block->attn_out_weight, (size_t)dim * dim);
        block->ffn_w1_bf16 = zi_f32_to_bf16(block->ffn_w1, (size_t)ffn_dim * dim);
        block->ffn_w2_bf16 = zi_f32_to_bf16(block->ffn_w2, (size_t)dim * ffn_dim);
        block->ffn_w3_bf16 = zi_f32_to_bf16(block->ffn_w3, (size_t)ffn_dim * dim);
        if (!block->attn_q_weight_bf16 || !block->attn_k_weight_bf16 ||
            !block->attn_v_weight_bf16 || !block->attn_out_weight_bf16 ||
            !block->ffn_w1_bf16 || !block->ffn_w2_bf16 || !block->ffn_w3_bf16) {
            return 0;
        }

        block->attn_qkv_weight_bf16 = zi_concat3_bf16(block->attn_q_weight_bf16, attn_mat_elems,
                                                       block->attn_k_weight_bf16, attn_mat_elems,
                                                       block->attn_v_weight_bf16, attn_mat_elems);
        block->ffn_w13_weight_bf16 = zi_concat_bf16(block->ffn_w1_bf16, ffn_mat_elems,
                                                    block->ffn_w3_bf16, ffn_mat_elems);
        /* NULL from concat is OK — forward pass falls back to separate weights */
        if (!block->attn_qkv_weight_bf16)
            fprintf(stderr, "zi_load_block: fused QKV alloc failed (%s), using separate weights\n", prefix);
        if (!block->ffn_w13_weight_bf16)
            fprintf(stderr, "zi_load_block: fused W1W3 alloc failed (%s), using separate weights\n", prefix);

        /* NULL out f32 pointers — don't free if they're mmap'd */
        if (mmap_gpu_f32) {
            block->attn_q_weight = NULL;
            block->attn_k_weight = NULL;
            block->attn_v_weight = NULL;
            block->attn_out_weight = NULL;
            block->ffn_w1 = NULL;
            block->ffn_w2 = NULL;
            block->ffn_w3 = NULL;
        } else {
            free(block->attn_q_weight); block->attn_q_weight = NULL;
            free(block->attn_k_weight); block->attn_k_weight = NULL;
            free(block->attn_v_weight); block->attn_v_weight = NULL;
            free(block->attn_out_weight); block->attn_out_weight = NULL;
            free(block->ffn_w1); block->ffn_w1 = NULL;
            free(block->ffn_w2); block->ffn_w2 = NULL;
            free(block->ffn_w3); block->ffn_w3 = NULL;
        }
    }
#else
    (void)mmap_f32_weights; (void)mmap_bf16; (void)mmap_gpu_f32;
#endif
    return 1;
}

static void zi_free_block(zi_block_t *block, int free_f32_weights) {
    /* Fused F32 weights are always heap-allocated (never mmap'd) */
    free(block->attn_qkv_weight_f32);
    free(block->ffn_w13_f32);
    block->attn_qkv_weight_f32 = NULL;
    block->ffn_w13_f32 = NULL;

    if (free_f32_weights) {
        free(block->attn_q_weight);
        free(block->attn_k_weight);
        free(block->attn_v_weight);
        free(block->attn_out_weight);
        free(block->attn_norm_q);
        free(block->attn_norm_k);
        free(block->attn_norm1);
        free(block->attn_norm2);
        free(block->ffn_w1);
        free(block->ffn_w2);
        free(block->ffn_w3);
        free(block->ffn_norm1);
        free(block->ffn_norm2);
        free(block->adaln_weight);
        free(block->adaln_bias);
    }
#ifdef USE_METAL
    if (!block->bf16_from_mmap) {
        free(block->attn_q_weight_bf16);
        free(block->attn_k_weight_bf16);
        free(block->attn_v_weight_bf16);
        free(block->attn_out_weight_bf16);
        free(block->ffn_w1_bf16);
        free(block->ffn_w2_bf16);
        free(block->ffn_w3_bf16);
    }
    /* Fused weights are heap-allocated by zi_concat3_bf16 / zi_concat_bf16
     * even when bf16_from_mmap=1 (mmap'd BF16 safetensors path).
     * Only skip free if they point into a cache mmap region. */
    if (!block->bf16_fused_from_cache) {
        free(block->attn_qkv_weight_bf16);
        free(block->ffn_w13_weight_bf16);
    }
    block->attn_qkv_weight_bf16 = NULL;
    block->ffn_w13_weight_bf16 = NULL;
#endif
}

/* ========================================================================
 * BF16 Weight Cache
 *
 * Pre-converts F32 weights to BF16 and stores them in a binary cache file.
 * On subsequent loads, the cache is mmap'd and weights are used zero-copy.
 * Eliminates the ~2.6s F32→BF16 conversion on every model load.
 *
 * Layout: [header 4096 bytes] [block 0 data] [block 1 data] ... [block N-1 data]
 * Per block: fused_qkv[3*dim*dim] + attn_out[dim*dim] + fused_w13[2*ffn_dim*dim]
 *            + ffn_w2[dim*ffn_dim] — all BF16 (uint16_t)
 * ======================================================================== */

#define ZI_CACHE_MAGIC   0x4942465A  /* "ZFBI" (Z-Image Fast BF16 Inference) */
#define ZI_CACHE_VERSION 1
#define ZI_CACHE_HEADER  4096        /* page-aligned header */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_blocks;
    uint32_t dim;
    uint32_t ffn_dim;
    uint32_t pad[3];
    /* Validation: total size of all source safetensors files */
    uint64_t source_total_size;
} zi_cache_header_t;

static size_t zi_cache_block_size(int dim, int ffn_dim) {
    size_t qkv   = (size_t)3 * dim * dim;
    size_t out    = (size_t)dim * dim;
    size_t w13   = (size_t)2 * ffn_dim * dim;
    size_t w2    = (size_t)dim * ffn_dim;
    return (qkv + out + w13 + w2) * sizeof(uint16_t);
}

static uint64_t zi_source_total_size(safetensors_file_t **files, int n_files) {
    uint64_t total = 0;
    for (int f = 0; f < n_files; f++) {
        if (!files[f]) continue;
        total += files[f]->file_size;
    }
    return total;
}

#ifdef USE_METAL
/* Try to load BF16 cache. Returns mmap'd base pointer or NULL. */
static void *zi_cache_open(const char *model_dir, int n_blocks, int dim, int ffn_dim,
                            safetensors_file_t **files, int n_files, size_t *out_size) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/transformer/.iris_bf16_cache", model_dir);

    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }

    size_t expected = ZI_CACHE_HEADER + (size_t)n_blocks * zi_cache_block_size(dim, ffn_dim);
    if ((size_t)st.st_size != expected) { close(fd); return NULL; }

    void *base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return NULL;

    zi_cache_header_t *hdr = (zi_cache_header_t *)base;
    if (hdr->magic != ZI_CACHE_MAGIC || hdr->version != ZI_CACHE_VERSION ||
        hdr->n_blocks != (uint32_t)n_blocks || hdr->dim != (uint32_t)dim ||
        hdr->ffn_dim != (uint32_t)ffn_dim) {
        munmap(base, st.st_size);
        return NULL;
    }

    /* Validate against source files */
    uint64_t src_size = zi_source_total_size(files, n_files);
    if (hdr->source_total_size != src_size) {
        munmap(base, st.st_size);
        return NULL;
    }

    madvise(base, st.st_size, MADV_SEQUENTIAL);
    *out_size = st.st_size;
    return base;
}

/* Set block BF16 pointers from cache mmap region. */
static void zi_cache_load_block(zi_block_t *block, void *cache_base,
                                 int block_idx, int dim, int ffn_dim) {
    size_t block_bytes = zi_cache_block_size(dim, ffn_dim);
    uint16_t *base = (uint16_t *)((char *)cache_base + ZI_CACHE_HEADER + (size_t)block_idx * block_bytes);

    size_t qkv_elems = (size_t)3 * dim * dim;
    size_t out_elems = (size_t)dim * dim;
    size_t w13_elems = (size_t)2 * ffn_dim * dim;

    block->attn_qkv_weight_bf16 = base;
    base += qkv_elems;

    block->attn_out_weight_bf16 = base;
    base += out_elems;

    block->ffn_w13_weight_bf16 = base;
    base += w13_elems;

    block->ffn_w2_bf16 = base;

    /* Individual pointers not stored in cache — fused path handles it */
    block->attn_q_weight_bf16 = NULL;
    block->attn_k_weight_bf16 = NULL;
    block->attn_v_weight_bf16 = NULL;
    block->ffn_w1_bf16 = NULL;
    block->ffn_w3_bf16 = NULL;

#ifdef USE_METAL
    block->bf16_from_mmap = 1;
    block->bf16_fused_from_cache = 1;
#endif
}

/* Write BF16 cache file from loaded blocks. */
static int zi_cache_write(const char *model_dir, zi_transformer_t *tf,
                           safetensors_file_t **files, int n_files) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/transformer/.iris_bf16_cache", model_dir);

    int total_blocks = tf->n_refiner * 2 + tf->n_layers;

    /* Write to temp file then rename for atomicity */
    char tmp_path[1024];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

    int fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 0;

    /* Write header */
    char header[ZI_CACHE_HEADER];
    memset(header, 0, sizeof(header));
    zi_cache_header_t *hdr = (zi_cache_header_t *)header;
    hdr->magic = ZI_CACHE_MAGIC;
    hdr->version = ZI_CACHE_VERSION;
    hdr->n_blocks = total_blocks;
    hdr->dim = tf->dim;
    hdr->ffn_dim = tf->ffn_dim;
    hdr->source_total_size = zi_source_total_size(files, n_files);
    if (write(fd, header, ZI_CACHE_HEADER) != ZI_CACHE_HEADER) {
        close(fd); unlink(tmp_path); return 0;
    }

    /* Write block data: noise_refiner, context_refiner, layers */
    zi_block_t *groups[3] = { tf->noise_refiner, tf->context_refiner, tf->layers };
    int counts[3] = { tf->n_refiner, tf->n_refiner, tf->n_layers };

    size_t qkv_elems = (size_t)3 * tf->dim * tf->dim;
    size_t out_elems = (size_t)tf->dim * tf->dim;
    size_t w13_elems = (size_t)2 * tf->ffn_dim * tf->dim;
    size_t w2_elems  = (size_t)tf->dim * tf->ffn_dim;

    for (int g = 0; g < 3; g++) {
        for (int i = 0; i < counts[g]; i++) {
            zi_block_t *b = &groups[g][i];
            if (write(fd, b->attn_qkv_weight_bf16, qkv_elems * 2) != (ssize_t)(qkv_elems * 2) ||
                write(fd, b->attn_out_weight_bf16, out_elems * 2) != (ssize_t)(out_elems * 2) ||
                write(fd, b->ffn_w13_weight_bf16, w13_elems * 2) != (ssize_t)(w13_elems * 2) ||
                write(fd, b->ffn_w2_bf16, w2_elems * 2) != (ssize_t)(w2_elems * 2)) {
                close(fd); unlink(tmp_path); return 0;
            }
        }
    }

    close(fd);
    rename(tmp_path, path);
    return 1;
}
/* ========================================================================
 * F16 Weight Cache (zero-copy Metal path)
 *
 * Stores weights as F16 (not BF16) with a page-aligned layout so that
 * each weight tensor can be wrapped directly as a Metal buffer via
 * newBufferWithBytesNoCopy — zero conversion, zero copy. On Apple
 * Silicon unified memory, the GPU reads directly from the mmap'd pages.
 *
 * Layout: [header 16384 bytes] [block 0 data] ... [block N-1 data]
 * Per block: fused_qkv + attn_out + fused_w13 + ffn_w2 — all F16.
 * All offsets and sizes are page-aligned (16384 on Apple Silicon).
 * ======================================================================== */

#define ZI_F16_CACHE_MAGIC   0x36314649  /* "IF16" */
#define ZI_F16_CACHE_VERSION 1
#define ZI_F16_CACHE_HEADER  16384       /* page-aligned header */

static void *zi_f16_cache_open(const char *model_dir, int n_blocks, int dim, int ffn_dim,
                                safetensors_file_t **files, int n_files, size_t *out_size) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/transformer/.iris_f16_cache", model_dir);

    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }

    size_t expected = ZI_F16_CACHE_HEADER + (size_t)n_blocks * zi_cache_block_size(dim, ffn_dim);
    if ((size_t)st.st_size != expected) { close(fd); return NULL; }

    void *base = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return NULL;

    zi_cache_header_t *hdr = (zi_cache_header_t *)base;
    if (hdr->magic != ZI_F16_CACHE_MAGIC || hdr->version != ZI_F16_CACHE_VERSION ||
        hdr->n_blocks != (uint32_t)n_blocks || hdr->dim != (uint32_t)dim ||
        hdr->ffn_dim != (uint32_t)ffn_dim) {
        munmap(base, st.st_size);
        return NULL;
    }

    uint64_t src_size = zi_source_total_size(files, n_files);
    if (hdr->source_total_size != src_size) {
        munmap(base, st.st_size);
        return NULL;
    }

    madvise(base, st.st_size, MADV_RANDOM);
    *out_size = st.st_size;
    return base;
}

/* Set block BF16 pointers to the F16 data from cache.
 * The inference path calls get_cached_bf16_as_f16_buffer() keyed by these
 * pointers — we pre-register them via iris_metal_register_f16_nocopy()
 * so the lookup returns the zero-copy Metal buffer immediately. */
static void zi_f16_cache_load_block(zi_block_t *block, void *cache_base,
                                     int block_idx, int dim, int ffn_dim) {
    size_t block_bytes = zi_cache_block_size(dim, ffn_dim);
    uint16_t *base = (uint16_t *)((char *)cache_base + ZI_F16_CACHE_HEADER +
                                   (size_t)block_idx * block_bytes);

    size_t qkv_elems = (size_t)3 * dim * dim;
    size_t out_elems = (size_t)dim * dim;
    size_t w13_elems = (size_t)2 * ffn_dim * dim;

    block->attn_qkv_weight_bf16 = base;  base += qkv_elems;
    block->attn_out_weight_bf16 = base;   base += out_elems;
    block->ffn_w13_weight_bf16 = base;    base += w13_elems;
    block->ffn_w2_bf16 = base;

    block->attn_q_weight_bf16 = NULL;
    block->attn_k_weight_bf16 = NULL;
    block->attn_v_weight_bf16 = NULL;
    block->ffn_w1_bf16 = NULL;
    block->ffn_w3_bf16 = NULL;
    block->bf16_from_mmap = 1;
    block->bf16_fused_from_cache = 1;
}

static inline void zi_bf16_to_f16_bulk(uint16_t *out, const uint16_t *in, size_t n) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        bfloat16x4_t b = vld1_bf16((const bfloat16_t *)(in + i));
        float32x4_t f32 = vcvt_f32_bf16(b);
        float16x4_t f16 = vcvt_f16_f32(f32);
        vst1_f16((__fp16 *)(out + i), f16);
    }
    for (; i < n; i++) {
        uint32_t sign = (in[i] >> 15) & 1;
        int32_t exp = (in[i] >> 7) & 0xFF;
        uint32_t mant = in[i] & 0x7F;
        if (exp == 0) { out[i] = (uint16_t)(sign << 15); continue; }
        if (exp == 0xFF) { out[i] = (uint16_t)((sign << 15) | 0x7C00 | (mant ? 0x200 : 0)); continue; }
        int32_t new_exp = exp - 127 + 15;
        if (new_exp <= 0) { out[i] = (uint16_t)(sign << 15); continue; }
        if (new_exp >= 31) { out[i] = (uint16_t)((sign << 15) | 0x7C00); continue; }
        out[i] = (uint16_t)((sign << 15) | (new_exp << 10) | (mant << 3));
    }
#else
    for (size_t i = 0; i < n; i++) {
        uint32_t sign = (in[i] >> 15) & 1;
        int32_t exp = (in[i] >> 7) & 0xFF;
        uint32_t mant = in[i] & 0x7F;
        if (exp == 0) { out[i] = (uint16_t)(sign << 15); continue; }
        if (exp == 0xFF) { out[i] = (uint16_t)((sign << 15) | 0x7C00 | (mant ? 0x200 : 0)); continue; }
        int32_t new_exp = exp - 127 + 15;
        if (new_exp <= 0) { out[i] = (uint16_t)(sign << 15); continue; }
        if (new_exp >= 31) { out[i] = (uint16_t)((sign << 15) | 0x7C00); continue; }
        out[i] = (uint16_t)((sign << 15) | (new_exp << 10) | (mant << 3));
    }
#endif
}

/* Write F16 cache by converting BF16 weights to F16.
 * One-time cost after first load; subsequent loads use zero-copy mmap. */
static int zi_f16_cache_write(const char *model_dir, zi_transformer_t *tf,
                               safetensors_file_t **files, int n_files) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/transformer/.iris_f16_cache", model_dir);

    int total_blocks = tf->n_refiner * 2 + tf->n_layers;

    char tmp_path[1024];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

    int fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 0;

    char header[ZI_F16_CACHE_HEADER];
    memset(header, 0, sizeof(header));
    zi_cache_header_t *hdr = (zi_cache_header_t *)header;
    hdr->magic = ZI_F16_CACHE_MAGIC;
    hdr->version = ZI_F16_CACHE_VERSION;
    hdr->n_blocks = total_blocks;
    hdr->dim = tf->dim;
    hdr->ffn_dim = tf->ffn_dim;
    hdr->source_total_size = zi_source_total_size(files, n_files);
    if (write(fd, header, ZI_F16_CACHE_HEADER) != ZI_F16_CACHE_HEADER) {
        close(fd); unlink(tmp_path); return 0;
    }

    zi_block_t *groups[3] = { tf->noise_refiner, tf->context_refiner, tf->layers };
    int counts[3] = { tf->n_refiner, tf->n_refiner, tf->n_layers };

    size_t qkv_elems = (size_t)3 * tf->dim * tf->dim;
    size_t out_elems = (size_t)tf->dim * tf->dim;
    size_t w13_elems = (size_t)2 * tf->ffn_dim * tf->dim;
    size_t w2_elems  = (size_t)tf->dim * tf->ffn_dim;

    size_t max_elems = w13_elems;
    uint16_t *f16_tmp = malloc(max_elems * sizeof(uint16_t));
    if (!f16_tmp) { close(fd); unlink(tmp_path); return 0; }

    for (int g = 0; g < 3; g++) {
        for (int i = 0; i < counts[g]; i++) {
            zi_block_t *b = &groups[g][i];
            struct { const uint16_t *src; size_t n; } weights[] = {
                { b->attn_qkv_weight_bf16, qkv_elems },
                { b->attn_out_weight_bf16, out_elems },
                { b->ffn_w13_weight_bf16,  w13_elems },
                { b->ffn_w2_bf16,          w2_elems },
            };
            for (int w = 0; w < 4; w++) {
                zi_bf16_to_f16_bulk(f16_tmp, weights[w].src, weights[w].n);
                ssize_t wb = weights[w].n * 2;
                if (write(fd, f16_tmp, wb) != wb) {
                    free(f16_tmp); close(fd); unlink(tmp_path); return 0;
                }
            }
        }
    }

    free(f16_tmp);
    close(fd);
    rename(tmp_path, path);
    return 1;
}
#endif /* USE_METAL */

/* Loads Z-Image transformer weights from sharded safetensors files.
 * When use_mmap is set and an F16/BF16 cache exists, weights are loaded zero-copy.
 * Otherwise converts F32→BF16 and writes both caches for next time.
 * Pre-warms Metal buffer cache after loading. */
zi_transformer_t *zi_transformer_load_safetensors(const char *model_dir,
                                                     int dim, int n_heads,
                                                     int n_layers, int n_refiner,
                                                     int cap_feat_dim, int in_channels,
                                                     int patch_size, float rope_theta,
                                                     const int *axes_dims,
                                                     int use_mmap) {
    zi_transformer_t *tf = calloc(1, sizeof(zi_transformer_t));
    if (!tf) return NULL;

    char name[256];

    /* Set config */
    tf->dim = dim;
    tf->n_heads = n_heads;
    tf->head_dim = dim / n_heads;
    tf->n_layers = n_layers;
    tf->n_refiner = n_refiner;
    tf->ffn_dim = (8 * dim / 3 + 255) / 256 * 256;  /* Round up to 256 */
    tf->in_channels = in_channels;
    tf->patch_size = patch_size;
    tf->adaln_dim = dim < 256 ? dim : 256;
    tf->rope_theta = rope_theta;
    tf->cap_feat_dim = cap_feat_dim;

    for (int i = 0; i < 3; i++) {
        tf->axes_dims[i] = axes_dims[i];
        tf->axes_lens[i] = 1024;  /* Default max positions */
    }

    /* Open safetensors files */
    char path[1024];

    /* Try index file first for sharded models */
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors.index.json", model_dir);
    FILE *idx_f = fopen(path, "r");

    safetensors_file_t *files[ZI_MAX_SHARDS] = {0};
    int n_files = 0;

    if (idx_f) {
        /* Sharded: parse index to find shard files */
        fseek(idx_f, 0, SEEK_END);
        long fsize = ftell(idx_f);
        fseek(idx_f, 0, SEEK_SET);
        char *json = (char *)malloc(fsize + 1);
        if (!json) {
            fclose(idx_f);
            goto error;
        }
        fread(json, 1, fsize, idx_f);
        json[fsize] = 0;
        fclose(idx_f);

        /* Find unique shard filenames */
        char seen[32][128];
        int n_seen = 0;
        char *p = json;
        while ((p = strstr(p, ".safetensors")) != NULL) {
            /* Find start of filename */
            char *end = p + strlen(".safetensors");
            char *start = p;
            while (start > json && *(start - 1) != '"') start--;

            int len = (int)(end - start);
            if (len < 128) {
                char fname[128];
                memcpy(fname, start, len);
                fname[len] = 0;

                /* Check if already seen */
                int found = 0;
                for (int i = 0; i < n_seen; i++) {
                    if (strcmp(seen[i], fname) == 0) { found = 1; break; }
                }
                if (!found && n_seen < ZI_MAX_SHARDS) {
                    strcpy(seen[n_seen], fname);
                    n_seen++;
                }
            }
            p = end;
        }
        free(json);

        /* Open each shard */
        for (int i = 0; i < n_seen && n_files < ZI_MAX_SHARDS; i++) {
            snprintf(path, sizeof(path), "%s/transformer/%s", model_dir, seen[i]);
            files[n_files] = safetensors_open(path);
            if (files[n_files]) n_files++;
        }
    } else {
        /* Single file */
        snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors", model_dir);
        files[0] = safetensors_open(path);
        if (files[0]) n_files = 1;
    }

    if (n_files == 0) {
        fprintf(stderr, "Z-Image: failed to open transformer safetensors\n");
        goto error;
    }

    tf->num_sf_files = n_files;
    for (int i = 0; i < n_files; i++) tf->sf_files[i] = files[i];

    if (iris_verbose)
        fprintf(stderr, "  Loading Z-Image transformer (%d shards)...\n", n_files);

    /* Determine FFN dimension from weights */
    const safetensor_t *w1_probe = NULL;
    for (int f = 0; f < n_files && !w1_probe; f++)
        w1_probe = safetensors_find(files[f], "layers.0.feed_forward.w1.weight");
    if (w1_probe) {
        tf->ffn_dim = (int)w1_probe->shape[0];
    }

    /* Determine t_embedder mid_size from weights */
    tf->t_emb_mid_size = 1024;  /* Default */
    const safetensor_t *t_probe = NULL;
    for (int f = 0; f < n_files && !t_probe; f++)
        t_probe = safetensors_find(files[f], "t_embedder.mlp.0.weight");
    if (t_probe) {
        tf->t_emb_mid_size = (int)t_probe->shape[0];
    }

    /* Check if GPU acceleration is available */
    int use_gpu = 0;
#ifdef USE_METAL
    if (iris_metal_available() && iris_metal_shaders_available()) {
        use_gpu = 1;
        tf->use_gpu = 1;
        if (iris_verbose)
            fprintf(stderr, "  Z-Image: GPU acceleration enabled (bf16 weights)\n");
    }
#endif
    /* GPU mmap: read weights directly from mmap'd safetensors.
     * For BF16 weights: zero-copy pointers (skip F32 roundtrip).
     * For F32 weights: mmap pointers feed directly into F32→BF16 conversion
     * (skip intermediate F32 allocation). */
    int mmap_bf16 = 0;
    int mmap_gpu_f32 = 0;
    if (use_gpu && use_mmap) {
        if (!zi_all_tensors_f32(files, n_files)) {
            mmap_bf16 = 1;
            tf->mmap_bf16_weights = 1;
            if (iris_verbose)
                fprintf(stderr, "  Z-Image: GPU mmap mode (zero-copy bf16 weights)\n");
        } else {
            mmap_gpu_f32 = 1;
            if (iris_verbose)
                fprintf(stderr, "  Z-Image: GPU mmap mode (mmap f32 → bf16 conversion)\n");
        }
    }
    /* BLAS/CPU fast-load mode: keep mmap files open and use direct f32 pointers. */
    int mmap_f32_weights = (!use_gpu && zi_all_tensors_f32(files, n_files));
    tf->mmap_f32_weights = mmap_f32_weights;
    if (mmap_f32_weights) {
        if (iris_verbose)
            fprintf(stderr, "  Z-Image: CPU mmap mode enabled (zero-copy f32 weights)\n");
    }

    /* Load timestep embedder */
    tf->t_emb_mlp0_weight = zi_get_tensor(files, n_files, "t_embedder.mlp.0.weight", mmap_f32_weights);
    tf->t_emb_mlp0_bias = zi_get_tensor(files, n_files, "t_embedder.mlp.0.bias", mmap_f32_weights);
    tf->t_emb_mlp2_weight = zi_get_tensor(files, n_files, "t_embedder.mlp.2.weight", mmap_f32_weights);
    tf->t_emb_mlp2_bias = zi_get_tensor(files, n_files, "t_embedder.mlp.2.bias", mmap_f32_weights);
    if (!tf->t_emb_mlp0_weight || !tf->t_emb_mlp0_bias ||
        !tf->t_emb_mlp2_weight || !tf->t_emb_mlp2_bias) {
        goto error;
    }

    /* Load caption embedder: RMSNorm + Linear */
    tf->cap_emb_norm = zi_get_tensor(files, n_files, "cap_embedder.0.weight", mmap_f32_weights);
    tf->cap_emb_linear_w = zi_get_tensor(files, n_files, "cap_embedder.1.weight", mmap_f32_weights);
    tf->cap_emb_linear_b = zi_get_tensor(files, n_files, "cap_embedder.1.bias", mmap_f32_weights);
    if (!tf->cap_emb_norm || !tf->cap_emb_linear_w || !tf->cap_emb_linear_b) {
        goto error;
    }

    /* Load image embedder */
    snprintf(name, sizeof(name), "all_x_embedder.%d-1.weight", patch_size);
    tf->x_emb_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_x_embedder.%d-1.bias", patch_size);
    tf->x_emb_bias = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    if (!tf->x_emb_weight || !tf->x_emb_bias) {
        goto error;
    }

    /* Pad tokens */
    tf->x_pad_token = zi_get_tensor(files, n_files, "x_pad_token", mmap_f32_weights);
    tf->cap_pad_token = zi_get_tensor(files, n_files, "cap_pad_token", mmap_f32_weights);
    if (!tf->x_pad_token || !tf->cap_pad_token) {
        goto error;
    }

    /* Allocate block arrays */
    tf->noise_refiner = calloc(n_refiner, sizeof(zi_block_t));
    tf->context_refiner = calloc(n_refiner, sizeof(zi_block_t));
    tf->layers = calloc(n_layers, sizeof(zi_block_t));
    if (!tf->noise_refiner || !tf->context_refiner || !tf->layers) goto error;

    int total_blocks = n_refiner * 2 + n_layers;
    int cache_loaded = 0;

#ifdef USE_METAL
    /* Try F16 cache first — true zero-copy into Metal (no BF16→F16 warmup needed) */
    if (use_gpu && use_mmap) {
        size_t f16_sz = 0;
        void *f16_cache = zi_f16_cache_open(model_dir, total_blocks, dim, tf->ffn_dim,
                                             files, n_files, &f16_sz);
        if (f16_cache) {
            tf->f16_cache_mmap = f16_cache;
            tf->f16_cache_size = f16_sz;
            tf->f16_from_cache = 1;
            tf->bf16_fused_from_cache = 1;

            int idx = 0;
            for (int i = 0; i < n_refiner; i++)
                zi_f16_cache_load_block(&tf->noise_refiner[i], f16_cache, idx++, dim, tf->ffn_dim);
            for (int i = 0; i < n_refiner; i++)
                zi_f16_cache_load_block(&tf->context_refiner[i], f16_cache, idx++, dim, tf->ffn_dim);
            for (int i = 0; i < n_layers; i++)
                zi_f16_cache_load_block(&tf->layers[i], f16_cache, idx++, dim, tf->ffn_dim);

            char norm_prefix[256];
            for (int i = 0; i < n_refiner; i++) {
                snprintf(norm_prefix, sizeof(norm_prefix), "noise_refiner.%d", i);
                if (!zi_load_block_norms(&tf->noise_refiner[i], files, n_files, norm_prefix, 1))
                    goto error;
            }
            for (int i = 0; i < n_refiner; i++) {
                snprintf(norm_prefix, sizeof(norm_prefix), "context_refiner.%d", i);
                if (!zi_load_block_norms(&tf->context_refiner[i], files, n_files, norm_prefix, 0))
                    goto error;
            }
            for (int i = 0; i < n_layers; i++) {
                snprintf(norm_prefix, sizeof(norm_prefix), "layers.%d", i);
                if (!zi_load_block_norms(&tf->layers[i], files, n_files, norm_prefix, 1))
                    goto error;
            }

            cache_loaded = 1;
            if (iris_verbose)
                fprintf(stderr, "  Z-Image: F16 cache loaded (zero-copy Metal buffers)\n");
        }
    }

    /* Fall back to BF16 cache (requires BF16→F16 warmup but faster than full load) */
    if (!cache_loaded && use_gpu && use_mmap) {
        size_t cache_sz = 0;
        void *cache = zi_cache_open(model_dir, total_blocks, dim, tf->ffn_dim,
                                     files, n_files, &cache_sz);
        if (cache) {
            tf->bf16_cache_mmap = cache;
            tf->bf16_cache_size = cache_sz;
            tf->bf16_fused_from_cache = 1;

            int idx = 0;
            for (int i = 0; i < n_refiner; i++)
                zi_cache_load_block(&tf->noise_refiner[i], cache, idx++, dim, tf->ffn_dim);
            for (int i = 0; i < n_refiner; i++)
                zi_cache_load_block(&tf->context_refiner[i], cache, idx++, dim, tf->ffn_dim);
            for (int i = 0; i < n_layers; i++)
                zi_cache_load_block(&tf->layers[i], cache, idx++, dim, tf->ffn_dim);

            char norm_prefix[256];
            for (int i = 0; i < n_refiner; i++) {
                snprintf(norm_prefix, sizeof(norm_prefix), "noise_refiner.%d", i);
                if (!zi_load_block_norms(&tf->noise_refiner[i], files, n_files, norm_prefix, 1))
                    goto error;
            }
            for (int i = 0; i < n_refiner; i++) {
                snprintf(norm_prefix, sizeof(norm_prefix), "context_refiner.%d", i);
                if (!zi_load_block_norms(&tf->context_refiner[i], files, n_files, norm_prefix, 0))
                    goto error;
            }
            for (int i = 0; i < n_layers; i++) {
                snprintf(norm_prefix, sizeof(norm_prefix), "layers.%d", i);
                if (!zi_load_block_norms(&tf->layers[i], files, n_files, norm_prefix, 1))
                    goto error;
            }

            cache_loaded = 1;
            if (iris_verbose)
                fprintf(stderr, "  Z-Image: BF16 cache loaded (zero-copy + F32 norms)\n");
        }
    }
#endif

    if (!cache_loaded) {
        /* Load noise refiner blocks */
        for (int i = 0; i < n_refiner; i++) {
            snprintf(name, sizeof(name), "noise_refiner.%d", i);
            if (!zi_load_block(&tf->noise_refiner[i], files, n_files, name, 1,
                               dim, tf->ffn_dim, use_gpu, mmap_f32_weights, mmap_bf16, mmap_gpu_f32)) {
                goto error;
            }
        }

        /* Load context refiner blocks (no modulation) */
        for (int i = 0; i < n_refiner; i++) {
            snprintf(name, sizeof(name), "context_refiner.%d", i);
            if (!zi_load_block(&tf->context_refiner[i], files, n_files, name, 0,
                               dim, tf->ffn_dim, use_gpu, mmap_f32_weights, mmap_bf16, mmap_gpu_f32)) {
                goto error;
            }
        }

        /* Load main transformer blocks */
        for (int i = 0; i < n_layers; i++) {
            snprintf(name, sizeof(name), "layers.%d", i);
            if (!zi_load_block(&tf->layers[i], files, n_files, name, 1,
                               dim, tf->ffn_dim, use_gpu, mmap_f32_weights, mmap_bf16, mmap_gpu_f32)) {
                goto error;
            }
            if ((i + 1) % 10 == 0) {
            }
        }

#ifdef USE_METAL
        /* Write BF16 cache for next time */
        if (use_gpu && use_mmap) {
            double t_cache = zi_time_ms();
            if (zi_cache_write(model_dir, tf, files, n_files)) {
                if (iris_verbose)
                    fprintf(stderr, "  Z-Image: BF16 cache written (%.1f ms)\n", zi_time_ms() - t_cache);
            }
        }
#endif
    }

    /* Load final layer */
    snprintf(name, sizeof(name), "all_final_layer.%d-1.adaLN_modulation.1.weight", patch_size);
    tf->final_layer.adaln_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_final_layer.%d-1.adaLN_modulation.1.bias", patch_size);
    tf->final_layer.adaln_bias = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_final_layer.%d-1.norm_final.weight", patch_size);
    tf->final_layer.norm_weight = zi_get_tensor_optional(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_final_layer.%d-1.linear.weight", patch_size);
    tf->final_layer.linear_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_final_layer.%d-1.linear.bias", patch_size);
    tf->final_layer.linear_bias = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    if (!tf->final_layer.adaln_weight || !tf->final_layer.adaln_bias ||
        !tf->final_layer.linear_weight || !tf->final_layer.linear_bias) {
        goto error;
    }

    /* Precompute RoPE tables */
    zi_precompute_rope(tf);

    /* Allocate initial working memory (will be resized as needed) */
    tf->work_alloc = 0;
    tf->work_x = NULL;
    tf->work_tmp = NULL;
    tf->work_qkv = NULL;
    tf->work_attn = NULL;
    tf->work_ffn = NULL;
    tf->max_seq = 0;

    /* Close safetensors files unless mmap mode needs them at inference time.
     * GPU mmap with F32 weights closes here since F32→BF16 conversion is done. */
    if (!mmap_f32_weights && !mmap_bf16) {
        for (int f = 0; f < n_files; f++) {
            if (tf->sf_files[f]) {
                safetensors_close(tf->sf_files[f]);
                tf->sf_files[f] = NULL;
            }
        }
        tf->num_sf_files = 0;
    }

    if (iris_verbose) {
        fprintf(stderr, "  Z-Image transformer loaded: dim=%d, heads=%d, layers=%d+%d+%d, ffn=%d\n",
                dim, n_heads, n_refiner, n_refiner, n_layers, tf->ffn_dim);
    }

    /* Pre-allocate reusable scratch buffers */
    tf->mod_scratch = (float *)malloc(4 * (size_t)dim * sizeof(float));
    tf->final_scale = (float *)malloc((size_t)dim * sizeof(float));
    /* final_normed is allocated lazily on first use (size depends on seq) */
    tf->final_normed = NULL;
    tf->final_normed_cap = 0;

    if (!tf->mod_scratch || !tf->final_scale) {
        goto error;
    }

#ifdef USE_METAL
    /* Pre-warm bf16->Metal buffer cache so first denoising step avoids misses. */
    iris_warmup_bf16_zimage(tf);

    /* Write F16 cache after first BF16 warmup (one-time cost) */
    if (use_gpu && use_mmap && !tf->f16_from_cache) {
        double t_f16 = zi_time_ms();
        if (zi_f16_cache_write(model_dir, tf, files, n_files)) {
            if (iris_verbose)
                fprintf(stderr, "  Z-Image: F16 cache written (%.1f ms)\n", zi_time_ms() - t_f16);
        }
    }
#endif

    return tf;

error:
    iris_transformer_free_zimage(tf);
    return NULL;
}

void iris_transformer_free_zimage(zi_transformer_t *tf) {
    if (!tf) return;

    int free_f32_weights = !tf->mmap_f32_weights;

    if (free_f32_weights) {
        free(tf->t_emb_mlp0_weight);
        free(tf->t_emb_mlp0_bias);
        free(tf->t_emb_mlp2_weight);
        free(tf->t_emb_mlp2_bias);
        free(tf->cap_emb_norm);
        free(tf->cap_emb_linear_w);
        free(tf->cap_emb_linear_b);
        free(tf->x_emb_weight);
        free(tf->x_emb_bias);
        free(tf->x_pad_token);
        free(tf->cap_pad_token);
    }

    if (tf->noise_refiner) {
        for (int i = 0; i < tf->n_refiner; i++)
            zi_free_block(&tf->noise_refiner[i], free_f32_weights);
        free(tf->noise_refiner);
    }
    if (tf->context_refiner) {
        for (int i = 0; i < tf->n_refiner; i++)
            zi_free_block(&tf->context_refiner[i], free_f32_weights);
        free(tf->context_refiner);
    }
    if (tf->layers) {
        for (int i = 0; i < tf->n_layers; i++)
            zi_free_block(&tf->layers[i], free_f32_weights);
        free(tf->layers);
    }

    if (free_f32_weights) {
        free(tf->final_layer.adaln_weight);
        free(tf->final_layer.adaln_bias);
        free(tf->final_layer.norm_weight);
        free(tf->final_layer.linear_weight);
        free(tf->final_layer.linear_bias);
    }

    for (int i = 0; i < tf->num_sf_files; i++) {
        if (tf->sf_files[i]) {
            safetensors_close(tf->sf_files[i]);
            tf->sf_files[i] = NULL;
        }
    }
    tf->num_sf_files = 0;

    for (int i = 0; i < 3; i++) {
        free(tf->rope_cos[i]);
        free(tf->rope_sin[i]);
    }

    free(tf->work_x);
    free(tf->work_tmp);
    free(tf->work_qkv);
    free(tf->work_attn);
    free(tf->work_ffn);
    free(tf->work_ffn_fused);

    /* Pre-allocated scratch buffers */
    free(tf->mod_scratch);
    free(tf->final_scale);
    free(tf->final_normed);

    /* Cached position IDs and masks */
    free(tf->cpu_cache_img_pos);
    free(tf->cpu_cache_cap_pos);
    free(tf->cpu_cache_img_mask);
    free(tf->cpu_cache_cap_mask);
    free(tf->cpu_cache_unified_pos);
    free(tf->cpu_cache_unified_mask);

    /* Pre-allocated forward buffers */
    free(tf->fwd_img_patches);
    free(tf->fwd_cap_normed);
    free(tf->fwd_step_mod);
    free(tf->fwd_gpu_final_scale);
    free(tf->fwd_gpu_final_shift);
    free(tf->fwd_gpu_final_scale_param);
    free(tf->fwd_final_out);
    free(tf->fwd_img_emb);
    free(tf->fwd_cap_emb);
    free(tf->fwd_cap_padded_feats);
    free(tf->fwd_img_out);

#ifdef USE_METAL
    if (tf->f16_cache_mmap) {
        munmap(tf->f16_cache_mmap, tf->f16_cache_size);
        tf->f16_cache_mmap = NULL;
    }
    if (tf->bf16_cache_mmap) {
        munmap(tf->bf16_cache_mmap, tf->bf16_cache_size);
        tf->bf16_cache_mmap = NULL;
    }
    zi_gpu_rope_cache_clear(tf);
#endif

    free(tf);
}
