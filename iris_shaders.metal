/*
 * iris_shaders.metal - Metal compute shaders for Iris inference
 *
 * These kernels accelerate operations that run on CPU otherwise:
 * - RMSNorm (used in QK normalization)
 * - LayerNorm + AdaLN modulation
 * - SiLU activation
 * - Softmax (row-wise)
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline float bf16_to_f32(ushort bf16);
inline ushort f32_to_bf16(float f32);

/* ========================================================================
 * RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
 * ======================================================================== */

/* RMSNorm kernel - processes one row per threadgroup
 * x: [seq, hidden], weight: [hidden], out: [seq, hidden]
 */
kernel void rms_norm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &hidden [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[32];  // One slot per SIMD group (max 8 groups for 256 threads)

    device const float *x_row = x + row * hidden;
    device float *out_row = out + row * hidden;

    // Compute partial sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = x_row[i];
        local_sum += val * val;
    }

    // SIMD shuffle reduction within each SIMD group
    float simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduce across SIMD groups
    uint num_simd_groups = (threads + 31) / 32;
    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute RMS inverse
    float rms_inv = rsqrt(shared_sum[0] / float(hidden) + eps);

    // Apply normalization with weight
    for (int i = tid; i < hidden; i += threads) {
        out_row[i] = x_row[i] * rms_inv * weight[i];
    }
}

/* QK RMSNorm (legacy) - single thread per (seq, head), serial over head_dim.
 * Kept as fallback. New code uses qk_rms_norm (threadgroup-parallel below).
 */
kernel void qk_rms_norm_legacy(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device const float *q_weight [[buffer(2)]],
    device const float *k_weight [[buffer(3)]],
    constant int &heads [[buffer(4)]],
    constant int &head_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]]  // (seq_idx, head_idx)
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    uint hidden = heads * head_dim;
    uint offset = seq_idx * hidden + head_idx * head_dim;

    // RMSNorm for Q
    float sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = q[offset + d];
        sum_sq += val * val;
    }
    float rms_inv = rsqrt(sum_sq / float(head_dim) + eps);
    for (int d = 0; d < head_dim; d++) {
        q[offset + d] = q[offset + d] * rms_inv * q_weight[d];
    }

    // RMSNorm for K
    sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = k[offset + d];
        sum_sq += val * val;
    }
    rms_inv = rsqrt(sum_sq / float(head_dim) + eps);
    for (int d = 0; d < head_dim; d++) {
        k[offset + d] = k[offset + d] * rms_inv * k_weight[d];
    }
}

/* QK RMSNorm - threadgroup-parallel version.
 * One threadgroup per (seq_idx, head_idx). Multiple threads cooperate on the
 * head_dim reduction via shared memory, then normalize in parallel.
 * Dispatched with threadgroupsPerGrid=(seq, heads, 1),
 * threadsPerThreadgroup=(min(128, head_dim), 1, 1).
 */
kernel void qk_rms_norm(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device const float *q_weight [[buffer(2)]],
    device const float *k_weight [[buffer(3)]],
    constant int &heads [[buffer(4)]],
    constant int &head_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint seq_idx = group_id.x;
    uint head_idx = group_id.y;

    uint hidden = heads * head_dim;
    device float *q_head = q + seq_idx * hidden + head_idx * head_dim;
    device float *k_head = k + seq_idx * hidden + head_idx * head_dim;

    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    // --- Q: parallel sum-of-squares ---
    float local_sum = 0.0f;
    for (int d = tid; d < head_dim; d += threads) {
        float val = q_head[d];
        local_sum += val * val;
    }

    float simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_rms = rsqrt(shared_sum[0] / float(head_dim) + eps);

    for (int d = tid; d < head_dim; d += threads) {
        q_head[d] = q_head[d] * q_rms * q_weight[d];
    }

    // Q normalization writes go to device memory (q_head[d]) — each thread
    // writes only its own elements and K reads different memory, but we must
    // ensure all Q writes are visible before any thread reuses shared_sum.
    // No barrier needed between Q writes and K accumulation since shared_sum
    // is overwritten below and Q/K don't alias.

    // --- K: parallel sum-of-squares ---
    local_sum = 0.0f;
    for (int d = tid; d < head_dim; d += threads) {
        float val = k_head[d];
        local_sum += val * val;
    }

    simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float k_rms = rsqrt(shared_sum[0] / float(head_dim) + eps);

    for (int d = tid; d < head_dim; d += threads) {
        k_head[d] = k_head[d] * k_rms * k_weight[d];
    }
}

/* ========================================================================
 * Fused QK RMSNorm + RoPE
 * In-place: norm Q and K heads, then apply RoPE rotation.
 * Eliminates 2 intermediate read+write passes.
 * cos/sin: [seq, head_dim], shared across heads.
 * ======================================================================== */

/* Legacy: single thread per (seq, head). Kept as fallback. */
kernel void qknorm_rope_legacy(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device const float *q_weight [[buffer(2)]],
    device const float *k_weight [[buffer(3)]],
    device const float *cos_freq [[buffer(4)]],
    device const float *sin_freq [[buffer(5)]],
    constant int &heads [[buffer(6)]],
    constant int &head_dim [[buffer(7)]],
    constant float &eps [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    uint hidden = heads * head_dim;
    uint offset = seq_idx * hidden + head_idx * head_dim;

    float sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = q[offset + d];
        sum_sq += val * val;
    }
    float rms_inv = rsqrt(sum_sq / float(head_dim) + eps);
    for (int d = 0; d < head_dim; d++) {
        q[offset + d] = q[offset + d] * rms_inv * q_weight[d];
    }

    sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = k[offset + d];
        sum_sq += val * val;
    }
    rms_inv = rsqrt(sum_sq / float(head_dim) + eps);
    for (int d = 0; d < head_dim; d++) {
        k[offset + d] = k[offset + d] * rms_inv * k_weight[d];
    }

    device const float *cos_row = cos_freq + seq_idx * head_dim;
    device const float *sin_row = sin_freq + seq_idx * head_dim;

    for (int d = 0; d < head_dim; d += 2) {
        float c = cos_row[d];
        float s = sin_row[d];

        float q0 = q[offset + d], q1 = q[offset + d + 1];
        q[offset + d]     = q0 * c - q1 * s;
        q[offset + d + 1] = q1 * c + q0 * s;

        float k0 = k[offset + d], k1 = k[offset + d + 1];
        k[offset + d]     = k0 * c - k1 * s;
        k[offset + d + 1] = k1 * c + k0 * s;
    }
}

/* Fused QK RMSNorm + RoPE - threadgroup-parallel version.
 * One threadgroup per (seq_idx, head_idx). Threads cooperate on the
 * head_dim reduction, then normalize + apply RoPE in parallel.
 * Dispatched with threadgroupsPerGrid=(seq, heads, 1),
 * threadsPerThreadgroup=(min(128, head_dim), 1, 1).
 * RoPE pairs (d, d+1) are handled by the thread owning element d when
 * d is even. With threads=128 and head_dim=128, each thread owns exactly
 * one element, so the even-indexed threads apply RoPE to their pair.
 */
kernel void qknorm_rope(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device const float *q_weight [[buffer(2)]],
    device const float *k_weight [[buffer(3)]],
    device const float *cos_freq [[buffer(4)]],
    device const float *sin_freq [[buffer(5)]],
    constant int &heads [[buffer(6)]],
    constant int &head_dim [[buffer(7)]],
    constant float &eps [[buffer(8)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint seq_idx = group_id.x;
    uint head_idx = group_id.y;

    uint hidden = heads * head_dim;
    device float *q_head = q + seq_idx * hidden + head_idx * head_dim;
    device float *k_head = k + seq_idx * hidden + head_idx * head_dim;

    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    // --- Q: parallel sum-of-squares ---
    float local_sum = 0.0f;
    for (int d = tid; d < head_dim; d += threads) {
        float val = q_head[d];
        local_sum += val * val;
    }

    float simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_rms = rsqrt(shared_sum[0] / float(head_dim) + eps);

    // Normalize Q
    for (int d = tid; d < head_dim; d += threads) {
        q_head[d] = q_head[d] * q_rms * q_weight[d];
    }

    // No barrier needed: Q writes go to device memory (q_head), K reads from
    // separate device memory (k_head), and shared_sum is overwritten below.

    // --- K: parallel sum-of-squares ---
    local_sum = 0.0f;
    for (int d = tid; d < head_dim; d += threads) {
        float val = k_head[d];
        local_sum += val * val;
    }

    simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float k_rms = rsqrt(shared_sum[0] / float(head_dim) + eps);

    // Normalize K
    for (int d = tid; d < head_dim; d += threads) {
        k_head[d] = k_head[d] * k_rms * k_weight[d];
    }

    // --- RoPE: apply rotation to normalized Q and K ---
    // Wait for all normalization writes to complete before reading them back
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float *cos_row = cos_freq + seq_idx * head_dim;
    device const float *sin_row = sin_freq + seq_idx * head_dim;

    // Each thread handles one pair: thread i processes elements (2*i, 2*i+1).
    // This uses all threads (no wasted odd-indexed threads).
    int half_dim = head_dim / 2;
    for (int p = (int)tid; p < half_dim; p += (int)threads) {
        int d = p * 2;
        float c = cos_row[d];
        float s = sin_row[d];

        float q0 = q_head[d], q1 = q_head[d + 1];
        q_head[d]     = q0 * c - q1 * s;
        q_head[d + 1] = q1 * c + q0 * s;

        float k0 = k_head[d], k1 = k_head[d + 1];
        k_head[d]     = k0 * c - k1 * s;
        k_head[d + 1] = k1 * c + k0 * s;
    }
}

/* ========================================================================
 * LayerNorm + AdaLN modulation
 * out = (1 + scale) * norm(x) + shift
 * where norm(x) = (x - mean) / sqrt(var + eps)
 * ======================================================================== */

kernel void adaln_norm(
    device const float *x [[buffer(0)]],
    device const float *shift [[buffer(1)]],
    device const float *scale [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &hidden [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    device const float *x_row = x + row * hidden;
    device float *out_row = out + row * hidden;

    // First pass: compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        local_sum += x_row[i];
    }

    float simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = shared_sum[0] / float(hidden);

    // Second pass: variance via Welford (numerically stable)
    float var_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }

    simd_result = simd_sum(var_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float var = shared_sum[0] / float(hidden);
    float std_inv = rsqrt(var + eps);

    // Apply LayerNorm + AdaLN modulation
    for (int i = tid; i < hidden; i += threads) {
        float norm = (x_row[i] - mean) * std_inv;
        out_row[i] = (1.0f + scale[i]) * norm + shift[i];
    }
}

/* ========================================================================
 * SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
 * ======================================================================== */

kernel void silu(
    device float *x [[buffer(0)]],
    constant int &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float val = x[gid];
        x[gid] = val / (1.0f + metal::fast::exp(-val));
    }
}

/* SiLU with multiply: gate = silu(gate) * up (SwiGLU style) */
kernel void silu_mul(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float g = gate[gid];
        float silu_g = g / (1.0f + metal::fast::exp(-g));
        gate[gid] = silu_g * up[gid];
    }
}

/* Fused split + SiLU + mul for SwiGLU FFN.
 * Reads fused [seq, 2*mlp_hidden] and writes out = silu(gate) * up
 * where gate = fused[:, :mlp_hidden], up = fused[:, mlp_hidden:]. */
kernel void split_silu_mul(
    device const float *fused [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant int &mlp_hidden [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int n = mlp_hidden;  // mlp_hidden passed; total elements inferred from dispatch
    int row = gid / n;
    int col = gid % n;
    int fused_stride = n * 2;
    float g = fused[row * fused_stride + col];
    float u = fused[row * fused_stride + n + col];
    out[gid] = (g / (1.0f + metal::fast::exp(-g))) * u;
}

/* ========================================================================
 * Fused RMSNorm + Gated Add:
 *   out[s,h] += gate[h] * (rms_norm(proj[s,:], weight) [h])
 * Eliminates the intermediate normalized tensor. Same RMS reduction
 * strategy as rms_norm (one threadgroup per sequence row).
 * ======================================================================== */

kernel void norm_gated_add(
    device const float *proj [[buffer(0)]],
    device float *out [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device const float *gate [[buffer(3)]],
    constant int &hidden [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    device const float *proj_row = proj + row * hidden;
    device float *out_row = out + row * hidden;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = proj_row[i];
        local_sum += val * val;
    }

    float simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = rsqrt(shared_sum[0] / float(hidden) + eps);

    for (int i = tid; i < hidden; i += threads) {
        out_row[i] += gate[i] * (proj_row[i] * rms_inv * weight[i]);
    }
}

/* Ungated variant for unmodulated (context_refiner) blocks:
 *   out[s,h] += rms_norm(proj[s,:], weight)[h] */
kernel void norm_add(
    device const float *proj [[buffer(0)]],
    device float *out [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    constant int &hidden [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    device const float *proj_row = proj + row * hidden;
    device float *out_row = out + row * hidden;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = proj_row[i];
        local_sum += val * val;
    }

    float simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = rsqrt(shared_sum[0] / float(hidden) + eps);

    for (int i = tid; i < hidden; i += threads) {
        out_row[i] += proj_row[i] * rms_inv * weight[i];
    }
}

/* ========================================================================
 * Gated Add: out += gate * proj
 * gate: [hidden], proj: [seq, hidden], out: [seq, hidden]
 * ======================================================================== */

kernel void gated_add(
    device float *out [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device const float *proj [[buffer(2)]],
    constant int &seq [[buffer(3)]],
    constant int &hidden [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]  // (seq_idx, hidden_idx)
) {
    uint s = pos.x;
    uint h = pos.y;
    if (s < uint(seq) && h < uint(hidden)) {
        uint idx = s * hidden + h;
        out[idx] += gate[h] * proj[idx];
    }
}

/* ========================================================================
 * Split Fused QKV+MLP Output
 * fused: [seq, fused_dim] where fused_dim = hidden*3 + mlp_hidden*2
 * Splits into: q, k, v [seq, hidden], gate, up [seq, mlp_hidden]
 * ======================================================================== */

kernel void split_qkv_mlp(
    device const float *fused [[buffer(0)]],  // [seq, fused_dim]
    device float *q [[buffer(1)]],            // [seq, hidden]
    device float *k [[buffer(2)]],            // [seq, hidden]
    device float *v [[buffer(3)]],            // [seq, hidden]
    device float *gate [[buffer(4)]],         // [seq, mlp_hidden]
    device float *up [[buffer(5)]],           // [seq, mlp_hidden]
    constant int &seq [[buffer(6)]],
    constant int &hidden [[buffer(7)]],
    constant int &mlp_hidden [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]  // (seq_idx, element_idx within largest output)
) {
    uint s = pos.x;
    uint e = pos.y;

    if (s >= uint(seq)) return;

    int fused_dim = hidden * 3 + mlp_hidden * 2;
    device const float *row = fused + s * fused_dim;

    // Copy Q (dims 0 to hidden-1)
    if (e < uint(hidden)) {
        q[s * hidden + e] = row[e];
        k[s * hidden + e] = row[hidden + e];
        v[s * hidden + e] = row[hidden * 2 + e];
    }

    // Copy gate and up (dims 0 to mlp_hidden-1)
    if (e < uint(mlp_hidden)) {
        gate[s * mlp_hidden + e] = row[hidden * 3 + e];
        up[s * mlp_hidden + e] = row[hidden * 3 + mlp_hidden + e];
    }
}

/* ========================================================================
 * Concat Attention + MLP outputs for fused projection
 * attn: [seq, hidden], mlp: [seq, mlp_hidden]
 * out: [seq, hidden + mlp_hidden]
 * ======================================================================== */

kernel void concat_attn_mlp(
    device const float *attn [[buffer(0)]],   // [seq, hidden]
    device const float *mlp [[buffer(1)]],    // [seq, mlp_hidden]
    device float *out [[buffer(2)]],          // [seq, hidden + mlp_hidden]
    constant int &seq [[buffer(3)]],
    constant int &hidden [[buffer(4)]],
    constant int &mlp_hidden [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]  // (seq_idx, element_idx)
) {
    uint s = pos.x;
    uint e = pos.y;

    if (s >= uint(seq)) return;

    int out_dim = hidden + mlp_hidden;
    device float *out_row = out + s * out_dim;

    // Copy attention output
    if (e < uint(hidden)) {
        out_row[e] = attn[s * hidden + e];
    }

    // Copy MLP output
    if (e < uint(mlp_hidden)) {
        out_row[hidden + e] = mlp[s * mlp_hidden + e];
    }
}

/* ========================================================================
 * Softmax (row-wise): out[i] = exp(x[i] - max) / sum(exp(x - max))
 * ======================================================================== */

kernel void softmax(
    device float *x [[buffer(0)]],
    constant int &rows [[buffer(1)]],
    constant int &cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_max[32];
    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    device float *row_ptr = x + row * cols;

    // Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += threads) {
        local_max = max(local_max, row_ptr[i]);
    }

    // SIMD max reduction
    float simd_max_val = simd_max(local_max);
    if (simd_lane_id == 0) shared_max[simd_group_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_max[tid];
        val = simd_max(val);
        if (tid == 0) shared_max[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = shared_max[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += threads) {
        float e = metal::fast::exp(row_ptr[i] - max_val);
        row_ptr[i] = e;  // Store exp temporarily
        local_sum += e;
    }

    // SIMD sum reduction
    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = shared_sum[0];

    // Normalize
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int i = tid; i < cols; i += threads) {
        row_ptr[i] *= inv_sum;
    }
}

/* ========================================================================
 * RoPE (Rotary Position Embeddings)
 * ======================================================================== */

/* Apply 2D RoPE to Q or K tensor
 * x: [seq, heads*head_dim]
 * cos, sin: [seq, head_dim]  (precomputed frequencies)
 */
kernel void apply_rope_2d(
    device float *x [[buffer(0)]],
    device const float *cos_freq [[buffer(1)]],
    device const float *sin_freq [[buffer(2)]],
    constant int &seq [[buffer(3)]],
    constant int &heads [[buffer(4)]],
    constant int &head_dim [[buffer(5)]],
    constant int &axis_dim [[buffer(6)]],  // 32 for FLUX
    uint2 pos [[thread_position_in_grid]]  // (seq_idx, head_idx)
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    if (seq_idx >= uint(seq) || head_idx >= uint(heads)) return;

    uint hidden = heads * head_dim;
    device float *vec = x + seq_idx * hidden + head_idx * head_dim;
    device const float *cos_row = cos_freq + seq_idx * head_dim;
    device const float *sin_row = sin_freq + seq_idx * head_dim;

    // RoPE rotation for each axis (4 axes of 32 dims each = 128)
    int half_axis = axis_dim / 2;  // 16

    for (int axis = 0; axis < 4; axis++) {
        int axis_offset = axis * axis_dim;
        for (int d = 0; d < half_axis; d++) {
            int i0 = axis_offset + d;
            int i1 = axis_offset + half_axis + d;

            float c = cos_row[i0];
            float s = sin_row[i0];

            float x0 = vec[i0];
            float x1 = vec[i1];

            vec[i0] = x0 * c - x1 * s;
            vec[i1] = x0 * s + x1 * c;
        }
    }
}

/* Apply 2D RoPE to bf16 Q or K tensor
 * x: [seq, heads*head_dim] (bf16)
 * cos, sin: [seq, head_dim] (f32)
 */
kernel void apply_rope_2d_bf16(
    device ushort *x [[buffer(0)]],
    device const float *cos_freq [[buffer(1)]],
    device const float *sin_freq [[buffer(2)]],
    constant int &seq [[buffer(3)]],
    constant int &heads [[buffer(4)]],
    constant int &head_dim [[buffer(5)]],
    constant int &axis_dim [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    if (seq_idx >= uint(seq) || head_idx >= uint(heads)) return;

    uint hidden = heads * head_dim;
    device ushort *vec = x + seq_idx * hidden + head_idx * head_dim;
    device const float *cos_row = cos_freq + seq_idx * head_dim;
    device const float *sin_row = sin_freq + seq_idx * head_dim;

    (void)axis_dim;
    for (int d = 0; d < head_dim; d += 2) {
        float c = cos_row[d];
        float s = sin_row[d];

        float x0 = bf16_to_f32(vec[d]);
        float x1 = bf16_to_f32(vec[d + 1]);

        vec[d] = f32_to_bf16(x0 * c - x1 * s);
        vec[d + 1] = f32_to_bf16(x1 * c + x0 * s);
    }
}

/* ========================================================================
 * Unified RoPE for Text+Image (Single Block Forward)
 * Applies different frequency tables to text and image portions in one pass.
 * Text portion: positions [0, img_offset)
 * Image portion: positions [img_offset, seq)
 * ======================================================================== */
kernel void apply_rope_unified(
    device float *x [[buffer(0)]],
    device const float *txt_cos [[buffer(1)]],
    device const float *txt_sin [[buffer(2)]],
    device const float *img_cos [[buffer(3)]],
    device const float *img_sin [[buffer(4)]],
    constant int &seq [[buffer(5)]],
    constant int &img_offset [[buffer(6)]],
    constant int &heads [[buffer(7)]],
    constant int &head_dim [[buffer(8)]],
    constant int &axis_dim [[buffer(9)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    if (seq_idx >= uint(seq) || head_idx >= uint(heads)) return;

    uint hidden = heads * head_dim;
    device float *vec = x + seq_idx * hidden + head_idx * head_dim;

    // Select appropriate frequency table based on position
    device const float *cos_row;
    device const float *sin_row;

    if (seq_idx < uint(img_offset)) {
        // Text portion: use text frequencies indexed by seq_idx
        cos_row = txt_cos + seq_idx * head_dim;
        sin_row = txt_sin + seq_idx * head_dim;
    } else {
        // Image portion: use image frequencies indexed by (seq_idx - img_offset)
        uint img_idx = seq_idx - uint(img_offset);
        cos_row = img_cos + img_idx * head_dim;
        sin_row = img_sin + img_idx * head_dim;
    }

    // RoPE rotation: apply to consecutive pairs (d, d+1) matching CPU implementation
    // cos[d] == cos[d+1] due to repeat_interleave in frequency generation
    for (int d = 0; d < head_dim; d += 2) {
        float c = cos_row[d];
        float s = sin_row[d];

        float x0 = vec[d];
        float x1 = vec[d + 1];

        // Complex rotation: (x0 + i*x1) * (cos + i*sin)
        vec[d] = x0 * c - x1 * s;
        vec[d + 1] = x1 * c + x0 * s;
    }
}

/* ========================================================================
 * Fused Non-Causal Attention for Transformer (FLUX)
 * Processes all heads in parallel without causal masking.
 *
 * This kernel computes one output row per threadgroup:
 * out[query_idx, head] = softmax(Q @ K^T * scale) @ V
 *
 * Works directly on [seq, heads*head_dim] layout without transpose.
 * Supports different Q and K/V sequence lengths (for joint attention).
 * ======================================================================== */

constant int FUSED_V_TILE = 8;

kernel void attention_fused(
    device const float *Q [[buffer(0)]],      // [seq_q, heads * head_dim]
    device const float *K [[buffer(1)]],      // [seq_k, heads * head_dim]
    device const float *V [[buffer(2)]],      // [seq_k, heads * head_dim]
    device float *out [[buffer(3)]],          // [seq_q, heads * head_dim]
    constant int &seq_q [[buffer(4)]],        // Query sequence length
    constant int &seq_k [[buffer(5)]],        // Key/Value sequence length
    constant int &num_heads [[buffer(6)]],
    constant int &head_dim [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    threadgroup float *shared_scores [[threadgroup(0)]],  // [seq_k] dynamic
    uint3 tg_pos [[threadgroup_position_in_grid]],   // (query_idx, head_idx, 0)
    uint3 tid_pos [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Shared memory for reductions (shared_scores is dynamic via threadgroup(0))
    threadgroup float shared_reduce[32];
    // Q cached in threadgroup memory — avoids re-reading from device per key position
    threadgroup float shared_q[128];   // head_dim is always 128
    // V tile buffer for coalesced cooperative loading in Phase 6
    // FUSED_V_TILE rows * (128+1) cols = 1032 floats per row, +1 padding avoids bank conflicts
    // Total static threadgroup: 128 (reduce) + 512 (q) + 8*129*4 = 4128 (v) = ~4.8 KB
    threadgroup float shared_v[FUSED_V_TILE * (128 + 1)];  // +1 padding per row to avoid bank conflicts

    int query_idx = tg_pos.x;
    int head_idx = tg_pos.y;
    uint tid = tid_pos.x;
    uint threads = tg_size.x;
    uint num_simd_groups = (threads + 31) / 32;

    // Guard: shared_q[128] is fixed-size
    if (head_dim > 128) return;
    if (query_idx >= seq_q || head_idx >= num_heads) return;

    int hidden = num_heads * head_dim;
    int v_stride = head_dim + 1;  // Padded stride for bank conflict avoidance

    // Pointers to this position's Q and output (layout: [seq, heads*head_dim])
    device const float *q_row = Q + query_idx * hidden + head_idx * head_dim;
    device float *out_row = out + query_idx * hidden + head_idx * head_dim;

    // K and V have same layout, head offset is same
    device const float *K_head = K + head_idx * head_dim;
    device const float *V_head = V + head_idx * head_dim;

    // Cache Q row in threadgroup memory (read once, used for every key position)
    for (int d = tid; d < head_dim; d += threads) {
        shared_q[d] = q_row[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 1: Compute Q @ K^T ==========
    float local_max = -INFINITY;

    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        // Dot product: Q[query_idx, head] . K[key_idx, head]
        float dot = 0.0f;
        device const float *k_row = K_head + key_idx * hidden;
        for (int d = 0; d < head_dim; d++) {
            dot += shared_q[d] * k_row[d];
        }
        float score = dot * scale;
        shared_scores[key_idx] = score;
        local_max = max(local_max, score);
    }

    // ========== Phase 2: Find global max (SIMD) ==========
    float simd_max_val = simd_max(local_max);
    if (simd_lane_id == 0) shared_reduce[simd_group_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_reduce[tid];
        val = simd_max(val);
        if (tid == 0) shared_reduce[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = shared_reduce[0];

    // ========== Phase 3: Compute exp(score - max) and sum ==========
    float local_sum = 0.0f;
    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        float e = metal::fast::exp(shared_scores[key_idx] - max_val);
        shared_scores[key_idx] = e;
        local_sum += e;
    }

    // ========== Phase 4: Find total sum (SIMD) ==========
    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_reduce[simd_group_id] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_reduce[tid];
        val = simd_sum(val);
        if (tid == 0) shared_reduce[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = (shared_reduce[0] > 0.0f) ? (1.0f / shared_reduce[0]) : 0.0f;

    // ========== Phase 5: Normalize scores ==========
    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        shared_scores[key_idx] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 6: Compute output = scores @ V (tiled) ==========
    // Per-thread accumulator for each dimension this thread owns.
    // With threads=256 and head_dim=128, each thread handles at most 1 dim.
    // 16 slots supports head_dim up to 4096 with threads=256, and handles
    // small threadgroup sizes (threads as low as 8 with head_dim=128).
    float acc[16] = {0};

    for (int tile_start = 0; tile_start < seq_k; tile_start += FUSED_V_TILE) {
        int tile_size = min(FUSED_V_TILE, seq_k - tile_start);

        // Cooperative load: all threads load V[tile_start:tile_end, :] into
        // shared_v[tile_size, head_dim] with padded stride. Contiguous device reads across threads.
        int total_elems = tile_size * head_dim;
        for (int idx = (int)tid; idx < total_elems; idx += (int)threads) {
            int k = idx / head_dim;
            int d = idx % head_dim;
            shared_v[k * v_stride + d] = V_head[(tile_start + k) * hidden + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate: each thread processes its assigned dimensions
        int dim_slot = 0;
        for (int d = (int)tid; d < head_dim; d += (int)threads) {
            if (dim_slot >= 16) break;
            for (int k = 0; k < tile_size; k++) {
                acc[dim_slot] += shared_scores[tile_start + k] * shared_v[k * v_stride + d];
            }
            dim_slot++;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    int dim_slot = 0;
    for (int d = (int)tid; d < head_dim; d += (int)threads) {
        if (dim_slot >= 16) break;
        out_row[d] = acc[dim_slot];
        dim_slot++;
    }
}

/* ========================================================================
 * Flash Attention (Tiled Online Softmax)
 *
 * Drop-in replacement for attention_fused. Tiles over K/V in chunks of
 * TILE_K, maintaining a running max and sum (online softmax) so the full
 * [seq_k] score vector never materializes. No sequence length limit.
 *
 * Threadgroup memory: (head_dim + TILE_K + 512) * 4 bytes ≈ 3.5 KB
 * (vs up to 30 KB for attention_fused at large seq_k).
 * ======================================================================== */

constant int FLASH_TILE_K = 256;

kernel void attention_flash(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &seq_q [[buffer(4)]],
    constant int &seq_k [[buffer(5)]],
    constant int &num_heads [[buffer(6)]],
    constant int &head_dim [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    threadgroup float *shared_mem [[threadgroup(0)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid_pos [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    int query_idx = tg_pos.x;
    int head_idx = tg_pos.y;
    uint tid = tid_pos.x;
    uint threads = tg_size.x;
    uint num_simd_groups = (threads + 31) / 32;

    if (query_idx >= seq_q || head_idx >= num_heads) return;

    int hidden = num_heads * head_dim;

    threadgroup float *shared_q      = shared_mem;
    threadgroup float *shared_scores = shared_mem + head_dim;
    threadgroup float *shared_reduce = shared_scores + FLASH_TILE_K;
    // Only 32 slots needed for SIMD reduce (was 256+256 = 512)

    device const float *q_row = Q + query_idx * hidden + head_idx * head_dim;
    device float *out_row     = out + query_idx * hidden + head_idx * head_dim;
    device const float *K_head = K + head_idx * head_dim;
    device const float *V_head = V + head_idx * head_dim;

    for (int d = (int)tid; d < head_dim; d += (int)threads) {
        shared_q[d] = q_row[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    /* Max accumulator slots: ceil(head_dim / threads).
     * 16 supports head_dim up to 4096 with threads=256, and handles
     * small threadgroup sizes (threads as low as 8 with head_dim=128). */
    float acc[16] = {0};

    int num_tiles = (seq_k + FLASH_TILE_K - 1) / FLASH_TILE_K;

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * FLASH_TILE_K;
        int tile_end = min(tile_start + FLASH_TILE_K, seq_k);
        int tile_len = tile_end - tile_start;

        float local_tile_max = -INFINITY;
        for (int k = (int)tid; k < tile_len; k += (int)threads) {
            int key_idx = tile_start + k;
            device const float *k_row = K_head + key_idx * hidden;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += shared_q[d] * k_row[d];
            }
            float score = dot * scale;
            shared_scores[k] = score;
            local_tile_max = max(local_tile_max, score);
        }

        // SIMD max reduction for tile max
        float simd_max_val = simd_max(local_tile_max);
        if (simd_lane_id == 0) shared_reduce[simd_group_id] = simd_max_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < num_simd_groups) {
            float val = shared_reduce[tid];
            val = simd_max(val);
            if (tid == 0) shared_reduce[0] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = shared_reduce[0];

        float new_max = max(running_max, tile_max);
        float correction = (running_max > -INFINITY) ? metal::fast::exp(running_max - new_max) : 0.0f;

        running_sum *= correction;
        int dim_idx = 0;
        for (int d = (int)tid; d < head_dim; d += (int)threads) {
            if (dim_idx >= 16) break;
            acc[dim_idx] *= correction;
            dim_idx++;
        }
        running_max = new_max;

        float local_tile_sum = 0.0f;
        for (int k = (int)tid; k < tile_len; k += (int)threads) {
            float e = metal::fast::exp(shared_scores[k] - new_max);
            shared_scores[k] = e;
            local_tile_sum += e;
        }

        // SIMD sum reduction for tile sum
        float simd_sum_val = simd_sum(local_tile_sum);
        if (simd_lane_id == 0) shared_reduce[simd_group_id] = simd_sum_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < num_simd_groups) {
            float val = shared_reduce[tid];
            val = simd_sum(val);
            if (tid == 0) shared_reduce[0] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running_sum += shared_reduce[0];

        dim_idx = 0;
        for (int d = (int)tid; d < head_dim; d += (int)threads) {
            if (dim_idx >= 16) break;
            float v_acc = 0.0f;
            for (int k = 0; k < tile_len; k++) {
                int key_idx = tile_start + k;
                v_acc += shared_scores[k] * V_head[key_idx * hidden + d];
            }
            acc[dim_idx] += v_acc;
            dim_idx++;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    int dim_idx = 0;
    for (int d = (int)tid; d < head_dim; d += (int)threads) {
        if (dim_idx >= 16) break;
        out_row[d] = acc[dim_idx] * inv_sum;
        dim_idx++;
    }
}

/* ========================================================================
 * Fused Causal Attention for Text Encoder (Qwen3)
 * Processes all heads in parallel with causal masking and GQA support.
 *
 * This kernel computes one output row per threadgroup:
 * out[query_idx, head] = softmax(Q @ K^T * scale + causal_mask) @ V
 *
 * GQA: Multiple Q heads share the same K/V heads
 * (e.g., 32 Q heads / 8 KV heads = 4 Q heads per KV)
 * ======================================================================== */

/* Fused causal attention - one threadgroup per (query_pos, head) pair
 * Q: [seq, num_q_heads * head_dim]
 * K: [seq, num_kv_heads * head_dim]
 * V: [seq, num_kv_heads * head_dim]
 * out: [seq, num_q_heads * head_dim]
 * attn_mask: [seq] - 1 for valid tokens, 0 for padding (optional, can be null)
 */
kernel void causal_attention_fused(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *out [[buffer(3)]],
    device const int *attn_mask [[buffer(4)]],  // Attention mask (1=valid, 0=padding)
    constant int &seq [[buffer(5)]],
    constant int &num_q_heads [[buffer(6)]],
    constant int &num_kv_heads [[buffer(7)]],
    constant int &head_dim [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant int &use_mask [[buffer(10)]],  // Whether to apply attn_mask
    uint3 tg_pos [[threadgroup_position_in_grid]],   // (query_idx, head_idx, 0)
    uint3 tid_pos [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Guard: shared_scores is fixed at 512 entries
    if (seq > 512) return;

    // Shared memory for scores and reductions
    threadgroup float shared_scores[512];  // For attention scores (up to 512 seq len)
    threadgroup float shared_reduce[32];

    int query_idx = tg_pos.x;
    int head_idx = tg_pos.y;
    uint tid = tid_pos.x;
    uint threads = tg_size.x;
    uint num_simd_groups = (threads + 31) / 32;

    if (query_idx >= seq || head_idx >= num_q_heads) return;

    // GQA: map Q head to KV head
    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head_idx = head_idx / heads_per_kv;

    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    // Pointers to this head's Q, K, V
    device const float *q_row = Q + query_idx * q_dim + head_idx * head_dim;
    device const float *K_head = K + kv_head_idx * head_dim;
    device const float *V_head = V + kv_head_idx * head_dim;
    device float *out_row = out + query_idx * q_dim + head_idx * head_dim;

    // ========== Phase 1: Compute Q @ K^T with causal mask ==========
    // Each thread computes scores for a subset of key positions
    float local_max = -INFINITY;

    for (int key_idx = tid; key_idx < seq; key_idx += threads) {
        // Causal mask: only attend to positions <= query_idx
        // Attention mask: only attend to valid tokens (mask[key_idx] != 0)
        bool masked = (key_idx > query_idx);
        if (use_mask && attn_mask[key_idx] == 0) {
            masked = true;
        }

        if (masked) {
            shared_scores[key_idx] = -INFINITY;
        } else {
            // Dot product: Q[query_idx, head] · K[key_idx, kv_head]
            float dot = 0.0f;
            device const float *k_row = K_head + key_idx * kv_dim;
            for (int d = 0; d < head_dim; d++) {
                dot += q_row[d] * k_row[d];
            }
            float score = dot * scale;
            shared_scores[key_idx] = score;
            local_max = max(local_max, score);
        }
    }

    // ========== Phase 2: Find global max (SIMD reduction) ==========
    float simd_max_val = simd_max(local_max);
    if (simd_lane_id == 0) shared_reduce[simd_group_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_reduce[tid];
        val = simd_max(val);
        if (tid == 0) shared_reduce[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = shared_reduce[0];

    // ========== Phase 3: Compute exp(score - max) and sum ==========
    float local_sum = 0.0f;
    for (int key_idx = tid; key_idx < seq; key_idx += threads) {
        float score = shared_scores[key_idx];
        if (score > -1e30f) {  // Not masked
            float e = metal::fast::exp(score - max_val);
            shared_scores[key_idx] = e;
            local_sum += e;
        } else {
            shared_scores[key_idx] = 0.0f;
        }
    }

    // ========== Phase 4: Find total sum (SIMD reduction) ==========
    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_reduce[simd_group_id] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_reduce[tid];
        val = simd_sum(val);
        if (tid == 0) shared_reduce[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = shared_reduce[0];
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

    // ========== Phase 5: Normalize scores ==========
    for (int key_idx = tid; key_idx < seq; key_idx += threads) {
        shared_scores[key_idx] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 6: Compute output = scores @ V ==========
    // Each thread computes a subset of output dimensions
    // Masked positions have score==0, so 0*v_val==0 — no branch needed.
    for (int d = tid; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (int key_idx = 0; key_idx < seq; key_idx++) {
            acc += shared_scores[key_idx] * V_head[key_idx * kv_dim + d];
        }
        out_row[d] = acc;
    }
}

/* ========================================================================
 * Half-Precision Batched Matrix Multiply for Attention
 *
 * Tiled implementation with f32 accumulation for numerical stability.
 * Works directly with half-precision data.
 * ======================================================================== */

constant uint TILE_SIZE = 16;

/* Batched matmul for Q @ K^T (transposes K)
 * Q: [batch, M, K] (half)
 * K: [batch, N, K] (half) - note: N is seq_k, accessed transposed
 * out: [batch, M, N] (half)
 * For attention: M=seq_q, N=seq_k, K=head_dim
 */
kernel void batched_matmul_half_qkt(
    device const half *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant int &M [[buffer(3)]],      // seq_q
    constant int &N [[buffer(4)]],      // seq_k
    constant int &K_dim [[buffer(5)]],  // head_dim
    constant int &batch [[buffer(6)]],  // num_heads
    constant float &scale [[buffer(7)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]
) {
    threadgroup half A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup half B_tile[TILE_SIZE][TILE_SIZE];

    uint b = group_id.z;
    uint col = group_id.x * TILE_SIZE + tid.x;
    uint row = group_id.y * TILE_SIZE + tid.y;

    // Batch offsets
    uint q_batch_offset = b * M * K_dim;
    uint k_batch_offset = b * N * K_dim;
    uint out_batch_offset = b * M * N;

    float sum = 0.0f;
    uint numTiles = (K_dim + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint tiledK = t * TILE_SIZE + tid.x;
        // Load Q tile: Q[b, row, tiledK]
        if (row < (uint)M && tiledK < (uint)K_dim) {
            A_tile[tid.y][tid.x] = Q[q_batch_offset + row * K_dim + tiledK];
        } else {
            A_tile[tid.y][tid.x] = 0.0h;
        }

        uint tiledK_row = t * TILE_SIZE + tid.y;
        // Load K tile transposed: K[b, col, tiledK_row] -> access K^T
        if (col < (uint)N && tiledK_row < (uint)K_dim) {
            B_tile[tid.y][tid.x] = K[k_batch_offset + col * K_dim + tiledK_row];
        } else {
            B_tile[tid.y][tid.x] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product with f32 accumulation
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(A_tile[tid.y][k]) * float(B_tile[k][tid.x]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result with scale
    if (row < (uint)M && col < (uint)N) {
        out[out_batch_offset + row * N + col] = half(sum * scale);
    }
}

/* Batched matmul for scores @ V (no transpose)
 * scores: [batch, M, K] (half) - K is seq_k
 * V: [batch, K, N] (half)
 * out: [batch, M, N] (half)
 * For attention: M=seq_q, K=seq_k, N=head_dim
 */
kernel void batched_matmul_half_sv(
    device const half *scores [[buffer(0)]],
    device const half *V [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant int &M [[buffer(3)]],      // seq_q
    constant int &K_dim [[buffer(4)]],  // seq_k
    constant int &N [[buffer(5)]],      // head_dim
    constant int &batch [[buffer(6)]],  // num_heads
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]
) {
    threadgroup half A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup half B_tile[TILE_SIZE][TILE_SIZE];

    uint b = group_id.z;
    uint col = group_id.x * TILE_SIZE + tid.x;
    uint row = group_id.y * TILE_SIZE + tid.y;

    // Batch offsets
    uint scores_batch_offset = b * M * K_dim;
    uint v_batch_offset = b * K_dim * N;
    uint out_batch_offset = b * M * N;

    float sum = 0.0f;
    uint numTiles = (K_dim + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint tiledK = t * TILE_SIZE + tid.x;
        // Load scores tile: scores[b, row, tiledK]
        if (row < (uint)M && tiledK < (uint)K_dim) {
            A_tile[tid.y][tid.x] = scores[scores_batch_offset + row * K_dim + tiledK];
        } else {
            A_tile[tid.y][tid.x] = 0.0h;
        }

        uint tiledK_row = t * TILE_SIZE + tid.y;
        // Load V tile: V[b, tiledK_row, col]
        if (tiledK_row < (uint)K_dim && col < (uint)N) {
            B_tile[tid.y][tid.x] = V[v_batch_offset + tiledK_row * N + col];
        } else {
            B_tile[tid.y][tid.x] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product with f32 accumulation
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(A_tile[tid.y][k]) * float(B_tile[k][tid.x]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < (uint)M && col < (uint)N) {
        out[out_batch_offset + row * N + col] = half(sum);
    }
}

/* Softmax for half-precision attention scores
 * scores: [batch, M, N] (half)
 * Applies row-wise softmax in-place
 */
kernel void softmax_half(
    device half *scores [[buffer(0)]],
    constant int &total_rows [[buffer(1)]],
    constant int &N [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (row >= (uint)total_rows) return;

    device half *row_data = scores + row * N;
    threadgroup float shared_max[32];
    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    // Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += threads) {
        local_max = max(local_max, float(row_data[i]));
    }

    float simd_max_val = simd_max(local_max);
    if (simd_lane_id == 0) shared_max[simd_group_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_max[tid];
        val = simd_max(val);
        if (tid == 0) shared_max[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = shared_max[0];

    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += threads) {
        float exp_val = metal::fast::exp(float(row_data[i]) - max_val);
        row_data[i] = half(exp_val);  // Store temporarily
        local_sum += exp_val;
    }

    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = shared_sum[0];
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

    // Normalize
    for (int i = tid; i < N; i += threads) {
        row_data[i] = half(float(row_data[i]) * inv_sum);
    }
}

/* ========================================================================
 * BFloat16 Native Kernels
 * These kernels accept bf16 inputs and produce bf16 outputs with f32
 * internal computation for numerical stability.
 * ======================================================================== */

/* Helper: bf16 <-> f32 conversion functions */
inline float bf16_to_f32(ushort bf16) {
    uint bits = uint(bf16) << 16;
    return as_type<float>(bits);
}

inline ushort f32_to_bf16(float f32) {
    uint bits = as_type<uint>(f32);
    // Inf/NaN: exponent all-1s — truncate without rounding to preserve payload
    if ((bits & 0x7F800000u) == 0x7F800000u) return ushort(bits >> 16);
    // Round to nearest even
    uint lsb = (bits >> 16) & 1;
    uint rounding = 0x7FFF + lsb;
    bits += rounding;
    return ushort(bits >> 16);
}

/* RMSNorm for bf16: out = x * rsqrt(mean(x^2) + eps) * weight
 * x: [seq, hidden] (bf16), weight: [hidden] (bf16), out: [seq, hidden] (bf16)
 */
kernel void rms_norm_bf16(
    device const ushort *x [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device ushort *out [[buffer(2)]],
    constant int &hidden [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    device const ushort *x_row = x + row * hidden;
    device ushort *out_row = out + row * hidden;

    // Compute partial sum of squares in f32
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = bf16_to_f32(x_row[i]);
        local_sum += val * val;
    }

    float simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = rsqrt(shared_sum[0] / float(hidden) + eps);

    // Apply normalization with weight, output bf16
    for (int i = tid; i < hidden; i += threads) {
        float val = bf16_to_f32(x_row[i]);
        float w = bf16_to_f32(weight[i]);
        out_row[i] = f32_to_bf16(val * rms_inv * w);
    }
}

/* QK RMSNorm for bf16 (legacy) - single thread per (seq, head). Kept as fallback. */
kernel void qk_rms_norm_bf16_legacy(
    device ushort *q [[buffer(0)]],
    device ushort *k [[buffer(1)]],
    device const ushort *q_weight [[buffer(2)]],
    device const ushort *k_weight [[buffer(3)]],
    constant int &heads [[buffer(4)]],
    constant int &head_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    uint hidden = heads * head_dim;
    uint offset = seq_idx * hidden + head_idx * head_dim;

    // RMSNorm for Q
    float sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = bf16_to_f32(q[offset + d]);
        sum_sq += val * val;
    }
    float rms_inv = rsqrt(sum_sq / float(head_dim) + eps);
    for (int d = 0; d < head_dim; d++) {
        float val = bf16_to_f32(q[offset + d]);
        float w = bf16_to_f32(q_weight[d]);
        q[offset + d] = f32_to_bf16(val * rms_inv * w);
    }

    // RMSNorm for K
    sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = bf16_to_f32(k[offset + d]);
        sum_sq += val * val;
    }
    rms_inv = rsqrt(sum_sq / float(head_dim) + eps);
    for (int d = 0; d < head_dim; d++) {
        float val = bf16_to_f32(k[offset + d]);
        float w = bf16_to_f32(k_weight[d]);
        k[offset + d] = f32_to_bf16(val * rms_inv * w);
    }
}

/* QK RMSNorm for bf16 - threadgroup-parallel version.
 * One threadgroup per (seq_idx, head_idx). Threads cooperate on the
 * head_dim reduction via shared memory.
 * Dispatched with threadgroupsPerGrid=(seq, heads, 1),
 * threadsPerThreadgroup=(min(128, head_dim), 1, 1).
 */
kernel void qk_rms_norm_bf16(
    device ushort *q [[buffer(0)]],
    device ushort *k [[buffer(1)]],
    device const ushort *q_weight [[buffer(2)]],
    device const ushort *k_weight [[buffer(3)]],
    constant int &heads [[buffer(4)]],
    constant int &head_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint seq_idx = group_id.x;
    uint head_idx = group_id.y;

    uint hidden = heads * head_dim;
    device ushort *q_head = q + seq_idx * hidden + head_idx * head_dim;
    device ushort *k_head = k + seq_idx * hidden + head_idx * head_dim;

    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    // --- Q: parallel sum-of-squares ---
    float local_sum = 0.0f;
    for (int d = tid; d < head_dim; d += threads) {
        float val = bf16_to_f32(q_head[d]);
        local_sum += val * val;
    }

    float simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_rms = rsqrt(shared_sum[0] / float(head_dim) + eps);

    for (int d = tid; d < head_dim; d += threads) {
        float val = bf16_to_f32(q_head[d]);
        float w = bf16_to_f32(q_weight[d]);
        q_head[d] = f32_to_bf16(val * q_rms * w);
    }

    // No barrier needed: Q and K are separate device memory regions,
    // and shared_sum is overwritten below.

    // --- K: parallel sum-of-squares ---
    local_sum = 0.0f;
    for (int d = tid; d < head_dim; d += threads) {
        float val = bf16_to_f32(k_head[d]);
        local_sum += val * val;
    }

    simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float k_rms = rsqrt(shared_sum[0] / float(head_dim) + eps);

    for (int d = tid; d < head_dim; d += threads) {
        float val = bf16_to_f32(k_head[d]);
        float w = bf16_to_f32(k_weight[d]);
        k_head[d] = f32_to_bf16(val * k_rms * w);
    }
}

/* Per-head RMSNorm for bf16 - single tensor version for GQA support
 * x: [seq, heads * head_dim] (bf16, modified in-place)
 * weight: [head_dim] (bf16)
 * Dispatched with (seq, heads) threadgroups
 */
kernel void head_rms_norm_bf16(
    device ushort *x [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    constant int &heads [[buffer(2)]],
    constant int &head_dim [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    if (head_idx >= uint(heads)) return;

    uint hidden = heads * head_dim;
    uint offset = seq_idx * hidden + head_idx * head_dim;

    // Compute RMS
    float sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = bf16_to_f32(x[offset + d]);
        sum_sq += val * val;
    }
    float rms_inv = rsqrt(sum_sq / float(head_dim) + eps);

    // Normalize and scale
    for (int d = 0; d < head_dim; d++) {
        float val = bf16_to_f32(x[offset + d]);
        float w = bf16_to_f32(weight[d]);
        x[offset + d] = f32_to_bf16(val * rms_inv * w);
    }
}

/* LayerNorm + AdaLN for bf16
 * out = (1 + scale) * norm(x) + shift
 */
kernel void adaln_norm_bf16(
    device const ushort *x [[buffer(0)]],
    device const ushort *shift [[buffer(1)]],
    device const ushort *scale [[buffer(2)]],
    device ushort *out [[buffer(3)]],
    constant int &hidden [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    device const ushort *x_row = x + row * hidden;
    device ushort *out_row = out + row * hidden;

    // First pass: compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        local_sum += bf16_to_f32(x_row[i]);
    }

    float simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = shared_sum[0] / float(hidden);

    // Second pass: variance via Welford (numerically stable)
    float var_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float diff = bf16_to_f32(x_row[i]) - mean;
        var_sum += diff * diff;
    }

    simd_result = simd_sum(var_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float var = shared_sum[0] / float(hidden);
    float std_inv = rsqrt(var + eps);

    for (int i = tid; i < hidden; i += threads) {
        float val = bf16_to_f32(x_row[i]);
        float s = bf16_to_f32(scale[i]);
        float sh = bf16_to_f32(shift[i]);
        float norm = (val - mean) * std_inv;
        out_row[i] = f32_to_bf16((1.0f + s) * norm + sh);
    }
}

/* SiLU for bf16: x * sigmoid(x) */
kernel void silu_bf16(
    device ushort *x [[buffer(0)]],
    constant int &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float val = bf16_to_f32(x[gid]);
        x[gid] = f32_to_bf16(val / (1.0f + metal::fast::exp(-val)));
    }
}

/* SiLU with multiply for bf16: gate = silu(gate) * up */
kernel void silu_mul_bf16(
    device ushort *gate [[buffer(0)]],
    device const ushort *up [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float g = bf16_to_f32(gate[gid]);
        float silu_g = g / (1.0f + metal::fast::exp(-g));
        float u = bf16_to_f32(up[gid]);
        gate[gid] = f32_to_bf16(silu_g * u);
    }
}

/* Gated add for bf16: out += gate * proj */
kernel void gated_add_bf16(
    device ushort *out [[buffer(0)]],
    device const ushort *gate [[buffer(1)]],
    device const ushort *proj [[buffer(2)]],
    constant int &seq [[buffer(3)]],
    constant int &hidden [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint s = pos.x;
    uint h = pos.y;
    if (s < uint(seq) && h < uint(hidden)) {
        uint idx = s * hidden + h;
        float o = bf16_to_f32(out[idx]);
        float g = bf16_to_f32(gate[h]);
        float p = bf16_to_f32(proj[idx]);
        out[idx] = f32_to_bf16(o + g * p);
    }
}

/* Simple element-wise add for bf16: out = a + b */
kernel void add_bf16(
    device const ushort *a [[buffer(0)]],
    device const ushort *b [[buffer(1)]],
    device ushort *out [[buffer(2)]],
    constant int &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float va = bf16_to_f32(a[gid]);
        float vb = bf16_to_f32(b[gid]);
        out[gid] = f32_to_bf16(va + vb);
    }
}

/* RoPE for bf16: applies rotary position embeddings */
kernel void apply_rope_unified_bf16(
    device ushort *x [[buffer(0)]],
    device const float *txt_cos [[buffer(1)]],
    device const float *txt_sin [[buffer(2)]],
    device const float *img_cos [[buffer(3)]],
    device const float *img_sin [[buffer(4)]],
    constant int &seq [[buffer(5)]],
    constant int &img_offset [[buffer(6)]],
    constant int &heads [[buffer(7)]],
    constant int &head_dim [[buffer(8)]],
    constant int &axis_dim [[buffer(9)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint seq_idx = pos.x;
    uint head_idx = pos.y;

    if (seq_idx >= uint(seq) || head_idx >= uint(heads)) return;

    uint hidden = heads * head_dim;
    device ushort *vec = x + seq_idx * hidden + head_idx * head_dim;

    device const float *cos_row;
    device const float *sin_row;

    if (seq_idx < uint(img_offset)) {
        cos_row = txt_cos + seq_idx * head_dim;
        sin_row = txt_sin + seq_idx * head_dim;
    } else {
        uint img_idx = seq_idx - uint(img_offset);
        cos_row = img_cos + img_idx * head_dim;
        sin_row = img_sin + img_idx * head_dim;
    }

    for (int d = 0; d < head_dim; d += 2) {
        float c = cos_row[d];
        float s = sin_row[d];

        float x0 = bf16_to_f32(vec[d]);
        float x1 = bf16_to_f32(vec[d + 1]);

        vec[d] = f32_to_bf16(x0 * c - x1 * s);
        vec[d + 1] = f32_to_bf16(x1 * c + x0 * s);
    }
}

/* Batched matmul Q @ K^T for bf16 with f32 accumulation */
kernel void batched_matmul_bf16_qkt(
    device const ushort *Q [[buffer(0)]],
    device const ushort *K [[buffer(1)]],
    device ushort *out [[buffer(2)]],
    constant int &M [[buffer(3)]],      // seq_q
    constant int &N [[buffer(4)]],      // seq_k
    constant int &K_dim [[buffer(5)]],  // head_dim
    constant int &batch [[buffer(6)]],  // num_heads
    constant float &scale [[buffer(7)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]
) {
    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];

    uint b = group_id.z;
    uint col = group_id.x * TILE_SIZE + tid.x;
    uint row = group_id.y * TILE_SIZE + tid.y;

    uint q_batch_offset = b * M * K_dim;
    uint k_batch_offset = b * N * K_dim;
    uint out_batch_offset = b * M * N;

    float sum = 0.0f;
    uint numTiles = (K_dim + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint tiledK = t * TILE_SIZE + tid.x;
        if (row < (uint)M && tiledK < (uint)K_dim) {
            A_tile[tid.y][tid.x] = bf16_to_f32(Q[q_batch_offset + row * K_dim + tiledK]);
        } else {
            A_tile[tid.y][tid.x] = 0.0f;
        }

        uint tiledK_row = t * TILE_SIZE + tid.y;
        if (col < (uint)N && tiledK_row < (uint)K_dim) {
            B_tile[tid.y][tid.x] = bf16_to_f32(K[k_batch_offset + col * K_dim + tiledK_row]);
        } else {
            B_tile[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[tid.y][k] * B_tile[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < (uint)M && col < (uint)N) {
        out[out_batch_offset + row * N + col] = f32_to_bf16(sum * scale);
    }
}

/* Batched matmul scores @ V for bf16 with f32 accumulation */
kernel void batched_matmul_bf16_sv(
    device const ushort *scores [[buffer(0)]],
    device const ushort *V [[buffer(1)]],
    device ushort *out [[buffer(2)]],
    constant int &M [[buffer(3)]],      // seq_q
    constant int &K_dim [[buffer(4)]],  // seq_k
    constant int &N [[buffer(5)]],      // head_dim
    constant int &batch [[buffer(6)]],  // num_heads
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]
) {
    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];

    uint b = group_id.z;
    uint col = group_id.x * TILE_SIZE + tid.x;
    uint row = group_id.y * TILE_SIZE + tid.y;

    uint scores_batch_offset = b * M * K_dim;
    uint v_batch_offset = b * K_dim * N;
    uint out_batch_offset = b * M * N;

    float sum = 0.0f;
    uint numTiles = (K_dim + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint tiledK = t * TILE_SIZE + tid.x;
        if (row < (uint)M && tiledK < (uint)K_dim) {
            A_tile[tid.y][tid.x] = bf16_to_f32(scores[scores_batch_offset + row * K_dim + tiledK]);
        } else {
            A_tile[tid.y][tid.x] = 0.0f;
        }

        uint tiledK_row = t * TILE_SIZE + tid.y;
        if (tiledK_row < (uint)K_dim && col < (uint)N) {
            B_tile[tid.y][tid.x] = bf16_to_f32(V[v_batch_offset + tiledK_row * N + col]);
        } else {
            B_tile[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[tid.y][k] * B_tile[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < (uint)M && col < (uint)N) {
        out[out_batch_offset + row * N + col] = f32_to_bf16(sum);
    }
}

/* Softmax for bf16 attention scores (f32 internal computation) */
kernel void softmax_bf16(
    device ushort *scores [[buffer(0)]],
    constant int &total_rows [[buffer(1)]],
    constant int &N [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (row >= (uint)total_rows) return;

    device ushort *row_data = scores + row * N;
    threadgroup float shared_max[32];
    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    // Find max in f32
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += threads) {
        local_max = max(local_max, bf16_to_f32(row_data[i]));
    }

    float simd_max_val = simd_max(local_max);
    if (simd_lane_id == 0) shared_max[simd_group_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_max[tid];
        val = simd_max(val);
        if (tid == 0) shared_max[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = shared_max[0];

    // Compute exp and sum in f32
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += threads) {
        float exp_val = metal::fast::exp(bf16_to_f32(row_data[i]) - max_val);
        row_data[i] = f32_to_bf16(exp_val);
        local_sum += exp_val;
    }

    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = (shared_sum[0] > 0.0f) ? (1.0f / shared_sum[0]) : 0.0f;

    // Normalize and store as bf16
    for (int i = tid; i < N; i += threads) {
        row_data[i] = f32_to_bf16(bf16_to_f32(row_data[i]) * inv_sum);
    }
}

/* Convert f32 tensor to bf16 */
kernel void f32_to_bf16_convert(
    device const float *input [[buffer(0)]],
    device ushort *output [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        output[gid] = f32_to_bf16(input[gid]);
    }
}

/* Convert bf16 tensor to f32 */
kernel void bf16_to_f32_convert(
    device const ushort *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        output[gid] = bf16_to_f32(input[gid]);
    }
}

/* Batched bf16 linear: out = x @ W^T where W is bf16 weights
 * x: [batch, in_features] (bf16)
 * W: [out_features, in_features] (bf16)
 * out: [batch, out_features] (bf16)
 * Uses f32 accumulation for numerical stability
 */
kernel void linear_bf16(
    device const ushort *x [[buffer(0)]],
    device const ushort *W [[buffer(1)]],
    device ushort *out [[buffer(2)]],
    constant int &batch [[buffer(3)]],
    constant int &in_features [[buffer(4)]],
    constant int &out_features [[buffer(5)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]
) {
    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];

    uint row = group_id.y * TILE_SIZE + tid.y;  // batch index
    uint col = group_id.x * TILE_SIZE + tid.x;  // output feature

    float sum = 0.0f;
    uint numTiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint tiledK = t * TILE_SIZE + tid.x;
        // Load x[row, tiledK]
        if (row < (uint)batch && tiledK < (uint)in_features) {
            A_tile[tid.y][tid.x] = bf16_to_f32(x[row * in_features + tiledK]);
        } else {
            A_tile[tid.y][tid.x] = 0.0f;
        }

        uint tiledK_row = t * TILE_SIZE + tid.y;
        // Load W[col, tiledK_row] (transposed access)
        if (col < (uint)out_features && tiledK_row < (uint)in_features) {
            B_tile[tid.y][tid.x] = bf16_to_f32(W[col * in_features + tiledK_row]);
        } else {
            B_tile[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[tid.y][k] * B_tile[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < (uint)batch && col < (uint)out_features) {
        out[row * out_features + col] = f32_to_bf16(sum);
    }
}

/* Split fused QKV+MLP output into separate tensors (bf16 version)
 * fused: [seq, hidden*3 + mlp_hidden*2] (bf16)
 * q, k, v: [seq, hidden] (bf16)
 * gate, up: [seq, mlp_hidden] (bf16)
 */
kernel void split_qkv_mlp_bf16(
    device const ushort *fused [[buffer(0)]],
    device ushort *q [[buffer(1)]],
    device ushort *k [[buffer(2)]],
    device ushort *v [[buffer(3)]],
    device ushort *gate [[buffer(4)]],
    device ushort *up [[buffer(5)]],
    constant int &seq [[buffer(6)]],
    constant int &hidden [[buffer(7)]],
    constant int &mlp_hidden [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint s = pos.x;
    uint e = pos.y;

    if (s >= uint(seq)) return;

    int fused_dim = hidden * 3 + mlp_hidden * 2;
    device const ushort *row = fused + s * fused_dim;

    if (e < uint(hidden)) {
        q[s * hidden + e] = row[e];
        k[s * hidden + e] = row[hidden + e];
        v[s * hidden + e] = row[hidden * 2 + e];
    }

    if (e < uint(mlp_hidden)) {
        gate[s * mlp_hidden + e] = row[hidden * 3 + e];
        up[s * mlp_hidden + e] = row[hidden * 3 + mlp_hidden + e];
    }
}

/* Concat attention + MLP outputs (bf16 version)
 * attn: [seq, hidden] (bf16)
 * mlp: [seq, mlp_hidden] (bf16)
 * out: [seq, hidden + mlp_hidden] (bf16)
 */
kernel void concat_attn_mlp_bf16(
    device const ushort *attn [[buffer(0)]],
    device const ushort *mlp [[buffer(1)]],
    device ushort *out [[buffer(2)]],
    constant int &seq [[buffer(3)]],
    constant int &hidden [[buffer(4)]],
    constant int &mlp_hidden [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint s = pos.x;
    uint e = pos.y;

    if (s >= uint(seq)) return;

    int out_dim = hidden + mlp_hidden;
    device ushort *out_row = out + s * out_dim;

    if (e < uint(hidden)) {
        out_row[e] = attn[s * hidden + e];
    }

    if (e < uint(mlp_hidden)) {
        out_row[hidden + e] = mlp[s * mlp_hidden + e];
    }
}

/* Concatenate two bf16 sequences along seq dimension:
 * out = [a; b], where a: [seq_a, hidden], b: [seq_b, hidden]
 */
kernel void concat_seq_bf16(
    device const ushort *a [[buffer(0)]],
    device const ushort *b [[buffer(1)]],
    device ushort *out [[buffer(2)]],
    constant int &seq_a [[buffer(3)]],
    constant int &seq_b [[buffer(4)]],
    constant int &hidden [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint s = pos.x;
    uint h = pos.y;
    uint total_seq = uint(seq_a + seq_b);

    if (s >= total_seq || h >= uint(hidden)) return;

    if (s < uint(seq_a)) {
        out[s * hidden + h] = a[s * hidden + h];
    } else {
        uint b_idx = s - uint(seq_a);
        out[s * hidden + h] = b[b_idx * hidden + h];
    }
}

/* Slice a bf16 sequence along seq dimension:
 * out[s, h] = in[(s + start), h], out: [seq_out, hidden]
 */
kernel void slice_seq_bf16(
    device const ushort *in [[buffer(0)]],
    device ushort *out [[buffer(1)]],
    constant int &seq_out [[buffer(2)]],
    constant int &hidden [[buffer(3)]],
    constant int &start [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint s = pos.x;
    uint h = pos.y;

    if (s >= uint(seq_out) || h >= uint(hidden)) return;

    out[s * hidden + h] = in[(s + uint(start)) * hidden + h];
}

/* ========================================================================
 * BF16 Transpose kernels for attention
 * ======================================================================== */

/* Transpose for attention input: [seq, heads*head_dim] -> [heads, seq, head_dim]
 * in:  [seq, heads * head_dim] (bf16)
 * out: [heads, seq, head_dim] (bf16)
 */
kernel void transpose_to_heads_bf16(
    device const ushort *in [[buffer(0)]],
    device ushort *out [[buffer(1)]],
    constant int &seq [[buffer(2)]],
    constant int &heads [[buffer(3)]],
    constant int &head_dim [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint h = pos.z;      // head index
    uint s = pos.y;      // sequence position
    uint d = pos.x;      // head_dim position

    if (h >= uint(heads) || s >= uint(seq) || d >= uint(head_dim)) return;

    // Input layout: [seq, heads * head_dim] - row s, column h*head_dim + d
    uint in_idx = s * (heads * head_dim) + h * head_dim + d;

    // Output layout: [heads, seq, head_dim] - head h, row s, column d
    uint out_idx = h * (seq * head_dim) + s * head_dim + d;

    out[out_idx] = in[in_idx];
}

/* Transpose for attention output: [heads, seq, head_dim] -> [seq, heads*head_dim]
 * in:  [heads, seq, head_dim] (bf16)
 * out: [seq, heads * head_dim] (bf16)
 */
kernel void transpose_from_heads_bf16(
    device const ushort *in [[buffer(0)]],
    device ushort *out [[buffer(1)]],
    constant int &seq [[buffer(2)]],
    constant int &heads [[buffer(3)]],
    constant int &head_dim [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint h = pos.z;      // head index
    uint s = pos.y;      // sequence position
    uint d = pos.x;      // head_dim position

    if (h >= uint(heads) || s >= uint(seq) || d >= uint(head_dim)) return;

    // Input layout: [heads, seq, head_dim] - head h, row s, column d
    uint in_idx = h * (seq * head_dim) + s * head_dim + d;

    // Output layout: [seq, heads * head_dim] - row s, column h*head_dim + d
    uint out_idx = s * (heads * head_dim) + h * head_dim + d;

    out[out_idx] = in[in_idx];
}

/* ========================================================================
 * Fused Non-Causal Attention for BF16 Pipeline
 * Same algorithm as attention_fused but with bf16 I/O and f32 computation.
 *
 * This kernel computes one output row per threadgroup:
 * out[query_idx, head] = softmax(Q @ K^T * scale) @ V
 *
 * Works directly on [seq, heads*head_dim] layout without transpose.
 * bf16 input/output, f32 accumulation for numerical stability.
 * ======================================================================== */

kernel void attention_fused_bf16(
    device const ushort *Q [[buffer(0)]],      // [seq_q, heads * head_dim] bf16
    device const ushort *K [[buffer(1)]],      // [seq_k, heads * head_dim] bf16
    device const ushort *V [[buffer(2)]],      // [seq_k, heads * head_dim] bf16
    device ushort *out [[buffer(3)]],          // [seq_q, heads * head_dim] bf16
    constant int &seq_q [[buffer(4)]],         // Query sequence length
    constant int &seq_k [[buffer(5)]],         // Key/Value sequence length
    constant int &num_heads [[buffer(6)]],
    constant int &head_dim [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    threadgroup float *shared_scores [[threadgroup(0)]],  // [seq_k] dynamic
    uint3 tg_pos [[threadgroup_position_in_grid]],   // (query_idx, head_idx, 0)
    uint3 tid_pos [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Shared memory for reductions (shared_scores is dynamic via threadgroup(0))
    threadgroup float shared_reduce[32];

    int query_idx = tg_pos.x;
    int head_idx = tg_pos.y;
    uint tid = tid_pos.x;
    uint threads = tg_size.x;
    uint num_simd_groups = (threads + 31) / 32;

    // Guard: shared_q[128] is fixed-size
    if (head_dim > 128) return;
    if (query_idx >= seq_q || head_idx >= num_heads) return;

    int hidden = num_heads * head_dim;

    // Pointers to this position's Q and output (layout: [seq, heads*head_dim])
    device const ushort *q_row = Q + query_idx * hidden + head_idx * head_dim;
    device ushort *out_row = out + query_idx * hidden + head_idx * head_dim;

    // K and V have same layout, head offset is same
    device const ushort *K_head = K + head_idx * head_dim;
    device const ushort *V_head = V + head_idx * head_dim;

    // Cache Q in threadgroup memory (converted to f32)
    threadgroup float shared_q[128];  // Max head_dim = 128
    for (int d = tid; d < head_dim; d += threads) {
        shared_q[d] = bf16_to_f32(q_row[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 1: Compute Q @ K^T ==========
    float local_max = -INFINITY;

    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        // Dot product: Q[query_idx, head] · K[key_idx, head]
        float dot = 0.0f;
        device const ushort *k_row = K_head + key_idx * hidden;
        for (int d = 0; d < head_dim; d++) {
            dot += shared_q[d] * bf16_to_f32(k_row[d]);
        }
        float score = dot * scale;
        shared_scores[key_idx] = score;
        local_max = max(local_max, score);
    }

    // ========== Phase 2: Find global max (SIMD) ==========
    float simd_max_val = simd_max(local_max);
    if (simd_lane_id == 0) shared_reduce[simd_group_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_reduce[tid];
        val = simd_max(val);
        if (tid == 0) shared_reduce[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = shared_reduce[0];

    // ========== Phase 3: Compute exp(score - max) and sum ==========
    float local_sum = 0.0f;
    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        float e = metal::fast::exp(shared_scores[key_idx] - max_val);
        shared_scores[key_idx] = e;
        local_sum += e;
    }

    // ========== Phase 4: Find total sum (SIMD) ==========
    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_reduce[simd_group_id] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_reduce[tid];
        val = simd_sum(val);
        if (tid == 0) shared_reduce[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = (shared_reduce[0] > 0.0f) ? (1.0f / shared_reduce[0]) : 0.0f;

    // ========== Phase 5: Normalize scores ==========
    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        shared_scores[key_idx] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 6: Compute output = scores @ V ==========
    for (int d = tid; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (int key_idx = 0; key_idx < seq_k; key_idx++) {
            float v_val = bf16_to_f32(V_head[key_idx * hidden + d]);
            acc += shared_scores[key_idx] * v_val;
        }
        out_row[d] = f32_to_bf16(acc);
    }
}

/* ========================================================================
 * BF16 Causal Attention with GQA Support
 *
 * Fused kernel for causal (decoder) attention with bf16 I/O and f32 compute.
 * Supports Grouped Query Attention where num_q_heads > num_kv_heads.
 *
 * Q: [seq, num_q_heads * head_dim] (bf16)
 * K: [seq, num_kv_heads * head_dim] (bf16)
 * V: [seq, num_kv_heads * head_dim] (bf16)
 * out: [seq, num_q_heads * head_dim] (bf16)
 * attn_mask: [seq] - 1 for valid tokens, 0 for padding (optional)
 *
 * Each threadgroup processes one (query_pos, head) pair.
 * ======================================================================== */

kernel void causal_attention_fused_bf16(
    device const ushort *Q [[buffer(0)]],
    device const ushort *K [[buffer(1)]],
    device const ushort *V [[buffer(2)]],
    device ushort *out [[buffer(3)]],
    device const int *attn_mask [[buffer(4)]],
    constant int &seq [[buffer(5)]],
    constant int &num_q_heads [[buffer(6)]],
    constant int &num_kv_heads [[buffer(7)]],
    constant int &head_dim [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant int &use_mask [[buffer(10)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid_pos [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Guards: shared_scores[512] and shared_q[128] are fixed-size
    if (seq > 512) return;
    if (head_dim > 128) return;

    // Shared memory for scores and reductions
    threadgroup float shared_scores[512];
    threadgroup float shared_reduce[32];
    threadgroup float shared_q[128];  // Cache Q for this query position

    int query_idx = tg_pos.x;
    int head_idx = tg_pos.y;
    uint tid = tid_pos.x;
    uint threads = tg_size.x;
    uint num_simd_groups = (threads + 31) / 32;

    if (query_idx >= seq || head_idx >= num_q_heads) return;

    // GQA: map Q head to KV head
    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head_idx = head_idx / heads_per_kv;

    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    // Pointers to this head's Q, K, V (bf16)
    device const ushort *q_row = Q + query_idx * q_dim + head_idx * head_dim;
    device const ushort *K_head = K + kv_head_idx * head_dim;
    device const ushort *V_head = V + kv_head_idx * head_dim;
    device ushort *out_row = out + query_idx * q_dim + head_idx * head_dim;

    // Load Q into shared memory (convert bf16 -> f32)
    for (int d = tid; d < head_dim; d += threads) {
        shared_q[d] = bf16_to_f32(q_row[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 1: Compute Q @ K^T with causal mask ==========
    float local_max = -INFINITY;

    for (int key_idx = tid; key_idx < seq; key_idx += threads) {
        // Causal mask: only attend to positions <= query_idx
        // Attention mask: only attend to valid tokens
        bool masked = (key_idx > query_idx);
        if (use_mask && attn_mask[key_idx] == 0) {
            masked = true;
        }

        if (masked) {
            shared_scores[key_idx] = -INFINITY;
        } else {
            // Dot product: Q[query_idx, head] · K[key_idx, kv_head]
            float dot = 0.0f;
            device const ushort *k_row = K_head + key_idx * kv_dim;
            for (int d = 0; d < head_dim; d++) {
                dot += shared_q[d] * bf16_to_f32(k_row[d]);
            }
            float score = dot * scale;
            shared_scores[key_idx] = score;
            local_max = max(local_max, score);
        }
    }

    // ========== Phase 2: Find global max (SIMD reduction) ==========
    float simd_max_val = simd_max(local_max);
    if (simd_lane_id == 0) shared_reduce[simd_group_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_reduce[tid];
        val = simd_max(val);
        if (tid == 0) shared_reduce[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = shared_reduce[0];

    // ========== Phase 3: Compute exp(score - max) and sum ==========
    float local_sum = 0.0f;
    for (int key_idx = tid; key_idx < seq; key_idx += threads) {
        float score = shared_scores[key_idx];
        if (score > -1e30f) {
            float e = metal::fast::exp(score - max_val);
            shared_scores[key_idx] = e;
            local_sum += e;
        } else {
            shared_scores[key_idx] = 0.0f;
        }
    }

    // ========== Phase 4: Find total sum (SIMD reduction) ==========
    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_reduce[simd_group_id] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_reduce[tid];
        val = simd_sum(val);
        if (tid == 0) shared_reduce[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = (shared_reduce[0] > 0.0f) ? (1.0f / shared_reduce[0]) : 0.0f;

    // ========== Phase 5: Normalize scores ==========
    for (int key_idx = tid; key_idx < seq; key_idx += threads) {
        shared_scores[key_idx] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== Phase 6: Compute output = scores @ V ==========
    // Masked positions have score==0, so 0*v_val==0 — no branch needed.
    for (int d = tid; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (int key_idx = 0; key_idx < seq; key_idx++) {
            acc += shared_scores[key_idx] * bf16_to_f32(V_head[key_idx * kv_dim + d]);
        }
        out_row[d] = f32_to_bf16(acc);
    }
}

/* ========================================================================
 * BF16 RoPE for Text-Only (Qwen3 style)
 *
 * Applies rotary position embeddings to Q and K tensors.
 * Unlike the unified RoPE, this is for decoder-only text encoding.
 *
 * Q: [seq, num_q_heads * head_dim] (bf16) - modified in-place
 * K: [seq, num_kv_heads * head_dim] (bf16) - modified in-place
 * cos: [seq, head_dim/2] (f32) - precomputed cos(pos * freq)
 * sin: [seq, head_dim/2] (f32) - precomputed sin(pos * freq)
 *
 * RoPE formula: for each pair (x0, x1):
 *   x0' = x0 * cos - x1 * sin
 *   x1' = x0 * sin + x1 * cos
 * ======================================================================== */

kernel void rope_bf16(
    device ushort *Q [[buffer(0)]],
    device ushort *K [[buffer(1)]],
    device const float *cos_cache [[buffer(2)]],
    device const float *sin_cache [[buffer(3)]],
    constant int &seq [[buffer(4)]],
    constant int &num_q_heads [[buffer(5)]],
    constant int &num_kv_heads [[buffer(6)]],
    constant int &head_dim [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]  // (pos, head)
) {
    int pos = gid.x;
    int head = gid.y;

    if (pos >= seq) return;

    int half_dim = head_dim / 2;

    // Apply RoPE to Q heads
    if (head < num_q_heads) {
        int q_dim = num_q_heads * head_dim;
        device ushort *q_head = Q + pos * q_dim + head * head_dim;

        for (int i = 0; i < half_dim; i++) {
            float x0 = bf16_to_f32(q_head[i]);
            float x1 = bf16_to_f32(q_head[i + half_dim]);
            float c = cos_cache[pos * half_dim + i];
            float s = sin_cache[pos * half_dim + i];

            q_head[i] = f32_to_bf16(x0 * c - x1 * s);
            q_head[i + half_dim] = f32_to_bf16(x0 * s + x1 * c);
        }
    }

    // Apply RoPE to K heads (fewer heads due to GQA)
    if (head < num_kv_heads) {
        int kv_dim = num_kv_heads * head_dim;
        device ushort *k_head = K + pos * kv_dim + head * head_dim;

        for (int i = 0; i < half_dim; i++) {
            float x0 = bf16_to_f32(k_head[i]);
            float x1 = bf16_to_f32(k_head[i + half_dim]);
            float c = cos_cache[pos * half_dim + i];
            float s = sin_cache[pos * half_dim + i];

            k_head[i] = f32_to_bf16(x0 * c - x1 * s);
            k_head[i + half_dim] = f32_to_bf16(x0 * s + x1 * c);
        }
    }
}

/* ========================================================================
 * F32 VAE Shaders — GPU-resident VAE decoder operations
 * ======================================================================== */

/* GroupNorm f32: one threadgroup per (batch, group) pair.
 * Layout: [batch, channels, H, W] (NCHW)
 * Each group has channels_per_group consecutive channels.
 * Computes mean and variance over all spatial positions × channels in the group. */
kernel void group_norm_f32(
    device const float *x [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device const float *beta [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &channels [[buffer(4)]],
    constant int &spatial [[buffer(5)]],       /* H * W */
    constant int &channels_per_group [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id_val [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    /* group_id encodes (batch, group) */
    int num_groups = channels / channels_per_group;
    int batch_idx = group_id / num_groups;
    int group_idx = group_id % num_groups;

    int c_start = group_idx * channels_per_group;
    int group_size = channels_per_group * spatial;

    /* Base offset for this batch element */
    device const float *x_batch = x + batch_idx * channels * spatial;
    device float *out_batch = out + batch_idx * channels * spatial;

    /* First pass: compute mean
     * Use incremental c/s tracking to avoid per-element integer division. */
    float local_sum = 0.0f;
    {
        int c = c_start + (int)tid / spatial;
        int s = (int)tid % spatial;
        for (int i = tid; i < group_size; i += threads) {
            local_sum += x_batch[c * spatial + s];
            /* Advance (c, s) by 'threads' positions */
            s += threads;
            while (s >= spatial) { s -= spatial; c++; }
        }
    }

    float simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id_val] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = shared_sum[0] / float(group_size);

    /* Second pass: variance via Welford (numerically stable)
     * Same incremental c/s tracking. */
    float var_local = 0.0f;
    {
        int c = c_start + (int)tid / spatial;
        int s = (int)tid % spatial;
        for (int i = tid; i < group_size; i += threads) {
            float diff = x_batch[c * spatial + s] - mean;
            var_local += diff * diff;
            s += threads;
            while (s >= spatial) { s -= spatial; c++; }
        }
    }

    simd_result = simd_sum(var_local);
    if (simd_lane_id == 0) shared_sum[simd_group_id_val] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float var = shared_sum[0] / float(group_size);
    float inv_std = rsqrt(var + eps);

    /* Apply normalization with gamma/beta per channel
     * Same incremental c/s tracking. */
    {
        int c = c_start + (int)tid / spatial;
        int s = (int)tid % spatial;
        for (int i = tid; i < group_size; i += threads) {
            int idx = c * spatial + s;
            float val = (x_batch[idx] - mean) * inv_std;
            out_batch[idx] = gamma[c] * val + beta[c];
            s += threads;
            while (s >= spatial) { s -= spatial; c++; }
        }
    }
}

/* Fused GroupNorm + Swish: normalize then apply x*sigmoid(x) in one pass */
kernel void group_norm_swish_f32(
    device const float *x [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device const float *beta [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &channels [[buffer(4)]],
    constant int &spatial [[buffer(5)]],
    constant int &channels_per_group [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id_val [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[32];
    uint num_simd_groups = (threads + 31) / 32;

    int num_groups = channels / channels_per_group;
    int batch_idx = group_id / num_groups;
    int group_idx = group_id % num_groups;

    int c_start = group_idx * channels_per_group;
    int group_size = channels_per_group * spatial;

    device const float *x_batch = x + batch_idx * channels * spatial;
    device float *out_batch = out + batch_idx * channels * spatial;

    /* First pass: compute mean
     * Use incremental c/s tracking to avoid per-element integer division. */
    float local_sum = 0.0f;
    {
        int c = c_start + (int)tid / spatial;
        int s = (int)tid % spatial;
        for (int i = tid; i < group_size; i += threads) {
            local_sum += x_batch[c * spatial + s];
            s += threads;
            while (s >= spatial) { s -= spatial; c++; }
        }
    }

    float simd_result = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_sum[simd_group_id_val] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = shared_sum[0] / float(group_size);

    /* Second pass: variance via Welford (numerically stable)
     * Same incremental c/s tracking. */
    float var_local = 0.0f;
    {
        int c = c_start + (int)tid / spatial;
        int s = (int)tid % spatial;
        for (int i = tid; i < group_size; i += threads) {
            float diff = x_batch[c * spatial + s] - mean;
            var_local += diff * diff;
            s += threads;
            while (s >= spatial) { s -= spatial; c++; }
        }
    }

    simd_result = simd_sum(var_local);
    if (simd_lane_id == 0) shared_sum[simd_group_id_val] = simd_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        float val = shared_sum[tid];
        val = simd_sum(val);
        if (tid == 0) shared_sum[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float var = shared_sum[0] / float(group_size);
    float inv_std = rsqrt(var + eps);

    /* Apply fused norm + swish with incremental c/s tracking. */
    {
        int c = c_start + (int)tid / spatial;
        int s = (int)tid % spatial;
        for (int i = tid; i < group_size; i += threads) {
            int idx = c * spatial + s;
            float normed = gamma[c] * ((x_batch[idx] - mean) * inv_std) + beta[c];
            out_batch[idx] = normed / (1.0f + metal::fast::exp(-normed));
            s += threads;
            while (s >= spatial) { s -= spatial; c++; }
        }
    }
}

/* Swish f32: out = x * sigmoid(x), in-place safe (out can alias x) */
kernel void swish_f32(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float v = x[gid];
        out[gid] = v / (1.0f + metal::fast::exp(-v));
    }
}

/* Add f32: out = a + b, in-place safe (out can alias a or b) */
kernel void add_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        out[gid] = a[gid] + b[gid];
    }
}

/* Upsample nearest 2x f32: [B, C, H, W] -> [B, C, 2H, 2W]
 * Each thread writes one output element. */
kernel void upsample_nearest_2x_f32(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant int &channels [[buffer(2)]],
    constant int &in_h [[buffer(3)]],
    constant int &in_w [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int out_h = in_h * 2;
    int out_w = in_w * 2;
    int out_spatial = out_h * out_w;
    int total = channels * out_spatial;  /* batch=1 assumed */
    if (gid >= uint(total)) return;

    int c = gid / out_spatial;
    int rem = gid % out_spatial;
    int oy = rem / out_w;
    int ox = rem % out_w;

    int iy = oy / 2;
    int ix = ox / 2;

    out[c * out_spatial + oy * out_w + ox] = x[c * in_h * in_w + iy * in_w + ix];
}

/* 2D matrix transpose with optional scale: [rows, cols] -> [cols, rows]
 * out[c * rows + r] = in[r * cols + c] * scale
 * Used by VAE attention to convert [C, HW] <-> [HW, C]. */
kernel void transpose_2d_scale_f32(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant int &rows [[buffer(2)]],
    constant int &cols [[buffer(3)]],
    constant float &scale [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]
) {
    uint c = pos.x;
    uint r = pos.y;
    if (r >= uint(rows) || c >= uint(cols)) return;
    out[c * rows + r] = in[r * cols + c] * scale;
}
