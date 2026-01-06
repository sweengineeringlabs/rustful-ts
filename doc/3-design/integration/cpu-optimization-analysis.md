# CPU Optimization Analysis

> **Status:** Complete
> **Date:** 2025-01-02

## Executive Summary

Investigated CPU-level optimizations for GPT-2 inference. Found that **memory copies are necessary** for BLAS efficiency, and element-wise operations are NOT the bottleneck.

| Optimization | Expected | Actual | Verdict |
|--------------|----------|--------|---------|
| Zero-copy tensor views (MP-006) | 1.5-2x | **Negative** | Blocked |
| SIMD softmax/GELU (MP-008) | 1.2-1.5x | **~0%** | Not a bottleneck |
| Fast exp/tanh approximations | 1.2x | **~0%** | Reverted |

**Current Performance:** 2.07 tok/s (GPT-2 small, BLAS + KV-cache)

## Investigation Details

### MP-006: Zero-Copy Tensor Views

**Hypothesis:** Replacing `slice().to_owned()` with zero-copy views would eliminate allocation overhead in KV-cache access.

**Approach:**
1. Modified `matmul_batch_4d` to use `ndarray::ArrayView` instead of copying
2. Used `ndarray::s![]` macro for direct slicing into 4D tensors

**Code Change:**
```rust
// Before (copies data)
let a_batch: Vec<f32> = a_slice[a_start..a_start + m * k].to_vec();
let a_2d = Array2::from_shape_vec((m, k), a_batch)?;

// After (view-based)
let a_4d = self.data.view().into_dimensionality::<Ix4>()?;
let a_2d = a_4d.slice(s![b_idx, h_idx, .., ..]);
```

**Result:** Performance DECREASED from 2.07 tok/s to 1.52 tok/s (27% slower)

**Root Cause:**
- Tensors after `transpose()` have non-contiguous memory layouts
- `ndarray::dot()` on non-contiguous views cannot use optimized BLAS routines
- BLAS (SGEMM) requires contiguous memory with known strides
- Copying to contiguous buffers enables BLAS vectorization

**Lesson:** The copy overhead is small compared to the BLAS speedup from contiguous memory access patterns.

### MP-008: SIMD Softmax/GELU

**Hypothesis:** Element-wise operations (exp, tanh) are bottlenecks that can be accelerated with fast approximations.

**Approach:**
1. Implemented `fast_exp()` using polynomial approximation with range reduction
2. Implemented `fast_tanh()` using exp-based formula
3. Applied to softmax and GELU activation functions

**Code:**
```rust
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    let k = (x * LOG2_E).floor();
    let r = x - k * LN_2;
    let exp_r = 1.0 + r + 0.5*r*r + 0.166667*r*r*r;
    exp_r * f32::from_bits(((127 + k as i32) as u32) << 23)
}
```

**Result:** No measurable improvement (~0%)

**Root Cause:**
- BLAS matmul dominates inference time (~70%)
- Element-wise ops are memory-bound, not compute-bound
- Compiler already optimizes std::f32::exp with SIMD where beneficial
- Fast approximations add complexity without benefit

### Profiling Breakdown

| Operation | % of Time | Optimizable? |
|-----------|-----------|--------------|
| BLAS matmul (Q×K, attn×V) | ~70% | Already optimized by OpenBLAS |
| Memory allocation/copy | ~15% | Necessary for BLAS efficiency |
| Softmax/GELU | ~10% | NOT a bottleneck |
| Other (mask, reshape) | ~5% | Minor |

## Conclusions

### What Works
1. **System BLAS** (OpenBLAS) - 10-100x faster than pure Rust
2. **Thread tuning** (OPENBLAS_NUM_THREADS=4) - 36% improvement
3. **KV-cache** with pre-allocation - 42% improvement
4. **Contiguous memory copies** - Required for BLAS efficiency

### What Doesn't Work (for CPU)
1. **Zero-copy views** - Slower due to non-contiguous memory access
2. **Fast math approximations** - No improvement, adds complexity
3. **Parallelization of small ops** - Thread overhead exceeds gains

### Remaining Options

| Option | Expected Gain | Effort | Notes |
|--------|---------------|--------|-------|
| **INT8 Quantization** | 2-4x | Medium | Reduces memory bandwidth |
| Fused attention kernel | 2-3x | High | Custom BLAS-like implementation |
| GPU acceleration | 10-50x | High | CUDA/Metal support |

## Recommendations

1. **For 2-4x CPU speedup:** Implement INT8 quantization (MP-005)
   - Reduces memory bandwidth (biggest bottleneck)
   - INT8 SIMD processes 8 values per instruction vs 4 for f32
   - Weights can be quantized offline, activations dynamically

2. **For 10x+ speedup:** Add GPU support
   - CUDA for NVIDIA GPUs
   - Metal for Apple Silicon
   - Requires significant architectural changes

3. **Do NOT pursue:**
   - Zero-copy views without BLAS stride support
   - Fast math approximations for element-wise ops
   - Micro-optimizations on non-bottleneck code
