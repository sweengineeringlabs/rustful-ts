# BLAS Performance Analysis

> **Status:** Complete
> **Date:** 2024-12-31

## Executive Summary

GPT-2 inference performance was optimized from **1.1 tok/s to 2.2 tok/s** (2x improvement) through:

1. **BLAS thread tuning** - Set `OPENBLAS_NUM_THREADS=4` (+36%)
2. **KV-cache implementation** - Pre-allocated buffers with in-place updates (+42%)

## Final Results

| Configuration | Throughput | Improvement |
|---------------|------------|-------------|
| Baseline (default threads, no cache) | 1.1 tok/s | - |
| + OPENBLAS_NUM_THREADS=4 | 1.5 tok/s | +36% |
| + Pre-allocated KV-Cache | **2.2 tok/s** | **+100% total** |

### Benchmark Details

| Metric | Without Cache | With Cache |
|--------|--------------|------------|
| Time (2 prompts, 50 tok each) | 66.0s | 46.4s |
| Throughput | 1.52 tok/s | 2.16 tok/s |
| Speedup | - | 1.4x |

## Root Causes Identified

### 1. Thread Contention (Fixed)

| Thread Config | Wall Time | Throughput |
|---------------|-----------|------------|
| 1 thread | 81.6s | 1.5 tok/s |
| **4 threads** | **71.4s** | **1.7 tok/s** |
| All cores (default) | 113.4s | 1.1 tok/s |

**Problem:** Default OpenBLAS uses all cores, causing contention for small matrices.

**Solution:** `export OPENBLAS_NUM_THREADS=4`

### 2. Missing KV-Cache (Fixed)

**Problem:** Generation recomputed full attention for all tokens on each step.

```
Without cache: O(n²) - regenerate all previous tokens
With cache:    O(n)  - only compute new token
```

**Solution:** Implemented `CachedTextGenerator` with pre-allocated buffers.

## Implementation

### Files Modified

| File | Changes |
|------|---------|
| `rustml-core/src/tensor.rs` | Added `set_slice()` for in-place updates |
| `rustml-nn/src/attention.rs` | `LayerKvCache` with pre-allocation, `forward_with_cache()` |
| `rustml-nlp/src/gpt.rs` | `GptKvCache`, `forward_with_cache()`, `new_cache()` |
| `rustml-nlp/src/generation.rs` | `CachedTextGenerator` |

### Key Components

```rust
// Pre-allocated KV-cache
pub struct LayerKvCache {
    k: Tensor,      // [B, H, max_seq, D] - pre-allocated
    v: Tensor,      // [B, H, max_seq, D] - pre-allocated
    len: usize,     // Current filled length
}

impl LayerKvCache {
    fn append(&mut self, k_new: &Tensor, v_new: &Tensor) {
        // In-place update using set_slice (no allocation)
        self.k.set_slice(2, self.len, k_new)?;
        self.v.set_slice(2, self.len, v_new)?;
        self.len += new_len;
    }
}
```

## Usage

```rust
use rustml_nlp::CachedTextGenerator;

let generator = CachedTextGenerator::new(&model);
let output = generator.generate(&input, &config)?;
```

```bash
# Optimal BLAS configuration
export OPENBLAS_NUM_THREADS=4
cargo run --release
```

## Remaining Performance Gap

Current: **2.2 tok/s** vs PyTorch: **10-30 tok/s**

### Why Still Slower?

| Bottleneck | Impact | Fix |
|------------|--------|-----|
| `get_k()`/`get_v()` slicing | Creates tensors each attention call | Zero-copy views |
| Reshape/permute ops | Memory copies | Contiguous layouts |
| Dynamic shapes | Prevents compile-time optimization | Static tensor sizes |
| Pure CPU | No GPU acceleration | CUDA/Metal support |

### Future Optimizations

| Optimization | Expected Impact | Effort |
|--------------|-----------------|--------|
| Zero-copy cache access | 1.5-2x | Medium |
| Fused attention kernel | 2-3x | High |
| GPU acceleration | 10-50x | High |
| INT8 quantization | 2-4x | Medium |

## Conclusion

Achieved **2x speedup** with BLAS tuning and KV-cache. Further gains require:
- Architectural changes (zero-copy operations)
- GPU support
- Quantization

For production use, recommend:
1. Set `OPENBLAS_NUM_THREADS=4`
2. Use `CachedTextGenerator`
3. Consider GPU acceleration for higher throughput
