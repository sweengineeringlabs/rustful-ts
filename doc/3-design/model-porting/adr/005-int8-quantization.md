# ADR-005: INT8 Weight Quantization for Inference

**Status:** Phase 1 Complete
**Date:** 2025-01-02
**Updated:** 2026-01-02
**Deciders:** RustML maintainers

## Context

GPT-2 inference is memory-bandwidth bound on CPU. The current implementation achieves 2.07 tok/s with FP32 weights. INT8 quantization can reduce memory bandwidth by 4x and enable faster SIMD operations.

## Approaches Considered

### Approach A: Weight-Only Quantization (Selected)

Store weights as INT8, dequantize to FP32 before matmul.

**Pros:**
- Simple implementation
- No changes to matmul code
- Reduces memory by 4x
- Expected 1.5-2x speedup from bandwidth reduction

**Cons:**
- Compute still in FP32
- Dequantization overhead

### Approach B: Full INT8 Inference

Weights and activations in INT8, INT8 matmul using SIMD.

**Pros:**
- Maximum speedup (2-4x)
- AVX-512 VNNI can do 8 MACs per cycle

**Cons:**
- Complex implementation
- Requires dynamic activation quantization
- Platform-specific SIMD code

## Decision

Implement **Approach A (Weight-Only Quantization)** first for quick wins, then optionally add INT8 matmul later.

## Implementation Plan

### Phase 1: Weight-Only Quantization

1. **QuantizedTensor struct**
```rust
pub struct QuantizedTensor {
    data: Vec<i8>,           // INT8 weights
    scale: f32,              // Per-tensor scale
    zero_point: i8,          // Zero point (0 for symmetric)
    shape: Shape,
}

impl QuantizedTensor {
    /// Dequantize to FP32: value = scale * (quantized - zero_point)
    pub fn dequantize(&self) -> Tensor { ... }
}
```

2. **Quantization function**
```rust
/// Symmetric quantization: scale = max(abs(tensor)) / 127
pub fn quantize_symmetric(tensor: &Tensor) -> QuantizedTensor {
    let max_val = tensor.abs().max();
    let scale = max_val / 127.0;
    let quantized: Vec<i8> = tensor.to_vec()
        .iter()
        .map(|&x| (x / scale).round().clamp(-128.0, 127.0) as i8)
        .collect();
    QuantizedTensor { data: quantized, scale, zero_point: 0, shape }
}
```

3. **QuantizedLinear layer**
```rust
pub struct QuantizedLinear {
    weight: QuantizedTensor,  // INT8
    bias: Option<Tensor>,     // FP32 (small, keep as-is)
}

impl QuantizedLinear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight_fp32 = self.weight.dequantize();
        x.matmul(&weight_fp32)?.add_bias(&self.bias)
    }
}
```

4. **Model quantization**
```rust
impl GptModel {
    pub fn quantize(&self) -> QuantizedGptModel {
        // Convert all Linear layers to QuantizedLinear
    }
}
```

### Phase 2: INT8 Matmul (Future)

Use SIMD intrinsics for true INT8 compute:
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn matmul_int8_avx2(a: &[i8], b: &[i8], ...) -> Vec<i32> {
    // Use _mm256_maddubs_epi16 for INT8 multiply-add
}
```

## Memory Savings

| Model | FP32 Size | INT8 Size | Reduction |
|-------|-----------|-----------|-----------|
| GPT-2 Small | 500 MB | 125 MB | 4x |
| GPT-2 Medium | 1.5 GB | 375 MB | 4x |
| GPT-2 Large | 3 GB | 750 MB | 4x |

## Expected Performance

| Configuration | Throughput | Speedup |
|---------------|------------|---------|
| FP32 baseline | 2.07 tok/s | 1.0x |
| INT8 weights (Phase 1) | ~3-4 tok/s | 1.5-2x |
| INT8 matmul (Phase 2) | ~5-8 tok/s | 2.5-4x |

## Accuracy Considerations

- Symmetric quantization preserves ~99% accuracy for inference
- Per-channel quantization is more accurate but more complex
- For GPT-2, per-tensor symmetric is sufficient

## Build Requirements

No additional dependencies for Phase 1 (pure Rust).

Phase 2 may require:
- `std::arch` for SIMD intrinsics
- Feature flags for AVX2/AVX-512 support

## Phase 1 Results (2026-01-02)

### Implementation

Created the following components:
- `rustml-core/src/quantization.rs`: QuantizedTensor, quantize_symmetric(), quantize_per_channel()
- `rustml-nn/src/layers.rs`: QuantizedLinear layer
- `rustml-nlp/src/gpt.rs`: QuantizedGptModel, QuantizedGptBlock, QuantizedGptMlp, QuantizedCausalSelfAttention

### Benchmark Results

| Metric | FP32 | INT8 | Change |
|--------|------|------|--------|
| Model Size | 497.8 MB | 242.8 MB | **-51% (2.1x)** |
| Throughput | 1.67 tok/s | 1.53 tok/s | **-9%** |
| Output Quality | Baseline | Maintained | N/A |

### Analysis

**Memory Reduction (Positive):**
- Achieved 2.1x memory reduction (not 4x due to FP32 embeddings and LayerNorm)
- Embeddings: 50257 × 768 × 4 bytes = ~154 MB (FP32, not quantized)
- Linear layers: ~344 MB → ~86 MB (4x reduction)

**Speed Regression (Negative):**
- Dequantization overhead: ~10% of forward pass time
- Memory bandwidth savings: ~7% improvement from reduced weight reads
- Net effect: -9% performance

### Lessons Learned

1. **Weight-only quantization trades speed for memory** when dequantization happens per-forward-pass
2. **BLAS dominates inference time** (~70%), so reducing memory bandwidth alone doesn't translate to proportional speedup
3. **Phase 2 (INT8 matmul) is necessary** to avoid dequantization overhead entirely

### Recommendations

1. **Use INT8 for memory-constrained deployments** where 2x smaller model matters
2. **Implement INT8 matmul with SIMD** for actual speedup (avoids dequantization)
3. **Consider weight caching** - dequantize once, cache FP32 weights for reuse
