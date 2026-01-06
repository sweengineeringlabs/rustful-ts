# Model Porting Backlog

> **TLDR:** Prioritized list of work items for the model porting pipeline.
>
> *Migrated from rustml-hub*

## Current State

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Porting Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HubApi ──► WeightMapper ──► ArchitectureCheck ──► Model    │
│    │            │                   │                 │     │
│ download    name mapping      detect variant    from_state  │
│ safetensors  HF → RustML        CI / CD           _dict     │
│                                                     │       │
│                                          ┌──────────┴───┐   │
│                                          │  transpose   │   │
│                                          │  [out,in] →  │   │
│                                          │  [in,out]    │   │
│                                          └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Backlog

### Priority 1: Validation

| ID | Item | Status | Description |
|----|------|--------|-------------|
| MP-001 | Test with real data | Done | Validated pre-trained model produces bounded, finite, deterministic outputs |

### Priority 2: Completeness

| ID | Item | Status | Description |
|----|------|--------|-------------|
| MP-002 | Implement LayerNorm properly | Done | Per-sample normalization with 5 passing tests |

### Priority 3: Extensibility

| ID | Item | Status | Description |
|----|------|--------|-------------|
| MP-003 | Add another model | Done | BERT implemented in rustml-nlp crate with weight mapper |

### Priority 3.5: Additional Timeseries Models

| ID | Item | Status | Description |
|----|------|--------|-------------|
| TS-001 | DLinear | Done | Simple linear baseline - decomposition + linear layers. Fast, interpretable. |
| TS-002 | Autoformer | Open | Auto-Correlation mechanism for long-range dependencies. ICLR 2022. |
| TS-003 | FEDformer | Open | Frequency Enhanced Decomposition Transformer. ICML 2022. |
| TS-004 | TimesNet | Open | 2D variation modeling via FFT for multi-periodic patterns. ICLR 2023. |
| TS-005 | iTransformer | Open | Inverted Transformer - attention on variate dimension. ICLR 2024. |

**References:**
- DLinear: "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2023)
- Autoformer: "Autoformer: Decomposition Transformers with Auto-Correlation" (Wu et al., 2022)
- FEDformer: "FEDformer: Frequency Enhanced Decomposed Transformer" (Zhou et al., 2022)
- TimesNet: "TimesNet: Temporal 2D-Variation Modeling" (Wu et al., 2023)
- iTransformer: "iTransformer: Inverted Transformers Are Effective for Time Series" (Liu et al., 2024)

### Priority 4: Capability

| ID | Item | Status | Description |
|----|------|--------|-------------|
| MP-004 | Training support | Done | Phase 1 MVP complete: GradientTape, loss functions, SGD + AdamW optimizers |

### Priority 5: Training SPI Layer

| ID | Item | Status | Description |
|----|------|--------|-------------|
| SPI-001 | Loss SPI | Done | Trait + MseLoss, CrossEntropyLoss, HuberLoss, BceLoss implementations |
| SPI-002 | Initializer SPI | Done | Weight initialization strategies: Xavier, Kaiming/He, Normal, Uniform, Orthogonal |
| SPI-003 | Optimizer re-exports | Done | Re-export Optimizer/Scheduler traits from rustml-optim |
| SPI-004 | Hyperparameter Tuning SPI | Done | Tuner trait + GridSearch, RandomSearch, BayesianOptimization implementations |

**Initializer SPI Design (SPI-002):**
```
train-spi/src/init/
├── mod.rs           # Initializer trait + re-exports
├── xavier.rs        # Xavier/Glorot: sqrt(2/(fan_in+fan_out))
├── kaiming.rs       # Kaiming/He: sqrt(2/fan_in) - better for ReLU
├── normal.rs        # Normal(mean, std)
├── uniform.rs       # Uniform(low, high)
└── orthogonal.rs    # Orthogonal - good for RNNs
```

**Trait:**
```rust
pub trait Initializer: Send + Sync {
    fn initialize(&self, shape: &[usize]) -> Tensor;
    fn name(&self) -> &str;
}
```

**Rationale:** Xavier initialization bug caused PatchTST outputs to explode (MAE $677,098 → $39.50 after fix). Different architectures need different initialization:
- ReLU networks: Kaiming/He preserves variance
- Transformers: Xavier with small gain
- RNNs/LSTMs: Orthogonal prevents vanishing gradients

**Hyperparameter Tuning SPI Design (SPI-004):**
```
train-spi/src/tune/
├── mod.rs           # Tuner trait + Direction + TuneResult + re-exports
├── space.rs         # SearchSpace, Parameter, ParameterValue
├── trial.rs         # TrialConfig, TrialResult, TrialStatus
├── grid.rs          # GridSearch - exhaustive grid evaluation
├── random.rs        # RandomSearch - random sampling
└── bayesian.rs      # BayesianOptimization - TPE-style adaptive search
```

**Tuner Trait:**
```rust
pub trait Tuner: Send + Sync {
    fn suggest(&mut self, space: &SearchSpace) -> Option<TrialConfig>;
    fn report(&mut self, result: TrialResult);
    fn is_complete(&self) -> bool;
    fn best_trial(&self) -> Option<&TrialResult>;
    fn trials(&self) -> &[TrialResult];
    fn name(&self) -> &str;
    fn direction(&self) -> Direction;
    fn objective(&self) -> &str;
}
```

**Search Strategies:**
- **GridSearch**: Exhaustive evaluation of parameter grid with configurable resolution
- **RandomSearch**: Uniform random sampling (efficient for high-dimensional spaces)
- **BayesianOptimization**: TPE-style adaptive search using kernel density estimation

---

### Priority 6: CPU Performance Optimization

| ID | Item | Status | Expected Gain | Actual | Description |
|----|------|--------|---------------|--------|-------------|
| MP-005 | INT8 quantization | Done | 2-4x | **2.1x mem** | Weight-only quantization: 2.1x memory reduction. Phase 2 adds AVX2/VNNI/GPU backends. |
| MP-006 | Zero-copy tensor views | Blocked | 1.5-2x | **Negative** | Requires architectural changes; ndarray views slower than copies for BLAS |
| MP-007 | Fused attention kernel | Done | 2-3x | **~0%** | Implemented but not a bottleneck for short sequences |
| MP-008 | SIMD vectorization | Tested | 1.2-1.5x | **~0%** | Fast exp/tanh approximations - NOT a bottleneck |
| MP-009 | Revert fast approximations | Done | - | - | Reverted fast_exp/fast_tanh - no benefit, adds complexity |
| MP-014 | AVX-512 VNNI support | Done | 2-4x | **TBD** | Native INT8 dot products via dpbusd. Requires VNNI-capable CPU. |
| MP-015 | GPU acceleration (wgpu) | Done | 5-10x | **TBD** | Tiled matmul shader. Requires GPU with Vulkan/Metal/DX12. |

**Current Baseline:** 2.07 tok/s (GPT-2 small, CPU with BLAS + KV-cache)
**Target:** 5-8 tok/s (interactive generation threshold)

**INT8 Quantization Results (MP-005):**
- Memory reduction: 2.1x (497.8 MB → 242.8 MB)
- Speed impact: -19% with dequantization, -500% with naive SIMD INT8 matmul
- Output quality: Maintained (greedy sampling produces coherent text)

**Phase 2 SIMD Results (with pre-transposed weights):**
| Configuration | Tok/s | vs FP32 |
|---------------|-------|---------|
| FP32 Baseline | 1.75 | 1.00x |
| INT8 dequantize | 1.64 | 1.07x slower |
| INT8 SIMD matmul | 1.33 | 1.32x slower |
| INT8 fused attn | 1.62 | 1.08x slower |

**Optimization History:**
- Initial SIMD matmul: 6.03x slower (strided memory access)
- With per-forward transpose: 1.68x slower
- With pre-computed transpose: 1.32x slower (4.5x improvement from initial)

**Root Cause Analysis:**
- Naive SIMD cannot beat BLAS due to lack of blocking/tiling optimizations
- BLAS uses assembly-level register blocking, prefetching, cache-aware tiling
- Fused attention has minimal impact - attention is not the bottleneck for short sequences

**Conclusion:**
- INT8 useful for memory reduction (2.1x), not speed on CPU with naive SIMD
- Pre-transposed weights reduce SIMD overhead significantly (6.03x → 1.32x slower)
- For CPU speedup beyond BLAS, need: AVX-512 VNNI, blocked SIMD, or GPU acceleration
- Current BLAS + KV-cache approach is near-optimal for single-threaded CPU inference

**Compute Backend Architecture (MP-014, MP-015):**
```
ComputeBackend::Cpu      → BLAS FP32 matmul (default)
ComputeBackend::CpuAvx2  → AVX2 SIMD with INT8 weights
ComputeBackend::CpuVnni  → AVX-512 VNNI native INT8×INT8
ComputeBackend::Gpu      → wgpu tiled matmul shader
```

Features:
- Runtime backend selection via `set_compute_backend()`
- Auto-detection with `auto_detect_backend()` and `enable_best_cpu()`
- GPU requires `--features gpu` compile flag

**Bottleneck Analysis (Updated 2025-01-02):**
- ~~Element-wise ops (softmax, GELU) not vectorized~~ - Tested, NOT a bottleneck
- ~~Tensor slicing allocations in attention~~ - View-based approach slower due to non-contiguous memory
- BLAS matmul dominates (~70% of time) - requires contiguous memory for optimal performance
- **Memory copies ARE necessary** for BLAS efficiency with non-contiguous layouts

**Key Finding:** ndarray views on non-contiguous data (after transpose/permute) result in slower BLAS operations than copying to contiguous buffers first. The current copy-based approach is actually optimal for CPU BLAS.

---

## Architecture Gaps

Items identified during implementation that need addressing:

| ID | Gap | Status | Description |
|----|-----|--------|-------------|
| MP-010 | HubPortable trait | Open | No trait/interface for portable models. Each model implements its own `from_hub_weights` |
| MP-011 | Model registry | Open | No registry of supported models. Users must know which models are implemented |
| MP-012 | Provider abstraction | Open | No SPI for adding new source frameworks (ONNX, TensorFlow) |
| MP-013 | Weight mapping registry | Open | Each model has its own mapper. No unified mapping system |

---

## Completed

| ID | Item | Completed | Description |
|----|------|-----------|-------------|
| MP-001 | Validate pre-trained model | 2024-12-31 | Bounded, finite, deterministic outputs confirmed |
| MP-002 | LayerNorm implementation | 2025-12-31 | Per-sample normalization with 5 passing tests |
| MP-003 | BERT model porting | 2025-12-31 | rustml-nlp crate with BertModel, weight mapper, GELU, Embedding lookup |
| MP-004 | Training support Phase 1 | 2025-12-31 | GradientTape, loss functions, SGD + AdamW optimizers |
| MP-100 | CI mode for PatchTST | 2024-12-31 | Channel-Independent mode implementation |
| MP-101 | Weight transposition | 2024-12-31 | PyTorch [out,in] → RustML [in,out] |
| MP-102 | BatchNorm1d | 2024-12-31 | Proper implementation with running stats |
| MP-103 | Architecture detection | 2024-12-31 | Auto-detect CI vs CD mode |
| MP-104 | Model Porting docs | 2024-12-31 | WHAT/WHY/HOW documentation with ADRs |

---

## Notes

### On HubPortable Trait (MP-010)

Proposed interface:

```rust
pub trait HubPortable: Sized {
    type Config;

    /// Load from HuggingFace weights with auto-detection
    fn from_hub_weights(
        config: Self::Config,
        weights: HashMap<String, Tensor>,
    ) -> Result<Self>;

    /// Check architecture compatibility
    fn architecture_check(
        weights: &HashMap<String, Tensor>,
        config: &Self::Config,
    ) -> ArchitectureCheck;

    /// Return weight mapping function
    fn weight_mapper() -> fn(HashMap<String, Tensor>) -> WeightMappingResult;
}
```

### On Model Registry (MP-011)

Proposed structure:

```rust
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

pub struct ModelInfo {
    pub name: &'static str,
    pub hf_repos: Vec<&'static str>,  // Known compatible repos
    pub architecture: &'static str,
    pub variants: Vec<&'static str>,
}

// Usage
let registry = ModelRegistry::default();
let info = registry.get("patchtst")?;
println!("Compatible repos: {:?}", info.hf_repos);
```

### On BERT Model Porting (MP-003)

**Selected Model:** `google-bert/bert-base-uncased` (440 MB, 12 layers, 768 hidden, 110M params)

**Key Differences from PatchTST:**
| Aspect | PatchTST | BERT |
|--------|----------|------|
| Normalization | BatchNorm | LayerNorm |
| FFN Activation | ReLU | GELU |
| Embeddings | Linear projection | Token lookup |
| Architecture Variants | CI/CD modes | Single architecture |

**Required New Components:**
- New crate: `rustml-nlp` with `bert.rs`, `bert_encoder.rs`
- New activation: GELU in `rustml-nn/src/activation.rs`
- Embedding lookup implementation (currently placeholder)
- Weight mapper: `map_bert_weights()`, `check_bert_compatibility()`

**Implementation Phases:**
1. Layer enhancements (Embedding lookup, GELU) - 2-3 days
2. Weight mapping functions - 1-2 days
3. BERT model structure - 3-4 days
4. Testing & validation - 2-3 days

### On Training Support (MP-004)

**Current State:** Inference-only. Optimizers exist but are placeholders (`todo!()`).

**Phase 1: MVP Training (~10-12 days)**
- Computation graph for operation recording
- Backward functions: matmul, add, mul, relu, softmax
- Loss functions: MSE, CrossEntropy
- `Tensor.backward()` method
- `Optimizer.step()` implementation (AdamW, SGD)

**Phase 2: Full Training (~15-18 days)**
- Module trait with `parameters()`, `train()` methods
- Gradient clipping (`clip_grad_norm`)
- DataLoader Iterator implementation
- Additional loss functions (L1, Huber, BCE)
- Checkpointing (save/load state)

**Dependency Chain:**
```
Computation Graph → Backward Functions → Loss Functions
                          ↓
                   Tensor.backward()
                          ↓
Module Trait → Optimizer.step() → Training Loop
```
