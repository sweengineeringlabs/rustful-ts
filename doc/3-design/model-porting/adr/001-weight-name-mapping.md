# ADR-001: Weight Name Mapping Strategy

**Status:** Accepted
**Date:** 2024-12-31
**Deciders:** RustML maintainers

## Context

When loading pre-trained weights from HuggingFace, weight tensor names follow PyTorch/Transformers conventions that differ from RustML's naming conventions.

Example discrepancy:
- HuggingFace: `model.encoder.layers.0.ff.0.weight`
- RustML: `encoder_layers.0.ff.linear1.weight`

We need a strategy to bridge this gap.

## Decision

Implement explicit **per-model weight mapping functions** that translate HuggingFace names to RustML names.

```rust
pub fn map_patchtst_weights(hf_weights: HashMap<String, Tensor>) -> WeightMappingResult {
    // Explicit mapping rules for PatchTST
}

pub fn map_patchtst_key(hf_key: &str) -> Option<String> {
    // Pattern matching on key structure
}
```

## Alternatives Considered

### 1. Universal Mapping (Rejected)

Create a single mapping function that works for all models.

**Rejected because:**
- Different model families have incompatible naming conventions
- BERT, GPT, PatchTST all use different patterns
- Would require complex heuristics prone to errors

### 2. Config-based Mapping (Rejected)

Store mapping rules in external configuration files.

**Rejected because:**
- Adds deployment complexity
- Mapping errors would only surface at runtime
- Harder to version control with model changes

### 3. Adopt HuggingFace Names (Rejected)

Use HuggingFace naming convention in RustML.

**Rejected because:**
- RustML names are more idiomatic for Rust
- Would create tight coupling to HuggingFace
- Harder to support other frameworks later

## Consequences

### Positive

- Type-safe mapping with compile-time guarantees
- Clear error messages when mapping fails
- Each model's mapping is self-contained
- Easy to test mapping logic

### Negative

- Must implement mapping for each new model
- Mapping must be updated if HuggingFace changes conventions
- Some duplication across model mappers

## Implementation

Location: `rustml-hub/src/weight_mapper.rs`

```rust
fn map_patchtst_key(hf_key: &str) -> Option<String> {
    // Patch embedding
    if hf_key == "model.encoder.embedder.input_embedding.weight" {
        return Some("patch_embedding.weight".to_string());
    }

    // Encoder layers with pattern matching
    if hf_key.starts_with("model.encoder.layers.") {
        // Extract layer index and map sub-components
    }

    None // Unmapped
}
```
