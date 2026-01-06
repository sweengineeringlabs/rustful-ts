# ADR-002: Tensor Transposition for Linear Layers

**Status:** Accepted
**Date:** 2024-12-31
**Deciders:** RustML maintainers

## Context

PyTorch and RustML use different conventions for Linear layer weight matrices:

| Framework | Weight Shape | Forward Pass |
|-----------|--------------|--------------|
| PyTorch | `[out_features, in_features]` | `y = x @ W.T + b` |
| RustML | `[in_features, out_features]` | `y = x @ W + b` |

When loading PyTorch weights into RustML, the shapes are incompatible.

Example:
- PyTorch Linear(128, 512) weight: `[512, 128]`
- RustML Linear(128, 512) weight: `[128, 512]`

## Decision

**Transpose Linear weights during loading** in `Linear::from_state_dict()`.

```rust
impl Linear {
    pub fn from_state_dict(state_dict: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let weight = state_dict.get(&format!("{}weight", prefix))?;

        // Transpose from PyTorch [out, in] to RustML [in, out]
        let weight = weight.transpose(0, 1)?;

        // Bias doesn't need transposition (1D)
        let bias = state_dict.get(&format!("{}bias", prefix)).cloned();

        Self::from_weights(weight, bias)
    }
}
```

## Alternatives Considered

### 1. Transpose in Forward Pass (Rejected)

Keep PyTorch layout and transpose during every forward pass.

```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    x.matmul(&self.weight.transpose(0, 1)?)  // Transpose every time
}
```

**Rejected because:**
- Runtime overhead on every forward pass
- Transposition is O(n²) memory copy
- Inference performance suffers

### 2. Adopt PyTorch Convention (Rejected)

Change RustML to use PyTorch's weight layout.

**Rejected because:**
- Breaking change for existing RustML users
- PyTorch convention is historical (Lua Torch compatibility)
- RustML's `y = x @ W` is more intuitive mathematically

### 3. Store Both Layouts (Rejected)

Keep original and transposed weights.

**Rejected because:**
- Doubles memory usage
- Complexity in managing two copies
- No clear benefit

## Consequences

### Positive

- One-time transposition cost at load time
- Forward pass is optimal (no runtime transpose)
- RustML maintains clean mathematical convention
- Compatible with weights from any PyTorch model

### Negative

- Loading is slightly slower (transpose operation)
- Documentation must explain the convention difference
- Contributors must remember to transpose in from_state_dict

## Implementation

Location: `rustml-nn/src/layers.rs`

The transposition is documented in the method:

```rust
/// Load from a weight dictionary with prefix
///
/// Note: PyTorch stores Linear weights as [out_features, in_features],
/// but RustML uses [in_features, out_features]. This method automatically
/// transposes the weight to match RustML convention.
pub fn from_state_dict(...) -> Result<Self> {
    let weight = weight.transpose(0, 1)?;
    ...
}
```
