# ADR-003: Automatic Architecture Variant Detection

**Status:** Accepted
**Date:** 2024-12-31
**Deciders:** RustML maintainers

## Context

Some model architectures have multiple variants that share the same name but have different internal structures. For PatchTST:

| Variant | Description | Weight Shape |
|---------|-------------|--------------|
| Channel-Independent (CI) | Each channel processed separately | `[d_model, patch_len]` |
| Channel-Dependent (CD) | All channels combined | `[d_model, patch_len × enc_in]` |

HuggingFace models typically use CI mode, but the variant isn't explicitly stated in config.json. Users shouldn't need to know which variant a model uses.

## Decision

**Automatically detect architecture variant by inspecting weight tensor shapes.**

```rust
pub enum ArchitectureCheck {
    ChannelIndependent,
    ChannelDependent,
    Unknown { expected_ci: usize, expected_cd: usize, got: usize },
}

pub fn check_patchtst_compatibility(
    weights: &HashMap<String, Tensor>,
    config: &CompatibilityConfig,
) -> ArchitectureCheck {
    let embedding_weight = weights.get("patch_embedding.weight")?;
    let in_features = embedding_weight.shape()[1];

    let expected_ci = config.patch_len;                    // e.g., 12
    let expected_cd = config.patch_len * config.enc_in;    // e.g., 84

    if in_features == expected_ci {
        ArchitectureCheck::ChannelIndependent
    } else if in_features == expected_cd {
        ArchitectureCheck::ChannelDependent
    } else {
        ArchitectureCheck::Unknown { expected_ci, expected_cd, got: in_features }
    }
}
```

## Alternatives Considered

### 1. Require User to Specify (Rejected)

Add `channel_independent: bool` as required config parameter.

```rust
let config = PatchTSTConfig {
    channel_independent: true,  // User must know this
    ...
};
```

**Rejected because:**
- Users shouldn't need to know internal architecture details
- Error-prone if user guesses wrong
- Most users just want "load this model and run it"

### 2. Parse from config.json (Rejected)

Look for a field in HuggingFace config indicating variant.

**Rejected because:**
- HuggingFace doesn't standardize this field
- Would fail silently if field missing
- Different model families use different conventions

### 3. Try Both and See Which Works (Rejected)

Attempt to load with CI mode, if shapes don't match, try CD mode.

**Rejected because:**
- Unclear error messages on failure
- Wasteful computation
- Could mask other errors

## Consequences

### Positive

- Zero configuration required from users
- Works with any HuggingFace model automatically
- Clear error message if variant is unknown
- Detection logic is testable and deterministic

### Negative

- Must implement detection logic per model architecture
- Relies on specific weight being present (patch_embedding)
- New variants require code changes

## Implementation

Location: `rustml-hub/src/weight_mapper.rs`

The detection is used in `from_hub_weights`:

```rust
impl PatchTST {
    pub fn from_hub_weights(mut config: PatchTSTConfig, hf_weights: HashMap<String, Tensor>) -> Result<Self> {
        let check = check_patchtst_compatibility(&hf_weights, &compat_config);

        match &check {
            ArchitectureCheck::ChannelIndependent => {
                config.channel_independent = true;
                eprintln!("Detected Channel-Independent mode from HuggingFace weights");
            }
            ArchitectureCheck::ChannelDependent => {
                config.channel_independent = false;
            }
            ArchitectureCheck::Unknown { expected_ci, expected_cd, got } => {
                return Err(Error::Other(format!(
                    "Unknown architecture: expected CI={} or CD={}, got {}",
                    expected_ci, expected_cd, got
                )));
            }
        }

        // Proceed with loading...
    }
}
```

## Future Work

- Add detection for other model architectures (BERT variants, GPT sizes)
- Consider trait-based detection for extensibility
- Add logging/telemetry for variant statistics
