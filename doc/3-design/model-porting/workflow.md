# Model Porting Workflow

> **TLDR:** The model porting workflow consists of 6 steps: Download → Parse → Map → Detect → Transpose → Load. Each step transforms the data closer to RustML's internal format.

## Table of Contents

- [Workflow Steps](#workflow-steps)
- [Step Details](#step-details)
- [Transformations Reference](#transformations-reference)
- [Troubleshooting](#troubleshooting)

---

## Workflow Steps

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   Step 1          Step 2          Step 3          Step 4           │
│  ┌───────┐       ┌───────┐       ┌───────┐       ┌───────┐         │
│  │DOWNLOAD│──────►│ PARSE │──────►│  MAP  │──────►│DETECT │         │
│  │       │       │       │       │       │       │       │         │
│  │.safe- │       │HashMap│       │RustML │       │CI/CD  │         │
│  │tensors│       │<HF>   │       │names  │       │mode   │         │
│  └───────┘       └───────┘       └───────┘       └───────┘         │
│                                                       │             │
│                                                       ▼             │
│                                  Step 6          Step 5            │
│                                 ┌───────┐       ┌───────┐          │
│                                 │ LOAD  │◄──────│TRANSP.│          │
│                                 │       │       │       │          │
│                                 │Model  │       │[in,out│          │
│                                 │struct │       │]layout│          │
│                                 └───────┘       └───────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step Details

### Step 1: Download

**Input:** Repository ID (e.g., `ibm-granite/granite-timeseries-patchtst`)
**Output:** Local file paths to `config.json` and `model.safetensors`

```rust
let api = HubApi::new();
let bundle = api.download_model("ibm-granite/granite-timeseries-patchtst").await?;
// bundle.config -> ModelConfig
// bundle.weight_files -> Vec<PathBuf>
```

**What happens:**
1. Resolve repository URL on HuggingFace Hub
2. Check local cache for existing files
3. Download `config.json` for model configuration
4. Download `model.safetensors` (or sharded files)
5. Store in cache directory (`~/.cache/rustml/`)

### Step 2: Parse

**Input:** Binary safetensors file
**Output:** `HashMap<String, Tensor>` with HuggingFace naming

```rust
let hf_weights = bundle.load_tensors()?;
// Keys: "model.encoder.layers.0.ff.0.weight", etc.
```

**What happens:**
1. Read safetensors header (JSON metadata)
2. Parse tensor metadata (dtype, shape, offset)
3. Memory-map tensor data
4. Convert to RustML `Tensor` objects

**Example keys:**
```
model.encoder.embedder.input_embedding.weight    [128, 12]
model.encoder.layers.0.self_attn.q_proj.weight   [128, 128]
model.encoder.layers.0.ff.0.weight               [512, 128]
model.encoder.layers.0.norm_sublayer1.batchnorm.weight      [128]
model.encoder.layers.0.norm_sublayer1.batchnorm.running_mean [128]
head.projection.weight                           [96, 128]
```

### Step 3: Map

**Input:** `HashMap<String, Tensor>` with HuggingFace naming
**Output:** `HashMap<String, Tensor>` with RustML naming

```rust
let mapping = map_patchtst_weights(hf_weights);
// mapping.mapped -> HashMap with RustML keys
// mapping.unmapped -> Vec of keys that couldn't be mapped
```

**What happens:**
1. Parse each HuggingFace key
2. Apply naming transformation rules
3. Collect mapped and unmapped keys

**Name transformations:**

| HuggingFace | RustML |
|-------------|--------|
| `model.encoder.embedder.input_embedding.weight` | `patch_embedding.weight` |
| `model.encoder.layers.0.self_attn.q_proj.weight` | `encoder_layers.0.self_attn.q_proj.weight` |
| `model.encoder.layers.0.ff.0.weight` | `encoder_layers.0.ff.linear1.weight` |
| `model.encoder.layers.0.ff.3.weight` | `encoder_layers.0.ff.linear2.weight` |
| `model.encoder.layers.0.norm_sublayer1.batchnorm.weight` | `encoder_layers.0.norm1.weight` |
| `model.encoder.layers.0.norm_sublayer1.batchnorm.running_mean` | `encoder_layers.0.norm1.running_mean` |
| `head.projection.weight` | `head.weight` |

### Step 4: Detect

**Input:** Mapped weights + model configuration
**Output:** `ArchitectureCheck` enum indicating variant

```rust
let check = check_patchtst_compatibility(&mapped_weights, &config);
match check {
    ArchitectureCheck::ChannelIndependent => { /* HuggingFace default */ }
    ArchitectureCheck::ChannelDependent => { /* Alternative mode */ }
    ArchitectureCheck::Unknown { .. } => { /* Unsupported variant */ }
}
```

**What happens:**
1. Extract `patch_embedding.weight` shape
2. Compare input dimension against expected values:
   - CI mode: `in_features == patch_len`
   - CD mode: `in_features == patch_len * enc_in`
3. Return detected architecture

**Detection logic:**
```
patch_embedding.weight shape = [d_model, in_features]

if in_features == patch_len (12):
    → Channel-Independent (CI)
    → Each channel processed separately

if in_features == patch_len * enc_in (12 * 7 = 84):
    → Channel-Dependent (CD)
    → All channels combined
```

### Step 5: Transpose

**Input:** Tensors in PyTorch layout `[out_features, in_features]`
**Output:** Tensors in RustML layout `[in_features, out_features]`

```rust
// Inside Linear::from_state_dict()
let weight = weight.transpose(0, 1)?;
```

**What happens:**
1. For each Linear layer weight
2. Swap first two dimensions
3. Result matches RustML's `x @ weight` convention

**Why needed:**
```
PyTorch:  y = x @ W.T    where W is [out, in]
RustML:   y = x @ W      where W is [in, out]
```

### Step 6: Load

**Input:** Mapped, transposed weights
**Output:** Fully initialized model struct

```rust
let model = PatchTST::from_pretrained(config, mapped_weights)?;
```

**What happens:**
1. Create model struct
2. For each component:
   - Call `from_state_dict()` with appropriate prefix
   - Load weight tensors
   - Initialize layer
3. Return complete model ready for inference

**Components loaded:**

| Component | Prefix | Parameters |
|-----------|--------|------------|
| `patch_embedding` | `patch_embedding.` | weight, bias |
| `positional_encoding` | - | encoding tensor |
| `encoder_layers[i].self_attn` | `encoder_layers.{i}.self_attn.` | q,k,v,out projections |
| `encoder_layers[i].ff` | `encoder_layers.{i}.ff.` | linear1, linear2 |
| `encoder_layers[i].norm1` | `encoder_layers.{i}.norm1.` | weight, bias, running_mean, running_var |
| `encoder_layers[i].norm2` | `encoder_layers.{i}.norm2.` | weight, bias, running_mean, running_var |
| `head` | `head.` | weight, bias |

---

## Transformations Reference

### Tensor Shape Changes

| Layer Type | PyTorch Shape | RustML Shape | Transformation |
|------------|---------------|--------------|----------------|
| Linear weight | `[out, in]` | `[in, out]` | `transpose(0, 1)` |
| Linear bias | `[out]` | `[out]` | None |
| BatchNorm weight | `[features]` | `[features]` | None |
| BatchNorm running_mean | `[features]` | `[features]` | None |
| Positional encoding | `[seq, dim]` | `[1, seq, dim]` | `reshape` |

### Name Pattern Rules

```
HuggingFace Pattern                      →  RustML Pattern
─────────────────────────────────────────────────────────────
model.encoder.embedder.input_embedding   →  patch_embedding
model.encoder.positional_encoder.*       →  positional_encoding
model.encoder.layers.{N}.self_attn.*     →  encoder_layers.{N}.self_attn.*
model.encoder.layers.{N}.ff.0.*          →  encoder_layers.{N}.ff.linear1.*
model.encoder.layers.{N}.ff.3.*          →  encoder_layers.{N}.ff.linear2.*
model.encoder.layers.{N}.norm_sublayer1.batchnorm.*  →  encoder_layers.{N}.norm1.*
model.encoder.layers.{N}.norm_sublayer3.batchnorm.*  →  encoder_layers.{N}.norm2.*
head.projection.*                        →  head.*
```

---

## Troubleshooting

### Common Errors

#### `ShapeMismatch { expected: [X, Y], got: [Y, X] }`

**Cause:** Weight not transposed.
**Solution:** Ensure `from_state_dict` calls `transpose(0, 1)` for Linear weights.

#### `Missing weight: {key}`

**Cause:** Weight name mapping failed.
**Solution:** Check `map_patchtst_key()` covers this key pattern.

#### `ArchitectureCheck::Unknown`

**Cause:** Model variant not recognized.
**Solution:** Inspect `patch_embedding.weight` shape and add new variant detection.

#### `Missing running_mean: {key}`

**Cause:** BatchNorm running stats not mapped.
**Solution:** Ensure weight mapper includes `running_mean` and `running_var` patterns.

---

## See Also

- [Overview](overview.md) - High-level introduction
- [Architecture](architecture.md) - Component design
- [ADR-002: Tensor Transposition](adr/002-tensor-transposition.md) - Why we transpose
