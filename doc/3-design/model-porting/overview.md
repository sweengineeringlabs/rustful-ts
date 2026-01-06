# Model Porting

> **TLDR:** Model porting enables loading pre-trained models from external frameworks (PyTorch/HuggingFace) into RustML for inference, handling weight format conversion, name mapping, tensor transposition, and architecture detection automatically.

## Table of Contents

- [WHAT](#what)
- [WHY](#why)
- [HOW](#how)
- [Related Documentation](#related-documentation)

---

## WHAT

Model Porting is RustML's system for loading pre-trained neural network weights from external frameworks into native Rust implementations.

### Capabilities

| Capability | Description |
|------------|-------------|
| **Weight Download** | Fetch models from HuggingFace Hub via API |
| **Format Conversion** | Parse safetensors format into RustML Tensors |
| **Name Mapping** | Translate framework-specific weight names to RustML convention |
| **Tensor Transposition** | Convert PyTorch `[out, in]` layout to RustML `[in, out]` |
| **Architecture Detection** | Auto-detect model variants (e.g., CI vs CD mode) |
| **Running Stats** | Load BatchNorm statistics for proper inference |

### Supported Frameworks

| Source Framework | Format | Status |
|------------------|--------|--------|
| HuggingFace/PyTorch | safetensors | Supported |
| ONNX | onnx | Planned |
| TensorFlow | SavedModel | Planned |

### Supported Models

| Model | Architecture | HuggingFace Example |
|-------|--------------|---------------------|
| PatchTST | Time-series Transformer | `ibm-granite/granite-timeseries-patchtst` |

---

## WHY

### Problem

Implementing a neural network architecture is only half the work. Without pre-trained weights, the model is like a **car without fuel**—the engine works, but you can't go anywhere useful.

Training models from scratch requires:
- Large datasets
- Significant compute resources
- Hyperparameter tuning expertise
- Days to weeks of training time

### Solution

Model porting allows RustML to leverage the vast ecosystem of pre-trained models on HuggingFace Hub:
- **66,000+** pre-trained models available
- **Zero training** required
- **Immediate inference** capability
- **Production-ready** weights from researchers and companies

### Benefits

| Benefit | Impact |
|---------|--------|
| No training required | Save days/weeks of compute |
| Leverage existing research | Use SOTA models immediately |
| Framework independence | Run PyTorch models in pure Rust |
| Deployment flexibility | No Python runtime dependency |

---

## HOW

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Model Porting Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐           │
│  │  HubApi  │───►│ WeightMapper │───►│ ArchitectureChk │           │
│  │          │    │              │    │                 │           │
│  │ download │    │  HF → RustML │    │  detect variant │           │
│  │ safetens │    │  name mapping│    │    CI / CD      │           │
│  └──────────┘    └──────────────┘    └────────┬────────┘           │
│                                               │                     │
│                                               ▼                     │
│                                    ┌─────────────────────┐         │
│                                    │   from_state_dict   │         │
│                                    │                     │         │
│                                    │  ┌───────────────┐  │         │
│                                    │  │   transpose   │  │         │
│                                    │  │ [out,in] →    │  │         │
│                                    │  │ [in,out]      │  │         │
│                                    │  └───────────────┘  │         │
│                                    │                     │         │
│                                    │  Load into Model    │         │
│                                    └─────────────────────┘         │
│                                               │                     │
│                                               ▼                     │
│                                    ┌─────────────────────┐         │
│                                    │     Inference       │         │
│                                    │   model.forward()   │         │
│                                    └─────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Crate | Responsibility |
|-----------|-------|----------------|
| `HubApi` | `rustml-hub` | Download models from HuggingFace |
| `WeightMapper` | `rustml-hub` | Map weight names between conventions |
| `ArchitectureCheck` | `rustml-hub` | Detect model architecture variants |
| `from_state_dict` | `rustml-nn` | Load weights with tensor transposition |
| `BatchNorm1d` | `rustml-nn` | Handle running statistics for normalization |

### Usage Example

```rust
use rustml_hub::HubApi;
use rustml_timeseries::{PatchTST, PatchTSTConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Download from HuggingFace
    let api = HubApi::new();
    let bundle = api.download_model("ibm-granite/granite-timeseries-patchtst").await?;

    // 2. Load tensors
    let hf_weights = bundle.load_tensors()?;

    // 3. Create config from bundle
    let config = PatchTSTConfig {
        d_model: bundle.config.d_model,
        n_heads: bundle.config.n_heads,
        // ... other config
        ..Default::default()
    };

    // 4. Load model (auto-detects CI/CD mode, maps weights, transposes)
    let model = PatchTST::from_hub_weights(config, hf_weights)?;

    // 5. Run inference
    let input = Tensor::randn([1, 512, 7]);
    let output = model.forward(&input)?;

    Ok(())
}
```

### Step-by-Step Process

See [workflow.md](workflow.md) for detailed step-by-step breakdown.

---

## Related Documentation

- [Architecture](architecture.md) - Component design and interfaces
- [Workflow](workflow.md) - Detailed porting process
- [ADR-001: Weight Name Mapping](adr/001-weight-name-mapping.md) - Why we map names
- [ADR-002: Tensor Transposition](adr/002-tensor-transposition.md) - PyTorch vs RustML layout
- [ADR-003: Architecture Detection](adr/003-architecture-detection.md) - Auto-detecting variants
