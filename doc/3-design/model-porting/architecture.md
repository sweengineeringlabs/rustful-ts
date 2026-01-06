# Model Porting Architecture

> **TLDR:** The model porting system uses a layered pipeline architecture with clear separation between download, mapping, detection, and loading phases. Each layer is independent and testable.

## Table of Contents

- [Layer Architecture](#layer-architecture)
- [Component Interfaces](#component-interfaces)
- [Data Flow](#data-flow)
- [Extension Points](#extension-points)

---

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Consumer Layer                              │
│                                                                     │
│   PatchTST::from_hub_weights()    BERT::from_hub_weights()         │
│   (Model-specific entry points)                                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                       Orchestration Layer                           │
│                                                                     │
│   ┌─────────────────┐   ┌──────────────────┐   ┌────────────────┐  │
│   │  WeightMapper   │   │ ArchitectureCheck│   │  from_state_   │  │
│   │                 │   │                  │   │  dict methods  │  │
│   │  map_*_weights()│   │  check_*_compat()│   │                │  │
│   └─────────────────┘   └──────────────────┘   └────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                        Transport Layer                              │
│                                                                     │
│   ┌─────────────────┐   ┌──────────────────┐                       │
│   │     HubApi      │   │   ModelBundle    │                       │
│   │                 │   │                  │                       │
│   │  download_model │   │  load_tensors()  │                       │
│   │  fetch_config   │   │  config          │                       │
│   └─────────────────┘   └──────────────────┘                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                         Format Layer                                │
│                                                                     │
│   ┌─────────────────┐   ┌──────────────────┐                       │
│   │ SafetensorReader│   │   Tensor         │                       │
│   │                 │   │                  │                       │
│   │  parse bytes    │   │  transpose()     │                       │
│   │  to tensors     │   │  reshape()       │                       │
│   └─────────────────┘   └──────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Crate | Responsibility |
|-------|-------|----------------|
| **Consumer** | `rustml-timeseries`, etc. | Model-specific loading logic |
| **Orchestration** | `rustml-hub` | Weight mapping, architecture detection |
| **Transport** | `rustml-hub` | HuggingFace API, file download, caching |
| **Format** | `rustml-format`, `rustml-core` | Binary parsing, tensor operations |

---

## Component Interfaces

### HubApi

```rust
pub struct HubApi {
    client: reqwest::Client,
    cache_dir: PathBuf,
    token: Option<String>,
}

impl HubApi {
    pub fn new() -> Self;
    pub fn with_token(token: String) -> Self;
    pub fn with_cache_dir(dir: PathBuf) -> Self;

    pub async fn download_model(&self, repo_id: &str) -> Result<ModelBundle>;
    pub async fn download_model_with_progress(
        &self,
        repo_id: &str,
        progress: Option<Arc<dyn Fn(u64, u64)>>,
    ) -> Result<ModelBundle>;
}
```

### ModelBundle

```rust
pub struct ModelBundle {
    pub config: ModelConfig,
    pub weight_files: Vec<PathBuf>,
}

impl ModelBundle {
    pub fn load_tensors(&self) -> Result<HashMap<String, Tensor>>;
}
```

### WeightMapper

```rust
/// Map HuggingFace weight names to RustML convention
pub fn map_patchtst_weights(
    hf_weights: HashMap<String, Tensor>
) -> WeightMappingResult;

pub struct WeightMappingResult {
    pub mapped: HashMap<String, Tensor>,
    pub unmapped: Vec<String>,
}
```

### ArchitectureCheck

```rust
pub enum ArchitectureCheck {
    ChannelIndependent,
    ChannelDependent,
    Unknown {
        expected_ci: usize,
        expected_cd: usize,
        got: usize,
    },
}

pub fn check_patchtst_compatibility(
    weights: &HashMap<String, Tensor>,
    config: &CompatibilityConfig,
) -> ArchitectureCheck;
```

### Layer from_state_dict

```rust
impl Linear {
    /// Load from state dict with automatic transposition
    pub fn from_state_dict(
        state_dict: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<Self>;
}

impl BatchNorm1d {
    /// Load including running_mean and running_var
    pub fn from_state_dict(
        state_dict: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<Self>;
}
```

---

## Data Flow

### Weight Tensor Journey

```
HuggingFace Hub
      │
      │ HTTP GET model.safetensors
      ▼
┌─────────────────┐
│  Binary Bytes   │  safetensors format
└────────┬────────┘
         │
         │ SafetensorReader::parse()
         ▼
┌─────────────────┐
│ HashMap<String, │  HuggingFace naming:
│   Tensor>       │  "model.encoder.layers.0.ff.0.weight"
└────────┬────────┘
         │
         │ map_patchtst_weights()
         ▼
┌─────────────────┐
│ HashMap<String, │  RustML naming:
│   Tensor>       │  "encoder_layers.0.ff.linear1.weight"
└────────┬────────┘
         │
         │ Linear::from_state_dict()
         │   └─► tensor.transpose(0, 1)
         ▼
┌─────────────────┐
│  Linear {       │  PyTorch [out,in] → RustML [in,out]
│    weight,      │
│    bias,        │
│  }              │
└─────────────────┘
```

### Config Flow

```
config.json (HuggingFace)
      │
      │ serde_json::from_str()
      ▼
┌─────────────────┐
│  ModelConfig    │  Generic config struct
└────────┬────────┘
         │
         │ Field extraction
         ▼
┌─────────────────┐
│  PatchTSTConfig │  Model-specific config
│  {              │
│    d_model,     │
│    n_heads,     │
│    ...          │
│  }              │
└─────────────────┘
```

---

## Extension Points

### Adding a New Source Framework

To add support for a new model format (e.g., ONNX):

1. **Format Layer**: Add parser in `rustml-format`
   ```rust
   pub fn read_onnx(path: &Path) -> Result<HashMap<String, Tensor>>;
   ```

2. **Transport Layer**: Add download method in `HubApi`
   ```rust
   pub async fn download_onnx_model(&self, url: &str) -> Result<ModelBundle>;
   ```

### Adding a New Model Architecture

To add support for a new model (e.g., BERT):

1. **Weight Mapper**: Add mapping function
   ```rust
   pub fn map_bert_weights(hf_weights: HashMap<String, Tensor>) -> WeightMappingResult;
   ```

2. **Architecture Check**: Add compatibility checker
   ```rust
   pub fn check_bert_compatibility(weights: &HashMap<String, Tensor>) -> ArchitectureCheck;
   ```

3. **Model**: Implement `from_hub_weights`
   ```rust
   impl BERT {
       pub fn from_hub_weights(config: BERTConfig, hf_weights: HashMap<String, Tensor>) -> Result<Self>;
   }
   ```

### Interface Trait (Future)

```rust
/// Trait for models that support HuggingFace weight loading
pub trait HubPortable: Sized {
    type Config;

    fn from_hub_weights(
        config: Self::Config,
        weights: HashMap<String, Tensor>,
    ) -> Result<Self>;

    fn architecture_check(
        weights: &HashMap<String, Tensor>,
        config: &Self::Config,
    ) -> ArchitectureCheck;
}
```

---

## See Also

- [Overview](overview.md) - High-level introduction
- [Workflow](workflow.md) - Step-by-step process
- [ADR-001: Weight Name Mapping](adr/001-weight-name-mapping.md)
