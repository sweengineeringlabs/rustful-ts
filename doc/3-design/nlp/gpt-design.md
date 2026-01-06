# GPT Model Implementation Plan for RustML

## Overview

Implement a GPT-2 compatible model in RustML that can:
- Load pre-trained weights from HuggingFace
- Generate text autoregressively
- Support temperature, top-k, and top-p sampling

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GptModel                              │
├─────────────────────────────────────────────────────────────┤
│  wte: Embedding (tokens)     wpe: Embedding (positions)     │
│  blocks: Vec<GptBlock>       ln_f: LayerNorm                │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                        GptBlock                              │
├─────────────────────────────────────────────────────────────┤
│  ln_1 → CausalSelfAttention → residual                      │
│  ln_2 → GptMlp (fc→GELU→proj) → residual                    │
└─────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                   CausalSelfAttention                        │
├─────────────────────────────────────────────────────────────┤
│  1. QKV = c_attn(x)          [B, T, 3*C]                    │
│  2. Split Q, K, V            [B, T, C] each                 │
│  3. Reshape to heads         [B, H, T, C/H]                 │
│  4. scores = Q @ K.T / √d    [B, H, T, T]                   │
│  5. Causal mask (future=-∞)                                 │
│  6. attn = softmax(scores)                                  │
│  7. out = attn @ V → reshape → c_proj                       │
└─────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    TextGenerator                             │
├─────────────────────────────────────────────────────────────┤
│  Loop: logits → temperature → top_k/top_p → sample → append │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Tensor Operations
**File:** `/home/adentic/rustml/rustml-core/src/tensor.rs`

Add missing operations:
```rust
pub fn slice(&self, dim: i64, start: usize, end: usize) -> Result<Tensor>
pub fn select(&self, dim: i64, index: usize) -> Result<Tensor>
pub fn cat(tensors: &[&Tensor], dim: i64) -> Result<Tensor>
pub fn tril(size: usize) -> Tensor  // Lower triangular
pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Tensor>
pub fn argmax(&self, dim: i64) -> Result<Tensor>
```

### Phase 2: Causal Self-Attention
**File:** `/home/adentic/rustml/rustml-nn/src/attention.rs`

New `CausalSelfAttention` struct with:
- Combined QKV projection (GPT-2 style)
- Proper multi-head reshaping
- Scaling by √d_k
- Causal masking (triangular mask)

### Phase 3: GPT Model
**File:** `/home/adentic/rustml/rustml-nlp/src/gpt.rs` (new)

```rust
pub struct GptConfig {
    pub vocab_size: usize,      // 50257
    pub n_positions: usize,     // 1024
    pub n_embd: usize,          // 768 (small)
    pub n_layer: usize,         // 12 (small)
    pub n_head: usize,          // 12 (small)
}

pub struct GptMlp { c_fc, c_proj }
pub struct GptBlock { ln_1, attn, ln_2, mlp }
pub struct GptModel { config, wte, wpe, blocks, ln_f }
```

### Phase 4: Weight Mapping
**File:** `/home/adentic/rustml/rustml-hub/src/weight_mapper.rs`

Map HuggingFace GPT-2 keys:
```
transformer.wte.weight       → wte.weight
transformer.wpe.weight       → wpe.weight
transformer.h.{i}.ln_1.*     → blocks.{i}.ln_1.*
transformer.h.{i}.attn.*     → blocks.{i}.attn.*
transformer.h.{i}.mlp.*      → blocks.{i}.mlp.*
transformer.ln_f.*           → ln_f.*
```

### Phase 5: Text Generation
**File:** `/home/adentic/rustml/rustml-nlp/src/generation.rs` (new)

```rust
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

pub struct TextGenerator { model }
impl TextGenerator {
    pub fn generate(&self, input_ids: &Tensor, config: &GenerationConfig) -> Result<Tensor>
}
```

### Phase 6: Tokenizer (Minimal)
**File:** `/home/adentic/rustml/rustml-nlp/src/tokenizer/bpe.rs` (new)

Simple BPE tokenizer:
- Load vocab.json and merges.txt from GPT-2
- encode(text) → token IDs
- decode(ids) → text

## Files to Create

| File | Description |
|------|-------------|
| `rustml-nlp/src/gpt.rs` | GptConfig, GptBlock, GptMlp, GptModel |
| `rustml-nlp/src/generation.rs` | TextGenerator with sampling |
| `rustml-nlp/src/tokenizer/mod.rs` | Tokenizer module |
| `rustml-nlp/src/tokenizer/bpe.rs` | BPE tokenizer |
| `rustml-nlp/examples/gpt2_generate.rs` | Example: generate text |

## Files to Modify

| File | Changes |
|------|---------|
| `rustml-core/src/tensor.rs` | Add slice, select, cat, tril, masked_fill, argmax |
| `rustml-nn/src/attention.rs` | Add CausalSelfAttention |
| `rustml-hub/src/weight_mapper.rs` | Add map_gpt2_weights() |
| `rustml-nlp/src/lib.rs` | Export gpt, generation, tokenizer modules |

## GPT-2 Variants

| Variant | n_embd | n_layer | n_head | Params |
|---------|--------|---------|--------|--------|
| Small   | 768    | 12      | 12     | 124M   |
| Medium  | 1024   | 24      | 16     | 355M   |
| Large   | 1280   | 36      | 20     | 774M   |
| XL      | 1600   | 48      | 25     | 1.5B   |

## Key Algorithms

**Causal Mask:**
```
[0, -∞, -∞, -∞]
[0,  0, -∞, -∞]
[0,  0,  0, -∞]
[0,  0,  0,  0]
```

**Generation Loop:**
```
for _ in 0..max_tokens:
    logits = model.forward(current_ids)
    next_logits = logits[:, -1, :] / temperature
    if top_k: filter to top k
    if top_p: nucleus filter
    next_token = sample(softmax(next_logits))
    current_ids = concat(current_ids, next_token)
```

## Example Usage (Target)

```rust
use rustml_nlp::{GptModel, GptConfig, TextGenerator, GenerationConfig};
use rustml_nlp::tokenizer::BpeTokenizer;
use rustml_hub::HubApi;

// Load model
let api = HubApi::new();
let bundle = api.download_model("openai-community/gpt2").await?;
let weights = bundle.load_tensors()?;
let model = GptModel::from_hub_weights(GptConfig::gpt2_small(), weights)?;

// Load tokenizer
let tokenizer = BpeTokenizer::from_files("vocab.json", "merges.txt")?;

// Generate
let prompt = "The quick brown fox";
let input_ids = tokenizer.encode(prompt);
let input_tensor = Tensor::from_vec(...)?;

let generator = TextGenerator::new(&model);
let output = generator.generate(&input_tensor, &GenerationConfig {
    max_new_tokens: 50,
    temperature: 0.8,
    top_k: Some(50),
    ..Default::default()
})?;

println!("{}", tokenizer.decode(&output_ids));
```

## Dependencies

**Cargo.toml additions:**
```toml
rand = "0.8"           # For sampling
serde_json = "1.0"     # For tokenizer vocab loading
```

---

## Detailed Design

### CausalSelfAttention Implementation

```rust
pub struct CausalSelfAttention {
    c_attn: Linear,      // Combined Q, K, V projection [n_embd, 3*n_embd]
    c_proj: Linear,      // Output projection [n_embd, n_embd]
    n_head: usize,
    head_dim: usize,
    n_embd: usize,
    attn_dropout: Dropout,
    resid_dropout: Dropout,
}

impl CausalSelfAttention {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // 1. Combined QKV projection
        let qkv = self.c_attn.forward(x)?;  // [B, T, 3*C]

        // 2. Split into Q, K, V
        let (q, k, v) = split_qkv(&qkv, self.n_embd)?;

        // 3. Reshape: [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        let q = q.view([batch, seq_len, self.n_head, self.head_dim])?
                 .permute([0, 2, 1, 3])?;
        let k = k.view([batch, seq_len, self.n_head, self.head_dim])?
                 .permute([0, 2, 1, 3])?;
        let v = v.view([batch, seq_len, self.n_head, self.head_dim])?
                 .permute([0, 2, 1, 3])?;

        // 4. Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)?)?
                      .mul_scalar(1.0 / scale);

        // 5. Apply causal mask
        let mask = Tensor::tril(seq_len);
        let scores = scores.masked_fill(&mask.eq(0)?, f32::NEG_INFINITY)?;

        // 6. Softmax + dropout
        let attn = scores.softmax(-1)?;
        let attn = self.attn_dropout.forward(&attn);

        // 7. Weighted sum of values
        let out = attn.matmul(&v)?;  // [B, H, T, D]

        // 8. Reshape back: [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        let out = out.permute([0, 2, 1, 3])?
                     .reshape([batch, seq_len, self.n_embd])?;

        // 9. Output projection + dropout
        let out = self.c_proj.forward(&out)?;
        Ok(self.resid_dropout.forward(&out))
    }
}
```

### GptModel Full Implementation

```rust
pub struct GptModel {
    pub config: GptConfig,
    wte: Embedding,           // [vocab_size, n_embd]
    wpe: Embedding,           // [n_positions, n_embd]
    drop: Dropout,
    blocks: Vec<GptBlock>,
    ln_f: LayerNorm,
}

impl GptModel {
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;

        // Position indices
        let positions = Tensor::arange(0, seq_len as i64)?;

        // Embeddings
        let tok_emb = self.wte.forward(input_ids)?;  // [B, T, C]
        let pos_emb = self.wpe.forward(&positions)?; // [T, C]

        let mut x = tok_emb.add(&pos_emb)?;
        x = self.drop.forward(&x);

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Final layer norm
        x = self.ln_f.forward(&x)?;

        // LM head (weight tying with wte)
        let logits = x.matmul(&self.wte.weight().transpose(0, 1)?)?;

        Ok(logits)  // [B, T, vocab_size]
    }
}
```

### Sampling Algorithms

**Top-K Sampling:**
```rust
fn top_k_filter(logits: &Tensor, k: usize) -> Result<Tensor> {
    let (sorted_logits, sorted_indices) = logits.sort_descending(-1)?;
    let threshold = sorted_logits.select(-1, k - 1)?;
    let mask = logits.lt(&threshold)?;
    logits.masked_fill(&mask, f32::NEG_INFINITY)
}
```

**Top-P (Nucleus) Sampling:**
```rust
fn top_p_filter(logits: &Tensor, p: f32) -> Result<Tensor> {
    let probs = logits.softmax(-1)?;
    let (sorted_probs, sorted_indices) = probs.sort_descending(-1)?;
    let cumsum = sorted_probs.cumsum(-1)?;

    // Find cutoff where cumsum > p
    let cutoff_mask = cumsum.gt(p)?;
    // Shift mask right by 1 (keep first token above threshold)
    let cutoff_mask = cutoff_mask.roll(1, -1)?;

    // Apply mask in original order
    let mask = cutoff_mask.scatter(-1, &sorted_indices)?;
    logits.masked_fill(&mask, f32::NEG_INFINITY)
}
```

**Multinomial Sampling:**
```rust
fn sample_multinomial(probs: &Tensor) -> Result<usize> {
    let probs_vec = probs.to_vec();
    let mut rng = rand::thread_rng();
    let sample: f32 = rng.gen();

    let mut cumsum = 0.0;
    for (i, &p) in probs_vec.iter().enumerate() {
        cumsum += p;
        if cumsum >= sample {
            return Ok(i);
        }
    }
    Ok(probs_vec.len() - 1)
}
```

### BPE Tokenizer

```rust
pub struct BpeTokenizer {
    encoder: HashMap<String, usize>,    // token -> id
    decoder: HashMap<usize, String>,    // id -> token
    bpe_ranks: HashMap<(String, String), usize>,
}

impl BpeTokenizer {
    pub fn from_files(vocab_path: &str, merges_path: &str) -> Result<Self>

    pub fn encode(&self, text: &str) -> Vec<usize> {
        // 1. Pre-tokenize (split on whitespace, handle special chars)
        // 2. Apply BPE merges iteratively
        // 3. Look up token IDs
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        // 1. Look up tokens
        // 2. Join and clean up (Ġ -> space)
    }
}
```

### Weight Mapping (HuggingFace → RustML)

| HuggingFace Key | RustML Key | Notes |
|-----------------|------------|-------|
| `transformer.wte.weight` | `wte.weight` | Token embeddings |
| `transformer.wpe.weight` | `wpe.weight` | Position embeddings |
| `transformer.h.{i}.ln_1.weight` | `blocks.{i}.ln_1.weight` | Pre-attention norm |
| `transformer.h.{i}.ln_1.bias` | `blocks.{i}.ln_1.bias` | |
| `transformer.h.{i}.attn.c_attn.weight` | `blocks.{i}.attn.c_attn.weight` | Transpose! |
| `transformer.h.{i}.attn.c_attn.bias` | `blocks.{i}.attn.c_attn.bias` | |
| `transformer.h.{i}.attn.c_proj.weight` | `blocks.{i}.attn.c_proj.weight` | Transpose! |
| `transformer.h.{i}.attn.c_proj.bias` | `blocks.{i}.attn.c_proj.bias` | |
| `transformer.h.{i}.ln_2.weight` | `blocks.{i}.ln_2.weight` | Pre-MLP norm |
| `transformer.h.{i}.ln_2.bias` | `blocks.{i}.ln_2.bias` | |
| `transformer.h.{i}.mlp.c_fc.weight` | `blocks.{i}.mlp.c_fc.weight` | Transpose! |
| `transformer.h.{i}.mlp.c_fc.bias` | `blocks.{i}.mlp.c_fc.bias` | |
| `transformer.h.{i}.mlp.c_proj.weight` | `blocks.{i}.mlp.c_proj.weight` | Transpose! |
| `transformer.h.{i}.mlp.c_proj.bias` | `blocks.{i}.mlp.c_proj.bias` | |
| `transformer.ln_f.weight` | `ln_f.weight` | Final layer norm |
| `transformer.ln_f.bias` | `ln_f.bias` | |
| `lm_head.weight` | (skip) | Tied to wte |

**Note:** GPT-2 uses Conv1D which stores weights as [out, in], need transpose to [in, out].

---

## Save Location

After implementation, copy this plan to:
`/home/adentic/rustml/rustml-nlp/doc/gpt-design.md`
