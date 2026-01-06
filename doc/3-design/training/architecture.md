# RustML Training Framework - SEA Architecture

## Overview

The training framework follows the **SEA (SPI-Engine-API)** layered architecture pattern, which separates concerns into three distinct layers with clear dependency directions.

## Why SEA?

SEA is used when a framework needs **extensibility through provider implementations**:

- **Multiple providers** implement the same interface (GPT2, BERT, PatchTST all implement `TrainableModule`)
- **Strategy pattern** - the framework calls into provider code (inversion of control)
- **Stable contracts** - providers and consumers can evolve independently

If we only needed consumers to call into the framework with no extensibility, a simple API layer would suffice.

## Layer Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                           train-spi                              │
│                    Service Provider Interface                    │
│                                                                  │
│  Traits: TrainableModule, DataModule, Callback, Metric          │
│  Types:  Batch, TrainStepOutput, ValStepOutput, EpochMetrics    │
│                                                                  │
│  Purpose: Extension points that providers implement              │
│  Depends on: External crates only (rustml-core, rustml-optim)   │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ extends (uses SPI types)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                           train-api                              │
│                     Application Programming Interface            │
│                                                                  │
│  Traits: TrainerApi                                              │
│  Types:  TrainerConfig, FitResult, ValMetrics, Precision        │
│                                                                  │
│  Purpose: Consumer-facing contract                               │
│  Depends on: train-spi                                           │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ implements
                              │
┌─────────────────────────────────────────────────────────────────┐
│                          train-core                              │
│                        Engine / Core                             │
│                                                                  │
│  Structs: Trainer, EarlyStopping, ModelCheckpoint, ProgressBar  │
│                                                                  │
│  Purpose: Implementation of TrainerApi                           │
│  Depends on: train-api, train-spi                                │
└─────────────────────────────────────────────────────────────────┘
```

## Dependency Direction

```
SPI ← API ← Core
```

- **SPI**: No dependencies on other SEA layers (only external crates)
- **API**: Depends on SPI, uses SPI types in trait signatures
- **Core**: Depends on API and SPI, implements API traits

## Call Directions

There are two distinct call directions in SEA:

### 1. Consumer → API (Standard API calls)

Consumers call the framework through the API layer:

```rust
// Consumer code
let config = TrainerConfig::new(10).with_lr(1e-4);
let trainer = Trainer::new(config);
let result = trainer.fit(&mut model, &mut data)?;
println!("Best loss: {}", result.best_val_loss);
```

### 2. Framework → SPI (Inversion of Control)

The framework calls into provider implementations through SPI traits:

```rust
// Inside train-core/src/trainer.rs (Framework)
impl TrainerApi for Trainer {
    fn fit<M, D>(&mut self, model: &mut M, data: &mut D) -> Result<FitResult>
    where
        M: TrainableModule,  // SPI trait bound
        D: DataModule,       // SPI trait bound
    {
        data.setup(Stage::Fit)?;                    // Framework calls SPI

        for epoch in 0..self.config.max_epochs {
            let loader = data.train_dataloader()?;  // Framework calls SPI

            for (batch_idx, batch) in loader.enumerate() {
                let output = model.training_step(&batch, batch_idx)?;  // Framework calls SPI
                // ... optimizer step ...
            }

            model.on_epoch_end(epoch, &metrics);    // Framework calls SPI
        }
    }
}
```

The framework doesn't know concrete types - it only knows SPI traits. This enables:

```
Trainer.fit()  ──►  model.training_step()  ──►  Gpt2.training_step()
   (Core)              (SPI trait)               (Provider impl)
```

## Who Implements What

| Layer | Implemented By | Example |
|-------|---------------|---------|
| **SPI traits** | Model/Data providers | `impl TrainableModule for Gpt2` |
| **API traits** | Core engine | `impl TrainerApi for Trainer` |
| **API types** | Used by consumers | `TrainerConfig::new(10)` |

## Crate Contents

### train-spi

```
src/
├── traits/
│   ├── trainable.rs   # TrainableModule - models implement this
│   ├── data.rs        # DataModule - data loaders implement this
│   ├── callback.rs    # Callback - training hooks
│   └── metric.rs      # Metric - custom metrics (accuracy, F1)
├── model/
│   ├── batch.rs       # Batch - input/target tensor pair
│   ├── output.rs      # TrainStepOutput, ValStepOutput
│   ├── metrics.rs     # EpochMetrics
│   └── config.rs      # Stage, SchedulerInterval
└── error/
    └── mod.rs         # TrainError
```

### train-api

```
src/
├── traits/
│   └── trainer.rs     # TrainerApi - consumer-facing trait
└── model/
    ├── config.rs      # TrainerConfig, Precision, Strategy
    └── result.rs      # FitResult, ValMetrics
```

### train-core

```
src/
├── trainer.rs         # Trainer implements TrainerApi
└── callbacks/
    ├── early_stopping.rs
    ├── checkpoint.rs
    └── progress.rs
```

## Example: Implementing a Provider

Model providers implement SPI traits:

```rust
use rustml_train::{TrainableModule, Batch, TrainStepOutput};

impl TrainableModule for Gpt2 {
    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        // GPT2 forward pass
    }

    fn training_step(&mut self, batch: &Batch, batch_idx: usize) -> Result<TrainStepOutput> {
        let logits = self.forward(batch.input())?;
        let loss = cross_entropy(&logits, batch.target())?;
        Ok(TrainStepOutput::new(loss))
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        // Return mutable refs to all trainable parameters
    }

    fn parameters(&self) -> Vec<&Tensor> {
        // Return refs to all parameters
    }
}
```

## Example: Consumer Usage

Consumers use the API:

```rust
use rustml_train::{Trainer, TrainerConfig, TrainerApi, EarlyStopping};

// Configure
let config = TrainerConfig::new(10)
    .with_lr(5e-5)
    .with_gradient_accumulation(4);

// Create trainer with callbacks
let mut trainer = Trainer::new(config)
    .add_callback(EarlyStopping::new(3, 0.001));

// Train (fit is from TrainerApi)
let result = trainer.fit(&mut model, &mut data)?;

// Use results
println!("Trained {} epochs", result.epochs_trained);
println!("Best val loss: {:.4}", result.best_val_loss);
```

## Facade Crate

The `rustml-train-sea` crate is a facade that re-exports all public types:

```rust
// Re-export from SPI (for model implementers)
pub use train_spi::{TrainableModule, DataModule, Callback, Metric, ...};

// Re-export from API (for consumers)
pub use train_api::{TrainerApi, TrainerConfig, FitResult, ...};

// Re-export from Core (implementation)
pub use train_core::{Trainer, EarlyStopping, ModelCheckpoint, ...};
```

Users only need one import:

```rust
use rustml_train_sea::*;
```

## Summary

| Aspect | SPI | API | Core |
|--------|-----|-----|------|
| **Purpose** | Extension points | Consumer contract | Implementation |
| **Contains** | Traits for providers | Traits + types for consumers | Concrete structs |
| **Implemented by** | Model/data providers | Core engine | - |
| **Called by** | Core engine | Consumers | - |
| **Depends on** | External only | SPI | API + SPI |
