# Build Sizes

Release build artifact sizes for rustful-ts.

## Binaries

| Artifact | Size | Description |
|----------|------|-------------|
| rustful | 1.2M | CLI tool |
| rustful-server | 1.4M | HTTP API server |
| librustful_wasm.so | 420K | WASM library (shared object) |

## Domain Crates

### SPI Layer (Traits Only)

Minimal dependencies, suitable for implementing custom algorithms.

| Crate | Size |
|-------|------|
| predictor-spi | 8K |
| detector-spi | 12K |
| pipeline-spi | 8K |
| signal-spi | 36K |
| **Total** | **64K** |

### Core Layer (Types & Utilities)

| Crate | Size |
|-------|------|
| predictor-core | 96K |
| detector-core | 104K |
| pipeline-core | 28K |
| signal-core | 16K |
| **Total** | **244K** |

### API Layer (Implementations)

| Crate | Size |
|-------|------|
| predictor-api | 908K |
| detector-api | 208K |
| pipeline-api | 172K |
| signal-api | 48K |
| **Total** | **1.3M** |

### Facade Layer (Unified API)

| Crate | Size |
|-------|------|
| predictor-facade | 8K |
| detector-facade | 8K |
| pipeline-facade | 8K |
| signal-facade | 8K |
| **Total** | **32K** |

## Legacy Crates

| Crate | Size |
|-------|------|
| algorithm | 1.4M |
| data | 792K |
| automl | 572K |
| forecast | 364K |
| financial | 308K |
| anomaly | 264K |

## Notes

- All sizes are from `cargo build --release` with LTO enabled
- SPI crates are intentionally minimal for extensibility
- Facade crates are thin re-export layers
- WASM size can be further reduced with `wasm-opt`
