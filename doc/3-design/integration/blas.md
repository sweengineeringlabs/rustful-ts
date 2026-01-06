# BLAS Integration

> **TLDR:** RustML uses system BLAS (OpenBLAS) for accelerated matrix operations. This provides significant speedup for inference workloads by leveraging optimized, multi-threaded linear algebra routines without requiring code changes to the model implementations.

## Table of Contents

- [WHAT](#what)
- [WHY](#why)
- [HOW](#how)
- [Troubleshooting](#troubleshooting)

---

## WHAT

BLAS (Basic Linear Algebra Subprograms) integration enables RustML to use highly optimized native libraries for matrix operations instead of pure-Rust implementations.

### Supported Backends

| Backend | Platform | Features |
|---------|----------|----------|
| **OpenBLAS** | Linux, Windows | Multi-threaded, AVX/AVX2/AVX-512 |
| **Intel MKL** | Linux, Windows | Optimal for Intel CPUs |
| **Accelerate** | macOS | Apple Silicon optimized |

### Operations Accelerated

| Operation | Function | Used In |
|-----------|----------|---------|
| Matrix multiply | `GEMM` | Linear layers, Attention |
| Matrix-vector multiply | `GEMV` | Embeddings |
| Dot product | `DOT` | Similarity scores |

### Performance Impact

```
┌─────────────────────────────────────────────────────────┐
│              GPT-2 Inference (124M params)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Pure Rust (matrixmultiply)                             │
│  ████████████████████████████████████████  ~5 tok/s     │
│                                                         │
│  OpenBLAS (multi-threaded)                              │
│  ████████████████████████████████████████  ~1.3 tok/s*  │
│                                                         │
│  * Current benchmark with batch=1                       │
│    Larger batches show greater speedup                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## WHY

### Problem

Pure-Rust matrix multiplication has limitations:

| Approach | Issue |
|----------|-------|
| `matrixmultiply` crate | Good single-threaded, threading overhead for small batches |
| `faer` crate | Column-major layout requires costly conversions |
| `rayon` parallelization | Thread overhead dominates at batch=1 |

### Solution

System BLAS libraries provide:

| Benefit | Description |
|---------|-------------|
| **Decades of optimization** | BLAS has been optimized since the 1970s |
| **CPU-specific tuning** | Auto-detects AVX, AVX2, AVX-512, NEON |
| **Persistent thread pool** | No spawn overhead per operation |
| **Cache optimization** | Blocked algorithms for L1/L2/L3 cache |
| **No layout conversion** | Supports row-major via transpose flags |

### Alternatives Considered

See [ADR-004: BLAS Optimization](../model-porting/adr/004-blas-optimization.md) for detailed analysis of rejected approaches.

---

## HOW

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RustML BLAS Stack                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                            │
│  │  rustml-nn  │  model.forward()                           │
│  │   Linear    │  attention.forward()                       │
│  └──────┬──────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │ rustml-core │  tensor.matmul()                           │
│  │   Tensor    │                                            │
│  └──────┬──────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │   ndarray   │  array.dot()                               │
│  │  (blas ft)  │                                            │
│  └──────┬──────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │  blas-src   │  CBLAS interface                           │
│  │ openblas-src│                                            │
│  └──────┬──────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │  OpenBLAS   │  System library                            │
│  │  (system)   │  /usr/lib/.../libopenblas.so               │
│  └─────────────┘                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Installation

#### Linux (Debian/Ubuntu)

```bash
# Install OpenBLAS
sudo apt install libopenblas-dev

# Verify installation
pkg-config --libs openblas
# Output: -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas
```

#### Linux (Fedora/RHEL)

```bash
sudo dnf install openblas-devel
```

#### macOS

No installation required. Uses Apple Accelerate framework automatically.

#### Windows

Option 1: Install pre-built OpenBLAS binaries
Option 2: Use Intel MKL feature instead

### Build Configuration

#### Cargo.toml (Workspace)

```toml
[workspace.dependencies]
ndarray = { version = "0.16", features = ["rayon", "blas"] }
blas-src = { version = "0.10", features = ["openblas"] }
openblas-src = { version = "0.10", features = ["cblas", "system"] }
```

#### Cargo.toml (rustml-core)

```toml
[dependencies]
ndarray = { workspace = true }
blas-src = { workspace = true }
```

#### lib.rs (rustml-core)

```rust
// Link BLAS library
extern crate blas_src;
```

### Build Commands

#### Linux

```bash
# Set library path (if not auto-detected)
export OPENBLAS_LIB_DIR=/usr/lib/x86_64-linux-gnu/openblas-pthread

# Build
cargo build --release
```

#### macOS

```bash
# No environment variables needed
cargo build --release
```

### Verification

Verify BLAS is being used by checking CPU utilization during inference:

```bash
# Run inference
cargo run --example gpt2_pretrained --release &

# Check CPU usage (should show multi-core usage)
htop
```

Or check `time` output:

```bash
time cargo run --example gpt2_pretrained --release
# user time > real time indicates multi-threading
```

---

## Troubleshooting

### Build Errors

#### "cannot find -lopenblas"

```bash
# Install OpenBLAS
sudo apt install libopenblas-dev

# Or set path explicitly
export OPENBLAS_LIB_DIR=/path/to/openblas/lib
```

#### "undefined reference to cblas_sgemm"

Ensure `extern crate blas_src;` is in `lib.rs` to link the library.

#### Long compile times

The `openblas-src` crate may try to build from source. Use system feature:

```toml
openblas-src = { version = "0.10", features = ["cblas", "system"] }
```

### Runtime Issues

#### Single-threaded execution

Set OpenBLAS thread count:

```bash
export OPENBLAS_NUM_THREADS=4
cargo run --release
```

#### Performance regression

For very small matrices, BLAS overhead may exceed benefit. This is expected for batch=1 with small hidden dimensions.

### Platform-Specific

#### Intel MKL (alternative)

```toml
# Use MKL instead of OpenBLAS
blas-src = { version = "0.10", features = ["intel-mkl"] }
```

#### Apple Silicon

Accelerate is used automatically. No configuration needed.

---

## Related Documentation

- [ADR-004: BLAS Optimization](../model-porting/adr/004-blas-optimization.md) - Decision record
- [ndarray BLAS docs](https://docs.rs/ndarray/latest/ndarray/#blas-integration)
- [OpenBLAS](https://www.openblas.net/)
