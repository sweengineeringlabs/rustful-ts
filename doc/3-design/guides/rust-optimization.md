# Rust Optimization and Release Builds

**Audience**: Developers writing benchmarks, performance-critical code, or debugging release builds

## WHAT: Compiler Optimizations

Rust's release builds (`--release`) enable aggressive compiler optimizations that transform your code for maximum performance. These optimizations can fundamentally change how code executes.

**Key optimizations**:
- Dead code elimination
- Function inlining
- Loop unrolling
- Constant propagation
- SIMD vectorization

## WHY: Understanding Matters

### The Benchmark Problem

Optimizations that help production code can break benchmarks:

```rust
// This benchmark measures NOTHING in release mode
fn broken_benchmark() {
    for _ in 0..1000 {
        let mut model = Arima::new(1, 1, 0).unwrap();
        model.fit(&data).unwrap();
        // model is never used - compiler removes the entire computation
    }
}
```

The compiler sees that `model` is never read after creation, so it eliminates the `fit()` call entirely. The loop runs in nanoseconds because it does nothing.

### Dead Code Elimination

The compiler analyzes data flow to find computations whose results are never used:

```
Source code:          What compiler sees:     What actually runs:
─────────────────────────────────────────────────────────────────
let x = expensive();  x is never read         (nothing)
let y = cheap();      y is returned           let y = cheap();
return y;                                     return y;
```

**Symptoms of eliminated code**:
- Impossibly fast benchmarks (nanoseconds for complex operations)
- No scaling with input size (1K and 100K take same time)
- Results don't change when algorithm is modified

## HOW: Preventing Unwanted Optimization

### Solution: `std::hint::black_box`

`black_box` is an optimization barrier that tells the compiler "pretend this value escapes to external code":

```rust
use std::hint::black_box;

fn correct_benchmark() {
    for _ in 0..1000 {
        let mut model = Arima::new(1, 1, 0).unwrap();
        model.fit(&data).unwrap();
        black_box(model);  // Compiler must keep the computation
    }
}
```

**How it works**:
1. Compiler cannot prove the value is unused
2. Must preserve all computations that produce the value
3. Actual runtime cost is zero (it's a compiler hint, not real code)

### Benchmark Pattern

Our standard benchmark harness:

```rust
use std::hint::black_box;
use std::time::Instant;

fn bench<F, R>(name: &str, iterations: u32, mut f: F)
where
    F: FnMut() -> R,  // Must return a value
{
    // Warmup (prime caches, trigger JIT if applicable)
    for _ in 0..3 {
        black_box(f());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(f());  // Prevent elimination
    }
    let elapsed = start.elapsed();

    println!("{}: {:?}/iter", name, elapsed / iterations);
}

// Usage - closure MUST return the result
bench("ARIMA fit", 100, || {
    let mut model = Arima::new(1, 1, 0).unwrap();
    model.fit(&data).unwrap();
    model  // Return it so black_box can capture it
});
```

### Common Mistakes

| Wrong | Right |
|-------|-------|
| `let _ = model.fit(&data);` | `model.fit(&data).unwrap(); model` |
| `model.predict(10);` | `black_box(model.predict(10))` |
| `for x in data { process(x); }` | `for x in data { black_box(process(x)); }` |

## Release Profile Settings

Our `Cargo.toml` release profile:

```toml
[profile.release]
opt-level = 3      # Maximum optimization
lto = true         # Link-time optimization across crates
codegen-units = 1  # Better optimization, slower compile
```

**LTO (Link-Time Optimization)**: Enables cross-crate inlining and dead code elimination. A function in `predictor-api` can be inlined into `wasm` if beneficial.

## Verifying Benchmarks Are Real

Check for proper scaling:

| Data Size | Expected (O(n)) | Suspicious |
|-----------|-----------------|------------|
| 1K | 1.5µs | 2ns |
| 10K | 15µs | 2ns |
| 100K | 150µs | 2ns |

If times don't scale with input, the computation is being eliminated.

## Additional Resources

- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [cargo-show-asm](https://github.com/pacak/cargo-show-asm) - Inspect generated assembly
- [criterion.rs](https://bheisler.github.io/criterion.rs/book/) - Statistical benchmarking framework
