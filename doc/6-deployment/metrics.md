# Project Metrics

Tracking key metrics for rustful-ts.

**Last Updated**: 2025-12-27

## Lines of Code

| Crate | Lines |
|-------|-------|
| algorithm | 3,294 |
| predictor | 2,096 |
| data | 1,458 |
| automl | 1,187 |
| cli | 897 |
| forecast | 845 |
| anomaly | 587 |
| detector | 542 |
| wasm | 521 |
| financial | 465 |
| pipeline | 292 |
| signal | 169 |
| bench-harness | 109 |
| server | 100 |
| **Total** | **~95,000** |

## Build Sizes

Release build with LTO enabled.

| Artifact | Size |
|----------|------|
| CLI binary | 1.2 MB |
| Server binary | 1.4 MB |
| WASM module | 420 KB |
| SPI layer (all) | 64 KB |

See [build-sizes.md](build-sizes.md) for detailed breakdown.

## Performance Benchmarks

### Predictor Algorithms

| Operation | Time/iter |
|-----------|-----------|
| SES fit (1K) | 1.5µs |
| SES fit (10K) | 15.5µs |
| SES fit (100K) | 157µs |
| Holt-Winters fit (1K) | 4.3µs |
| ARIMA(1,1,0) fit (1K) | 5.7µs |
| Linear Regression fit (10K) | 56µs |
| KNN predict(10) | 206µs |

### Detector Algorithms

| Operation | Time/iter |
|-----------|-----------|
| Z-Score fit (10K) | 14.6µs |
| Z-Score detect (10K) | 18.6µs |
| IQR fit (10K) | 201µs |
| IQR detect (10K) | 29.3µs |

See [performance.md](performance.md) for full benchmark results.

## Complexity Analysis

| Algorithm | Fit | Predict |
|-----------|-----|---------|
| SES | O(n) | O(1) |
| Holt-Winters | O(n) | O(h) |
| ARIMA | O(n) | O(h) |
| Linear Regression | O(n) | O(1) |
| KNN | O(n) | O(n*k) |
| Z-Score | O(n) | O(n) |
| IQR | O(n log n) | O(n) |

## Test Coverage

| Type | Count |
|------|-------|
| Integration tests | 158 |

Run: `cargo test --workspace`

## Running Metrics

```bash
# Lines of code
find crates -name "*.rs" | xargs wc -l | tail -1

# Build sizes
cargo build --release && ls -lh target/release/{cli,server}

# Benchmarks
cargo bench -p algorithm --bench performance
cargo bench -p detector-api --bench performance
```
