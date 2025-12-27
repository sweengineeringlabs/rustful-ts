# Performance Benchmarks

Benchmark results for rustful-ts algorithms on optimized release builds.

## Test Environment

- Build: `cargo bench` (release profile with optimizations)
- Harness: Custom timing with `std::hint::black_box` to prevent dead code elimination

## Predictor Algorithms

### Preprocessing (10K points)

| Operation | Time/iter |
|-----------|-----------|
| normalize | 22.4µs |
| standardize | 28.3µs |
| difference(1) | 34.2µs |
| difference(2) | 42.5µs |

### Metrics (10K points)

| Operation | Time/iter |
|-----------|-----------|
| mae | 8.6µs |
| mse | 7.5µs |
| rmse | 8.3µs |

### Simple Exponential Smoothing

| Operation | Time/iter |
|-----------|-----------|
| fit (1K points) | 1.5µs |
| fit (10K points) | 15.5µs |
| fit (100K points) | 157µs |
| predict(100) | 34ns |

### Holt-Winters

| Operation | Time/iter |
|-----------|-----------|
| fit (1K, period=12) | 4.3µs |
| fit (10K, period=12) | 43.4µs |
| predict(100) | 408ns |

### ARIMA

| Operation | Time/iter |
|-----------|-----------|
| ARIMA(1,1,0) fit (1K) | 5.7µs |
| ARIMA(2,1,1) fit (1K) | 9.9µs |
| predict(100) | 1.9µs |

### Linear Regression

| Operation | Time/iter |
|-----------|-----------|
| fit (1K points) | 5.7µs |
| fit (10K points) | 56µs |
| fit (100K points) | 562µs |

### KNN

| Operation | Time/iter |
|-----------|-----------|
| fit (1K, k=5, w=10) | 57µs |
| predict(10) | 206µs |

### Moving Average

| Operation | Time/iter |
|-----------|-----------|
| SMA(20) fit (10K) | 36.9µs |
| SMA(100) fit (10K) | 36.4µs |

## Detector Algorithms

### Z-Score Detector

| Operation | Time/iter |
|-----------|-----------|
| fit (1K points) | 1.4µs |
| fit (10K points) | 14.6µs |
| fit (100K points) | 151µs |
| detect (10K) | 18.6µs |
| score (10K) | 13.3µs |

### IQR Detector

| Operation | Time/iter |
|-----------|-----------|
| fit (1K points) | 11µs |
| fit (10K points) | 201µs |
| fit (100K points) | 3.5ms |
| detect (10K) | 29.3µs |

## Summary

- **SES/Holt/ARIMA**: Low-microsecond fit times with linear O(n) scaling
- **Linear Regression**: Linear scaling O(n)
- **KNN**: Prediction is O(n*k) due to distance computation
- **Z-Score**: Linear O(n) fit due to mean/stddev calculation
- **IQR**: O(n log n) fit due to sorting for quartile calculation

## Running Benchmarks

```bash
# Algorithm benchmarks
cargo bench -p algorithm --bench performance

# Detector benchmarks
cargo bench -p detector-api --bench performance
```
