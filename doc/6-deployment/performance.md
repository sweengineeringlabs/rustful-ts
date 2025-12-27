# Performance Benchmarks

Benchmark results for rustful-ts algorithms on optimized release builds.

## Test Environment

- Build: `cargo bench` (release profile with optimizations)
- Harness: Custom timing (no criterion overhead)

## Predictor Algorithms

### Preprocessing (10K points)

| Operation | Time/iter |
|-----------|-----------|
| normalize | 22.4µs |
| standardize | 27.6µs |
| difference(1) | 33.1µs |
| difference(2) | 41.5µs |

### Simple Exponential Smoothing

| Operation | Time/iter |
|-----------|-----------|
| fit (1K points) | ~1.7µs |
| fit (10K points) | ~2µs |
| fit (100K points) | ~6µs |
| predict(100) | 18ns |

### Holt-Winters

| Operation | Time/iter |
|-----------|-----------|
| fit (1K, period=12) | 4.2µs |
| fit (10K, period=12) | 44.5µs |
| predict(100) | 354ns |

### ARIMA

| Operation | Time/iter |
|-----------|-----------|
| ARIMA(1,1,0) fit (1K) | 5.7µs |
| ARIMA(2,1,1) fit (1K) | 15.4µs |
| predict(100) | 1.9µs |

### Linear Regression

| Operation | Time/iter |
|-----------|-----------|
| fit (1K points) | 5.5µs |
| fit (10K points) | 54.5µs |
| fit (100K points) | 551µs |

### KNN

| Operation | Time/iter |
|-----------|-----------|
| fit (1K, k=5, w=10) | 61.9µs |
| predict(10) | 191µs |

### Moving Average

| Operation | Time/iter |
|-----------|-----------|
| SMA(20) fit (10K) | 35.8µs |
| SMA(100) fit (10K) | 35.5µs |

## Detector Algorithms

### Z-Score Detector

| Operation | Time/iter |
|-----------|-----------|
| fit (1K points) | ~2ns |
| fit (10K points) | ~2ns |
| fit (100K points) | ~6ns |
| detect (10K) | 19.1µs |
| score (10K) | 13.7µs |

### IQR Detector

| Operation | Time/iter |
|-----------|-----------|
| fit (1K points) | 11µs |
| fit (10K points) | 213µs |
| fit (100K points) | 3.5ms |
| detect (10K) | 29µs |

## Summary

- **SES/Holt/ARIMA**: Sub-microsecond to low-microsecond fit times
- **Linear Regression**: Linear scaling O(n)
- **KNN**: Prediction is O(n*k) due to distance computation
- **Z-Score**: Near-instant fit, fast detection
- **IQR**: O(n log n) fit due to sorting

## Running Benchmarks

```bash
# Algorithm benchmarks
cargo bench -p algorithm --bench performance

# Detector benchmarks
cargo bench -p detector-api --bench performance
```
