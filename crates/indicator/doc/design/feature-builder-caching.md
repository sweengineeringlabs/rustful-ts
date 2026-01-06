# FeatureBuilder Caching Architecture

The `FeatureBuilder` in QuantLang's ML module uses a two-phase caching approach to efficiently compute technical indicator features for machine learning models.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     FeatureBuilder.build()                       │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Precomputation                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ SMA Cache   │  │ EMA Cache   │  │ RSI Cache   │              │
│  │ {10: [...]} │  │ {12: [...]} │  │ {14: [...]} │              │
│  │ {20: [...]} │  │ {26: [...]} │  │             │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: Assembly (O(1) lookups per feature)                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ for each sample:                                         │    │
│  │   row = [sma_cache[10][i], sma_cache[20][i],            │    │
│  │          ema_cache[12][i], rsi_cache[14][i], ...]       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Indicator Precomputation

Before iterating through samples, `build()` precomputes all indicators once for the entire price series and stores them in HashMaps keyed by period:

```rust
// Internal cache structure
sma_cache: HashMap<period, Vec<f64>>  // {10 -> [sma values], 20 -> [sma values]}
ema_cache: HashMap<period, Vec<f64>>  // {12 -> [ema values], 26 -> [ema values]}
rsi_cache: HashMap<period, Vec<f64>>  // {14 -> [rsi values]}
```

Each indicator is computed in a single O(n) pass:

| Indicator | Algorithm | Time Complexity |
|-----------|-----------|-----------------|
| SMA | Rolling sum with sliding window | O(n) |
| EMA | Recursive: `ema[i] = α * price[i] + (1-α) * ema[i-1]` | O(n) |
| RSI | Smoothed gain/loss averages | O(n) |

## Phase 2: Feature Assembly

When building feature rows, indicators are retrieved via O(1) HashMap lookup instead of being recomputed for each sample:

```rust
// Without caching (naive approach)
for sample in 0..n_samples {
    for feature in features {
        row.push(compute_sma(prices, period, sample));  // O(period) per sample!
    }
}

// With caching (FeatureBuilder approach)
let sma_cache = precompute_all_smas(prices);  // O(n) once
for sample in 0..n_samples {
    for feature in features {
        row.push(sma_cache[period][sample]);  // O(1) lookup!
    }
}
```

## Automatic Deduplication

The cache automatically deduplicates shared computations:

```rust
let builder = FeatureBuilder::new()
    .add_sma(20)              // SMA-20 computed
    .add_sma(20)              // Reuses cached SMA-20 (no recomputation)
    .add_ema(12)              // EMA-12 computed
    .add_macd(12, 26, 9);     // Reuses EMA-12, adds EMA-26
```

### Cross-Indicator Sharing

MACD internally requires two EMAs (fast and slow). The cache handles this transparently:

```rust
// This configuration:
.add_ema(12)
.add_ema(26)
.add_macd(12, 26, 9)

// Only computes:
// - EMA-12: once (shared between explicit EMA and MACD)
// - EMA-26: once (shared between explicit EMA and MACD)
```

## Complexity Analysis

### Without Caching

```
Time: O(samples × features × avg_period)
```

For each sample, each feature requires O(period) computation.

### With Caching

```
Time: O(n × unique_indicators) + O(samples × features)
      ─────────────────────────   ─────────────────────
           Precomputation              Assembly
```

### Example Comparison

| Scenario | Without Caching | With Caching | Speedup |
|----------|-----------------|--------------|---------|
| 500 samples, 10 features, avg period 20 | 100,000 ops | 7,500 ops | **13x** |
| 1000 samples, 15 features, avg period 25 | 375,000 ops | 17,500 ops | **21x** |
| 5000 samples, 20 features, avg period 30 | 3,000,000 ops | 125,000 ops | **24x** |

## Memory Trade-off

Each cached indicator uses O(n) memory where n = price series length.

```
Memory = unique_indicators × n × sizeof(f64)
       = unique_indicators × n × 8 bytes
```

| Data Points | Unique Indicators | Cache Memory |
|-------------|-------------------|--------------|
| 1,000 | 5 | 40 KB |
| 1,000 | 10 | 80 KB |
| 10,000 | 10 | 800 KB |
| 100,000 | 10 | 8 MB |

For typical usage, this is negligible compared to the computational savings.

## Usage Example

```rust
use quantlang::runtime::timeseries_ai::{FeatureBuilder, GradientBoosting};

// Configure features with arbitrary indicator parameters
let builder = FeatureBuilder::new()
    .add_sma(10)                      // Short-term trend
    .add_sma(50)                      // Long-term trend
    .add_ema(12)                      // Fast EMA
    .add_ema(26)                      // Slow EMA
    .add_rsi(14)                      // Momentum
    .add_macd(12, 26, 9)              // Trend + momentum (reuses EMA-12, EMA-26)
    .add_bollinger_position(20, 2.0)  // Volatility position
    .add_roc(10)                      // Rate of change
    .add_std_dev(20)                  // Volatility
    .add_lags(5)                      // Autoregressive features
    .add_returns();                   // Price momentum

// Build features - indicators computed once, then assembled
let (features, targets) = builder.build(&prices);

// features: Vec<Vec<f64>> - each row is a feature vector
// targets: Vec<f64> - next-period returns for supervised learning

// Train model on indicator-based features
let mut model = GradientBoosting::with_params(100, 0.05, 4);
model.fit(&features, &targets);

// Make predictions
let predictions = model.predict(&test_features);
```

## Feature Normalization

All features are normalized for scale independence:

| Feature | Normalization | Range |
|---------|---------------|-------|
| SMA | `(sma - price) / price` | Relative deviation |
| EMA | `(ema - price) / price` | Relative deviation |
| RSI | `(rsi - 50) / 50` | [-1, 1] |
| MACD | `macd / price` | Relative to price |
| Bollinger | Position in band | [-1, 1] |
| ROC | `(price - prev) / prev` | Percentage change |
| StdDev | `std / price` | Relative volatility |
| Lag | `(price - lagged) / lagged` | Percentage change |
| Returns | `(price - prev) / prev` | Percentage change |

## Implementation Details

### Cache Method Signatures

```rust
impl FeatureBuilder {
    /// Precomputes all unique SMA periods
    fn compute_sma_cache(&self, prices: &[f64]) -> HashMap<usize, Vec<f64>>;

    /// Precomputes all unique EMA periods (including MACD dependencies)
    fn compute_ema_cache(&self, prices: &[f64]) -> HashMap<usize, Vec<f64>>;

    /// Precomputes all unique RSI periods
    fn compute_rsi_cache(&self, prices: &[f64]) -> HashMap<usize, Vec<f64>>;
}
```

### Thread Safety

The current implementation is single-threaded. Caches are local to the `build()` call and not shared across threads. For parallel feature building, each thread should create its own `FeatureBuilder` instance.

### Future Optimizations

Potential enhancements:
- **SIMD acceleration**: Use SIMD for indicator computation (already available in `indicator.rs`)
- **Parallel precomputation**: Compute different indicator types in parallel
- **Incremental updates**: Support adding new data points without full recomputation
- **Persistent cache**: Optionally persist cache for repeated builds with same data
