# Technical Indicators

This module provides a comprehensive set of technical analysis indicators for financial time series analysis.

## Moving Averages (8)

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **SMA** | Simple Moving Average | `period` |
| **EMA** | Exponential Moving Average | `period`, `alpha` (optional) |
| **WMA** | Weighted Moving Average | `period` |
| **DEMA** | Double Exponential Moving Average | `period` |
| **TEMA** | Triple Exponential Moving Average | `period` |
| **HMA** | Hull Moving Average | `period` |
| **KAMA** | Kaufman Adaptive Moving Average | `period`, `fast_period`, `slow_period` |
| **ZLEMA** | Zero-Lag Exponential Moving Average | `period` |

## Filters (3)

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **Kalman** | Kalman Filter for smoothing | `process_noise`, `measurement_noise` |
| **Median** | Median Filter (spike removal) | `period` |
| **Gaussian** | Gaussian Smoothing Filter | `period`, `sigma` |

## Oscillators (6)

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **RSI** | Relative Strength Index | `period`, `overbought`, `oversold` |
| **Stochastic** | Stochastic Oscillator | `k_period`, `d_period` |
| **Williams %R** | Williams Percent Range | `period`, `overbought`, `oversold` |
| **CCI** | Commodity Channel Index | `period`, `overbought`, `oversold` |
| **TRIX** | Triple Exponential Rate of Change | `period` |
| **Ultimate Oscillator** | Multi-timeframe momentum | `period1`, `period2`, `period3` |

## Trend Indicators (5)

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **MACD** | Moving Average Convergence Divergence | `fast_period`, `slow_period`, `signal_period` |
| **ADX** | Average Directional Index | `period`, `strong_trend` |
| **SuperTrend** | Trend-following based on ATR | `period`, `multiplier` |
| **Parabolic SAR** | Stop and Reverse | `af_start`, `af_step`, `af_max` |
| **Ichimoku** | Japanese charting technique | `tenkan`, `kijun`, `senkou_b` periods |

## Volatility Indicators (4)

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **Bollinger Bands** | SMA with standard deviation bands | `period`, `std_dev` |
| **ATR** | Average True Range | `period` |
| **Keltner Channels** | EMA with ATR-based bands | `ema_period`, `atr_period`, `multiplier` |
| **Donchian Channels** | Highest high / lowest low channels | `period` |

## Volume Indicators (4)

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **VWAP** | Volume Weighted Average Price | `reset_daily` |
| **OBV** | On-Balance Volume | `signal_period` (optional) |
| **MFI** | Money Flow Index | `period`, `overbought`, `oversold` |
| **CMF** | Chaikin Money Flow | `period` |

## Support/Resistance (2)

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **Pivot Points** | Support/resistance from prior HLC | `type` (Standard, Fibonacci, Woodie, Camarilla) |
| **Fibonacci** | Fibonacci retracement levels | `high`, `low`, `is_uptrend` |

## Bull/Bear Power (1)

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| **Elder Ray** | Bull and Bear Power | `period` |

---

## Usage Examples

### Basic Indicator Usage

```rust
use indicator_core::{RSI, BollingerBands, OHLCVSeries};

// Create price data
let data = OHLCVSeries::from_close(prices);

// Calculate RSI
let rsi = RSI::new(14);
let rsi_output = rsi.compute(&data)?;

// Calculate Bollinger Bands
let bb = BollingerBands::new(20, 2.0);
let bb_output = bb.compute(&data)?;
```

### Moving Averages

```rust
use indicator_core::{SMA, EMA, WMA, DEMA, TEMA, HMA, KAMA, ZLEMA};

let data = vec![100.0, 101.0, 102.0, 103.0, 104.0, ...];

// Simple Moving Average
let sma = SMA::new(20);
let sma_values = sma.calculate(&data);

// Exponential Moving Average
let ema = EMA::new(20);
let ema_values = ema.calculate(&data);

// Weighted Moving Average (more weight on recent)
let wma = WMA::new(20);
let wma_values = wma.calculate(&data);

// Double EMA (reduced lag)
let dema = DEMA::new(20);
let dema_values = dema.calculate(&data);

// Triple EMA (further reduced lag)
let tema = TEMA::new(20);
let tema_values = tema.calculate(&data);

// Hull Moving Average (smooth with low lag)
let hma = HMA::new(20);
let hma_values = hma.calculate(&data);

// Kaufman Adaptive MA (adapts to market efficiency)
let kama = KAMA::new(10, 2, 30);
let kama_values = kama.calculate(&data);

// Zero-Lag EMA (momentum-adjusted)
let zlema = ZLEMA::new(20);
let zlema_values = zlema.calculate(&data);
```

### Filters

```rust
use indicator_core::{KalmanFilter, MedianFilter, GaussianFilter};

// Kalman Filter for adaptive smoothing
let kalman = KalmanFilter::new(0.01, 0.1);
let smooth = kalman.calculate(&data);

// Median Filter removes spikes
let median = MedianFilter::new(5);
let filtered = median.calculate(&data);

// Gaussian Filter for smooth curves
let gaussian = GaussianFilter::new(5, 1.0);
let smooth = gaussian.calculate(&data);
```

### Volume Indicators

```rust
use indicator_core::{VWAP, OBV, MFI, CMF};

// VWAP
let vwap = VWAP::new();
let vwap_values = vwap.calculate(&high, &low, &close, &volume);

// On-Balance Volume
let obv = OBV::new();
let obv_values = obv.calculate(&close, &volume);

// Money Flow Index (volume-weighted RSI)
let mfi = MFI::new(14);
let mfi_values = mfi.calculate(&high, &low, &close, &volume);

// Chaikin Money Flow
let cmf = CMF::new(20);
let cmf_values = cmf.calculate(&high, &low, &close, &volume);
```

### ADX Trend Strength

```rust
use indicator_core::{ADX, SignalIndicator};

let adx = ADX::new(14);
let result = adx.calculate(&high, &low, &close);

// ADX measures trend strength
println!("ADX: {:?}", result.adx);      // 0-100, >25 = strong trend
println!("+DI: {:?}", result.plus_di);  // Bullish strength
println!("-DI: {:?}", result.minus_di); // Bearish strength
```

### Williams %R and CCI

```rust
use indicator_core::{WilliamsR, CCI};

// Williams %R (-100 to 0)
let wr = WilliamsR::new(14);
let wr_values = wr.calculate(&high, &low, &close);
// Above -20: Overbought, Below -80: Oversold

// Commodity Channel Index
let cci = CCI::new(20);
let cci_values = cci.calculate(&high, &low, &close);
// Above +100: Strong uptrend, Below -100: Strong downtrend
```

### TRIX and Ultimate Oscillator

```rust
use indicator_core::{TRIX, UltimateOscillator};

// TRIX - Triple EMA rate of change
let trix = TRIX::new(15);
let trix_values = trix.calculate(&close);
// Positive: Bullish, Negative: Bearish
// Zero crossovers signal trend changes

// Ultimate Oscillator - Multi-timeframe
let uo = UltimateOscillator::new(7, 14, 28);
let uo_values = uo.calculate(&high, &low, &close);
// Above 70: Overbought, Below 30: Oversold
```

### Ichimoku Cloud

```rust
use indicator_core::{Ichimoku, OHLCVSeries};

let ichimoku = Ichimoku::default(); // 9, 26, 52
let result = ichimoku.calculate(&high, &low, &close);

// Access all five components
println!("Tenkan-sen: {:?}", result.tenkan);
println!("Kijun-sen: {:?}", result.kijun);
println!("Senkou Span A: {:?}", result.senkou_a);
println!("Senkou Span B: {:?}", result.senkou_b);
println!("Chikou Span: {:?}", result.chikou);
```

### Pivot Points

```rust
use indicator_core::{PivotPoints, PivotType};

// Calculate standard pivot points
let pp = PivotPoints::new(PivotType::Standard);
let levels = pp.calculate(high, low, close);

println!("Pivot: {}", levels.pivot);
println!("R1: {}, R2: {}, R3: {}", levels.r1, levels.r2, levels.r3);
println!("S1: {}, S2: {}, S3: {}", levels.s1, levels.s2, levels.s3);

// Use Fibonacci pivots
let fib_pp = PivotPoints::fibonacci();
let fib_levels = fib_pp.calculate(high, low, close);
```

---

## SIMD Optimization

Enable SIMD acceleration with the `simd` feature:

```toml
[dependencies]
indicator-core = { version = "0.2", features = ["simd"] }
```

SIMD is automatically used when available (AVX2/SSE4.1 on x86_64).

---

## Default Parameters

| Indicator | Default Parameters |
|-----------|-------------------|
| SMA | period = 20 |
| EMA | period = 20 |
| WMA | period = 20 |
| DEMA | period = 20 |
| TEMA | period = 20 |
| HMA | period = 20 |
| KAMA | period = 10, fast = 2, slow = 30 |
| ZLEMA | period = 20 |
| Kalman | process_noise = 0.01, measurement_noise = 0.1 |
| Median | period = 5 |
| Gaussian | period = 5, sigma = 1.0 |
| RSI | period = 14, overbought = 70, oversold = 30 |
| Stochastic | k = 14, d = 3 |
| Williams %R | period = 14, overbought = -20, oversold = -80 |
| CCI | period = 20, overbought = 100, oversold = -100 |
| TRIX | period = 15 |
| Ultimate Oscillator | periods = 7, 14, 28 |
| MACD | fast = 12, slow = 26, signal = 9 |
| ADX | period = 14, strong_trend = 25 |
| Bollinger | period = 20, std_dev = 2.0 |
| ATR | period = 14 |
| Ichimoku | tenkan = 9, kijun = 26, senkou_b = 52 |
| SuperTrend | period = 10, multiplier = 3.0 |
| Parabolic SAR | af_start = 0.02, af_step = 0.02, af_max = 0.2 |
| Donchian | period = 20 |
| Keltner | ema = 20, atr = 10, multiplier = 2.0 |
| VWAP | cumulative |
| MFI | period = 14, overbought = 80, oversold = 20 |
| CMF | period = 20 |
| Elder Ray | period = 13 |

---

## Total: 33 Indicators
