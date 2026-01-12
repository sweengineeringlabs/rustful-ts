# NAS100 50-Indicator Optimization Benchmark

**Date:** 2026-01-12
**Version:** optimizer-core v0.2.0
**Commit:** 65e8afa (test: add comprehensive tests for all SPI crates)

## Methodology

| Parameter | Value |
|-----------|-------|
| Symbol | NAS100 |
| Timeframe | D1 (Daily) |
| Data Points | 1260 bars (~5 years) |
| Train/Test Split | 70% / 30% |
| Optimizer | Grid Search (parallel) |
| Objective | Sharpe Ratio |

### Validation Strategy
- **In-Sample (IS):** First 882 bars - used to find optimal parameters
- **Out-of-Sample (OOS):** Last 378 bars - used to validate performance
- **Robustness:** OOS/IS ratio - measures generalization capability

## Results Summary

| Metric | Value |
|--------|-------|
| Total Indicators | 50 |
| Positive OOS | 25 (50.0%) |
| Average OOS Sharpe | 0.0023 |
| Best OOS Indicator | UltimateOsc (1.6154) |
| Execution Time | <1s (release build) |

## Full Rankings by Out-of-Sample Sharpe

### Tier 1: Strong OOS Performance (>0.7)

| Rank | Indicator | IS Sharpe | OOS Sharpe | Robustness |
|------|-----------|-----------|------------|------------|
| 1 | UltimateOsc | 0.5955 | **1.6154** | 2.71 |
| 2 | DeMarker | 0.5504 | **1.2621** | 2.29 |
| 3 | VROC | 0.2751 | **1.2600** | 4.58 |
| 4 | LaguerreRSI | 0.8337 | **0.9553** | 1.15 |
| 5 | Schaff | 0.2359 | **0.8242** | 3.49 |
| 6 | RSI | 0.6353 | **0.8187** | 1.29 |
| 7 | Coppock | 0.4212 | **0.8095** | 1.92 |
| 8 | MACD | 0.4417 | **0.7897** | 1.79 |
| 9 | Stochastic | 0.4128 | **0.7294** | 1.77 |
| 10 | WilliamsR | 0.4128 | **0.7294** | 1.77 |
| 11 | PriceSMA | 0.5969 | **0.7253** | 1.22 |
| 12 | Keltner | 0.8512 | **0.7007** | 0.82 |

### Tier 2: Moderate OOS Performance (0.0 - 0.7)

| Rank | Indicator | IS Sharpe | OOS Sharpe | Robustness |
|------|-----------|-----------|------------|------------|
| 13 | CMO | 0.9303 | 0.6897 | 0.74 |
| 14 | StochRSI | 0.2287 | 0.6754 | 2.95 |
| 15 | Vortex | 0.2516 | 0.5503 | 2.19 |
| 16 | FisherTransform | -0.2356 | 0.4534 | -1.92 |
| 17 | Bollinger | 0.6688 | 0.3708 | 0.55 |
| 18 | Choppiness | 1.1989 | 0.3399 | 0.28 |
| 19 | ROC | 0.5059 | 0.3344 | 0.66 |
| 20 | TTMSqueeze | 0.4108 | 0.3032 | 0.74 |
| 21 | CCI | -0.1420 | 0.2833 | -2.00 |
| 22 | TSI | 0.3348 | 0.1922 | 0.57 |
| 23 | WMA | 0.2661 | 0.1551 | 0.58 |
| 24 | Aroon | 0.3288 | 0.1078 | 0.33 |
| 25 | SMA | 1.0836 | 0.0044 | 0.00 |

### Tier 3: Negative OOS Performance (<0.0)

| Rank | Indicator | IS Sharpe | OOS Sharpe | Robustness |
|------|-----------|-----------|------------|------------|
| 26 | ATR | 0.0000 | 0.0000 | 0.00 |
| 27 | PPO | 0.0000 | 0.0000 | 0.00 |
| 28 | HistVol | 0.0000 | 0.0000 | 0.00 |
| 29 | KAMA | 0.1778 | -0.1159 | -0.65 |
| 30 | EMA | 1.2604 | -0.1374 | -0.11 |
| 31 | OBV | 0.1564 | -0.1648 | -1.05 |
| 32 | SuperTrend | 0.3799 | -0.2438 | -0.64 |
| 33 | TRIX | 0.1150 | -0.3541 | -3.08 |
| 34 | PriceEMA | -0.0623 | -0.3884 | 6.23 |
| 35 | Momentum | 0.3509 | -0.4586 | -1.31 |
| 36 | ZLEMA | 0.5002 | -0.5273 | -1.05 |
| 37 | CGOscillator | -0.1592 | -0.5964 | 3.75 |
| 38 | ElderRay | -0.2617 | -0.5957 | 2.28 |
| 39 | HMA | 0.4502 | -0.6003 | -1.33 |
| 40 | ForceIndex | -0.0049 | -0.7187 | 147.03 |
| 41 | MFI | 0.5731 | -0.7217 | -1.26 |
| 42 | DPO | 0.7261 | -0.8007 | -1.10 |
| 43 | Donchian | 0.0000 | -0.8187 | 0.00 |
| 44 | ParabolicSAR | 0.2294 | -0.8281 | -3.61 |
| 45 | CMF | -0.1886 | -0.8495 | 4.50 |
| 46 | KST | 0.2208 | -0.8738 | -3.96 |
| 47 | TEMA | 0.0246 | -1.1876 | -48.26 |
| 48 | ADX | 0.1111 | -1.2035 | -10.83 |
| 49 | DEMA | 0.5535 | -1.6633 | -3.01 |
| 50 | AwesomeOsc | 0.9196 | -1.7170 | -1.87 |

## Key Insights

### Best Performers
1. **UltimateOsc** - Multi-timeframe momentum oscillator showed exceptional OOS performance
2. **DeMarker** - Exhaustion indicator proved robust across train/test periods
3. **VROC** - Volume-based indicator with highest robustness ratio (4.58)
4. **LaguerreRSI** - DSP-based oscillator with consistent performance

### Overfit Indicators (High IS, Low/Negative OOS)
- **SMA** (IS: 1.08, OOS: 0.00) - Classic overfitting example
- **EMA** (IS: 1.26, OOS: -0.14) - Fast MA prone to curve-fitting
- **AwesomeOsc** (IS: 0.92, OOS: -1.72) - Worst OOS despite good IS
- **Choppiness** (IS: 1.20, OOS: 0.34) - High IS doesn't translate

### Most Robust (OOS/IS > 2.0)
1. VROC (4.58)
2. Schaff (3.49)
3. StochRSI (2.95)
4. UltimateOsc (2.71)
5. DeMarker (2.29)
6. Vortex (2.19)

## Indicator Categories Performance

| Category | Avg OOS Sharpe | Best Performer |
|----------|----------------|----------------|
| Oscillators | 0.42 | UltimateOsc (1.62) |
| Volume | 0.00 | VROC (1.26) |
| Trend | -0.27 | Coppock (0.81) |
| Moving Averages | -0.30 | PriceSMA (0.73) |
| Bands/Channels | 0.09 | Keltner (0.70) |
| DSP | 0.18 | LaguerreRSI (0.96) |
| Composite | 0.18 | Schaff (0.82) |
| Volatility | 0.17 | Choppiness (0.34) |

## Reproduction

```bash
cargo run -p optimizer-core --example nas100_50_indicators --release
```

## Parameter Ranges Used

See `/crates/optimizer/optimizer-core/examples/nas100_50_indicators.rs` for exact parameter configurations.

## Notes

- All indicators use default signal generation logic (overbought/oversold, crossovers, etc.)
- Grid search ensures exhaustive parameter space exploration
- Results are specific to NAS100 daily data; other symbols/timeframes may vary
- Robustness values >1.0 indicate better OOS than IS performance (desirable)
