# Indicator Repainting Behavior

This document describes the repainting behavior of indicators in `indicator-core`.

## What is Repainting?

**Repainting** occurs when an indicator's historical values change after new price data arrives. This can make backtesting unreliable because the signals you see historically may not have existed at the time.

**Non-repainting** indicators produce fixed values once a bar closes. Only the current (live) bar recalculates as new ticks arrive.

## Non-Repainting Indicators

These indicators **never change historical values** once a bar is closed:

### Moving Averages (`moving_averages/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| SMA | No | Simple average of N bars |
| EMA | No | Exponential weighted |
| WMA | No | Linear weighted |
| DEMA | No | Double exponential |
| TEMA | No | Triple exponential |
| HMA | No | Hull moving average |
| KAMA | No | Kaufman adaptive |
| ZLEMA | No | Zero-lag EMA |
| SMMA | No | Smoothed MA |
| ALMA | No | Arnaud Legoux MA |
| FRAMA | No | Fractal adaptive |
| VIDYA | No | Variable index dynamic |
| T3 | No | Tillson T3 |
| TRIMA | No | Triangular MA |
| GMMA | No | Guppy multiple MA |
| SineWMA | No | Sine-weighted MA |

### Filters (`filters/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| KalmanFilter | No | State-space filter |
| MedianFilter | No | Median of N bars |
| GaussianFilter | No | Gaussian-weighted |
| SVHMA | No | Step variable horizontal MA |
| DeviationFilteredAverage | No | Deviation-filtered |
| StepVhfAdaptiveVMA | No | VHF-adaptive step MA |

### Oscillators (`oscillators/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| RSI | No | Relative strength index |
| Stochastic | No | Stochastic oscillator |
| WilliamsR | No | Williams %R |
| CCI | No | Commodity channel index |
| TRIX | No | Triple smoothed ROC |
| UltimateOscillator | No | Ultimate oscillator |
| ROC | No | Rate of change |
| Momentum | No | Price momentum |
| ChandeMomentum | No | Chande momentum |
| DeMarker | No | DeMarker oscillator |
| AwesomeOscillator | No | Awesome oscillator |
| AcceleratorOscillator | No | Accelerator oscillator |
| KST | No | Know Sure Thing |
| PPO | No | Percentage price oscillator |
| RVI | No | Relative vigor index |
| StochasticRSI | No | Stochastic RSI |
| ConnorsRSI | No | Connors RSI |
| TSI | No | True strength index |
| SMI | No | Stochastic momentum index |
| RMI | No | Relative momentum index |
| FisherTransform | No | Fisher transform |
| InverseFisherTransform | No | Inverse Fisher |
| Qstick | No | Qstick indicator |
| PMO | No | Price momentum oscillator |
| SpecialK | No | Special K |
| DisparityIndex | No | Disparity index |
| PrettyGoodOscillator | No | Pretty good oscillator |
| APO | No | Absolute price oscillator |
| ErgodicOscillator | No | Ergodic oscillator |
| PolarizedFractalEfficiency | No | PFE |
| IntradayMomentumIndex | No | Intraday momentum |
| RelativeVolatilityIndex | No | Relative volatility |
| DoubleStochastic | No | Double stochastic |

### Trend (`trend/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| MACD | No | Moving average convergence divergence |
| ADX | No | Average directional index |
| Ichimoku | No | Ichimoku cloud (displaced forward, not repaint) |
| SuperTrend | No | SuperTrend |
| ParabolicSAR | No | Parabolic stop and reverse |
| Alligator | No | Williams Alligator |
| Aroon | No | Aroon indicator |
| CoppockCurve | No | Coppock curve |
| DPO | No | Detrended price oscillator |
| GatorOscillator | No | Gator oscillator |
| McGinleyDynamic | No | McGinley dynamic |
| RainbowMA | No | Rainbow moving average |
| RandomWalkIndex | No | Random walk index |
| TrendDetectionIndex | No | Trend detection |
| TrendIntensityIndex | No | Trend intensity |
| VerticalHorizontalFilter | No | VHF |
| VortexIndicator | No | Vortex indicator |

### Volatility (`volatility/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| ATR | No | Average true range |
| HistoricalVolatility | No | Historical volatility |
| ChaikinVolatility | No | Chaikin volatility |
| MassIndex | No | Mass index |
| ParkinsonVolatility | No | Parkinson volatility |
| GarmanKlassVolatility | No | Garman-Klass volatility |
| RogersSatchellVolatility | No | Rogers-Satchell volatility |
| YangZhangVolatility | No | Yang-Zhang volatility |
| RealizedVolatility | No | Realized volatility |
| NormalizedATR | No | Normalized ATR |
| ChoppinessIndex | No | Choppiness index |
| UlcerIndex | No | Ulcer index |

### Volume (`volume/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| VWAP | No | Volume-weighted average price |
| OBV | No | On-balance volume |
| MFI | No | Money flow index |
| CMF | No | Chaikin money flow |
| VWMA | No | Volume-weighted MA |
| ADLine | No | Accumulation/distribution |
| ForceIndex | No | Force index |
| KlingerOscillator | No | Klinger oscillator |
| BalanceOfPower | No | Balance of power |
| EaseOfMovement | No | Ease of movement |
| VROC | No | Volume ROC |
| PVT | No | Price volume trend |
| NVI | No | Negative volume index |
| PVI | No | Positive volume index |
| WilliamsAD | No | Williams A/D |
| TwiggsMoneyFlow | No | Twiggs money flow |
| VolumeOscillator | No | Volume oscillator |
| NetVolume | No | Net volume |
| ChaikinOscillator | No | Chaikin oscillator |
| TWAP | No | Time-weighted average price |

### Bands (`bands/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| BollingerBands | No | Bollinger bands |
| KeltnerChannels | No | Keltner channels |
| DonchianChannels | No | Donchian channels |
| AccelerationBands | No | Acceleration bands |
| ChandelierExit | No | Chandelier exit |
| Envelope | No | Price envelope |
| HighLowBands | No | High-low bands |
| PriceChannel | No | Price channel |
| ProjectionBands | No | Projection bands |
| STARCBands | No | STARC bands |
| StandardErrorBands | No | Standard error bands |
| TironeLevels | No | Tirone levels |

### Support/Resistance (`support_resistance/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| PivotPoints | No | Calculated from prior period |
| Fibonacci | No | Static levels from input |

### Statistical (`statistical/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| StandardDeviation | No | Standard deviation |
| Variance | No | Variance |
| ZScore | No | Z-score |
| LinearRegression | No | Linear regression |
| Correlation | No | Correlation coefficient |
| Spread | No | Price spread |
| Ratio | No | Price ratio |
| ZScoreSpread | No | Z-score of spread |
| Autocorrelation | No | Autocorrelation |
| Skewness | No | Skewness |
| Kurtosis | No | Kurtosis |

### Risk (`risk/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| SharpeRatio | No | Sharpe ratio |
| SortinoRatio | No | Sortino ratio |
| CalmarRatio | No | Calmar ratio |
| MaxDrawdown | No | Maximum drawdown |
| ValueAtRisk | No | Value at risk |
| ConditionalVaR | No | Conditional VaR |
| Beta | No | Beta coefficient |
| Alpha | No | Alpha |
| TreynorRatio | No | Treynor ratio |
| InformationRatio | No | Information ratio |
| OmegaRatio | No | Omega ratio |
| GainLossRatio | No | Gain/loss ratio |

### DSP (`dsp/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| MESA | No | MESA adaptive MA |
| MAMA | No | MAMA/FAMA |
| SineWave | No | Sine wave indicator |
| HilbertTransform | No | Hilbert transform |
| CyberCycle | No | Cyber cycle |
| CGOscillator | No | Center of gravity |
| LaguerreRSI | No | Laguerre RSI |
| RoofingFilter | No | Roofing filter |
| Supersmoother | No | Supersmoother |
| Decycler | No | Decycler |

### Composite (`composite/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| TTMSqueeze | No | TTM Squeeze |
| ElderImpulse | No | Elder impulse system |
| ElderRay | No | Elder ray (bull/bear power) |
| SchaffTrendCycle | No | Schaff trend cycle |
| ElderTripleScreen | No | Elder triple screen |
| CommoditySelectionIndex | No | CSI |
| SqueezeMomentum | No | Squeeze momentum |
| TrendStrengthIndex | No | Trend strength |
| RegimeDetector | No | Market regime detection |

### Breadth (`breadth/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| AdvanceDeclineLine | No | A/D line |
| BreadthThrust | No | Breadth thrust |
| CumulativeVolumeIndex | No | Cumulative volume |
| McClellanOscillator | No | McClellan oscillator |
| McClellanSummationIndex | No | McClellan summation |
| PercentAboveMA | No | Percent above MA |
| PutCallRatio | No | Put/call ratio |
| TickIndex | No | TICK index |
| TRIN | No | TRIN/Arms index |
| HighLowIndex | No | High-low index |
| UpDownVolume | No | Up/down volume |

---

## Repainting Indicators

These indicators **may change historical values** as new data arrives. Use with caution in automated trading systems.

### Pattern (`pattern/`)
| Indicator | Repaints | Confirmation Bars | Notes |
|-----------|----------|-------------------|-------|
| ZigZag | **Yes** | Variable (depth) | Swing points update until confirmed |
| Fractals | **Yes** | N bars each side | Recent fractals can change |
| DarvasBox | Partial | Box completion | Box bounds may adjust |
| HeikinAshi | No | - | Smoothed candles, no repaint |
| Candlestick patterns | No | - | Doji, Hammer, Engulfing, etc. |

### Swing (`swing/`)
| Indicator | Repaints | Confirmation Bars | Notes |
|-----------|----------|-------------------|-------|
| SwingPoints | **Yes** | `strength` param | Requires N bars confirmation |
| PivotHighsLows | **Yes** | `left`/`right` params | Requires right-side confirmation |
| SwingIndex | No | - | Single-bar calculation |
| AccumulativeSwingIndex | No | - | Cumulative, no repaint |
| GannSwing | **Yes** | Swing confirmation | Swing direction can change |
| MarketStructure | **Yes** | Structure confirmation | HH/HL/LH/LL can change |
| OrderBlocks | **Partial** | Until invalidated | Blocks can be invalidated by price |
| FairValueGap | **Partial** | Until filled | Gaps can be filled |
| LiquidityVoids | **Partial** | Until filled | Voids can be filled |
| BreakOfStructure | **Partial** | Until invalidated | BOS/CHoCH can be invalidated |

### DeMark (`demark/`)
| Indicator | Repaints | Notes |
|-----------|----------|-------|
| TDSetup | **Partial** | Setup can fail before count reaches 9 |
| TDCountdown | **Partial** | Countdown can fail before 13 |
| TDSequential | **Partial** | Full sequence can fail |
| TDCombo | **Partial** | Combo can fail |
| TDREI | No | Single calculation |
| TDPOQ | No | Single calculation |
| TDPressure | No | Single calculation |
| TDDWave | **Yes** | Wave labeling can change |
| TDTrendFactor | No | Single calculation |

---

## Testing Non-Repainting Behavior

Non-repainting indicators have tests that verify:

1. **Historical stability**: Values at index N don't change when new data is added
2. **Incremental consistency**: Streaming calculation matches batch calculation
3. **Determinism**: Same input always produces same output

Example test pattern:
```rust
#[test]
fn test_no_repaint() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let indicator = SMA::new(3);

    // Calculate with initial data
    let result1 = indicator.calculate(&data);
    let historical_value = result1[3]; // Save value at index 3

    // Add new data
    let extended_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let result2 = indicator.calculate(&extended_data);

    // Historical value must not change
    assert_eq!(result2[3], historical_value, "Indicator repainted!");
}
```

## Testing Repainting Behavior

Repainting indicators have tests that verify:

1. **Expected repaint**: Values do change appropriately with new data
2. **Confirmation threshold**: After N bars, values become stable
3. **Repaint bounds**: Only recent values change, not ancient history

Example test pattern:
```rust
#[test]
fn test_repaint_within_confirmation_window() {
    let fractals = Fractals::new(2); // 2 bars each side

    let data1 = vec![1.0, 3.0, 2.0, 1.0, 0.5]; // Potential high at index 1
    let result1 = fractals.calculate(&data1);

    // Add data that invalidates the fractal
    let data2 = vec![1.0, 3.0, 2.0, 1.0, 0.5, 4.0]; // New high invalidates
    let result2 = fractals.calculate(&data2);

    // Value should have changed (repainted)
    assert_ne!(result1[1], result2[1], "Expected repaint didn't occur");
}

#[test]
fn test_no_repaint_after_confirmation() {
    let fractals = Fractals::new(2);

    let data = vec![1.0, 3.0, 2.0, 1.0, 0.5, 0.6, 0.7, 0.8];
    let result1 = fractals.calculate(&data);
    let confirmed_value = result1[1]; // Fully confirmed fractal

    // Add more data - confirmed fractal should not change
    let extended = vec![1.0, 3.0, 2.0, 1.0, 0.5, 0.6, 0.7, 0.8, 5.0, 6.0];
    let result2 = fractals.calculate(&extended);

    assert_eq!(result2[1], confirmed_value, "Confirmed value changed!");
}
```

---

## Recommendations

### For Backtesting
- Use only non-repainting indicators for signal generation
- If using repainting indicators, wait for confirmation period before acting
- Mark repainting indicator signals with their confirmation status

### For Live Trading
- Non-repainting: Safe to act on closed-bar signals immediately
- Repainting: Wait N bars after signal before acting, or use for visual analysis only

### For Visual Analysis
- Repainting indicators are useful for identifying patterns after the fact
- ZigZag is excellent for visual swing analysis but not for signals
- Fractals help identify support/resistance zones retrospectively
