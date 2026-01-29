# Indicator Backlog

> **Current:** 366 indicators | **Target:** 2148+ indicators
>
> **200 priority groups** covering technical analysis, quantitative finance, machine learning, alternative data, and industry-specific metrics.

## Priority 0: Core Implemented Indicators (Not Listed Elsewhere)

These indicators are already implemented in indicator-core but were not listed in early priorities.

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-A01 | ADX | Trend | Average Directional Index - trend strength | Medium | ✅ Done |
| IND-A02 | MACD | Trend | Moving Average Convergence Divergence | Medium | ✅ Done |
| IND-A03 | Ichimoku | Trend | Ichimoku Kinko Hyo cloud system | High | ✅ Done |
| IND-A04 | SuperTrend | Trend | ATR-based trend following | Medium | ✅ Done |
| IND-A05 | Parabolic SAR | Trend | Stop and Reverse trend system | Medium | ✅ Done |
| IND-A06 | McGinley Dynamic | Trend | Self-adjusting trend MA | Medium | ✅ Done |
| IND-A07 | Gator Oscillator | Trend | Alligator momentum histogram | Medium | ✅ Done |

## Priority 1: High-Value Additions

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-001 | ROC | Oscillator | Rate of Change - momentum as percentage | Low | ✅ Done |
| IND-002 | Momentum | Oscillator | Simple price change over N periods | Low | ✅ Done |
| IND-003 | Aroon | Trend | Aroon Up/Down/Oscillator - trend strength | Medium | ✅ Done |
| IND-004 | VWMA | Moving Average | Volume Weighted Moving Average | Low | ✅ Done |
| IND-005 | A/D Line | Volume | Accumulation/Distribution Line | Medium | ✅ Done |
| IND-006 | Force Index | Volume | Price × Volume momentum | Low | ✅ Done |

## Priority 2: Professional Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-007 | Awesome Oscillator | Oscillator | Bill Williams - 34/5 SMA difference | Low | ✅ Done |
| IND-008 | Accelerator Oscillator | Oscillator | Bill Williams - AO momentum | Low | ✅ Done |
| IND-009 | Gator Oscillator | Oscillator | Bill Williams - Alligator histogram | Medium | ✅ Done |
| IND-010 | Alligator | Trend | Bill Williams - 3 smoothed MAs | Medium | ✅ Done |
| IND-011 | Fractals | Pattern | Bill Williams - swing high/low detection | Medium | ✅ Done |
| IND-012 | Vortex Indicator | Trend | VI+ and VI- trend direction | Medium | ✅ Done |
| IND-013 | Chande Momentum | Oscillator | CMO - momentum oscillator | Low | ✅ Done |
| IND-014 | DeMarker | Oscillator | Exhaustion detection | Medium | ✅ Done |

## Priority 3: Volatility & Statistical

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-015 | Historical Volatility | Volatility | Annualized std dev of log returns | Low | ✅ Done |
| IND-016 | Chaikin Volatility | Volatility | EMA of high-low range change | Medium | ✅ Done |
| IND-017 | Mass Index | Volatility | Range expansion for reversals | Medium | ✅ Done |
| IND-018 | Standard Deviation | Statistical | Rolling std dev | Low | ✅ Done |
| IND-019 | Variance | Statistical | Rolling variance | Low | ✅ Done |
| IND-020 | Z-Score | Statistical | Standardized distance from mean | Low | ✅ Done |

## Priority 4: Additional MAs & Smoothing

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-021 | SMMA | Moving Average | Smoothed Moving Average (Wilder's) | Low | ✅ Done |
| IND-022 | ALMA | Moving Average | Arnaud Legoux MA - Gaussian weighted | Medium | ✅ Done |
| IND-023 | FRAMA | Moving Average | Fractal Adaptive MA | High | ✅ Done |
| IND-024 | VIDYA | Moving Average | Variable Index Dynamic Average | Medium | ✅ Done |
| IND-025 | T3 | Moving Average | Tillson T3 - smooth with low lag | Medium | ✅ Done |
| IND-026 | McGinley Dynamic | Moving Average | Self-adjusting MA | Medium | ✅ Done |

## Priority 5: Advanced & Composite

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-027 | KST | Oscillator | Know Sure Thing - weighted ROC | Medium | ✅ Done |
| IND-028 | PPO | Oscillator | Percentage Price Oscillator | Low | ✅ Done |
| IND-029 | DPO | Trend | Detrended Price Oscillator | Low | ✅ Done |
| IND-030 | Coppock Curve | Trend | Long-term momentum | Medium | ✅ Done |
| IND-031 | Balance of Power | Volume | (Close-Open)/(High-Low) | Low | ✅ Done |
| IND-032 | Ease of Movement | Volume | Price/volume relationship | Medium | ✅ Done |
| IND-033 | VROC | Volume | Volume Rate of Change | Low | ✅ Done |
| IND-034 | Klinger Oscillator | Volume | Volume-based trend | High | ✅ Done |
| IND-035 | Choppiness Index | Volatility | Trending vs ranging market | Medium | ✅ Done |
| IND-036 | RVI | Oscillator | Relative Vigor Index | Medium | ✅ Done |

## Priority 6: RSI Variants & Momentum

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-037 | Stochastic RSI | Oscillator | Stochastic of RSI - more sensitive | Medium | ✅ Done |
| IND-038 | Connors RSI | Oscillator | Composite: RSI + Streak RSI + ROC Percentile | High | ✅ Done |
| IND-039 | TSI | Oscillator | True Strength Index - double-smoothed momentum | Medium | ✅ Done |
| IND-040 | SMI | Oscillator | Stochastic Momentum Index | Medium | ✅ Done |
| IND-041 | RMI | Oscillator | Relative Momentum Index - RSI with momentum | Medium | ✅ Done |
| IND-042 | Fisher Transform | Oscillator | Normalize prices to Gaussian distribution | Medium | ✅ Done |
| IND-043 | Inverse Fisher Transform | Oscillator | Smooth oscillator signals | Medium | ✅ Done |

## Priority 7: Ehlers Indicators (DSP-Based)

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-044 | MESA | DSP | MESA Adaptive Moving Average | High | ✅ Done |
| IND-045 | MAMA | DSP | MESA Adaptive Moving Average with FAMA | High | ✅ Done |
| IND-046 | Sine Wave | Oscillator | Ehlers cycle detection | High | ✅ Done |
| IND-047 | Hilbert Transform | Filter | Phase/amplitude extraction | High | ✅ Done |
| IND-048 | Cyber Cycle | Oscillator | Ehlers bandpass cycle | High | ✅ Done |
| IND-049 | CG Oscillator | Oscillator | Center of Gravity | Medium | ✅ Done |
| IND-050 | Laguerre RSI | Oscillator | Ehlers 4-element Laguerre filter RSI | High | ✅ Done |
| IND-051 | Roofing Filter | Filter | Ehlers highpass + supersmoother | High | ✅ Done |

## Priority 8: Pattern & Structure

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-052 | Zig Zag | Pattern | Swing high/low connector | Medium | ✅ Done |
| IND-053 | Heikin Ashi | Transform | Smoothed candlesticks | Low | ✅ Done |
| IND-054 | Linear Regression | Statistical | Slope, intercept, R-squared | Medium | ✅ Done |
| IND-055 | Standard Error Bands | Volatility | Regression-based bands | Medium | ✅ Done |
| IND-056 | Price Channel | Volatility | Highest high / lowest low bands | Low | ✅ Done |
| IND-057 | Envelope | Volatility | MA with percentage bands | Low | ✅ Done |
| IND-058 | Chandelier Exit | Volatility | ATR-based trailing stop | Medium | ✅ Done |
| IND-059 | Anchored VWAP | Volume | VWAP from specific anchor point | Medium | ✅ Done |

## Priority 9: Market Regime & Composite

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-060 | TTM Squeeze | Composite | Bollinger inside Keltner detection | Medium | ✅ Done |
| IND-061 | Elder Impulse | Composite | EMA + MACD histogram signal | Medium | ✅ Done |
| IND-062 | Schaff Trend Cycle | Oscillator | MACD with stochastic smoothing | High | ✅ Done |
| IND-063 | Trend Intensity Index | Trend | Deviation-based trend strength | Medium | ✅ Done |
| IND-064 | Trend Detection Index | Trend | Composite trend detector | Medium | ✅ Done |
| IND-065 | Ulcer Index | Volatility | Downside risk measure | Medium | ✅ Done |
| IND-066 | Qstick | Oscillator | Open-close average | Low | ✅ Done |
| IND-067 | TRIN / Arms Index | Breadth | Advance/decline volume ratio | Medium | ✅ Done |
| IND-068 | McClellan Oscillator | Breadth | Breadth momentum | Medium | ✅ Done |

## Priority 10: Volume Advanced

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-069 | Volume Profile | Volume | Price-volume distribution | High | ✅ Done |
| IND-070 | Market Profile | Volume | TPO-based value area | High | ✅ Done |
| IND-071 | Volume Oscillator | Volume | Fast/slow volume MA difference | Low | ✅ Done |
| IND-072 | PVT | Volume | Price-Volume Trend | Low | ✅ Done |
| IND-073 | NVI | Volume | Negative Volume Index | Medium | ✅ Done |
| IND-074 | PVI | Volume | Positive Volume Index | Medium | ✅ Done |
| IND-075 | Williams A/D | Volume | Williams Accumulation/Distribution | Medium | ✅ Done |
| IND-076 | Twiggs Money Flow | Volume | Smoothed CMF variant | Medium | ✅ Done |

## Priority 11: Additional Specialized

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-077 | Rainbow MA | Trend | 10 recursive SMAs visualization | Low | ✅ Done |
| IND-078 | Rainbow Oscillator | Oscillator | Rainbow deviation | Medium | ✅ Done |
| IND-079 | Projection Oscillator | Oscillator | Linear regression oscillator | Medium | ✅ Done |
| IND-080 | Price Oscillator | Oscillator | MA difference (absolute) | Low | ✅ Done |
| IND-081 | Percentage Bands | Volatility | Fixed percentage from MA | Low | ✅ Done |
| IND-082 | Darvas Box | Pattern | Breakout boxes | Medium | ✅ Done |
| IND-083 | Keltner Original | Volatility | Original 10-day ATR version | Low | ✅ Done |
| IND-084 | Commodity Selection Index | Composite | ADXR + ATR composite | Medium | ✅ Done |

## Priority 12: Crypto & Modern

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-085 | NVT Ratio | Crypto | Network Value to Transactions | Low | ✅ Done |
| IND-086 | MVRV Ratio | Crypto | Market Value to Realized Value | Medium | ✅ Done |
| IND-087 | SOPR | Crypto | Spent Output Profit Ratio | Medium | ✅ Done |
| IND-088 | Hash Ribbons | Crypto | Mining difficulty MA crossover | Medium | ✅ Done |
| IND-089 | Fear & Greed Components | Sentiment | Volatility + momentum + volume composite | High | ✅ Done |
| IND-090 | Squeeze Momentum | Composite | LazyBear's squeeze with momentum | Medium | ✅ Done |

## Priority 13: DeMark Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-091 | TD Sequential | Pattern | DeMark 9-13 count exhaustion | High | ✅ Done |
| IND-092 | TD Combo | Pattern | Alternative DeMark count | High | ✅ Done |
| IND-093 | TD Setup | Pattern | Setup phase of Sequential | Medium | ✅ Done |
| IND-094 | TD Countdown | Pattern | Countdown phase of Sequential | Medium | ✅ Done |
| IND-095 | TD REI | Oscillator | Range Expansion Index | Medium | ✅ Done |
| IND-096 | TD POQ | Oscillator | Price Oscillator Qualifier | Medium | ✅ Done |
| IND-097 | TD Pressure | Oscillator | Buying/selling pressure | Medium | ✅ Done |
| IND-098 | TD D-Wave | Pattern | DeMark wave analysis | High | ✅ Done |

## Priority 14: Volatility Models

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-099 | Parkinson Volatility | Volatility | High-low range estimator | Low | ✅ Done |
| IND-100 | Garman-Klass Volatility | Volatility | OHLC volatility estimator | Medium | ✅ Done |
| IND-101 | Rogers-Satchell Volatility | Volatility | Drift-independent estimator | Medium | ✅ Done |
| IND-102 | Yang-Zhang Volatility | Volatility | Overnight + open-close + HL | Medium | ✅ Done |
| IND-103 | Realized Volatility | Volatility | Sum of squared returns | Low | ✅ Done |
| IND-104 | Close-to-Close Volatility | Volatility | Simple log return std dev | Low | ✅ Done |
| IND-105 | Volatility Cone | Volatility | Percentile bands over time | Medium | ✅ Done |
| IND-106 | Normalized ATR | Volatility | ATR as percentage of price | Low | ✅ Done |

## Priority 15: Risk Metrics

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-107 | Sharpe Ratio | Risk | Risk-adjusted return | Low | ✅ Done |
| IND-108 | Sortino Ratio | Risk | Downside risk-adjusted return | Medium | ✅ Done |
| IND-109 | Calmar Ratio | Risk | Return / max drawdown | Medium | ✅ Done |
| IND-110 | Information Ratio | Risk | Active return / tracking error | Medium | ✅ Done |
| IND-111 | Maximum Drawdown | Risk | Peak-to-trough decline | Low | ✅ Done |
| IND-112 | Value at Risk | Risk | VaR at confidence level | Medium | ✅ Done |
| IND-113 | Conditional VaR | Risk | Expected shortfall / CVaR | Medium | ✅ Done |
| IND-114 | Beta | Risk | Market sensitivity | Medium | ✅ Done |
| IND-115 | Alpha | Risk | Risk-adjusted excess return | Medium | ✅ Done |
| IND-116 | Treynor Ratio | Risk | Return per unit of beta | Medium | ✅ Done |

## Priority 16: Intermarket & Relative

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-117 | Relative Strength (Comparative) | Intermarket | Asset vs benchmark ratio | Low | ✅ Done |
| IND-118 | Correlation | Intermarket | Rolling correlation coefficient | Medium | ✅ Done |
| IND-119 | Cointegration Score | Intermarket | Pair trading stationarity test | High | ✅ Done |
| IND-120 | Spread | Intermarket | Price difference between assets | Low | ✅ Done |
| IND-121 | Ratio | Intermarket | Price ratio between assets | Low | ✅ Done |
| IND-122 | Z-Score Spread | Intermarket | Normalized spread for mean reversion | Medium | ✅ Done |
| IND-123 | Currency Strength | Intermarket | Multi-pair currency ranking | High | ✅ Done |
| IND-124 | Sector Rotation | Intermarket | Relative performance ranking | Medium | ✅ Done |

## Priority 17: Market Breadth Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-125 | Advance/Decline Line | Breadth | Cumulative A/D difference | Low | ✅ Done |
| IND-126 | McClellan Summation | Breadth | Cumulative McClellan Oscillator | Medium | ✅ Done |
| IND-127 | Breadth Thrust | Breadth | 10-day A/D thrust | Medium | ✅ Done |
| IND-128 | High-Low Index | Breadth | New highs vs new lows | Medium | ✅ Done |
| IND-129 | Percent Above MA | Breadth | % of stocks above 200 SMA | Medium | ✅ Done |
| IND-130 | Bullish Percent Index | Breadth | % of stocks with P&F buy signals | High | ✅ Done |
| IND-131 | Cumulative Volume Index | Breadth | Volume-weighted A/D | Medium | ✅ Done |
| IND-132 | New Highs/New Lows | Breadth | Daily NH-NL difference | Low | ✅ Done |

## Priority 18: Swing & Price Structure

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-133 | Swing Index | Swing | Wilder's swing index | Medium | ✅ Done |
| IND-134 | Accumulative Swing Index | Swing | Cumulative swing index | Medium | ✅ Done |
| IND-135 | Gann Swing | Swing | Gann-based swing detection | Medium | ✅ Done |
| IND-136 | Market Structure | Pattern | Higher highs/lower lows detection | Medium | ✅ Done |
| IND-137 | Order Blocks | Pattern | Institutional supply/demand zones | High | ✅ Done |
| IND-138 | Fair Value Gap | Pattern | Imbalance detection | Medium | ✅ Done |
| IND-139 | Liquidity Voids | Pattern | Unfilled price gaps | Medium | ✅ Done |
| IND-140 | Break of Structure | Pattern | Trend change detection | Medium | ✅ Done |

## Priority 19: Additional Bands & Channels

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-141 | STARC Bands | Volatility | Stoller Average Range Channel | Medium | ✅ Done |
| IND-142 | Tirone Levels | Support/Resistance | 1/3 - 2/3 retracement levels | Low | ✅ Done |
| IND-143 | Acceleration Bands | Volatility | Price Headley's bands | Medium | ✅ Done |
| IND-144 | Moving Average Envelope | Volatility | Percentage bands around MA | Low | ✅ Done |
| IND-145 | Projection Bands | Volatility | Linear regression bands | Medium | ✅ Done |
| IND-146 | Fractal Chaos Bands | Volatility | Williams fractal-based bands | Medium | ✅ Done |
| IND-147 | High-Low Bands | Volatility | Highest/lowest with MA | Low | ✅ Done |
| IND-148 | Prime Number Bands | Volatility | Prime number-based levels | Medium | ✅ Done |

## Priority 20: Additional Oscillators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-149 | PMO | Oscillator | Price Momentum Oscillator (DecisionPoint) | Medium | ✅ Done |
| IND-150 | Special K | Oscillator | Pring's summed ROC oscillator | Medium | ✅ Done |
| IND-151 | Disparity Index | Oscillator | Distance from MA as percentage | Low | ✅ Done |
| IND-152 | Pretty Good Oscillator | Oscillator | ATR-normalized distance from SMA | Low | ✅ Done |
| IND-153 | Absolute Price Oscillator | Oscillator | Fast MA - Slow MA (absolute) | Low | ✅ Done |
| IND-154 | Ergodic Oscillator | Oscillator | True Strength Index signal line | Medium | ✅ Done |
| IND-155 | Polarized Fractal Efficiency | Oscillator | Trend efficiency measure | Medium | ✅ Done |
| IND-156 | Intraday Momentum Index | Oscillator | Intraday RSI variant | Medium | ✅ Done |
| IND-157 | Relative Volatility Index | Oscillator | Volatility direction oscillator | Medium | ✅ Done |
| IND-158 | Double Stochastic | Oscillator | Stochastic of stochastic | Medium | ✅ Done |

## Priority 21: Volume Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-159 | TWAP | Volume | Time Weighted Average Price | Low | ✅ Done |
| IND-160 | Trade Volume Index | Volume | Tick-based accumulation | Medium | ✅ Done |
| IND-161 | Volume Zone Oscillator | Volume | Volume flow direction | Medium | ✅ Done |
| IND-162 | Net Volume | Volume | Up volume - down volume | Low | ✅ Done |
| IND-163 | Chaikin Oscillator | Volume | MACD of A/D line | Medium | ✅ Done |
| IND-164 | Elder's Thermometer | Volume | Price volatility measure | Medium | ✅ Done |
| IND-165 | Volume Price Confirmation | Volume | Price/volume trend agreement | Medium | ✅ Done |
| IND-166 | Volume Weighted MACD | Volume | VWMA-based MACD | Medium | ✅ Done |

## Priority 22: Moving Averages Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-167 | GMMA | Moving Average | Guppy Multiple MA (12 MAs) | Medium | ✅ Done |
| IND-168 | Rainbow MA | Moving Average | 10 recursive SMAs | Low | ✅ Done |
| IND-169 | Jurik MA | Moving Average | JMA approximation (low lag) | High | ✅ Done |
| IND-170 | Ehlers Supersmoother | Moving Average | 2-pole Butterworth filter | Medium | ✅ Done |
| IND-171 | Ehlers Decycler | Moving Average | High-pass complement | Medium | ✅ Done |
| IND-172 | Triangular MA | Moving Average | Double-smoothed SMA | Low | ✅ Done |
| IND-173 | LWMA | Moving Average | Linear Weighted MA (same as WMA) | Low | ✅ Done |
| IND-174 | Sine Weighted MA | Moving Average | Sine wave weighted | Medium | ✅ Done |
| IND-175 | Elastic Volume Weighted MA | Moving Average | Volume-elastic smoothing | High | ✅ Done |

## Priority 23: Elder Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-176 | Elder's SafeZone Stop | Stop | Directional stop loss | Medium | ✅ Done |
| IND-177 | Elder's AutoEnvelope | Volatility | Adaptive envelope | Medium | ✅ Done |
| IND-178 | Elder's Market Thermometer | Volatility | Intraday volatility | Medium | ✅ Done |
| IND-179 | Elder's Bull/Bear Power | Oscillator | Enhanced version | Medium | ✅ Done |
| IND-180 | Elder's Triple Screen | Composite | Multi-timeframe system | High | ✅ Done |

## Priority 24: Kase Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-181 | Kase Peak Oscillator | Oscillator | Volatility-adjusted momentum | High | ✅ Done |
| IND-182 | Kase Dev Stops | Stop | Deviation-based stops | High | ✅ Done |
| IND-183 | Kase Permission Stochastic | Oscillator | Modified stochastic | High | ✅ Done |
| IND-184 | Kase CD | Trend | Convergence/Divergence | High | ✅ Done |
| IND-185 | Kase Bars | Transform | Volatility-normalized bars | High | ✅ Done |

## Priority 25: Miscellaneous Advanced

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-186 | Random Walk Index | Trend | Trend vs random walk | Medium | ✅ Done |
| IND-187 | Vertical Horizontal Filter | Trend | Trending vs ranging | Medium | ✅ Done |
| IND-188 | Kaufman Efficiency Ratio | Trend | Price efficiency measure | Low | ✅ Done |
| IND-189 | Fractal Dimension | Statistical | Market complexity measure | High | ✅ Done |
| IND-190 | Hurst Exponent | Statistical | Long-term memory measure | High | ✅ Done |
| IND-191 | Detrended Fluctuation Analysis | Statistical | DFA for trending behavior | High | ✅ Done |
| IND-192 | Entropy | Statistical | Market randomness measure | High | ✅ Done |
| IND-193 | Autocorrelation | Statistical | Serial correlation | Medium | ✅ Done |
| IND-194 | Tick Index | Breadth | NYSE tick reading | Low | ✅ Done |
| IND-195 | Put/Call Ratio | Sentiment | Options sentiment | Low | ✅ Done |

## Priority 26: Candlestick Patterns

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-196 | Doji | Candlestick | Indecision pattern | Low | ✅ Done |
| IND-197 | Hammer / Hanging Man | Candlestick | Reversal single candle | Low | ✅ Done |
| IND-198 | Engulfing | Candlestick | Bullish/bearish engulfing | Low | ✅ Done |
| IND-199 | Harami | Candlestick | Inside bar pattern | Low | ✅ Done |
| IND-200 | Morning Star / Evening Star | Candlestick | 3-candle reversal | Medium | ✅ Done |
| IND-201 | Three White Soldiers / Black Crows | Candlestick | 3-candle continuation | Medium | ✅ Done |
| IND-202 | Piercing / Dark Cloud Cover | Candlestick | 2-candle reversal | Low | ✅ Done |
| IND-203 | Spinning Top | Candlestick | Small body, long wicks | Low | ✅ Done |
| IND-204 | Marubozu | Candlestick | Full body, no wicks | Low | ✅ Done |
| IND-205 | Tweezer Top / Bottom | Candlestick | Equal highs/lows | Low | ✅ Done |
| IND-206 | Three Inside Up/Down | Candlestick | Harami confirmation | Medium | ✅ Done |
| IND-207 | Three Outside Up/Down | Candlestick | Engulfing confirmation | Medium | ✅ Done |
| IND-208 | Abandoned Baby | Candlestick | Gapped reversal | Medium | ✅ Done |
| IND-209 | Shooting Star / Inverted Hammer | Candlestick | Long upper wick | Low | ✅ Done |
| IND-210 | Belt Hold | Candlestick | Gap with strong body | Low | ✅ Done |
| IND-211 | Kicking | Candlestick | Opposing Marubozu gap | Medium | ✅ Done |
| IND-212 | Three Line Strike | Candlestick | 4-candle pattern | Medium | ✅ Done |
| IND-213 | Tasuki Gap | Candlestick | Continuation gap | Medium | ✅ Done |
| IND-214 | Rising / Falling Three Methods | Candlestick | 5-candle continuation | Medium | ✅ Done |
| IND-215 | Counterattack | Candlestick | Opposing close at same level | Medium | |

## Priority 27: Order Flow & Microstructure

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-216 | Delta | Order Flow | Buy volume - sell volume | Medium | ✅ Done |
| IND-217 | Cumulative Delta | Order Flow | Running delta total | Medium | ✅ Done |
| IND-218 | Delta Divergence | Order Flow | Price vs delta divergence | Medium | ✅ Done |
| IND-219 | Volume Delta Percentage | Order Flow | Delta as % of volume | Low | ✅ Done |
| IND-220 | Imbalance Detector | Order Flow | Bid/ask volume imbalance | High | ✅ Done |
| IND-221 | Absorption Detector | Order Flow | Large orders absorbed | High | ✅ Done |
| IND-222 | POC (Point of Control) | Order Flow | Highest volume price | Medium | ✅ Done |
| IND-223 | Value Area | Order Flow | 70% volume range | Medium | ✅ Done |
| IND-224 | VWAP Bands | Order Flow | Standard deviation VWAP bands | Medium | ✅ Done |
| IND-225 | Bid-Ask Spread | Microstructure | Spread analysis | Low | |
| IND-226 | Order Book Imbalance | Microstructure | Bid/ask depth ratio | Medium | |
| IND-227 | Trade Flow Imbalance | Microstructure | Aggressive buyer/seller ratio | Medium | |

## Priority 28: Wyckoff Method

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-228 | Wyckoff Wave | Wyckoff | Composite market indicator | High | ✅ Done |
| IND-229 | Optimism-Pessimism Index | Wyckoff | Volume-based sentiment | Medium | ✅ Done |
| IND-230 | Force Index (Wyckoff) | Wyckoff | Price × volume force | Medium | ✅ Done |
| IND-231 | Technometer | Wyckoff | Overbought/oversold | Medium | ✅ Done |
| IND-232 | Selling Climax Detector | Wyckoff | High volume reversal | High | ✅ Done |
| IND-233 | Automatic Rally/Reaction | Wyckoff | Post-climax bounce | High | ✅ Done |
| IND-234 | Spring / Upthrust | Wyckoff | False breakout detection | High | ✅ Done |
| IND-235 | Sign of Strength / Weakness | Wyckoff | Trend quality | High | ✅ Done |

## Priority 29: Seasonality & Time

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-236 | Day of Week Effect | Seasonality | Weekday performance | Low | ✅ Done |
| IND-237 | Month of Year Effect | Seasonality | Monthly performance | Low | ✅ Done |
| IND-238 | Turn of Month | Seasonality | Month boundary effect | Low | ✅ Done |
| IND-239 | Holiday Effect | Seasonality | Pre/post holiday returns | Medium | ✅ Done |
| IND-240 | Quarterly Effect | Seasonality | Quarter-end flows | Low | ✅ Done |
| IND-241 | Options Expiration Effect | Seasonality | OpEx week behavior | Medium | ✅ Done |
| IND-242 | Lunar Cycle | Seasonality | Moon phase correlation | Low | ✅ Done |
| IND-243 | Intraday Seasonality | Seasonality | Hour-of-day patterns | Medium | ✅ Done |
| IND-244 | Rolling Seasonality | Seasonality | Multi-year seasonal pattern | Medium | ✅ Done |

## Priority 30: Options-Based

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-245 | Implied Volatility | Options | IV from option prices | High | ✅ Done |
| IND-246 | IV Rank | Options | IV percentile over year | Medium | ✅ Done |
| IND-247 | IV Percentile | Options | % of days IV was lower | Medium | ✅ Done |
| IND-248 | Volatility Skew | Options | Put/call IV difference | Medium | ✅ Done |
| IND-249 | Term Structure | Options | IV across expirations | Medium | ✅ Done |
| IND-250 | Put/Call Open Interest | Options | OI ratio | Low | ✅ Done |
| IND-251 | Max Pain | Options | Price with max option loss | Medium | |
| IND-252 | Gamma Exposure (GEX) | Options | Dealer gamma hedging level | High | |
| IND-253 | Delta Exposure (DEX) | Options | Net market delta | High | |
| IND-254 | Volatility Risk Premium | Options | IV - RV spread | Medium | |

## Priority 31: Factor Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-255 | Momentum Factor | Factor | 12-1 month return | Low | |
| IND-256 | Value Factor | Factor | Book-to-market ratio | Medium | |
| IND-257 | Size Factor | Factor | Market cap ranking | Low | ✅ Done |
| IND-258 | Quality Factor | Factor | ROE, debt, earnings stability | Medium | ✅ Done |
| IND-259 | Low Volatility Factor | Factor | Historical vol ranking | Medium | ✅ Done |
| IND-260 | Dividend Yield Factor | Factor | Yield ranking | Low | ✅ Done |
| IND-261 | Growth Factor | Factor | Earnings/revenue growth | Medium | ✅ Done |
| IND-262 | Liquidity Factor | Factor | Trading volume/turnover | Medium | ✅ Done |
| IND-263 | Reversal Factor | Factor | Short-term mean reversion | Medium | ✅ Done |
| IND-264 | Composite Factor Score | Factor | Multi-factor combination | High | ✅ Done |

## Priority 32: Crypto On-Chain Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-265 | Active Addresses | On-Chain | Daily active addresses | Low | ✅ Done |
| IND-266 | Transaction Count | On-Chain | Daily transactions | Low | ✅ Done |
| IND-267 | Transfer Volume | On-Chain | USD volume transferred | Low | ✅ Done |
| IND-268 | Exchange Inflow/Outflow | On-Chain | Exchange balance change | Medium | ✅ Done |
| IND-269 | Whale Transactions | On-Chain | Large transfers count | Medium | ✅ Done |
| IND-270 | HODL Waves | On-Chain | Coin age distribution | High | ✅ Done |
| IND-271 | Realized Cap | On-Chain | Cost basis market cap | Medium | ✅ Done |
| IND-272 | Thermocap Ratio | On-Chain | Market cap / miner revenue | Medium | ✅ Done |
| IND-273 | Puell Multiple | On-Chain | Daily issuance value / 365 MA | Medium | ✅ Done |
| IND-274 | Reserve Risk | On-Chain | HODL confidence vs price | High | ✅ Done |
| IND-275 | Stock-to-Flow | On-Chain | Scarcity model | Medium | ✅ Done |
| IND-276 | Stablecoin Supply Ratio | On-Chain | BTC cap / stablecoin cap | Medium | ✅ Done |

## Priority 33: Sentiment & Alternative Data

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-277 | Social Volume | Sentiment | Social media mentions | Medium | ✅ Done |
| IND-278 | Social Sentiment | Sentiment | Positive/negative ratio | High | ✅ Done |
| IND-279 | News Sentiment | Sentiment | NLP news analysis | High | ✅ Done |
| IND-280 | Google Trends | Sentiment | Search interest | Medium | ✅ Done |
| IND-281 | Reddit/Twitter Activity | Sentiment | Platform-specific metrics | Medium | ✅ Done |
| IND-282 | Commitment of Traders (COT) | Sentiment | Futures positioning | Medium | ✅ Done |
| IND-283 | Smart Money Index | Sentiment | Late day vs early day | Medium | ✅ Done |
| IND-284 | AAII Sentiment | Sentiment | Survey-based sentiment | Low | ✅ Done |
| IND-285 | VIX Term Structure | Sentiment | Contango/backwardation | Medium | ✅ Done |
| IND-286 | Insider Trading Ratio | Sentiment | Buy/sell ratio | Medium | ✅ Done |

## Priority 34: Machine Learning Based

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-287 | SVM Trend Classifier | ML | Support vector classification | High | ✅ Done |
| IND-288 | Random Forest Signal | ML | Ensemble tree prediction | High | |
| IND-289 | LSTM Prediction | ML | Neural network forecast | High | |
| IND-290 | K-Means Regime | ML | Clustering market states | High | |
| IND-291 | Isolation Forest Anomaly | ML | Outlier detection | High | |
| IND-292 | XGBoost Signal | ML | Gradient boosting prediction | High | |
| IND-293 | Autoencoder Anomaly | ML | Reconstruction error | High | ✅ Done |
| IND-294 | HMM Regime | ML | Hidden Markov Model states | High | ✅ Done |
| IND-295 | Reinforcement Learning Signal | ML | RL-based trading signal | High | ✅ Done |

## Priority 35: Fixed Income & Rates

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-296 | Yield Curve Slope | Rates | 10Y - 2Y spread | Low | ✅ Done |
| IND-297 | Yield Curve Curvature | Rates | Butterfly spread | Medium | ✅ Done |
| IND-298 | Real Yield | Rates | Nominal - inflation | Low | ✅ Done |
| IND-299 | Credit Spread | Rates | Corporate - Treasury | Low | ✅ Done |
| IND-300 | TED Spread | Rates | LIBOR - T-Bill | Low | ✅ Done |
| IND-301 | Swap Spread | Rates | Swap rate - Treasury | Low | ✅ Done |
| IND-302 | Duration | Rates | Interest rate sensitivity | Medium | ✅ Done |
| IND-303 | Convexity | Rates | Duration change rate | Medium | ✅ Done |
| IND-304 | Carry | Rates | Roll-down return | Medium | ✅ Done |
| IND-305 | Forward Rate | Rates | Implied future rate | Medium | |

## Priority 36: Forex Specific

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-306 | Currency Strength Index | Forex | Multi-pair strength | High | |
| IND-307 | Interest Rate Differential | Forex | Carry trade indicator | Low | |
| IND-308 | Real Effective Exchange Rate | Forex | Inflation-adjusted | Medium | |
| IND-309 | Purchasing Power Parity | Forex | Fair value estimate | Medium | |
| IND-310 | Forex Volatility Index | Forex | Currency implied vol | Medium | |
| IND-311 | Commitment of Traders (Forex) | Forex | Net positioning | Medium | ✅ Done |
| IND-312 | Central Bank Policy Indicator | Forex | Hawkish/dovish score | High | ✅ Done |
| IND-313 | Cross Rate Arbitrage | Forex | Triangular arbitrage | Medium | ✅ Done |

## Priority 37: Economic Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-314 | LEI (Leading Economic Index) | Economic | Conference Board LEI | Medium | |
| IND-315 | PMI Composite | Economic | Manufacturing + Services | Low | |
| IND-316 | Yield Curve Recession Prob | Economic | Probit model | Medium | |
| IND-317 | Unemployment Claims Trend | Economic | 4-week MA | Low | ✅ Done |
| IND-318 | Housing Starts Trend | Economic | YoY change | Low | ✅ Done |
| IND-319 | Consumer Confidence Delta | Economic | MoM change | Low | ✅ Done |
| IND-320 | ISM New Orders | Economic | Leading component | Low | ✅ Done |
| IND-321 | Real M2 Growth | Economic | Money supply change | Medium | ✅ Done |
| IND-322 | Financial Conditions Index | Economic | Composite stress index | High | ✅ Done |

## Priority 38: Additional Gann & Cycle

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-323 | Gann Fan | Gann | Trend lines at angles | Medium | ✅ Done |
| IND-324 | Gann Square of 9 | Gann | Price/time squaring | High | ✅ Done |
| IND-325 | Gann Hexagon | Gann | Hexagonal chart | High | ✅ Done |
| IND-326 | Planetary Cycles | Gann | Astronomical correlations | High | ✅ Done |
| IND-327 | Fibonacci Time Zones | Cycle | Fib time projections | Medium | ✅ Done |
| IND-328 | Cycle Finder | Cycle | Dominant cycle detection | High | ✅ Done |
| IND-329 | Fourier Transform | Cycle | Frequency decomposition | High | |
| IND-330 | Wavelet Transform | Cycle | Time-frequency analysis | High | |
| IND-331 | Spectral Analysis | Cycle | Power spectrum | High | |
| IND-332 | Hodrick-Prescott Filter | Cycle | Trend-cycle decomposition | Medium | |

## Priority 39: Additional Pattern Recognition

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-333 | Head and Shoulders | Pattern | Classic reversal | High | ✅ Done |
| IND-334 | Double Top / Bottom | Pattern | M/W patterns | Medium | ✅ Done |
| IND-335 | Triple Top / Bottom | Pattern | Extended reversal | Medium | ✅ Done |
| IND-336 | Cup and Handle | Pattern | Continuation pattern | High | ✅ Done |
| IND-337 | Wedge (Rising/Falling) | Pattern | Converging trend lines | Medium | ✅ Done |
| IND-338 | Flag / Pennant | Pattern | Continuation patterns | Medium | ✅ Done |
| IND-339 | Rectangle | Pattern | Range consolidation | Medium | ✅ Done |
| IND-340 | Triangle (Sym/Asc/Desc) | Pattern | Converging patterns | Medium | ✅ Done |
| IND-341 | Channel (Parallel) | Pattern | Trend channel | Medium | ✅ Done |
| IND-342 | Rounding Bottom / Top | Pattern | Saucer pattern | High | ✅ Done |
| IND-343 | Island Reversal | Pattern | Gapped isolation | Medium | ✅ Done |
| IND-344 | Diamond | Pattern | Rare reversal | High | ✅ Done |
| IND-345 | Broadening Formation | Pattern | Megaphone | Medium | ✅ Done |

## Priority 40: Harmonic Patterns

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-346 | Gartley Pattern | Harmonic | XABCD 0.618 retracement | High | |
| IND-347 | Butterfly Pattern | Harmonic | Extended Gartley | High | ✅ Done |
| IND-348 | Bat Pattern | Harmonic | 0.886 retracement | High | ✅ Done |
| IND-349 | Crab Pattern | Harmonic | 1.618 extension | High | ✅ Done |
| IND-350 | Shark Pattern | Harmonic | 0.886/1.13 pattern | High | ✅ Done |
| IND-351 | Cypher Pattern | Harmonic | Darren Oglesbee pattern | High | ✅ Done |
| IND-352 | ABCD Pattern | Harmonic | Simple harmonic | Medium | ✅ Done |
| IND-353 | Three Drives | Harmonic | Symmetrical drives | High | |
| IND-354 | 5-0 Pattern | Harmonic | Trend reversal | High | |
| IND-355 | Alternate Bat | Harmonic | 1.13 extension variant | High | |

## Priority 41: Elliott Wave

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-356 | Wave Degree Detector | Elliott | Identify wave degree | High | |
| IND-357 | Impulse Wave | Elliott | 5-wave motive pattern | High | |
| IND-358 | Corrective Wave | Elliott | 3-wave correction | High | |
| IND-359 | Wave Count Validator | Elliott | Rule validation | High | |
| IND-360 | Fibonacci Wave Targets | Elliott | Wave projections | High | |
| IND-361 | Diagonal Pattern | Elliott | Ending/leading diagonal | High | |
| IND-362 | Triangle (Elliott) | Elliott | Contracting/expanding | High | |
| IND-363 | Flat Correction | Elliott | 3-3-5 pattern | High | |
| IND-364 | Zigzag Correction | Elliott | 5-3-5 pattern | High | |
| IND-365 | Complex Correction | Elliott | WXY, WXYXZ patterns | High | |

## Priority 42: Volume Spread Analysis (VSA)

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-366 | No Demand | VSA | Low volume up bar | Medium | |
| IND-367 | No Supply | VSA | Low volume down bar | Medium | |
| IND-368 | Stopping Volume | VSA | High volume reversal | Medium | |
| IND-369 | Climactic Action | VSA | Extreme volume bar | Medium | |
| IND-370 | Test | VSA | Low volume test of support | Medium | |
| IND-371 | Upthrust (VSA) | VSA | False breakout up | Medium | |
| IND-372 | Shakeout | VSA | False breakout down | Medium | |
| IND-373 | Effort vs Result | VSA | Volume/price divergence | High | |
| IND-374 | Absorption Volume | VSA | Hidden accumulation | High | |
| IND-375 | Professional Activity | VSA | Smart money detection | High | |

## Priority 43: Session & Time-Based

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-376 | Asian Session Range | Session | Tokyo session H/L | Low | |
| IND-377 | London Session Range | Session | London session H/L | Low | ✅ Done |
| IND-378 | NY Session Range | Session | New York session H/L | Low | ✅ Done |
| IND-379 | Opening Range Breakout | Session | First N minutes range | Medium | |
| IND-380 | Initial Balance | Session | First hour range | Medium | |
| IND-381 | Power Hour | Session | Last hour activity | Medium | |
| IND-382 | Kill Zones | Session | High-activity periods | Medium | |
| IND-383 | Session VWAP | Session | Per-session VWAP | Medium | |
| IND-384 | Overnight Range | Session | Globex session range | Low | |
| IND-385 | Gap Analysis | Session | Open vs prior close | Low | |

## Priority 44: Fibonacci Extensions

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-386 | Fibonacci Extension | Fibonacci | 1.272, 1.618, 2.618 levels | Low | |
| IND-387 | Fibonacci Expansion | Fibonacci | ABC projection | Medium | |
| IND-388 | Fibonacci Arcs | Fibonacci | Curved support/resistance | Medium | |
| IND-389 | Fibonacci Fans | Fibonacci | Angled trend lines | Medium | ✅ Done |
| IND-390 | Fibonacci Channels | Fibonacci | Parallel channel levels | Medium | ✅ Done |
| IND-391 | Fibonacci Clusters | Fibonacci | Multi-swing confluence | High | ✅ Done |
| IND-392 | Auto Fibonacci | Fibonacci | Automatic swing detection | High | ✅ Done |
| IND-393 | Fibonacci Speed Resistance | Fibonacci | Time-price grid | Medium | ✅ Done |

## Priority 45: Market Internals

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-394 | NYSE Tick | Internals | Uptick - downtick stocks | Low | ✅ Done |
| IND-395 | NYSE TRIN (Arms) | Internals | A/D ratio / volume ratio | Medium | ✅ Done |
| IND-396 | VIX Fix | Internals | Synthetic VIX from price | Medium | ✅ Done |
| IND-397 | Cumulative Tick | Internals | Running tick total | Low | ✅ Done |
| IND-398 | TICK Extreme | Internals | Extreme readings | Low | ✅ Done |
| IND-399 | ADD (Advance/Decline) | Internals | Real-time A/D | Low | ✅ Done |
| IND-400 | UVOL/DVOL Ratio | Internals | Up/down volume ratio | Low | ✅ Done |
| IND-401 | Sector Rotation Model | Internals | Business cycle position | High | ✅ Done |
| IND-402 | Risk-On/Risk-Off | Internals | Cross-asset risk measure | High | ✅ Done |
| IND-403 | Intermarket Divergence | Internals | Asset class divergence | Medium | ✅ Done |

## Priority 46: Tail Risk & Extreme Events

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-404 | Skewness | Tail Risk | Distribution asymmetry | Medium | ✅ Done |
| IND-405 | Kurtosis | Tail Risk | Distribution tails | Medium | ✅ Done |
| IND-406 | Expected Shortfall | Tail Risk | CVaR / tail mean | Medium | ✅ Done |
| IND-407 | Tail Dependence | Tail Risk | Copula tail measure | High | ✅ Done |
| IND-408 | Jump Detection | Tail Risk | Price jump identifier | High | ✅ Done |
| IND-409 | Extreme Value Index | Tail Risk | GEV/GPD estimation | High | ✅ Done |
| IND-410 | Black Swan Index | Tail Risk | Tail event probability | High | ✅ Done |
| IND-411 | Crisis Alpha | Tail Risk | Performance in crashes | High | ✅ Done |
| IND-412 | Stress Index | Tail Risk | Market stress composite | High | ✅ Done |

## Priority 47: Commodities Specific

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-413 | Contango/Backwardation | Commodity | Futures curve shape | Medium | ✅ Done |
| IND-414 | Roll Yield | Commodity | Calendar spread return | Medium | ✅ Done |
| IND-415 | Convenience Yield | Commodity | Spot premium | Medium | ✅ Done |
| IND-416 | Inventory Levels | Commodity | Storage data | Low | ✅ Done |
| IND-417 | Crack Spread | Commodity | Refining margin | Low | ✅ Done |
| IND-418 | Crush Spread | Commodity | Soybean processing margin | Low | ✅ Done |
| IND-419 | Spark Spread | Commodity | Power generation margin | Low | |
| IND-420 | Freight Rates | Commodity | Baltic Dry Index | Low | |
| IND-421 | Seasonal Commodity | Commodity | Crop cycle patterns | Medium | |
| IND-422 | Weather Impact | Commodity | Temperature/precip deviation | Medium | |

## Priority 48: Credit & Default Risk

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-423 | CDS Spread | Credit | Credit default swap spread | Low | |
| IND-424 | Z-Spread | Credit | Zero-volatility spread | Medium | |
| IND-425 | OAS | Credit | Option-adjusted spread | High | |
| IND-426 | Default Probability | Credit | Merton model PD | High | |
| IND-427 | Distance to Default | Credit | DD measure | High | |
| IND-428 | Credit Rating Migration | Credit | Rating transition prob | High | |
| IND-429 | Recovery Rate Implied | Credit | From CDS/bond prices | High | |
| IND-430 | Sovereign CDS | Credit | Country default risk | Medium | |
| IND-431 | Credit Curve Slope | Credit | Short vs long CDS | Medium | |
| IND-432 | iTraxx/CDX Index | Credit | Credit index level | Low | |

## Priority 49: ETF Specific

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-433 | NAV Premium/Discount | ETF | Price vs NAV | Low | |
| IND-434 | Creation/Redemption Flow | ETF | Share creation activity | Medium | |
| IND-435 | Tracking Error | ETF | Deviation from benchmark | Medium | |
| IND-436 | Implied Liquidity | ETF | Underlying liquidity | Medium | |
| IND-437 | Authorized Participant Activity | ETF | AP arbitrage | High | |
| IND-438 | ETF Short Interest | ETF | Shares sold short | Low | |
| IND-439 | Sector ETF Rotation | ETF | Sector flow analysis | Medium | |
| IND-440 | Leveraged ETF Decay | ETF | Volatility drag | Medium | |

## Priority 50: Dispersion & Correlation Trading

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-441 | Implied Correlation | Dispersion | Index vs single stock IV | High | |
| IND-442 | Realized Correlation | Dispersion | Actual correlation | Medium | |
| IND-443 | Correlation Risk Premium | Dispersion | Implied - realized | High | |
| IND-444 | Dispersion Trade Signal | Dispersion | Entry/exit signals | High | |
| IND-445 | Correlation Regime | Dispersion | High/low correlation state | High | |
| IND-446 | Average Pairwise Correlation | Dispersion | Mean stock correlation | Medium | |
| IND-447 | Correlation Breakdown | Dispersion | Correlation instability | High | |
| IND-448 | Sector Correlation Matrix | Dispersion | Cross-sector correlation | Medium | |

## Priority 51: Earnings & Fundamental Events

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-449 | Earnings Surprise | Fundamental | Actual vs estimate | Low | |
| IND-450 | Post-Earnings Drift | Fundamental | PEAD measure | Medium | |
| IND-451 | Earnings Revision | Fundamental | Analyst estimate change | Low | |
| IND-452 | Earnings Quality | Fundamental | Accruals, persistence | Medium | |
| IND-453 | Revenue Surprise | Fundamental | Revenue beat/miss | Low | |
| IND-454 | Guidance Impact | Fundamental | Forward guidance effect | Medium | |
| IND-455 | Earnings Momentum | Fundamental | Sequential growth | Low | |
| IND-456 | Pre-Earnings Volatility | Fundamental | IV into earnings | Medium | |
| IND-457 | Earnings Call Sentiment | Fundamental | NLP transcript analysis | High | |

## Priority 52: Valuation Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-458 | P/E Ratio (TTM) | Valuation | Price to earnings | Low | |
| IND-459 | Forward P/E | Valuation | Price to forward earnings | Low | |
| IND-460 | PEG Ratio | Valuation | P/E to growth | Low | |
| IND-461 | P/B Ratio | Valuation | Price to book | Low | |
| IND-462 | P/S Ratio | Valuation | Price to sales | Low | |
| IND-463 | EV/EBITDA | Valuation | Enterprise value multiple | Medium | |
| IND-464 | FCF Yield | Valuation | Free cash flow yield | Medium | |
| IND-465 | Dividend Yield | Valuation | Annual dividend / price | Low | |
| IND-466 | CAPE (Shiller P/E) | Valuation | Cyclically adjusted P/E | Medium | |
| IND-467 | Buffett Indicator | Valuation | Market cap / GDP | Medium | |

## Priority 53: Flow & Positioning

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-468 | Fund Flow | Flow | Mutual fund/ETF flows | Medium | |
| IND-469 | Retail vs Institutional | Flow | Order size analysis | High | |
| IND-470 | Options Positioning | Flow | Net options exposure | Medium | |
| IND-471 | Futures Positioning | Flow | COT decomposition | Medium | |
| IND-472 | Short Interest Ratio | Flow | Days to cover | Low | |
| IND-473 | Securities Lending | Flow | Borrow rate / availability | Medium | |
| IND-474 | Margin Debt | Flow | Total margin balance | Low | |
| IND-475 | Hedge Fund Beta | Flow | HF market exposure | High | |
| IND-476 | CTA Positioning | Flow | Trend-follower exposure | High | |
| IND-477 | Risk Parity Allocation | Flow | Vol-weighted allocation | High | |

## Priority 54: ESG & Sustainability

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-478 | ESG Score | ESG | Composite ESG rating | Medium | |
| IND-479 | Carbon Intensity | ESG | Emissions per revenue | Medium | |
| IND-480 | Green Revenue | ESG | % sustainable revenue | Medium | |
| IND-481 | Governance Score | ESG | Corporate governance | Medium | |
| IND-482 | Controversy Score | ESG | Negative news/events | High | |
| IND-483 | ESG Momentum | ESG | Rating improvement | Medium | |
| IND-484 | Climate VaR | ESG | Climate risk exposure | High | |
| IND-485 | Stranded Asset Risk | ESG | Fossil fuel exposure | High | |

## Priority 55: Real Estate Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-486 | Cap Rate | Real Estate | NOI / Property value | Low | |
| IND-487 | REIT Discount to NAV | Real Estate | Price vs net asset value | Medium | |
| IND-488 | FFO Yield | Real Estate | Funds from operations yield | Medium | |
| IND-489 | Occupancy Rate | Real Estate | Leased space percentage | Low | |
| IND-490 | Same-Store NOI Growth | Real Estate | Comparable property growth | Medium | |
| IND-491 | Property Type Rotation | Real Estate | Sector allocation | Medium | |
| IND-492 | Mortgage Rate Spread | Real Estate | Rate vs Treasury | Low | |
| IND-493 | Housing Affordability | Real Estate | Income to price ratio | Medium | |

## Priority 56: Cross-Asset Signals

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-494 | Stock-Bond Correlation | Cross-Asset | Equity-fixed income corr | Medium | |
| IND-495 | Gold-Dollar Correlation | Cross-Asset | Inverse relationship | Medium | |
| IND-496 | Risk Parity Signal | Cross-Asset | Multi-asset allocation | High | |
| IND-497 | Carry Trade Signal | Cross-Asset | FX carry opportunities | Medium | ✅ Done |
| IND-498 | Global Macro Regime | Cross-Asset | Growth/inflation matrix | High | ✅ Done |
| IND-499 | Flight to Quality | Cross-Asset | Safe haven flows | Medium | ✅ Done |
| IND-500 | Real Asset Allocation | Cross-Asset | Inflation hedge signal | Medium | ✅ Done |

## Priority 57: Liquidity Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-501 | Amihud Illiquidity | Liquidity | Return / volume ratio | Medium | |
| IND-502 | Bid-Ask Bounce | Liquidity | Roll measure | Medium | |
| IND-503 | Kyle's Lambda | Liquidity | Price impact coefficient | High | |
| IND-504 | Turnover Ratio | Liquidity | Volume / shares outstanding | Low | |
| IND-505 | Zero Return Days | Liquidity | Days with no price change | Low | |
| IND-506 | Effective Spread | Liquidity | Trade vs mid price | Medium | |
| IND-507 | Market Depth | Liquidity | Order book volume | Medium | |
| IND-508 | Resilience | Liquidity | Price recovery speed | High | |
| IND-509 | Funding Liquidity | Liquidity | Repo rate / LIBOR-OIS | Medium | |
| IND-510 | Liquidity Mismatch | Liquidity | Asset vs liability liquidity | High | |

## Priority 58: Momentum Variants

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-511 | Cross-Sectional Momentum | Momentum | Relative performance rank | Medium | |
| IND-512 | Time-Series Momentum | Momentum | Own past return signal | Medium | |
| IND-513 | Dual Momentum | Momentum | Absolute + relative | Medium | |
| IND-514 | Momentum Crash Risk | Momentum | Tail risk of momentum | High | |
| IND-515 | Residual Momentum | Momentum | Alpha-adjusted momentum | High | |
| IND-516 | Industry Momentum | Momentum | Sector-relative strength | Medium | |
| IND-517 | 52-Week High Momentum | Momentum | Distance from high | Low | |
| IND-518 | Earnings Momentum | Momentum | Earnings-based signal | Medium | |
| IND-519 | Price Momentum Oscillator | Momentum | Smoothed momentum | Medium | |
| IND-520 | Acceleration | Momentum | Momentum of momentum | Medium | |

## Priority 59: Mean Reversion

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-521 | Ornstein-Uhlenbeck | Mean Rev | OU process parameters | High | |
| IND-522 | Half-Life | Mean Rev | Mean reversion speed | High | |
| IND-523 | Pair Spread | Mean Rev | Cointegrated pair spread | Medium | |
| IND-524 | RSI Mean Reversion | Mean Rev | Extreme RSI signals | Low | |
| IND-525 | Bollinger Mean Reversion | Mean Rev | Band touch signals | Low | |
| IND-526 | Statistical Arbitrage Signal | Mean Rev | Multi-factor residual | High | |
| IND-527 | Sector Rotation (MR) | Mean Rev | Sector reversion | Medium | |
| IND-528 | Contrarian Signal | Mean Rev | Opposite to trend | Medium | |

## Priority 60: Neural Network Specific

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-529 | CNN Pattern Recognition | Neural | Convolutional features | High | |
| IND-530 | Transformer Attention | Neural | Self-attention signal | High | |
| IND-531 | GRU Prediction | Neural | Gated recurrent unit | High | |
| IND-532 | Variational Autoencoder | Neural | Latent space anomaly | High | |
| IND-533 | GAN-Based Regime | Neural | Generative regime detection | High | |
| IND-534 | Neural Embedding | Neural | Learned representation | High | |
| IND-535 | Temporal Fusion | Neural | Multi-horizon forecast | High | |
| IND-536 | Graph Neural Network | Neural | Relational market structure | High | |

## Priority 61: Options Greeks

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-537 | Delta | Greeks | Price sensitivity | Medium | |
| IND-538 | Gamma | Greeks | Delta sensitivity | Medium | |
| IND-539 | Theta | Greeks | Time decay | Medium | |
| IND-540 | Vega | Greeks | Volatility sensitivity | Medium | |
| IND-541 | Rho | Greeks | Interest rate sensitivity | Medium | |
| IND-542 | Vanna | Greeks | Delta-vol cross | High | |
| IND-543 | Volga (Vomma) | Greeks | Vega convexity | High | |
| IND-544 | Charm | Greeks | Delta decay | High | |
| IND-545 | Speed | Greeks | Gamma change | High | |
| IND-546 | Color | Greeks | Gamma decay | High | |

## Priority 62: Volatility Surface

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-547 | ATM Volatility | Vol Surface | At-the-money IV | Medium | |
| IND-548 | Skew Slope | Vol Surface | Put-call IV gradient | Medium | |
| IND-549 | Smile Curvature | Vol Surface | Kurtosis of IV | High | |
| IND-550 | Term Structure Slope | Vol Surface | Front vs back month | Medium | |
| IND-551 | Risk Reversal | Vol Surface | 25-delta P-C spread | Medium | |
| IND-552 | Butterfly Spread | Vol Surface | 25-delta vs ATM | Medium | |
| IND-553 | Vol-of-Vol | Vol Surface | IV volatility | High | |
| IND-554 | Sticky Strike vs Delta | Vol Surface | Smile dynamics | High | |
| IND-555 | Local Volatility | Vol Surface | Dupire local vol | High | |
| IND-556 | SABR Parameters | Vol Surface | α, β, ρ, ν calibration | High | |

## Priority 63: Systemic Risk

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-557 | SRISK | Systemic | Capital shortfall measure | High | |
| IND-558 | CoVaR | Systemic | Conditional VaR | High | |
| IND-559 | DeltaCoVaR | Systemic | CoVaR contribution | High | |
| IND-560 | MES | Systemic | Marginal Expected Shortfall | High | |
| IND-561 | Network Centrality | Systemic | Interconnectedness | High | |
| IND-562 | Contagion Index | Systemic | Spillover measure | High | |
| IND-563 | Absorption Ratio | Systemic | Market fragility | High | |
| IND-564 | Turbulence Index | Systemic | Mahalanobis distance | High | |
| IND-565 | Financial Stress Index | Systemic | Composite stress | High | |
| IND-566 | Systemic CCA | Systemic | Contingent claims analysis | High | |

## Priority 64: Performance Metrics

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-567 | Information Coefficient | Performance | Signal-return correlation | Medium | |
| IND-568 | Transfer Coefficient | Performance | Implementation efficiency | Medium | |
| IND-569 | Hit Rate | Performance | % profitable trades | Low | |
| IND-570 | Win/Loss Ratio | Performance | Avg win / avg loss | Low | |
| IND-571 | Profit Factor | Performance | Gross profit / gross loss | Low | |
| IND-572 | Expectancy | Performance | Expected value per trade | Low | |
| IND-573 | Kelly Criterion | Performance | Optimal bet size | Medium | |
| IND-574 | Risk of Ruin | Performance | Probability of blowup | Medium | |
| IND-575 | Recovery Factor | Performance | Net profit / max DD | Low | |
| IND-576 | Ulcer Performance Index | Performance | Return / Ulcer Index | Medium | |

## Priority 65: Risk-Adjusted Metrics Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-577 | Omega Ratio | Risk-Adj | Gain-loss probability weighted | Medium | |
| IND-578 | Kappa Ratio | Risk-Adj | Generalized Sharpe | Medium | |
| IND-579 | Gain-Loss Ratio | Risk-Adj | Upside vs downside | Low | |
| IND-580 | Upside Potential Ratio | Risk-Adj | Upside vs downside risk | Medium | |
| IND-581 | Pain Ratio | Risk-Adj | Return / pain index | Medium | |
| IND-582 | Martin Ratio | Risk-Adj | Return / Ulcer Index | Medium | |
| IND-583 | Burke Ratio | Risk-Adj | Modified Sharpe | Medium | |
| IND-584 | Sterling Ratio | Risk-Adj | Return / avg drawdown | Medium | |
| IND-585 | Rachev Ratio | Risk-Adj | Tail gain vs tail loss | High | |
| IND-586 | Tail Ratio | Risk-Adj | 95th / 5th percentile | Medium | |

## Priority 66: Fixed Income Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-587 | Key Rate Duration | FI | Par-point sensitivity | High | |
| IND-588 | Spread Duration | FI | Spread sensitivity | Medium | |
| IND-589 | Effective Duration | FI | OAS-based duration | High | |
| IND-590 | Empirical Duration | FI | Regression-based | Medium | |
| IND-591 | DV01 | FI | Dollar value of 01 | Medium | |
| IND-592 | PV01 | FI | Present value of 01 | Medium | |
| IND-593 | Carry and Roll | FI | Expected return | Medium | |
| IND-594 | Breakeven Inflation | FI | TIPS vs nominal | Low | |
| IND-595 | Real Rate | FI | Inflation-adjusted yield | Low | |
| IND-596 | Swap Spread | FI | Swap vs Treasury | Low | |

## Priority 67: Derivatives & Structured

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-597 | Variance Swap Rate | Derivatives | Fair variance strike | High | |
| IND-598 | Volatility Swap Rate | Derivatives | Fair vol strike | High | |
| IND-599 | Corridor Variance | Derivatives | Range-bound variance | High | |
| IND-600 | Gamma Swap | Derivatives | Weighted variance | High | |
| IND-601 | Entropy Index | Derivatives | Vol surface entropy | High | |
| IND-602 | VIX Term Premium | Derivatives | VIX futures vs spot | Medium | |
| IND-603 | Variance Risk Premium | Derivatives | Implied vs realized var | Medium | |
| IND-604 | Put Spread Collar Cost | Derivatives | Hedging cost | Medium | |
| IND-605 | Skew Premium | Derivatives | OTM put premium | Medium | |
| IND-606 | Convexity Value | Derivatives | Gamma monetization | High | |

## Priority 68: Central Bank & Macro

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-607 | Fed Funds Probability | Macro | Rate hike/cut odds | Medium | |
| IND-608 | Taylor Rule | Macro | Implied policy rate | Medium | |
| IND-609 | Shadow Rate | Macro | Wu-Xia shadow rate | High | |
| IND-610 | QE Impact | Macro | Balance sheet effect | High | |
| IND-611 | Liquidity Injection | Macro | Central bank liquidity | Medium | |
| IND-612 | FX Intervention | Macro | CB currency intervention | Medium | |
| IND-613 | Reserve Accumulation | Macro | FX reserve change | Low | |
| IND-614 | Policy Divergence | Macro | Cross-CB rate spread | Medium | |
| IND-615 | Real Policy Rate | Macro | Rate minus inflation | Low | |
| IND-616 | Neutral Rate | Macro | r-star estimate | High | |

## Priority 69: Emerging Markets

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-617 | EM Spread (EMBI) | EM | Sovereign spread | Low | |
| IND-618 | Local vs Hard Currency | EM | Spread differential | Medium | |
| IND-619 | FX Reserve Coverage | EM | Reserves / imports | Low | |
| IND-620 | External Debt Ratio | EM | Debt / GDP | Low | |
| IND-621 | Current Account Balance | EM | CA / GDP | Low | |
| IND-622 | Political Risk Index | EM | Composite political risk | High | |
| IND-623 | EM Currency Carry | EM | Rate differential carry | Medium | |
| IND-624 | Capital Flow Tracker | EM | Portfolio flow direction | Medium | |
| IND-625 | Frontier Premium | EM | Frontier vs EM spread | Medium | |
| IND-626 | Dollarization Index | EM | USD usage in economy | Medium | |

## Priority 70: Alternative Investments

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-627 | Private Equity NAV | Alt | PE fund valuation | High | |
| IND-628 | Vintage Year Return | Alt | Cohort performance | Medium | |
| IND-629 | J-Curve Position | Alt | PE lifecycle stage | Medium | |
| IND-630 | Secondaries Discount | Alt | Secondary market price | Medium | |
| IND-631 | VC Deal Flow | Alt | Venture activity | Medium | |
| IND-632 | Dry Powder | Alt | Uninvested capital | Low | |
| IND-633 | MOIC | Alt | Multiple on invested capital | Low | |
| IND-634 | DPI | Alt | Distributions / paid-in | Low | |
| IND-635 | TVPI | Alt | Total value / paid-in | Low | |
| IND-636 | PME (Public Market Equivalent) | Alt | vs public benchmark | High | |

## Priority 71: Execution & Transaction Cost

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-637 | Implementation Shortfall | Execution | Decision vs execution | Medium | |
| IND-638 | VWAP Slippage | Execution | vs VWAP benchmark | Low | |
| IND-639 | TWAP Slippage | Execution | vs TWAP benchmark | Low | |
| IND-640 | Market Impact | Execution | Price impact of trade | High | |
| IND-641 | Timing Cost | Execution | Delay cost | Medium | |
| IND-642 | Opportunity Cost | Execution | Unfilled portion cost | Medium | |
| IND-643 | Spread Cost | Execution | Bid-ask crossing | Low | |
| IND-644 | Broker Performance | Execution | Execution quality rank | Medium | |
| IND-645 | Venue Analysis | Execution | Exchange comparison | Medium | |
| IND-646 | Toxicity Score | Execution | Adverse selection risk | High | |

## Priority 72: Bayesian & Probabilistic

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-647 | Bayesian Regression | Bayesian | Posterior estimates | High | |
| IND-648 | Kalman Filter State | Bayesian | State estimation | High | |
| IND-649 | Particle Filter | Bayesian | Sequential Monte Carlo | High | |
| IND-650 | Gaussian Process | Bayesian | GP regression | High | |
| IND-651 | Bayesian VAR | Bayesian | BVAR forecasts | High | |
| IND-652 | Posterior Probability | Bayesian | Model probability | High | |
| IND-653 | Bayes Factor | Bayesian | Model comparison | High | |
| IND-654 | Credible Interval | Bayesian | Posterior interval | Medium | |
| IND-655 | Prior Update | Bayesian | Belief revision | High | |
| IND-656 | Evidence | Bayesian | Marginal likelihood | High | |

## Priority 73: Time Series Econometrics

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-657 | Granger Causality | Econometric | Predictive causality | High | |
| IND-658 | Impulse Response | Econometric | Shock propagation | High | |
| IND-659 | Variance Decomposition | Econometric | Forecast error decomp | High | |
| IND-660 | Cointegration Rank | Econometric | Johansen test result | High | |
| IND-661 | VECM Forecast | Econometric | Error correction forecast | High | |
| IND-662 | Structural Break | Econometric | Regime change detection | High | |
| IND-663 | Unit Root Test | Econometric | Stationarity test | Medium | |
| IND-664 | ARCH/GARCH | Econometric | Volatility clustering | High | |
| IND-665 | DCC Correlation | Econometric | Dynamic conditional corr | High | |
| IND-666 | Copula Dependence | Econometric | Tail dependence | High | |

## Priority 74: Portfolio Analytics

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-667 | Tracking Error | Portfolio | vs Benchmark deviation | Medium | |
| IND-668 | Active Share | Portfolio | Position difference | Medium | |
| IND-669 | Style Drift | Portfolio | Factor exposure change | Medium | |
| IND-670 | Concentration Index | Portfolio | HHI of positions | Low | |
| IND-671 | Effective N | Portfolio | Diversification measure | Medium | |
| IND-672 | Marginal Contribution to Risk | Portfolio | Position risk contrib | High | |
| IND-673 | Component VaR | Portfolio | VaR decomposition | High | |
| IND-674 | Risk Budgeting | Portfolio | Risk allocation | High | |
| IND-675 | Factor Exposure | Portfolio | Multi-factor loadings | Medium | |
| IND-676 | Residual Risk | Portfolio | Idiosyncratic risk | Medium | |

## Priority 75: Sentiment Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-677 | Earnings Call Tone | Sentiment | NLP transcript analysis | High | |
| IND-678 | 10-K/10-Q Sentiment | Sentiment | Filing text analysis | High | |
| IND-679 | Patent Filing Activity | Sentiment | Innovation proxy | Medium | |
| IND-680 | Job Posting Trend | Sentiment | Hiring momentum | Medium | |
| IND-681 | Web Traffic Rank | Sentiment | Consumer interest | Medium | |
| IND-682 | App Downloads | Sentiment | Mobile engagement | Medium | |
| IND-683 | Satellite Imagery | Sentiment | Economic activity proxy | High | |
| IND-684 | Credit Card Spending | Sentiment | Consumer spending | High | |
| IND-685 | Geolocation Data | Sentiment | Foot traffic | High | |
| IND-686 | Supply Chain Disruption | Sentiment | Logistics stress | High | |

## Priority 76: Crypto DeFi

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-687 | TVL (Total Value Locked) | DeFi | Protocol deposits | Low | |
| IND-688 | DEX Volume | DeFi | Decentralized exchange vol | Low | |
| IND-689 | Yield Farming APY | DeFi | Farming returns | Medium | |
| IND-690 | Impermanent Loss | DeFi | LP opportunity cost | Medium | |
| IND-691 | Protocol Revenue | DeFi | Fee generation | Medium | |
| IND-692 | Token Velocity | DeFi | Transaction frequency | Medium | |
| IND-693 | Governance Activity | DeFi | DAO participation | Medium | |
| IND-694 | Smart Contract Risk | DeFi | Audit/exploit score | High | |
| IND-695 | Bridge Volume | DeFi | Cross-chain transfers | Medium | |
| IND-696 | Staking Ratio | DeFi | % tokens staked | Low | |

## Priority 77: Market Making & HFT

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-697 | Quote Stuffing | HFT | Order/cancel ratio | High | |
| IND-698 | Layering Detection | HFT | Spoofing pattern | High | |
| IND-699 | Latency Arbitrage | HFT | Speed advantage | High | |
| IND-700 | Maker/Taker Ratio | HFT | Liquidity provision | Medium | |
| IND-701 | Fill Rate | HFT | Order fill percentage | Low | |
| IND-702 | Adverse Selection | HFT | Toxic flow measure | High | |
| IND-703 | Queue Position | HFT | Order book priority | Medium | |
| IND-704 | Realized Spread | HFT | Post-trade spread | Medium | |
| IND-705 | Price Improvement | HFT | vs NBBO | Low | |
| IND-706 | Internalization Rate | HFT | Off-exchange execution | Medium | |

## Priority 78: Regime Detection Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-707 | Markov Regime | Regime | MS model states | High | |
| IND-708 | Threshold Model | Regime | TAR/SETAR regimes | High | |
| IND-709 | Smooth Transition | Regime | STAR model | High | |
| IND-710 | Structural VAR Regime | Regime | SVAR with breaks | High | |
| IND-711 | Volatility Regime | Regime | High/low vol states | Medium | |
| IND-712 | Trend Regime | Regime | Trending vs mean-revert | Medium | |
| IND-713 | Correlation Regime | Regime | Risk-on vs risk-off | Medium | |
| IND-714 | Liquidity Regime | Regime | Normal vs stressed | Medium | |
| IND-715 | Business Cycle Phase | Regime | Expansion/contraction | High | |
| IND-716 | Crisis Detector | Regime | Tail event warning | High | |

## Priority 79: Classic TA Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-717 | Weighted Close | Classic | (H+L+2C)/4 | Low | |
| IND-718 | Typical Price | Classic | (H+L+C)/3 | Low | |
| IND-719 | Median Price | Classic | (H+L)/2 | Low | |
| IND-720 | Average Price | Classic | (O+H+L+C)/4 | Low | |
| IND-721 | True Range | Classic | Wilder's TR | Low | |
| IND-722 | Average True Range Percent | Classic | ATR as % of close | Low | |
| IND-723 | Normalized Average True Range | Classic | ATR / ATR MA | Low | |
| IND-724 | Range | Classic | High - Low | Low | |
| IND-725 | Range Percent | Classic | Range / Close | Low | |
| IND-726 | Body Size | Classic | |Close - Open| | Low | |
| IND-727 | Upper Shadow | Classic | High - max(O,C) | Low | |
| IND-728 | Lower Shadow | Classic | min(O,C) - Low | Low | |
| IND-729 | Candle Range Ratio | Classic | Body / Range | Low | |
| IND-730 | Gap | Classic | Open - Prior Close | Low | |

## Priority 80: MetaTrader Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-731 | iAC | MT4 | Accelerator Oscillator | Low | |
| IND-732 | iAO | MT4 | Awesome Oscillator | Low | |
| IND-733 | iBands | MT4 | Bollinger Bands | Medium | |
| IND-734 | iBWMFI | MT4 | Bill Williams MFI | Medium | |
| IND-735 | iCCI | MT4 | Commodity Channel Index | Medium | |
| IND-736 | iDeMarker | MT4 | DeMarker | Medium | |
| IND-737 | iEnvelopes | MT4 | MA Envelopes | Low | |
| IND-738 | iForce | MT4 | Force Index | Low | |
| IND-739 | iFractals | MT4 | Fractals | Medium | |
| IND-740 | iGator | MT4 | Gator Oscillator | Medium | |
| IND-741 | iIchimoku | MT4 | Ichimoku Kinko Hyo | High | |
| IND-742 | iMomentum | MT4 | Momentum | Low | |
| IND-743 | iOBV | MT4 | On Balance Volume | Low | |
| IND-744 | iOsMA | MT4 | MACD Histogram | Low | |
| IND-745 | iSAR | MT4 | Parabolic SAR | Medium | |
| IND-746 | iStdDev | MT4 | Standard Deviation | Low | |
| IND-747 | iStochastic | MT4 | Stochastic Oscillator | Medium | |
| IND-748 | iWPR | MT4 | Williams %R | Low | |

## Priority 81: TradingView Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-749 | Supertrend | TV | ATR-based trend | Medium | |
| IND-750 | Volume Profile Visible Range | TV | VPVR | High | |
| IND-751 | Volume Profile Fixed Range | TV | VPFR | High | |
| IND-752 | Volume Profile Session | TV | Session VP | High | |
| IND-753 | Auto Fib Retracement | TV | Auto swing fibs | High | |
| IND-754 | Auto Fib Extension | TV | Auto extensions | High | |
| IND-755 | Pivot Points Standard | TV | Classic pivots | Low | |
| IND-756 | Pivot Points Camarilla | TV | Camarilla pivots | Low | |
| IND-757 | Pivot Points Woodie | TV | Woodie pivots | Low | |
| IND-758 | Pivot Points DeMark | TV | DeMark pivots | Medium | |
| IND-759 | Anchored VWAP | TV | Custom anchor VWAP | Medium | |
| IND-760 | Fixed Range Volume Profile | TV | Custom range VP | High | |
| IND-761 | Visible Range Volume Profile | TV | Visible VP | High | |
| IND-762 | Session Volume Profile | TV | Per-session VP | High | |
| IND-763 | Linear Regression Channel | TV | Regression channel | Medium | |
| IND-764 | Standard Error Channel | TV | Std error channel | Medium | |
| IND-765 | Raff Regression Channel | TV | Raff channel | Medium | |
| IND-766 | Zig Zag | TV | Swing point connector | Medium | |

## Priority 82: NinjaTrader Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-767 | Order Flow Cumulative Delta | NT | Cumulative delta | Medium | |
| IND-768 | Order Flow Volume Profile | NT | Volume at price | High | |
| IND-769 | Order Flow VWAP | NT | Volume weighted AP | Medium | |
| IND-770 | Market Analyzer | NT | Multi-indicator scanner | High | |
| IND-771 | Volumetric Bars | NT | Volume-based bars | High | |
| IND-772 | Tick Counter | NT | Tick analysis | Low | |
| IND-773 | Range Counter | NT | Range analysis | Low | |
| IND-774 | Current Day OHL | NT | Session OHLC | Low | |
| IND-775 | Prior Day OHLC | NT | Prior session OHLC | Low | |
| IND-776 | Weekly OHLC | NT | Weekly levels | Low | |
| IND-777 | Monthly OHLC | NT | Monthly levels | Low | |

## Priority 83: ThinkOrSwim Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-778 | TTM_Trend | ToS | John Carter trend | Medium | |
| IND-779 | TTM_Squeeze | ToS | John Carter squeeze | Medium | |
| IND-780 | TTM_Wave | ToS | Wave oscillator | Medium | |
| IND-781 | TTM_Scalper | ToS | Scalping indicator | Medium | |
| IND-782 | Sizzle Index | ToS | Options activity | Medium | |
| IND-783 | IV Percentile | ToS | IV rank | Medium | |
| IND-784 | Probability of ITM | ToS | Options probability | High | |
| IND-785 | Expected Move | ToS | 1 std dev range | Medium | |
| IND-786 | Greeks Display | ToS | Option greeks | Medium | |
| IND-787 | MoneyFlow | ToS | TOS money flow | Medium | |
| IND-788 | VolumeAvg | ToS | Volume average | Low | |
| IND-789 | RelativeVolume | ToS | vs Average volume | Low | |

## Priority 84: Bloomberg Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-790 | BOLL | BBG | Bollinger Bands | Medium | |
| IND-791 | MACD | BBG | MACD | Medium | |
| IND-792 | RSI | BBG | Relative Strength | Medium | |
| IND-793 | SMAVG | BBG | Simple MA | Low | |
| IND-794 | EMAVG | BBG | Exponential MA | Low | |
| IND-795 | DMI | BBG | Directional Movement | Medium | |
| IND-796 | CESI | BBG | Citi Economic Surprise | Medium | |
| IND-797 | FXCR | BBG | FX Carry | Medium | |
| IND-798 | BXII | BBG | Bloomberg DXY | Low | |
| IND-799 | VIX | BBG | Volatility Index | Low | |
| IND-800 | MOVE | BBG | Bond Volatility | Low | |
| IND-801 | GVZ | BBG | Gold Volatility | Low | |
| IND-802 | OVX | BBG | Oil Volatility | Low | |
| IND-803 | CVIX | BBG | Currency Volatility | Low | |

## Priority 85: Additional Candlestick Patterns

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-804 | Gravestone Doji | Candle | Long upper shadow doji | Low | |
| IND-805 | Dragonfly Doji | Candle | Long lower shadow doji | Low | |
| IND-806 | Long-Legged Doji | Candle | Long both shadows doji | Low | |
| IND-807 | Four Price Doji | Candle | O=H=L=C | Low | |
| IND-808 | Rickshaw Man | Candle | Midpoint doji | Low | |
| IND-809 | High Wave | Candle | Long shadows, small body | Low | |
| IND-810 | Concealing Baby Swallow | Candle | Rare bullish pattern | High | |
| IND-811 | Ladder Bottom | Candle | 5-candle reversal | High | |
| IND-812 | Ladder Top | Candle | 5-candle reversal | High | |
| IND-813 | Mat Hold | Candle | Continuation pattern | High | |
| IND-814 | Matching Low | Candle | Double bottom | Medium | |
| IND-815 | Matching High | Candle | Double top | Medium | |
| IND-816 | Meeting Lines | Candle | Same close reversal | Medium | |
| IND-817 | On Neck | Candle | Bearish continuation | Medium | |
| IND-818 | In Neck | Candle | Bearish continuation | Medium | |
| IND-819 | Thrusting | Candle | Weak bullish | Medium | |
| IND-820 | Separating Lines | Candle | Gap continuation | Medium | |
| IND-821 | Side by Side White | Candle | Bullish continuation | Medium | |
| IND-822 | Side by Side Black | Candle | Bearish continuation | Medium | |
| IND-823 | Stick Sandwich | Candle | Bullish reversal | Medium | |
| IND-824 | Tri-Star | Candle | Three doji reversal | High | |
| IND-825 | Unique Three River | Candle | Bullish reversal | High | |
| IND-826 | Upside Gap Three Methods | Candle | Bullish continuation | Medium | |
| IND-827 | Downside Gap Three Methods | Candle | Bearish continuation | Medium | |
| IND-828 | Hikkake Pattern | Candle | Inside bar failure | Medium | |
| IND-829 | Modified Hikkake | Candle | Inside bar variant | Medium | |
| IND-830 | Breakaway | Candle | 5-candle reversal | High | |
| IND-831 | Two Crows | Candle | Bearish pattern | Medium | |
| IND-832 | Three Stars in the South | Candle | Bullish reversal | High | |
| IND-833 | Advance Block | Candle | Bearish reversal | Medium | |
| IND-834 | Deliberation | Candle | Bearish reversal | Medium | |
| IND-835 | Identical Three Crows | Candle | Strong bearish | Medium | |
| IND-836 | Upside Tasuki Gap | Candle | Bullish continuation | Medium | |
| IND-837 | Downside Tasuki Gap | Candle | Bearish continuation | Medium | |

## Priority 86: Price Action Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-838 | Pin Bar | Price Action | Long wick rejection | Low | |
| IND-839 | Inside Bar | Price Action | NR4/NR7 | Low | |
| IND-840 | Outside Bar | Price Action | Engulfing bar | Low | |
| IND-841 | Fakey Pattern | Price Action | False breakout setup | Medium | |
| IND-842 | Two-Bar Reversal | Price Action | 2-bar pattern | Low | |
| IND-843 | Three-Bar Reversal | Price Action | 3-bar pattern | Low | |
| IND-844 | Key Reversal | Price Action | Gap reversal | Medium | |
| IND-845 | Hook Reversal | Price Action | Open-close hook | Low | |
| IND-846 | Island Cluster | Price Action | Multiple island bars | Medium | |
| IND-847 | Gap and Go | Price Action | Continuation after gap | Low | |
| IND-848 | Gap and Crap | Price Action | Gap fade | Low | |
| IND-849 | Pivot Point Reversal | Price Action | Swing high/low | Medium | |
| IND-850 | 1-2-3 Pattern | Price Action | Trend reversal | Medium | |
| IND-851 | OOPS Pattern | Price Action | Larry Williams OOPS | Low | |
| IND-852 | Holy Grail | Price Action | Linda Raschke setup | Medium | |
| IND-853 | Turtle Soup | Price Action | Failed breakout | Medium | |
| IND-854 | Turtle Soup Plus One | Price Action | Extended turtle soup | Medium | |
| IND-855 | 80-20 Pattern | Price Action | Range breakout | Low | |
| IND-856 | Momentum Pinball | Price Action | Linda Raschke | Medium | |
| IND-857 | Anti Pattern | Price Action | Continuation setup | Medium | |

## Priority 87: Larry Williams Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-858 | Williams %R | LW | Momentum oscillator | Low | |
| IND-859 | Williams Accumulation/Distribution | LW | Volume indicator | Medium | |
| IND-860 | Ultimate Oscillator | LW | Multi-TF momentum | Medium | |
| IND-861 | Williams VIX Fix | LW | Synthetic VIX | Medium | |
| IND-862 | COT Index | LW | COT positioning | Medium | |
| IND-863 | Commitment of Traders Net Position | LW | Net long/short | Low | |
| IND-864 | Commercial Index | LW | Commercial positioning | Medium | |
| IND-865 | Large Spec Index | LW | Large spec positioning | Medium | |
| IND-866 | Small Spec Index | LW | Small spec positioning | Medium | |
| IND-867 | POIV | LW | Proxy OBV | Medium | |
| IND-868 | WILLCO | LW | Williams Composite Index | High | |
| IND-869 | Sentiment Index | LW | Trader sentiment | Medium | |

## Priority 88: Linda Raschke Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-870 | 3/10 Oscillator | LR | 3-10 MACD variant | Low | |
| IND-871 | Keltner Channel | LR | ATR envelope | Medium | |
| IND-872 | ADXpress | LR | Smoothed ADX | Medium | |
| IND-873 | Momentum Pinball | LR | RSI momentum | Medium | |
| IND-874 | Holy Grail Setup | LR | ADX + retracement | Medium | |
| IND-875 | Turtle Soup | LR | Failed breakout | Medium | |
| IND-876 | Anti Pattern | LR | Retracement setup | Medium | |
| IND-877 | First Cross | LR | MA cross timing | Low | |
| IND-878 | Equilibrium | LR | Balance point | Medium | |
| IND-879 | ID/NR4 | LR | Inside day + NR4 | Low | |
| IND-880 | NR7 | LR | Narrowest range 7 | Low | |
| IND-881 | WR7 | LR | Widest range 7 | Low | |

## Priority 89: Tom DeMark Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-882 | TD Points | DeMark | Pivot points | Medium | |
| IND-883 | TD Lines | DeMark | Trend lines | Medium | |
| IND-884 | TD Camouflage | DeMark | Deceptive bars | High | |
| IND-885 | TD Clop | DeMark | Opening reversal | Medium | |
| IND-886 | TD Clopwin | DeMark | Opening range | Medium | |
| IND-887 | TD DeMarker I | DeMark | Short-term momentum | Medium | |
| IND-888 | TD DeMarker II | DeMark | Long-term momentum | Medium | |
| IND-889 | TD Range Expansion Index | DeMark | Momentum measure | Medium | |
| IND-890 | TD Rate of Change | DeMark | Price rate of change | Low | |
| IND-891 | TD Relative Retracement | DeMark | Fib retracement | Medium | |
| IND-892 | TD Reverse Range Expansion | DeMark | Reversal detection | High | |
| IND-893 | TD Trend Factor | DeMark | Trend strength | Medium | |
| IND-894 | TD Risk Level | DeMark | Support/resistance | High | |
| IND-895 | TD Alignment | DeMark | Multi-TF alignment | High | |
| IND-896 | TD Channel I | DeMark | Price channel | Medium | |
| IND-897 | TD Channel II | DeMark | Modified channel | Medium | |
| IND-898 | TD Propulsion | DeMark | Momentum thrust | High | |
| IND-899 | TD Differential | DeMark | Price differential | Medium | |
| IND-900 | TD Qualifier | DeMark | Setup qualifier | High | |

## Priority 90: John Bollinger Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-901 | %B | Bollinger | Band position | Low | |
| IND-902 | Bandwidth | Bollinger | Band width | Low | |
| IND-903 | BBTrend | Bollinger | Trend indicator | Medium | |
| IND-904 | BBMomentum | Bollinger | Momentum | Medium | |
| IND-905 | BBAccumulation | Bollinger | Volume indicator | Medium | |
| IND-906 | BBImpulse | Bollinger | Impulse indicator | Medium | |
| IND-907 | BBNormalize | Bollinger | Normalized price | Low | |
| IND-908 | BBPersist | Bollinger | Trend persistence | Medium | |
| IND-909 | BBStop | Bollinger | Trailing stop | Medium | |
| IND-910 | Squeeze Indicator | Bollinger | BB inside KC | Medium | |

## Priority 91: John Murphy Intermarket

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-911 | Dollar/Gold Ratio | Intermarket | USD vs Gold | Low | |
| IND-912 | Bond/Stock Ratio | Intermarket | TLT vs SPY | Low | |
| IND-913 | Stock/Commodity Ratio | Intermarket | SPY vs DBC | Low | |
| IND-914 | Copper/Gold Ratio | Intermarket | Economic indicator | Low | |
| IND-915 | Oil/Gold Ratio | Intermarket | Inflation indicator | Low | |
| IND-916 | High Yield Spread | Intermarket | Risk appetite | Low | |
| IND-917 | XLY/XLP Ratio | Intermarket | Consumer discretionary vs staples | Low | |
| IND-918 | Sector Rotation | Intermarket | Business cycle | High | |
| IND-919 | Global Relative Strength | Intermarket | Country ranking | Medium | |
| IND-920 | Asset Class Momentum | Intermarket | Cross-asset momentum | Medium | |

## Priority 92: Perry Kaufman Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-921 | Efficiency Ratio | Kaufman | Price efficiency | Low | |
| IND-922 | KAMA | Kaufman | Adaptive MA | Medium | |
| IND-923 | Adaptive Momentum | Kaufman | Variable momentum | Medium | |
| IND-924 | Adaptive Channel | Kaufman | Variable bands | Medium | |
| IND-925 | Adaptive RSI | Kaufman | Variable RSI | Medium | |
| IND-926 | Adaptive Stochastic | Kaufman | Variable stochastic | Medium | |
| IND-927 | Fractal Efficiency | Kaufman | Chaos measure | High | |
| IND-928 | Noise Elimination | Kaufman | Signal filtering | High | |
| IND-929 | Adaptive Trend | Kaufman | Trend filter | Medium | |
| IND-930 | System Performance | Kaufman | Backtest metrics | Medium | |

## Priority 93: Tushar Chande Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-931 | Chande Momentum Oscillator (CMO) | Chande | Momentum | Medium | |
| IND-932 | Variable Index Dynamic Average (VIDYA) | Chande | Adaptive MA | Medium | |
| IND-933 | Qstick | Chande | Open-close average | Low | |
| IND-934 | Intraday Momentum Index | Chande | IMI | Medium | |
| IND-935 | Stochastic RSI | Chande | StochRSI | Medium | |
| IND-936 | Range Action Verification Index | Chande | RAVI | Medium | |
| IND-937 | Chande Forecast Oscillator | Chande | Regression oscillator | Medium | |
| IND-938 | Aroon | Chande | Trend detection | Medium | |
| IND-939 | Aroon Oscillator | Chande | Aroon difference | Medium | |
| IND-940 | Dynamic Momentum Index | Chande | Variable RSI | Medium | |

## Priority 94: Welles Wilder Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-941 | RSI | Wilder | Relative Strength Index | Medium | |
| IND-942 | Average True Range | Wilder | ATR | Low | |
| IND-943 | Average Directional Index | Wilder | ADX | Medium | |
| IND-944 | Directional Movement +DI | Wilder | Plus DI | Medium | |
| IND-945 | Directional Movement -DI | Wilder | Minus DI | Medium | |
| IND-946 | Parabolic SAR | Wilder | Stop and Reverse | Medium | |
| IND-947 | Swing Index | Wilder | Single bar SI | Medium | |
| IND-948 | Accumulative Swing Index | Wilder | Cumulative SI | Medium | |
| IND-949 | Commodity Selection Index | Wilder | CSI | Medium | |
| IND-950 | Directional Movement Rating | Wilder | ADXR | Medium | |
| IND-951 | Volatility Index | Wilder | TR-based vol | Medium | |

## Priority 95: Gerald Appel Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-952 | MACD | Appel | Original MACD | Medium | |
| IND-953 | MACD Signal | Appel | MACD signal line | Medium | |
| IND-954 | MACD Histogram | Appel | MACD - Signal | Low | |
| IND-955 | MACD-2 | Appel | Modified MACD | Medium | |
| IND-956 | PPO | Appel | Percentage Price Oscillator | Low | |
| IND-957 | APO | Appel | Absolute Price Oscillator | Low | |
| IND-958 | Price Oscillator | Appel | General oscillator | Low | |
| IND-959 | Double MACD | Appel | MACD of MACD | Medium | |
| IND-960 | MACD Cross | Appel | Cross detection | Low | |
| IND-961 | Histogram Divergence | Appel | MACD divergence | Medium | |

## Priority 96: Marc Chaikin Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-962 | Chaikin A/D Line | Chaikin | Accumulation/Distribution | Medium | |
| IND-963 | Chaikin A/D Oscillator | Chaikin | A/D oscillator | Medium | |
| IND-964 | Chaikin Money Flow | Chaikin | CMF | Medium | |
| IND-965 | Chaikin Volatility | Chaikin | H-L volatility | Medium | |
| IND-966 | Chaikin Power Gauge | Chaikin | Multi-factor rating | High | |
| IND-967 | Chaikin Analytics | Chaikin | Stock rating | High | |
| IND-968 | Persistence of Money Flow | Chaikin | CMF streak | Medium | |
| IND-969 | Volume Accumulation | Chaikin | Volume analysis | Medium | |
| IND-970 | Price Trend Rating | Chaikin | Trend score | Medium | |

## Priority 97: Martin Pring Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-971 | KST (Know Sure Thing) | Pring | Weighted ROC sum | Medium | |
| IND-972 | KST Signal | Pring | KST MA | Medium | |
| IND-973 | Special K | Pring | Daily KST | Medium | |
| IND-974 | Pring's Momentum | Pring | ROC-based | Medium | |
| IND-975 | Diffusion Index | Pring | Breadth measure | Medium | |
| IND-976 | Pring MACD | Pring | Modified MACD | Medium | |
| IND-977 | Velocity | Pring | Rate of change | Low | |
| IND-978 | Volume Momentum | Pring | Volume oscillator | Medium | |
| IND-979 | Six Stage Business Cycle | Pring | Cycle phase | High | |
| IND-980 | Composite Momentum | Pring | Multi-TF | High | |

## Priority 98: Alexander Elder Indicators Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-981 | Force Index (13) | Elder | Short-term force | Low | |
| IND-982 | Force Index (2) | Elder | Ultra short-term | Low | |
| IND-983 | Elder Ray Bull Power | Elder | Bulls strength | Medium | |
| IND-984 | Elder Ray Bear Power | Elder | Bears strength | Medium | |
| IND-985 | Elder Impulse System | Elder | Trend + momentum | Medium | |
| IND-986 | Elder SafeZone Long | Elder | Long stop | Medium | |
| IND-987 | Elder SafeZone Short | Elder | Short stop | Medium | |
| IND-988 | Triple Screen | Elder | Multi-TF system | High | |
| IND-989 | Spike Indicator | Elder | Price spike | Low | |
| IND-990 | AutoEnvelope | Elder | Adaptive bands | Medium | |

## Priority 99: George Lane Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-991 | Fast Stochastic %K | Lane | Raw stochastic | Low | |
| IND-992 | Fast Stochastic %D | Lane | %K MA | Low | |
| IND-993 | Slow Stochastic %K | Lane | Smoothed %K | Low | |
| IND-994 | Slow Stochastic %D | Lane | Smoothed %D | Low | |
| IND-995 | Full Stochastic | Lane | Configurable | Medium | |
| IND-996 | Lane's Stochastic | Lane | Original | Medium | |
| IND-997 | %K Divergence | Lane | Divergence | Medium | |
| IND-998 | Stochastic Pop | Lane | Pop setup | Medium | |
| IND-999 | Stochastic Drop | Lane | Drop setup | Medium | |
| IND-1000 | Stochastic Crossover | Lane | Cross signal | Low | |

## Priority 100: Additional Statistical

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1001 | Mean | Statistical | Arithmetic mean | Low | |
| IND-1002 | Median | Statistical | Middle value | Low | |
| IND-1003 | Mode | Statistical | Most frequent | Medium | |
| IND-1004 | Range | Statistical | Max - Min | Low | |
| IND-1005 | Variance | Statistical | Squared deviations | Low | |
| IND-1006 | Standard Deviation | Statistical | Sqrt variance | Low | |
| IND-1007 | Coefficient of Variation | Statistical | StdDev / Mean | Low | |
| IND-1008 | Skewness | Statistical | Asymmetry | Medium | |
| IND-1009 | Kurtosis | Statistical | Tail weight | Medium | |
| IND-1010 | Percentile | Statistical | Nth percentile | Low | |
| IND-1011 | Quartiles | Statistical | Q1, Q2, Q3 | Low | |
| IND-1012 | IQR | Statistical | Q3 - Q1 | Low | |
| IND-1013 | MAD | Statistical | Mean absolute deviation | Low | |
| IND-1014 | Z-Score | Statistical | Standardized value | Low | |
| IND-1015 | T-Statistic | Statistical | T-test | Medium | |
| IND-1016 | P-Value | Statistical | Significance | Medium | |
| IND-1017 | Confidence Interval | Statistical | CI bounds | Medium | |
| IND-1018 | R-Squared | Statistical | Coefficient of determination | Medium | |
| IND-1019 | Adjusted R-Squared | Statistical | Adjusted CoD | Medium | |
| IND-1020 | F-Statistic | Statistical | F-test | Medium | |
| IND-1021 | AIC | Statistical | Akaike criterion | Medium | |
| IND-1022 | BIC | Statistical | Bayesian criterion | Medium | |
| IND-1023 | Durbin-Watson | Statistical | Autocorrelation test | Medium | |
| IND-1024 | Jarque-Bera | Statistical | Normality test | Medium | |
| IND-1025 | Shapiro-Wilk | Statistical | Normality test | Medium | |
| IND-1026 | Kolmogorov-Smirnov | Statistical | Distribution test | Medium | |
| IND-1027 | Anderson-Darling | Statistical | Distribution test | Medium | |
| IND-1028 | Augmented Dickey-Fuller | Statistical | Unit root test | High | |
| IND-1029 | KPSS | Statistical | Stationarity test | High | |
| IND-1030 | Phillips-Perron | Statistical | Unit root test | High | |

## Priority 101: Richard Arms Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1031 | TRIN (Arms Index) | Arms | A/D ratio / volume ratio | Medium | |
| IND-1032 | Open TRIN | Arms | Open-only calculation | Medium | |
| IND-1033 | TRIN-5 | Arms | 5-day MA of TRIN | Medium | |
| IND-1034 | TRIN-10 | Arms | 10-day MA of TRIN | Medium | |
| IND-1035 | Ease of Movement | Arms | Price/volume relationship | Medium | |
| IND-1036 | Ease of Movement MA | Arms | Smoothed EMV | Medium | |
| IND-1037 | Volume-Adjusted Moving Average | Arms | VAMA | Medium | |
| IND-1038 | Equivolume | Arms | Volume-width bars | High | |
| IND-1039 | CandleVolume | Arms | Volume-based candles | High | |
| IND-1040 | Arms Ease of Movement Value | Arms | EMV value | Medium | |

## Priority 102: Joe Granville Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1041 | On Balance Volume | Granville | Cumulative OBV | Low | |
| IND-1042 | OBV Trend | Granville | OBV MA crossover | Medium | |
| IND-1043 | OBV Divergence | Granville | Price vs OBV | Medium | |
| IND-1044 | Climax Indicator | Granville | Volume climax | Medium | |
| IND-1045 | Net Field Trend | Granville | Breadth indicator | High | |
| IND-1046 | Granville's New High/Low | Granville | NH-NL oscillator | Medium | |

## Priority 103: Stan Weinstein Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1047 | Stage Analysis | Weinstein | 4-stage cycle | High | |
| IND-1048 | 30-Week Moving Average | Weinstein | Weekly MA | Low | |
| IND-1049 | Relative Strength (Mansfield) | Weinstein | vs Market RS | Medium | |
| IND-1050 | Volume Confirmation | Weinstein | Volume pattern | Medium | |
| IND-1051 | Breakout Validation | Weinstein | Stage 2 breakout | High | |
| IND-1052 | Support/Resistance Levels | Weinstein | S/R zones | Medium | |

## Priority 104: William O'Neil / CANSLIM

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1053 | Relative Price Strength | CANSLIM | IBD RS Rating | Medium | |
| IND-1054 | EPS Rating | CANSLIM | Earnings strength | Medium | |
| IND-1055 | SMR Rating | CANSLIM | Sales/Margin/ROE | High | |
| IND-1056 | Composite Rating | CANSLIM | Overall score | High | |
| IND-1057 | Group Relative Strength | CANSLIM | Industry RS | Medium | |
| IND-1058 | Accumulation/Distribution Rating | CANSLIM | A/D grade | Medium | |
| IND-1059 | Up/Down Volume Ratio | CANSLIM | 50-day ratio | Medium | |
| IND-1060 | Sponsorship Rating | CANSLIM | Institutional quality | High | |
| IND-1061 | Cup with Handle | CANSLIM | Base pattern | High | |
| IND-1062 | Double Bottom | CANSLIM | Base pattern | High | |
| IND-1063 | Flat Base | CANSLIM | Base pattern | Medium | |
| IND-1064 | Saucer Base | CANSLIM | Base pattern | High | |
| IND-1065 | High Tight Flag | CANSLIM | Continuation pattern | High | |
| IND-1066 | Ascending Base | CANSLIM | Base pattern | High | |
| IND-1067 | IPO Base | CANSLIM | New issue pattern | High | |
| IND-1068 | Base on Base | CANSLIM | Stacked bases | High | |

## Priority 105: Victor Sperandeo Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1069 | 1-2-3 Reversal | Sperandeo | Trend change | Medium | |
| IND-1070 | 2B Pattern | Sperandeo | Failed breakout | Medium | |
| IND-1071 | Trendline Break | Sperandeo | Trend violation | Medium | |
| IND-1072 | Test of Low/High | Sperandeo | Support/resistance test | Medium | |
| IND-1073 | Change in Trend | Sperandeo | Trend reversal | Medium | |

## Priority 106: Richard Donchian Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1074 | Donchian Channel | Donchian | N-period high/low | Low | |
| IND-1075 | Donchian Channel Width | Donchian | Channel width | Low | |
| IND-1076 | Donchian Middle | Donchian | Channel midpoint | Low | |
| IND-1077 | 4-Week Rule | Donchian | 20-day breakout | Low | |
| IND-1078 | 5/20 Day Breakout | Donchian | Dual breakout | Medium | |
| IND-1079 | Turtle Trading Entry | Donchian | 20/55 day system | Medium | |
| IND-1080 | Turtle Trading Exit | Donchian | 10/20 day exit | Medium | |

## Priority 107: Mark Minervini Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1081 | Trend Template | Minervini | 8 criteria check | High | |
| IND-1082 | Volatility Contraction Pattern | Minervini | VCP | High | |
| IND-1083 | Pivot Point (Minervini) | Minervini | Buy point | Medium | |
| IND-1084 | Power Play | Minervini | Strong momentum | High | |
| IND-1085 | Cheat Setup | Minervini | Early entry | High | |
| IND-1086 | Low Cheat | Minervini | Low-risk entry | High | |
| IND-1087 | Pocket Pivot | Minervini | Volume signal | Medium | |

## Priority 108: Dan Zanger Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1088 | Zanger Volume | Zanger | Volume analysis | Medium | |
| IND-1089 | Bull Flag (Zanger) | Zanger | Tight flag | Medium | |
| IND-1090 | Cup Breakout | Zanger | Cup pattern | High | |
| IND-1091 | Handle Entry | Zanger | Handle breakout | High | |
| IND-1092 | Channel Breakout (Zanger) | Zanger | Ascending channel | Medium | |

## Priority 109: TradeStation Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1093 | RadarScreen | TS | Multi-symbol scanner | High | |
| IND-1094 | Matrix | TS | Heat map | High | |
| IND-1095 | OptionStation | TS | Options analytics | High | |
| IND-1096 | EasyLanguage Indicators | TS | Custom indicators | High | |
| IND-1097 | Walk Forward Optimizer | TS | WFO results | High | |
| IND-1098 | Market Depth | TS | DOM analysis | High | |
| IND-1099 | Time and Sales | TS | T&S analysis | Medium | |
| IND-1100 | Volume Delta (TS) | TS | Buy/sell volume | Medium | |

## Priority 110: Worden Indicators (TC2000)

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1101 | Balance of Power | Worden | BOP indicator | Medium | |
| IND-1102 | Time Segmented Volume | Worden | TSV | Medium | |
| IND-1103 | MoneyStream | Worden | Cumulative flow | Medium | |
| IND-1104 | Worden Stochastics | Worden | Modified stochastic | Medium | |
| IND-1105 | Linear Regression Percent | Worden | Regression oscillator | Medium | |
| IND-1106 | Personal Criteria Formulas | Worden | PCF custom | High | |

## Priority 111: eSignal Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1107 | Advanced GET Elliott | eSignal | Auto Elliott Wave | High | |
| IND-1108 | Profit Taking Index | eSignal | PTI | Medium | |
| IND-1109 | Type One Trade | eSignal | GET trade | High | |
| IND-1110 | Type Two Trade | eSignal | GET trade | High | |
| IND-1111 | Oscillator (GET) | eSignal | GET oscillator | Medium | |
| IND-1112 | XTL | eSignal | Trend indicator | Medium | |

## Priority 112: CBOE Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1113 | VIX | CBOE | S&P 500 volatility | Low | |
| IND-1114 | VIX9D | CBOE | 9-day VIX | Low | |
| IND-1115 | VIX3M | CBOE | 3-month VIX | Low | |
| IND-1116 | VIX6M | CBOE | 6-month VIX | Low | |
| IND-1117 | VIX1Y | CBOE | 1-year VIX | Low | |
| IND-1118 | VVIX | CBOE | VIX of VIX | Medium | |
| IND-1119 | SKEW | CBOE | Tail risk index | Medium | |
| IND-1120 | VIX Term Structure | CBOE | VIX curve | Medium | |
| IND-1121 | RVX | CBOE | Russell 2000 VIX | Low | |
| IND-1122 | VXN | CBOE | Nasdaq VIX | Low | |
| IND-1123 | VXD | CBOE | Dow VIX | Low | |
| IND-1124 | OVX | CBOE | Oil VIX | Low | |
| IND-1125 | GVZ | CBOE | Gold VIX | Low | |
| IND-1126 | EVZ | CBOE | Euro VIX | Low | |
| IND-1127 | VXEEM | CBOE | EM VIX | Low | |
| IND-1128 | VXEFA | CBOE | EAFE VIX | Low | |
| IND-1129 | Put/Call Ratio (Equity) | CBOE | Equity P/C | Low | |
| IND-1130 | Put/Call Ratio (Index) | CBOE | Index P/C | Low | |
| IND-1131 | Put/Call Ratio (Total) | CBOE | Total P/C | Low | |
| IND-1132 | TYVIX | CBOE | Treasury VIX | Low | |

## Priority 113: CME Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1133 | CME FedWatch | CME | Rate probability | Medium | |
| IND-1134 | Open Interest | CME | Futures OI | Low | |
| IND-1135 | Volume Profile (CME) | CME | Futures VP | High | |
| IND-1136 | Commitment of Traders | CME | COT report | Medium | |
| IND-1137 | Basis | CME | Futures - Spot | Low | |
| IND-1138 | Calendar Spread | CME | Front - Back | Low | |
| IND-1139 | Contango Measure | CME | Curve slope | Medium | |
| IND-1140 | Roll Yield | CME | Roll return | Medium | |

## Priority 114: S&P/MSCI Indices

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1141 | S&P 500 | Index | Large cap US | Low | |
| IND-1142 | S&P 400 | Index | Mid cap US | Low | |
| IND-1143 | S&P 600 | Index | Small cap US | Low | |
| IND-1144 | S&P 1500 | Index | Total US | Low | |
| IND-1145 | MSCI World | Index | Developed markets | Low | |
| IND-1146 | MSCI EAFE | Index | Ex-US developed | Low | |
| IND-1147 | MSCI EM | Index | Emerging markets | Low | |
| IND-1148 | MSCI ACWI | Index | All countries | Low | |
| IND-1149 | MSCI Frontier | Index | Frontier markets | Low | |
| IND-1150 | Factor Indices | Index | Smart beta | Medium | |

## Priority 115: Fixed Income Indices

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1151 | Bloomberg US Aggregate | FI Index | US bonds | Low | |
| IND-1152 | Bloomberg Global Aggregate | FI Index | Global bonds | Low | |
| IND-1153 | Bloomberg High Yield | FI Index | Junk bonds | Low | |
| IND-1154 | Bloomberg IG Corporate | FI Index | IG corporates | Low | |
| IND-1155 | Bloomberg TIPS | FI Index | Inflation-linked | Low | |
| IND-1156 | ICE BofA EM | FI Index | EM bonds | Low | |
| IND-1157 | JPM EMBI | FI Index | EM sovereigns | Low | |
| IND-1158 | JPM GBI-EM | FI Index | EM local | Low | |
| IND-1159 | Credit Suisse Leveraged Loan | FI Index | Loans | Low | |
| IND-1160 | iTraxx | FI Index | European CDS | Low | |
| IND-1161 | CDX | FI Index | US CDS | Low | |

## Priority 116: Commodity Indices

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1162 | Bloomberg Commodity | Cmdty Index | Broad commodities | Low | |
| IND-1163 | S&P GSCI | Cmdty Index | Goldman commodities | Low | |
| IND-1164 | CRB Index | Cmdty Index | Reuters commodities | Low | |
| IND-1165 | Baltic Dry Index | Cmdty Index | Shipping rates | Low | |
| IND-1166 | DBLCI | Cmdty Index | Deutsche Bank | Low | |
| IND-1167 | Rogers International | Cmdty Index | Jim Rogers | Low | |
| IND-1168 | Energy Select | Cmdty Index | Energy sub-index | Low | |
| IND-1169 | Metals Select | Cmdty Index | Metals sub-index | Low | |
| IND-1170 | Agriculture Select | Cmdty Index | Ags sub-index | Low | |

## Priority 117: Currency Indices

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1171 | DXY (Dollar Index) | FX Index | USD vs 6 majors | Low | |
| IND-1172 | Bloomberg Dollar Index | FX Index | USD vs 10 | Low | |
| IND-1173 | Fed Broad Dollar | FX Index | Trade-weighted | Low | |
| IND-1174 | Fed Major Currencies | FX Index | Major currencies | Low | |
| IND-1175 | Euro Index | FX Index | EUR trade-weighted | Low | |
| IND-1176 | JPY Index | FX Index | JPY trade-weighted | Low | |
| IND-1177 | GBP Index | FX Index | GBP trade-weighted | Low | |
| IND-1178 | EM Currency Index | FX Index | EM FX basket | Low | |
| IND-1179 | Carry Index | FX Index | Carry trade index | Medium | |
| IND-1180 | Momentum FX Index | FX Index | Trend-following FX | Medium | |

## Priority 118: Crypto Indices

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1181 | Bitcoin Dominance | Crypto Index | BTC market share | Low | |
| IND-1182 | Altcoin Season Index | Crypto Index | Alt vs BTC | Medium | |
| IND-1183 | Fear & Greed Index (Crypto) | Crypto Index | Sentiment | Medium | |
| IND-1184 | Crypto Volatility Index | Crypto Index | CVI | Medium | |
| IND-1185 | DeFi Pulse Index | Crypto Index | DeFi basket | Low | |
| IND-1186 | Metaverse Index | Crypto Index | Metaverse tokens | Low | |
| IND-1187 | NFT Index | Crypto Index | NFT-related tokens | Low | |
| IND-1188 | Layer 1 Index | Crypto Index | L1 chains | Low | |
| IND-1189 | Layer 2 Index | Crypto Index | L2 solutions | Low | |
| IND-1190 | Bitcoin Rainbow Chart | Crypto Index | Log regression bands | Medium | |

## Priority 119: Hedge Fund Indices

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1191 | HFRI Fund Weighted | HF Index | Broad hedge funds | Low | |
| IND-1192 | HFRI Equity Hedge | HF Index | Long/short equity | Low | |
| IND-1193 | HFRI Event Driven | HF Index | Event strategies | Low | |
| IND-1194 | HFRI Macro | HF Index | Global macro | Low | |
| IND-1195 | HFRI Relative Value | HF Index | Arbitrage | Low | |
| IND-1196 | Barclay CTA Index | HF Index | Managed futures | Low | |
| IND-1197 | SG CTA Index | HF Index | Société Générale | Low | |
| IND-1198 | SG Trend Index | HF Index | Trend followers | Low | |
| IND-1199 | Eurekahedge | HF Index | Global HF | Low | |
| IND-1200 | Credit Suisse HF | HF Index | CS indices | Low | |

## Priority 120: Alternative Data Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1201 | Satellite Parking Lots | Alt Data | Retail traffic | High | |
| IND-1202 | Satellite Oil Storage | Alt Data | Inventory proxy | High | |
| IND-1203 | Satellite Ship Tracking | Alt Data | Trade flows | High | |
| IND-1204 | Satellite Crop Analysis | Alt Data | Agriculture | High | |
| IND-1205 | Credit Card Transactions | Alt Data | Consumer spending | High | |
| IND-1206 | App Store Rankings | Alt Data | Digital products | Medium | |
| IND-1207 | Web Scraping Price | Alt Data | E-commerce prices | Medium | |
| IND-1208 | Job Postings | Alt Data | Employment trends | Medium | |
| IND-1209 | Patent Filings | Alt Data | Innovation | Medium | |
| IND-1210 | FDA Filings | Alt Data | Drug approvals | Medium | |
| IND-1211 | SEC Filings NLP | Alt Data | 10-K/10-Q analysis | High | |
| IND-1212 | Earnings Call NLP | Alt Data | Transcript analysis | High | |
| IND-1213 | News Sentiment | Alt Data | News NLP | High | |
| IND-1214 | Social Media Sentiment | Alt Data | Twitter/Reddit | High | |
| IND-1215 | Google Trends | Alt Data | Search interest | Medium | |
| IND-1216 | Yelp Reviews | Alt Data | Consumer sentiment | Medium | |
| IND-1217 | Glassdoor Reviews | Alt Data | Employee sentiment | Medium | |
| IND-1218 | Flight Data | Alt Data | Travel demand | Medium | |
| IND-1219 | Weather Data | Alt Data | Weather impact | Medium | |
| IND-1220 | Pollution Data | Alt Data | Industrial activity | Medium | |

## Priority 121: Insurance & Actuarial

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1221 | Combined Ratio | Insurance | Underwriting measure | Low | |
| IND-1222 | Loss Ratio | Insurance | Claims / Premium | Low | |
| IND-1223 | Expense Ratio | Insurance | Expenses / Premium | Low | |
| IND-1224 | Reserve Development | Insurance | Reserve changes | Medium | |
| IND-1225 | Cat Bond Spread | Insurance | Catastrophe risk | Medium | |
| IND-1226 | Reinsurance Rate-on-Line | Insurance | Pricing | Medium | |
| IND-1227 | ILS Performance | Insurance | Cat bond returns | Medium | |
| IND-1228 | Swiss Re Cat Index | Insurance | Catastrophe losses | Low | |
| IND-1229 | PCS Index | Insurance | Property claims | Low | |
| IND-1230 | Bermuda Cat Bond | Insurance | Bermuda index | Low | |

## Priority 122: Real Assets Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1231 | Timberland Index | Real Assets | Forest returns | Low | |
| IND-1232 | Farmland Index | Real Assets | Agricultural land | Low | |
| IND-1233 | Infrastructure Index | Real Assets | Infrastructure | Low | |
| IND-1234 | Art Index | Real Assets | Art market | High | |
| IND-1235 | Wine Index | Real Assets | Fine wine | Medium | |
| IND-1236 | Collectibles Index | Real Assets | Collectibles | High | |
| IND-1237 | Diamond Index | Real Assets | Diamond prices | Medium | |
| IND-1238 | Carbon Credit Index | Real Assets | Carbon market | Low | |
| IND-1239 | Water Rights Index | Real Assets | Water futures | Low | |
| IND-1240 | Renewable Energy Index | Real Assets | Clean energy | Low | |

## Priority 123: Quantitative Signals

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1241 | Low Vol Anomaly | Quant | Low volatility premium | Medium | |
| IND-1242 | Betting Against Beta | Quant | BAB factor | High | |
| IND-1243 | Quality Minus Junk | Quant | QMJ factor | High | |
| IND-1244 | Profitability Factor | Quant | Operating profitability | Medium | |
| IND-1245 | Investment Factor | Quant | Asset growth | Medium | |
| IND-1246 | Accruals Anomaly | Quant | Earnings quality | Medium | |
| IND-1247 | Net Stock Issues | Quant | Equity issuance | Medium | |
| IND-1248 | Asset Turnover | Quant | Sales efficiency | Low | |
| IND-1249 | Return on Assets | Quant | ROA | Low | |
| IND-1250 | Gross Profitability | Quant | Novy-Marx | Medium | |
| IND-1251 | Book to Market | Quant | Value factor | Low | |
| IND-1252 | Earnings to Price | Quant | E/P ratio | Low | |
| IND-1253 | Cash Flow to Price | Quant | CF/P ratio | Low | |
| IND-1254 | Sales to Price | Quant | S/P ratio | Low | |
| IND-1255 | Dividend to Price | Quant | D/P ratio | Low | |
| IND-1256 | Enterprise Multiple | Quant | EV/EBITDA | Medium | |
| IND-1257 | Debt to Equity | Quant | Leverage | Low | |
| IND-1258 | Interest Coverage | Quant | EBIT/Interest | Low | |
| IND-1259 | Current Ratio | Quant | Liquidity | Low | |
| IND-1260 | Quick Ratio | Quant | Acid test | Low | |

## Priority 124: Technical Analysis Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1261 | Double Smoothed Momentum | TA Ext | DSM | Medium | |
| IND-1262 | Triple Smoothed EMA | TA Ext | TRIX variant | Medium | |
| IND-1263 | Zero Lag TEMA | TA Ext | ZLTEMA | Medium | |
| IND-1264 | Hull Trend | TA Ext | HMA trend | Medium | |
| IND-1265 | Modular Filter | TA Ext | Jurik-style | High | |
| IND-1266 | Bandpass Filter | TA Ext | Ehlers bandpass | High | |
| IND-1267 | Highpass Filter | TA Ext | Ehlers highpass | High | |
| IND-1268 | Lowpass Filter | TA Ext | Butterworth | Medium | |
| IND-1269 | SuperSmoother 2-Pole | TA Ext | Ehlers SS2 | Medium | |
| IND-1270 | SuperSmoother 3-Pole | TA Ext | Ehlers SS3 | Medium | |
| IND-1271 | Instantaneous Trendline | TA Ext | Ehlers | High | |
| IND-1272 | Sinewave Indicator | TA Ext | Ehlers sine | High | |
| IND-1273 | Even Better Sinewave | TA Ext | Ehlers EBS | High | |
| IND-1274 | Autocorrelation Periodogram | TA Ext | Ehlers | High | |
| IND-1275 | Dominant Cycle | TA Ext | Ehlers DC | High | |
| IND-1276 | Cycle Period | TA Ext | Ehlers CP | High | |
| IND-1277 | Adaptive Cyber Cycle | TA Ext | Ehlers ACC | High | |
| IND-1278 | Adaptive CG | TA Ext | Ehlers ACG | High | |
| IND-1279 | Adaptive RSI (Ehlers) | TA Ext | Ehlers ARSI | High | |
| IND-1280 | Adaptive Stochastic (Ehlers) | TA Ext | Ehlers AS | High | |

## Priority 125: Market Microstructure Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1281 | PIN (Probability Informed) | Microstr | Information trading | High | |
| IND-1282 | VPIN | Microstr | Volume-synced PIN | High | |
| IND-1283 | Order Imbalance | Microstr | Buy-sell imbalance | Medium | |
| IND-1284 | Trade Imbalance | Microstr | Trade direction | Medium | |
| IND-1285 | Kyle's Lambda | Microstr | Price impact | High | |
| IND-1286 | Hasbrouck Information Share | Microstr | Price discovery | High | |
| IND-1287 | Gonzalo-Granger | Microstr | Cointegration share | High | |
| IND-1288 | Realized Variance | Microstr | High-freq variance | Medium | |
| IND-1289 | Bipower Variation | Microstr | Jump-robust var | High | |
| IND-1290 | Realized Kernel | Microstr | Noise-robust var | High | |
| IND-1291 | Realized Quarticity | Microstr | Fourth moment | High | |
| IND-1292 | Jump Test | Microstr | Barndorff-Nielsen | High | |
| IND-1293 | Noise Ratio | Microstr | Microstructure noise | High | |
| IND-1294 | Signature Plot | Microstr | Sampling frequency | High | |
| IND-1295 | Two-Scale Variance | Microstr | Zhang TSRV | High | |

## Priority 126: Behavioral Finance

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1296 | Disposition Effect | Behavioral | Sell winners early | High | |
| IND-1297 | Attention Anomaly | Behavioral | Attention-driven | High | |
| IND-1298 | Lottery Preference | Behavioral | Skewness preference | High | |
| IND-1299 | Anchoring Bias | Behavioral | 52-week high anchor | Medium | |
| IND-1300 | Herding Measure | Behavioral | Correlated trading | High | |
| IND-1301 | Overconfidence Measure | Behavioral | Excess trading | High | |
| IND-1302 | Sentiment Spread | Behavioral | Bull-bear spread | Medium | |
| IND-1303 | Retail Flow | Behavioral | Retail order flow | Medium | |
| IND-1304 | Noise Trader Risk | Behavioral | Sentiment risk | High | |
| IND-1305 | Limits to Arbitrage | Behavioral | Arb impediments | High | |

## Priority 127: Academic Research Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1306 | Fama-French 3 Factor | Academic | MKT, SMB, HML | High | |
| IND-1307 | Fama-French 5 Factor | Academic | +RMW, CMA | High | |
| IND-1308 | Carhart 4 Factor | Academic | +Momentum | High | |
| IND-1309 | Q-Factor Model | Academic | Hou-Xue-Zhang | High | |
| IND-1310 | Stambaugh-Yuan 4 Factor | Academic | Mispricing factors | High | |
| IND-1311 | Daniel-Hirshleifer-Sun 3 Factor | Academic | Behavioral factors | High | |
| IND-1312 | Pastor-Stambaugh Liquidity | Academic | Liquidity factor | High | |
| IND-1313 | Amihud Illiquidity | Academic | ILLIQ | Medium | |
| IND-1314 | Lo-MacKinlay Variance Ratio | Academic | Random walk test | High | |
| IND-1315 | Kim-Shephard-Chib SV | Academic | Stochastic vol | High | |
| IND-1316 | Heston Model | Academic | Stochastic vol | High | |
| IND-1317 | SABR Model | Academic | Stochastic alpha beta rho | High | |
| IND-1318 | Local Vol Dupire | Academic | Local volatility | High | |
| IND-1319 | Bates Jump-Diffusion | Academic | Jump model | High | |
| IND-1320 | Variance Gamma | Academic | VG model | High | |

## Priority 128: Risk Management Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1321 | Historical VaR | Risk Mgmt | Percentile VaR | Medium | |
| IND-1322 | Parametric VaR | Risk Mgmt | Normal VaR | Medium | |
| IND-1323 | Monte Carlo VaR | Risk Mgmt | Simulation VaR | High | |
| IND-1324 | Expected Shortfall | Risk Mgmt | CVaR / ES | Medium | |
| IND-1325 | Incremental VaR | Risk Mgmt | Marginal impact | High | |
| IND-1326 | Component VaR | Risk Mgmt | VaR decomposition | High | |
| IND-1327 | Stress VaR | Risk Mgmt | Stressed period | High | |
| IND-1328 | Stressed ES | Risk Mgmt | Stressed ES | High | |
| IND-1329 | Marginal ES | Risk Mgmt | Position ES | High | |
| IND-1330 | Risk Contribution | Risk Mgmt | Risk decomposition | High | |
| IND-1331 | Risk Budgeting | Risk Mgmt | Risk allocation | High | |
| IND-1332 | Tail Risk | Risk Mgmt | Extreme events | High | |
| IND-1333 | Drawdown Duration | Risk Mgmt | DD length | Medium | |
| IND-1334 | Time Under Water | Risk Mgmt | Recovery time | Medium | |
| IND-1335 | Conditional Drawdown | Risk Mgmt | CDaR | High | |

## Priority 129: Trading System Signals

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1336 | Golden Cross | Signal | 50/200 MA cross | Low | |
| IND-1337 | Death Cross | Signal | 200/50 MA cross | Low | |
| IND-1338 | MACD Cross | Signal | MACD/Signal cross | Low | |
| IND-1339 | RSI Overbought | Signal | RSI > 70 | Low | |
| IND-1340 | RSI Oversold | Signal | RSI < 30 | Low | |
| IND-1341 | Bollinger Squeeze | Signal | BB inside KC | Medium | |
| IND-1342 | Breakout Signal | Signal | N-day high/low | Low | |
| IND-1343 | Trend Following | Signal | CTA-style | Medium | |
| IND-1344 | Mean Reversion | Signal | Counter-trend | Medium | |
| IND-1345 | Pairs Trade Signal | Signal | Spread entry | High | |
| IND-1346 | Carry Trade Signal | Signal | Rate differential | Medium | |
| IND-1347 | Momentum Signal | Signal | ROC-based | Low | |
| IND-1348 | Value Signal | Signal | Valuation-based | Medium | |
| IND-1349 | Quality Signal | Signal | Quality-based | Medium | |
| IND-1350 | Composite Signal | Signal | Multi-factor | High | |

## Priority 130: All Historical TA Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1351 | McClellan Oscillator | Historical | Breadth oscillator | Medium | |
| IND-1352 | McClellan Summation | Historical | Cumulative McClellan | Medium | |
| IND-1353 | Haurlan Index | Historical | Smoothed breadth | Medium | |
| IND-1354 | Schultz A/T | Historical | Advances/Total | Low | |
| IND-1355 | Hughes Breadth | Historical | Breadth momentum | Medium | |
| IND-1356 | Sibbett Demand Index | Historical | Volume analysis | Medium | |
| IND-1357 | Fosback's NVI | Historical | Negative Volume Index | Medium | |
| IND-1358 | Fosback's PVI | Historical | Positive Volume Index | Medium | |
| IND-1359 | Herrick Payoff Index | Historical | Open interest | High | |
| IND-1360 | TRIX (Hutson) | Historical | Triple EMA ROC | Medium | |
| IND-1361 | Coppock Curve | Historical | Long-term momentum | Medium | |
| IND-1362 | Lambert CCI | Historical | Commodity Channel | Medium | |
| IND-1363 | Gopalakrishnan Range Index | Historical | Range measure | Medium | |
| IND-1364 | Natenberg Vol | Historical | Volatility estimate | High | |
| IND-1365 | Garman-Klass | Historical | OHLC volatility | Medium | |
| IND-1366 | Parkinson | Historical | H-L volatility | Low | |
| IND-1367 | Rogers-Satchell | Historical | Drift-adjusted vol | Medium | |
| IND-1368 | Yang-Zhang | Historical | Overnight vol | Medium | |
| IND-1369 | Chaikin Persistence | Historical | Money flow streak | Medium | |
| IND-1370 | Elder Thermometer | Historical | Volatility | Medium | |
| IND-1371 | Worden Money Stream | Historical | Volume analysis | Medium | |
| IND-1372 | Blau TSI | Historical | True Strength Index | Medium | |
| IND-1373 | Blau Ergodic | Historical | Ergodic oscillator | Medium | |
| IND-1374 | Blau Double Smooth | Historical | DSS | Medium | |
| IND-1375 | Blau True Range | Historical | TR variants | Medium | |
| IND-1376 | Chande TrendScore | Historical | Trend measure | Medium | |
| IND-1377 | Chande Volatility Index | Historical | VIX-like | Medium | |
| IND-1378 | Dorsey RMI | Historical | Relative Momentum | Medium | |
| IND-1379 | Ehlers Correlation Trend | Historical | CT indicator | High | |
| IND-1380 | Ehlers Convolution | Historical | Filter | High | |
| IND-1381 | Ehlers Decycler Oscillator | Historical | Trend | High | |
| IND-1382 | Ehlers Distance Coefficient | Historical | Filter | High | |
| IND-1383 | Ehlers Fisher Transform | Historical | Normalization | Medium | |
| IND-1384 | Ehlers Inverse Fisher | Historical | Inverse transform | Medium | |
| IND-1385 | Ehlers Predictive Moving Average | Historical | Predictive MA | High | |
| IND-1386 | Ehlers Reflex | Historical | Cycle indicator | High | |
| IND-1387 | Ehlers Relative Vigor Index | Historical | RVI | Medium | |
| IND-1388 | Ehlers Stochastic CG | Historical | SCG | High | |
| IND-1389 | Ehlers Super Smoother | Historical | SS filter | Medium | |
| IND-1390 | Ehlers Swiss Army Knife | Historical | Configurable filter | High | |
| IND-1391 | Ehlers Trend Flex | Historical | Trend indicator | High | |
| IND-1392 | Ehlers Trend Vigor | Historical | Trend strength | High | |
| IND-1393 | Ehlers Voss Predictor | Historical | Prediction | High | |
| IND-1394 | Ehlers Zero Lag | Historical | Zero-lag MA | Medium | |
| IND-1395 | Heikin Ashi Smoothed | Historical | Smoothed HA | Medium | |
| IND-1396 | Improved RSI | Historical | IRSI | Medium | |
| IND-1397 | LBR 3/10 | Historical | Raschke oscillator | Low | |
| IND-1398 | LBR Anti | Historical | Anti pattern | Medium | |
| IND-1399 | Laguerre Filter | Historical | Digital filter | High | |
| IND-1400 | Laguerre RSI | Historical | LRSI | High | |

## Priority 131: Additional Ehlers DSP Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1401 | Ehlers Adaptive Cyber Cycle | DSP | Adaptive cycle detection | High | |
| IND-1402 | Ehlers AGC | DSP | Automatic Gain Control | High | |
| IND-1403 | Ehlers Bandpass Filter | DSP | Two-pole bandpass | High | |
| IND-1404 | Ehlers Butterworth | DSP | Butterworth filter 2/3-pole | High | |
| IND-1405 | Ehlers Center of Gravity | DSP | CG oscillator | Medium | |
| IND-1406 | Ehlers Correlation Angle | DSP | Trend detection | High | |
| IND-1407 | Ehlers Cycle Period | DSP | Dominant cycle | High | |
| IND-1408 | Ehlers Damping Index | DSP | Damping measure | High | |
| IND-1409 | Ehlers Discriminator | DSP | Signal classification | High | |
| IND-1410 | Ehlers Even Better Sinewave | DSP | EBSW indicator | High | |
| IND-1411 | Ehlers Highpass Filter | DSP | DC removal | Medium | |
| IND-1412 | Ehlers Instantaneous Trendline | DSP | ITL | High | |
| IND-1413 | Ehlers Leading Indicator | DSP | Phase-shifted | High | |
| IND-1414 | Ehlers MAMA/FAMA | DSP | Adaptive MA pair | High | |
| IND-1415 | Ehlers Market Mode | DSP | Trend/cycle mode | High | |
| IND-1416 | Ehlers Modified Stochastic | DSP | Digital stochastic | Medium | |
| IND-1417 | Ehlers Phase Accumulation | DSP | Phase measure | High | |
| IND-1418 | Ehlers Quotient Transform | DSP | Signal normalization | Medium | |
| IND-1419 | Ehlers Sinewave Indicator | DSP | Cycle detection | High | |
| IND-1420 | Ehlers SNR | DSP | Signal-to-noise ratio | High | |
| IND-1421 | Ehlers Spearman Indicator | DSP | Rank correlation | Medium | |
| IND-1422 | Ehlers Stochastic RSI | DSP | Digital StochRSI | Medium | |
| IND-1423 | Ehlers Super Passband | DSP | Enhanced bandpass | High | |
| IND-1424 | Ehlers Trendflex | DSP | Trend oscillator | High | |
| IND-1425 | Ehlers Two-Pole Filter | DSP | Smoothing filter | Medium | |
| IND-1426 | Ehlers Unified DC | DSP | DC indicator | High | |
| IND-1427 | Ehlers Universal Oscillator | DSP | Configurable osc | High | |
| IND-1428 | Ehlers Variable Bandpass | DSP | Adaptive bandpass | High | |
| IND-1429 | Ehlers Voss Filter | DSP | Voss predictor | High | |
| IND-1430 | Ehlers Weighted MA | DSP | EWMA | Medium | |

## Priority 132: Japanese Candlestick Continuation Patterns

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1431 | Rising Three Methods | Continuation | Bullish 5-bar pattern | Medium | |
| IND-1432 | Falling Three Methods | Continuation | Bearish 5-bar pattern | Medium | |
| IND-1433 | Mat Hold | Continuation | Bull continuation | Medium | |
| IND-1434 | Upside Gap Three Methods | Continuation | Gap continuation | Medium | |
| IND-1435 | Downside Gap Three Methods | Continuation | Gap continuation | Medium | |
| IND-1436 | Tasuki Gap | Continuation | Gap pattern | Medium | |
| IND-1437 | Side-by-Side White Lines | Continuation | Bull pattern | Medium | |
| IND-1438 | Separating Lines | Continuation | Trend continuation | Medium | |
| IND-1439 | In Neck | Continuation | Bearish | Low | |
| IND-1440 | On Neck | Continuation | Bearish | Low | |
| IND-1441 | Thrusting | Continuation | Bearish | Low | |
| IND-1442 | Window (Gap) | Continuation | Japanese gap | Low | |
| IND-1443 | Kicking | Continuation | Strong signal | Medium | |
| IND-1444 | Kicking Bull/Bear | Continuation | Directional kicking | Medium | |
| IND-1445 | Meeting Lines | Continuation | Reversal | Medium | |

## Priority 133: Japanese Candlestick Rare Patterns

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1446 | Unique Three River | Candlestick | Rare bottom | Medium | |
| IND-1447 | Concealing Baby Swallow | Candlestick | Rare bullish | High | |
| IND-1448 | Ladder Top | Candlestick | Reversal | Medium | |
| IND-1449 | Ladder Bottom | Candlestick | Reversal | Medium | |
| IND-1450 | Tower Top | Candlestick | Reversal | Medium | |
| IND-1451 | Tower Bottom | Candlestick | Reversal | Medium | |
| IND-1452 | Two Crows | Candlestick | Bearish | Medium | |
| IND-1453 | Upside Gap Two Crows | Candlestick | Bearish | Medium | |
| IND-1454 | Three Inside Up | Candlestick | Bull reversal | Medium | |
| IND-1455 | Three Inside Down | Candlestick | Bear reversal | Medium | |
| IND-1456 | Three Outside Up | Candlestick | Bull reversal | Medium | |
| IND-1457 | Three Outside Down | Candlestick | Bear reversal | Medium | |
| IND-1458 | Three Stars in South | Candlestick | Rare bullish | High | |
| IND-1459 | Advance Block | Candlestick | Weakening trend | Medium | |
| IND-1460 | Stalled Pattern | Candlestick | Slowing trend | Medium | |

## Priority 134: Advanced Chart Patterns

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1461 | Adam & Eve Double | Pattern | Variation of double | High | |
| IND-1462 | Pipe Top | Pattern | Reversal | Medium | |
| IND-1463 | Pipe Bottom | Pattern | Reversal | Medium | |
| IND-1464 | Scallops | Pattern | Ascending/descending | High | |
| IND-1465 | Big W | Pattern | Bottom pattern | High | |
| IND-1466 | Big M | Pattern | Top pattern | High | |
| IND-1467 | Three Drives | Pattern | Harmonic | High | |
| IND-1468 | AB=CD | Pattern | Harmonic pattern | High | |
| IND-1469 | Shark | Pattern | Harmonic | High | |
| IND-1470 | Cypher | Pattern | Harmonic | High | |
| IND-1471 | Anti-Pattern | Pattern | Harmonic counter | High | |
| IND-1472 | 5-0 Pattern | Pattern | Harmonic | High | |
| IND-1473 | Alternate AB=CD | Pattern | Harmonic variation | High | |
| IND-1474 | Extended Gartley | Pattern | Harmonic | High | |
| IND-1475 | Deep Crab | Pattern | Harmonic | High | |

## Priority 135: Market Profile Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1476 | TPO Count | Profile | Time at Price | Medium | |
| IND-1477 | Value Area High | Profile | VAH | Medium | |
| IND-1478 | Value Area Low | Profile | VAL | Medium | |
| IND-1479 | Point of Control | Profile | POC | Medium | |
| IND-1480 | Initial Balance | Profile | First hour range | Medium | |
| IND-1481 | Single Prints | Profile | Low volume nodes | Medium | |
| IND-1482 | Poor High | Profile | Weak rejection | Medium | |
| IND-1483 | Poor Low | Profile | Weak rejection | Medium | |
| IND-1484 | Excess | Profile | Strong rejection | Medium | |
| IND-1485 | Ledge | Profile | Balance area | Medium | |
| IND-1486 | Profile Distribution | Profile | Shape classification | High | |
| IND-1487 | Profile Migration | Profile | Shift detection | High | |
| IND-1488 | Virgin POC | Profile | Untested POC | Medium | |
| IND-1489 | Naked VPOC | Profile | Untested volume POC | Medium | |
| IND-1490 | Composite Profile | Profile | Multi-day profile | High | |

## Priority 136: Footprint Chart Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1491 | Bid/Ask Delta | Footprint | Buy vs sell volume | High | |
| IND-1492 | Cumulative Delta | Footprint | Running delta | High | |
| IND-1493 | Delta Imbalance | Footprint | Stacked imbalance | High | |
| IND-1494 | Finished Auction | Footprint | Exhaustion signal | High | |
| IND-1495 | Unfinished Auction | Footprint | Continuation | High | |
| IND-1496 | Stacked Bid/Ask | Footprint | Order flow strength | High | |
| IND-1497 | Volume at Price | Footprint | Price-level volume | Medium | |
| IND-1498 | Aggressive Buyer/Seller | Footprint | Market order flow | High | |
| IND-1499 | Absorption Pattern | Footprint | Limit order absorption | High | |
| IND-1500 | Iceberg Order Detection | Footprint | Hidden order size | High | |

## Priority 137: Depth of Market (DOM) Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1501 | Book Imbalance | DOM | Bid/Ask ratio | Medium | |
| IND-1502 | Book Pressure | DOM | Depth weighted | High | |
| IND-1503 | Spoofing Detection | DOM | Order manipulation | High | |
| IND-1504 | Layering Detection | DOM | Multi-level spoof | High | |
| IND-1505 | Quote Stuffing | DOM | Noise detection | High | |
| IND-1506 | Pull Rate | DOM | Order cancellation | High | |
| IND-1507 | Fill Rate | DOM | Execution probability | High | |
| IND-1508 | Queue Position | DOM | Time priority | High | |
| IND-1509 | Book Resilience | DOM | Recovery speed | High | |
| IND-1510 | Depth Momentum | DOM | Book change rate | High | |

## Priority 138: Tick Data Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1511 | Tick Volume | Tick | Count per bar | Low | |
| IND-1512 | Tick Speed | Tick | Ticks per second | Medium | |
| IND-1513 | Trade Intensity | Tick | Arrival rate | Medium | |
| IND-1514 | Trade Size Distribution | Tick | Lot size analysis | Medium | |
| IND-1515 | Dollar Volume | Tick | Price × Volume | Low | |
| IND-1516 | VWAP Deviation | Tick | Distance from VWAP | Medium | |
| IND-1517 | Trade Clustering | Tick | Time clustering | High | |
| IND-1518 | Price Impact | Tick | Trade impact | High | |
| IND-1519 | Spread Dynamics | Tick | Spread behavior | Medium | |
| IND-1520 | Trade Classification | Tick | Lee-Ready algorithm | Medium | |

## Priority 139: Implied Volatility Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1521 | IV Rank | Options | Percentile ranking | Medium | |
| IND-1522 | IV Percentile | Options | Historical percentile | Medium | |
| IND-1523 | IV Skew Slope | Options | Put-call skew | High | |
| IND-1524 | Term Structure Slope | Options | Time structure | High | |
| IND-1525 | Vol of Vol | Options | Volatility of IV | High | |
| IND-1526 | Risk Reversal | Options | 25-delta skew | High | |
| IND-1527 | Butterfly Spread | Options | Wings pricing | High | |
| IND-1528 | IV Curve Fitting | Options | SABR/SVI model | High | |
| IND-1529 | Variance Swap Level | Options | Fair variance | High | |
| IND-1530 | Variance Risk Premium | Options | Realized vs implied | High | |

## Priority 140: Earnings & Events Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1531 | Earnings Momentum | Fundamental | EPS surprise | Medium | |
| IND-1532 | Estimate Revisions | Fundamental | Analyst changes | Medium | |
| IND-1533 | Pre-Earnings Drift | Fundamental | PEAD | Medium | |
| IND-1534 | Post-Earnings Drift | Fundamental | PEAD | Medium | |
| IND-1535 | Earnings Quality | Fundamental | Accrual analysis | High | |
| IND-1536 | Guidance Sentiment | Fundamental | Management tone | High | |
| IND-1537 | Whisper Number | Fundamental | Street estimate | Medium | |
| IND-1538 | Earnings Surprise Score | Fundamental | Standardized surprise | Medium | |
| IND-1539 | Analyst Dispersion | Fundamental | Estimate spread | Medium | |
| IND-1540 | Revision Momentum | Fundamental | Estimate trend | Medium | |

## Priority 141: Sector & Industry Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1541 | Sector Rotation | Sector | Business cycle | Medium | |
| IND-1542 | Industry Momentum | Sector | Relative strength | Medium | |
| IND-1543 | Sector Beta | Sector | Market sensitivity | Medium | |
| IND-1544 | Sector Correlation | Sector | Cross-sector links | Medium | |
| IND-1545 | Industry Breadth | Sector | Participation | Medium | |
| IND-1546 | Sector Leadership | Sector | Rank tracking | Medium | |
| IND-1547 | Pair Divergence | Sector | Related pair spread | Medium | |
| IND-1548 | Sector RSI | Sector | Sector momentum | Medium | |
| IND-1549 | Industry Score | Sector | Composite ranking | Medium | |
| IND-1550 | GICS Flow | Sector | Capital movement | High | |

## Priority 142: Currency-Specific Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1551 | Real Effective XR | Forex | Trade-weighted | High | |
| IND-1552 | Covered Interest Parity | Forex | CIP deviation | High | |
| IND-1553 | Carry Trade Index | Forex | Yield differential | Medium | |
| IND-1554 | FX Volatility Term | Forex | Vol term structure | High | |
| IND-1555 | Risk Reversal 25D | Forex | Skew measure | High | |
| IND-1556 | Butterfly 25D | Forex | Smile measure | High | |
| IND-1557 | FX Positioning | Forex | IMM positions | Medium | |
| IND-1558 | Dollar Smile | Forex | USD behavior | High | |
| IND-1559 | PPP Deviation | Forex | Purchasing power | Medium | |
| IND-1560 | BEER | Forex | Behavioral equilibrium | High | |

## Priority 143: Bond-Specific Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1561 | Yield Curve Shape | Fixed Income | Steepness/curvature | Medium | |
| IND-1562 | Butterfly Spread | Fixed Income | 2s5s10s | Medium | |
| IND-1563 | Term Premium | Fixed Income | ACM model | High | |
| IND-1564 | Breakeven Inflation | Fixed Income | TIPS spread | Medium | |
| IND-1565 | Real Rate | Fixed Income | TIPS yield | Medium | |
| IND-1566 | Credit Curve | Fixed Income | Spread term structure | High | |
| IND-1567 | Z-Spread | Fixed Income | Zero-vol spread | High | |
| IND-1568 | OAS | Fixed Income | Option-adjusted spread | High | |
| IND-1569 | Key Rate Duration | Fixed Income | Partial durations | High | |
| IND-1570 | Effective Duration | Fixed Income | Price sensitivity | Medium | |

## Priority 144: Commodity-Specific Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1571 | Contango/Backwardation | Commodity | Term structure | Medium | |
| IND-1572 | Roll Yield | Commodity | Calendar spread | Medium | |
| IND-1573 | Basis | Commodity | Cash-futures | Medium | |
| IND-1574 | Convenience Yield | Commodity | Storage value | High | |
| IND-1575 | Inventory Surprise | Commodity | EIA/API | Medium | |
| IND-1576 | Crack Spread | Commodity | Refining margin | Medium | |
| IND-1577 | Crush Spread | Commodity | Soybean processing | Medium | |
| IND-1578 | Spark Spread | Commodity | Power generation | Medium | |
| IND-1579 | Dark Spread | Commodity | Coal power margin | Medium | |
| IND-1580 | Calendar Spread | Commodity | Time spread | Medium | |

## Priority 145: Crypto-Specific Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1581 | Hash Rate | Crypto | Network security | Medium | |
| IND-1582 | Hash Ribbon | Crypto | Miner capitulation | Medium | |
| IND-1583 | Difficulty Adjustment | Crypto | Mining difficulty | Medium | |
| IND-1584 | Realized Cap | Crypto | UTXO-based cap | High | |
| IND-1585 | Realized Price | Crypto | Average cost basis | High | |
| IND-1586 | SOPR | Crypto | Spent Output Profit Ratio | High | |
| IND-1587 | aSOPR | Crypto | Adjusted SOPR | High | |
| IND-1588 | Entity-Adjusted SOPR | Crypto | Entity-based | High | |
| IND-1589 | Coin Days Destroyed | Crypto | Long-term holder | High | |
| IND-1590 | Dormancy Flow | Crypto | Spending behavior | High | |
| IND-1591 | Reserve Risk | Crypto | HODLer conviction | High | |
| IND-1592 | RHODL Ratio | Crypto | Realized HODL | High | |
| IND-1593 | Thermo Cap | Crypto | Miner revenue | High | |
| IND-1594 | Exchange Whale Ratio | Crypto | Large deposits | High | |
| IND-1595 | Stablecoin Supply Ratio | Crypto | Buying power | Medium | |

## Priority 146: Machine Learning Signals Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1596 | Gradient Boosting Signal | ML | XGBoost/LightGBM | High | |
| IND-1597 | Random Forest Prob | ML | Ensemble | High | |
| IND-1598 | SVM Classification | ML | Support Vector | High | |
| IND-1599 | Naive Bayes Score | ML | Probabilistic | Medium | |
| IND-1600 | K-Nearest Neighbors | ML | Instance-based | Medium | |
| IND-1601 | Logistic Regression | ML | Linear classifier | Medium | |
| IND-1602 | Neural Network Signal | ML | MLP/DNN | High | |
| IND-1603 | LSTM Forecast | ML | Sequence model | High | |
| IND-1604 | Transformer Attention | ML | Attention weights | High | |
| IND-1605 | Autoencoder Anomaly | ML | Reconstruction error | High | |
| IND-1606 | Variational Autoencoder | ML | Latent space | High | |
| IND-1607 | GAN-Based | ML | Generative signal | High | |
| IND-1608 | Reinforcement Learning | ML | Policy signal | High | |
| IND-1609 | Ensemble Voting | ML | Multi-model | High | |
| IND-1610 | Stacking Meta-Learner | ML | Meta-learning | High | |

## Priority 147: NLP & Sentiment Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1611 | News Sentiment Score | NLP | Article analysis | High | |
| IND-1612 | Twitter Sentiment | NLP | Social media | High | |
| IND-1613 | Reddit Wallstreetbets | NLP | Retail sentiment | High | |
| IND-1614 | StockTwits Sentiment | NLP | Platform-specific | High | |
| IND-1615 | Earnings Call Tone | NLP | Transcript analysis | High | |
| IND-1616 | SEC Filing Sentiment | NLP | 10-K/10-Q analysis | High | |
| IND-1617 | Management Tone Change | NLP | QoQ comparison | High | |
| IND-1618 | Keyword Frequency | NLP | Term tracking | Medium | |
| IND-1619 | Topic Modeling | NLP | LDA/NMF | High | |
| IND-1620 | Named Entity Extraction | NLP | NER | High | |

## Priority 148: Alternative Data Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1621 | Satellite Parking Lots | Alt Data | Retail traffic | High | |
| IND-1622 | Satellite Oil Storage | Alt Data | Inventory | High | |
| IND-1623 | Ship Tracking (AIS) | Alt Data | Trade flow | High | |
| IND-1624 | Flight Data | Alt Data | Business activity | High | |
| IND-1625 | Credit Card Transactions | Alt Data | Consumer spending | High | |
| IND-1626 | Web Traffic | Alt Data | Site visits | High | |
| IND-1627 | App Downloads | Alt Data | Mobile engagement | High | |
| IND-1628 | App Usage Time | Alt Data | Engagement depth | High | |
| IND-1629 | Job Postings | Alt Data | Hiring indicator | High | |
| IND-1630 | Patent Filings | Alt Data | Innovation | High | |
| IND-1631 | GitHub Activity | Alt Data | Developer interest | Medium | |
| IND-1632 | Google Trends | Alt Data | Search interest | Medium | |
| IND-1633 | Weather Impact | Alt Data | Sector sensitivity | High | |
| IND-1634 | Geolocation Traffic | Alt Data | Foot traffic | High | |
| IND-1635 | Supply Chain Mapping | Alt Data | Supplier network | High | |

## Priority 149: Risk Parity & Portfolio

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1636 | Risk Parity Weight | Portfolio | Equal risk | High | |
| IND-1637 | Minimum Variance Weight | Portfolio | Min vol | High | |
| IND-1638 | Maximum Diversification | Portfolio | Div ratio | High | |
| IND-1639 | Hierarchical Risk Parity | Portfolio | HRP | High | |
| IND-1640 | Black-Litterman | Portfolio | BL views | High | |
| IND-1641 | Mean-CVaR Optimal | Portfolio | Tail risk | High | |
| IND-1642 | Factor Risk Decomposition | Portfolio | Factor attribution | High | |
| IND-1643 | Marginal Risk Contribution | Portfolio | MRC | High | |
| IND-1644 | Component VaR | Portfolio | CVaR breakdown | High | |
| IND-1645 | Tracking Error | Portfolio | Benchmark deviation | Medium | |

## Priority 150: Execution Analytics

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1646 | Implementation Shortfall | Execution | IS | Medium | |
| IND-1647 | VWAP Slippage | Execution | VWAP deviation | Medium | |
| IND-1648 | TWAP Slippage | Execution | TWAP deviation | Medium | |
| IND-1649 | Market Impact Model | Execution | Price impact | High | |
| IND-1650 | Optimal Execution Time | Execution | Almgren-Chriss | High | |
| IND-1651 | Participation Rate | Execution | Volume share | Medium | |
| IND-1652 | Reversion Capture | Execution | Post-trade reversal | Medium | |
| IND-1653 | Spread Capture | Execution | Bid-ask capture | Medium | |
| IND-1654 | Fill Rate Analysis | Execution | Execution probability | Medium | |
| IND-1655 | Venue Analysis | Execution | Exchange comparison | High | |

## Priority 151: Point & Figure Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1656 | P&F Chart Builder | P&F | Box construction | High | |
| IND-1657 | P&F Bullish Signal | P&F | Buy pattern | High | |
| IND-1658 | P&F Bearish Signal | P&F | Sell pattern | High | |
| IND-1659 | P&F Double Top | P&F | Breakout pattern | High | |
| IND-1660 | P&F Double Bottom | P&F | Reversal pattern | High | |
| IND-1661 | P&F Triple Top | P&F | Strong breakout | High | |
| IND-1662 | P&F Triple Bottom | P&F | Strong reversal | High | |
| IND-1663 | P&F Catapult | P&F | Follow-through | High | |
| IND-1664 | P&F High Pole | P&F | Exhaustion | High | |
| IND-1665 | P&F Low Pole | P&F | Capitulation | High | |

## Priority 152: Renko Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1666 | Renko Chart Builder | Renko | Brick construction | Medium | |
| IND-1667 | Renko Trend | Renko | Brick color sequence | Medium | |
| IND-1668 | Renko Double Top/Bottom | Renko | Reversal pattern | Medium | |
| IND-1669 | Renko Breakout | Renko | Continuation | Medium | |
| IND-1670 | Renko ATR-Based | Renko | Volatility-adjusted | Medium | |
| IND-1671 | Renko Brick Count | Renko | Momentum measure | Low | |
| IND-1672 | Renko With Wicks | Renko | Enhanced display | Medium | |
| IND-1673 | Renko-Heikin Hybrid | Renko | Combined smoothing | Medium | |

## Priority 153: Kagi Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1674 | Kagi Chart Builder | Kagi | Line construction | Medium | |
| IND-1675 | Kagi Shoulder | Kagi | Resistance level | Medium | |
| IND-1676 | Kagi Waist | Kagi | Support level | Medium | |
| IND-1677 | Kagi Trend Reversal | Kagi | Thick/thin change | Medium | |
| IND-1678 | Kagi Record High | Kagi | Breakout | Medium | |
| IND-1679 | Kagi Record Low | Kagi | Breakdown | Medium | |
| IND-1680 | Kagi Double Window | Kagi | Strong reversal | Medium | |
| IND-1681 | Kagi Multi-Level | Kagi | Multiple reversals | High | |

## Priority 154: Three Line Break Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1682 | Three Line Break Chart | TLB | Line construction | Medium | |
| IND-1683 | TLB Turnaround | TLB | Reversal signal | Medium | |
| IND-1684 | TLB White Lines | TLB | Bull continuation | Medium | |
| IND-1685 | TLB Black Lines | TLB | Bear continuation | Medium | |
| IND-1686 | TLB Record Session | TLB | Strong move | Medium | |
| IND-1687 | TLB Trend Strength | TLB | Consecutive count | Low | |

## Priority 155: Linear Regression Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1688 | Linear Regression Slope | Regression | Trend rate | Medium | |
| IND-1689 | Linear Regression Intercept | Regression | Starting point | Low | |
| IND-1690 | Linear Regression Forecast | Regression | Prediction | Medium | |
| IND-1691 | Linear Regression R-Squared | Regression | Fit quality | Medium | |
| IND-1692 | Raff Regression Channel | Regression | Standard error bands | Medium | |
| IND-1693 | Polynomial Regression | Regression | Non-linear fit | High | |
| IND-1694 | LOWESS Smoothing | Regression | Local regression | High | |
| IND-1695 | Theil-Sen Estimator | Regression | Robust regression | High | |
| IND-1696 | Piecewise Linear | Regression | Segmented trend | High | |
| IND-1697 | Quantile Regression | Regression | Percentile regression | High | |

## Priority 156: Correlation & Cointegration

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1698 | Rolling Correlation | Correlation | Moving correlation | Medium | |
| IND-1699 | Rank Correlation (Spearman) | Correlation | Non-parametric | Medium | |
| IND-1700 | Kendall Tau | Correlation | Concordance | Medium | |
| IND-1701 | Partial Correlation | Correlation | Controlling for third | High | |
| IND-1702 | Correlation Breakdown | Correlation | Crisis detection | High | |
| IND-1703 | DCC-GARCH | Correlation | Dynamic conditional | High | |
| IND-1704 | Cointegration (Engle-Granger) | Cointegration | Two-step test | High | |
| IND-1705 | Cointegration (Johansen) | Cointegration | Multi-var test | High | |
| IND-1706 | Error Correction Model | Cointegration | ECM | High | |
| IND-1707 | Half-Life of Mean Reversion | Cointegration | Reversion speed | High | |

## Priority 157: Wavelet Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1708 | Haar Wavelet | Wavelet | Basic decomposition | High | |
| IND-1709 | Daubechies Wavelet | Wavelet | DB4/DB8 | High | |
| IND-1710 | Morlet Wavelet | Wavelet | Complex wavelet | High | |
| IND-1711 | Wavelet Denoising | Wavelet | Noise removal | High | |
| IND-1712 | Wavelet Trend | Wavelet | Low-frequency component | High | |
| IND-1713 | Wavelet Cycle | Wavelet | Band-specific cycles | High | |
| IND-1714 | Multi-Resolution Analysis | Wavelet | MRA decomposition | High | |
| IND-1715 | Wavelet Coherence | Wavelet | Time-frequency correlation | High | |
| IND-1716 | Cross-Wavelet Transform | Wavelet | Relationship analysis | High | |
| IND-1717 | Wavelet Variance | Wavelet | Scale-dependent variance | High | |

## Priority 158: Fourier & Spectral

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1718 | FFT Spectrum | Spectral | Frequency analysis | High | |
| IND-1719 | Dominant Period | Spectral | Main cycle length | High | |
| IND-1720 | Spectral Density | Spectral | Power spectrum | High | |
| IND-1721 | Periodogram | Spectral | Raw spectrum | High | |
| IND-1722 | Welch PSD | Spectral | Averaged spectrum | High | |
| IND-1723 | Lomb-Scargle | Spectral | Uneven sampling | High | |
| IND-1724 | Multitaper Spectrum | Spectral | Robust estimate | High | |
| IND-1725 | Spectral Centroid | Spectral | Frequency center | High | |
| IND-1726 | Spectral Entropy | Spectral | Complexity measure | High | |
| IND-1727 | Harmonic Ratio | Spectral | Overtone analysis | High | |

## Priority 159: Entropy & Complexity

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1728 | Shannon Entropy | Entropy | Information content | Medium | |
| IND-1729 | Sample Entropy | Entropy | Time series complexity | High | |
| IND-1730 | Approximate Entropy | Entropy | ApEn | High | |
| IND-1731 | Permutation Entropy | Entropy | Ordinal patterns | High | |
| IND-1732 | Multiscale Entropy | Entropy | Scale-dependent | High | |
| IND-1733 | Transfer Entropy | Entropy | Information flow | High | |
| IND-1734 | Mutual Information | Entropy | Dependence measure | High | |
| IND-1735 | Hurst Exponent | Complexity | Long-term memory | High | |
| IND-1736 | Fractal Dimension | Complexity | Self-similarity | High | |
| IND-1737 | Lyapunov Exponent | Complexity | Chaos measure | High | |

## Priority 160: GARCH Family

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1738 | GARCH(1,1) | Volatility | Standard GARCH | High | |
| IND-1739 | EGARCH | Volatility | Exponential GARCH | High | |
| IND-1740 | GJR-GARCH | Volatility | Asymmetric | High | |
| IND-1741 | TGARCH | Volatility | Threshold GARCH | High | |
| IND-1742 | IGARCH | Volatility | Integrated GARCH | High | |
| IND-1743 | FIGARCH | Volatility | Fractional GARCH | High | |
| IND-1744 | CGARCH | Volatility | Component GARCH | High | |
| IND-1745 | NGARCH | Volatility | Nonlinear GARCH | High | |
| IND-1746 | APARCH | Volatility | Asymmetric Power | High | |
| IND-1747 | GARCH-M | Volatility | GARCH in Mean | High | |

## Priority 161: Jump & Realized Volatility

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1748 | Realized Volatility | Realized | RV from high-frequency | High | |
| IND-1749 | Bipower Variation | Realized | Jump-robust RV | High | |
| IND-1750 | Jump Detection (BNS) | Jump | Barndorff-Nielsen test | High | |
| IND-1751 | Realized Kernels | Realized | Noise-robust RV | High | |
| IND-1752 | Realized Range | Realized | Parkinson extension | High | |
| IND-1753 | Realized Quarticity | Realized | Fourth moment | High | |
| IND-1754 | Jump Variation | Jump | Jump contribution | High | |
| IND-1755 | Integrated Variance | Realized | Continuous variation | High | |
| IND-1756 | Realized Skewness | Realized | Third moment | High | |
| IND-1757 | Realized Kurtosis | Realized | Fourth moment ratio | High | |

## Priority 162: Regime Detection Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1758 | Markov Switching | Regime | MS model | High | |
| IND-1759 | Hidden Markov Model | Regime | HMM | High | |
| IND-1760 | Structural Break (Chow) | Regime | Break detection | High | |
| IND-1761 | CUSUM Test | Regime | Cumulative sum | High | |
| IND-1762 | Bai-Perron | Regime | Multiple breaks | High | |
| IND-1763 | Threshold Autoregression | Regime | TAR model | High | |
| IND-1764 | Smooth Transition | Regime | STAR model | High | |
| IND-1765 | Regime Probability | Regime | State probability | High | |
| IND-1766 | Duration Model | Regime | Time in regime | High | |
| IND-1767 | Volatility Regime | Regime | Vol clustering | High | |

## Priority 163: Copula & Tail Dependence

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1768 | Gaussian Copula | Copula | Normal dependence | High | |
| IND-1769 | Student-t Copula | Copula | Fat-tail dependence | High | |
| IND-1770 | Clayton Copula | Copula | Lower tail | High | |
| IND-1771 | Gumbel Copula | Copula | Upper tail | High | |
| IND-1772 | Frank Copula | Copula | Symmetric | High | |
| IND-1773 | Joe Copula | Copula | Upper tail extreme | High | |
| IND-1774 | Tail Dependence Coef | Copula | Extreme dependence | High | |
| IND-1775 | Time-Varying Copula | Copula | Dynamic dependence | High | |
| IND-1776 | Vine Copula | Copula | Multi-dimensional | High | |
| IND-1777 | Copula-GARCH | Copula | Combined model | High | |

## Priority 164: Factor Models Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1778 | CAPM Alpha | Factor | Market-adjusted | Medium | |
| IND-1779 | CAPM Beta | Factor | Market sensitivity | Medium | |
| IND-1780 | Fama-French SMB | Factor | Size factor | Medium | |
| IND-1781 | Fama-French HML | Factor | Value factor | Medium | |
| IND-1782 | Fama-French RMW | Factor | Profitability | Medium | |
| IND-1783 | Fama-French CMA | Factor | Investment | Medium | |
| IND-1784 | Carhart Momentum | Factor | UMD factor | Medium | |
| IND-1785 | BAB (Betting Against Beta) | Factor | Low-beta premium | High | |
| IND-1786 | QMJ (Quality Minus Junk) | Factor | Quality factor | High | |
| IND-1787 | Factor Exposure | Factor | Loading analysis | High | |

## Priority 165: Style & Attribution

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1788 | Style Analysis | Style | Sharpe style | High | |
| IND-1789 | Returns-Based Style | Style | RBS analysis | High | |
| IND-1790 | Holdings-Based Style | Style | HBS analysis | High | |
| IND-1791 | Active Share | Style | Benchmark deviation | Medium | |
| IND-1792 | Tracking Difference | Style | Return deviation | Medium | |
| IND-1793 | R-Squared to Benchmark | Style | Explained variance | Medium | |
| IND-1794 | Brinson Attribution | Attribution | Sector allocation | High | |
| IND-1795 | Factor Attribution | Attribution | Factor contribution | High | |
| IND-1796 | Fixed Income Attribution | Attribution | Key rate contribution | High | |
| IND-1797 | Multi-Period Attribution | Attribution | Time linking | High | |

## Priority 166: Private Equity & Alternatives

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1798 | IRR (Internal Rate) | PE | Net return | Medium | |
| IND-1799 | TVPI | PE | Total value | Medium | |
| IND-1800 | DPI | PE | Distributed value | Medium | |
| IND-1801 | RVPI | PE | Residual value | Medium | |
| IND-1802 | J-Curve Position | PE | Lifecycle stage | Medium | |
| IND-1803 | PME (Public Market Eq) | PE | Benchmark comparison | High | |
| IND-1804 | Direct Alpha | PE | Abnormal return | High | |
| IND-1805 | Vintage Year Return | PE | Cohort return | Medium | |
| IND-1806 | Commitment Pacing | PE | Cash flow planning | High | |
| IND-1807 | NAV Bridge | PE | Value reconciliation | High | |

## Priority 167: Hedge Fund Specific

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1808 | Omega Ratio | HF | Gain/loss threshold | Medium | |
| IND-1809 | Kappa Ratio | HF | Generalized Sortino | Medium | |
| IND-1810 | Tail Ratio | HF | Right/left tail | Medium | |
| IND-1811 | Gain-to-Pain | HF | Sum gains/losses | Low | |
| IND-1812 | Profit Factor | HF | Gross profit/loss | Low | |
| IND-1813 | Expectancy | HF | Average trade result | Low | |
| IND-1814 | K-Ratio | HF | Kestner ratio | Medium | |
| IND-1815 | Lake Ratio | HF | Drawdown area | Medium | |
| IND-1816 | Ulcer Performance Index | HF | Risk-adjusted by Ulcer | Medium | |
| IND-1817 | CPC Index | HF | Commodity Pool | Medium | |

## Priority 168: Tax & After-Tax

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1818 | Tax Cost Ratio | Tax | Tax drag | Medium | |
| IND-1819 | After-Tax Return | Tax | Net of taxes | Medium | |
| IND-1820 | Tax Alpha | Tax | Tax efficiency | Medium | |
| IND-1821 | Tax Lot Optimization | Tax | HIFO/FIFO analysis | High | |
| IND-1822 | Wash Sale Tracking | Tax | Loss harvesting | High | |
| IND-1823 | Tax Loss Harvesting Opp | Tax | Opportunity score | Medium | |
| IND-1824 | Unrealized Gains | Tax | Embedded gains | Medium | |
| IND-1825 | Tax Efficiency Score | Tax | Composite measure | Medium | |

## Priority 169: Currency Overlay

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1826 | Currency Return | FX Overlay | Local vs USD | Medium | |
| IND-1827 | Hedge Ratio | FX Overlay | FX exposure | Medium | |
| IND-1828 | Forward Premium | FX Overlay | IR differential | Medium | |
| IND-1829 | Hedge Cost | FX Overlay | Forward cost | Medium | |
| IND-1830 | Currency Contribution | FX Overlay | Return attribution | Medium | |
| IND-1831 | Passive Hedge | FX Overlay | Rule-based | Medium | |
| IND-1832 | Active Currency | FX Overlay | Tactical | High | |
| IND-1833 | Currency Beta | FX Overlay | Dollar exposure | Medium | |

## Priority 170: ESG Extended

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1834 | Carbon Intensity | ESG | CO2/revenue | Medium | |
| IND-1835 | Carbon Footprint | ESG | Portfolio emissions | Medium | |
| IND-1836 | WACI | ESG | Weighted avg carbon | Medium | |
| IND-1837 | Temperature Alignment | ESG | Paris agreement | High | |
| IND-1838 | Brown Share | ESG | Fossil fuel revenue | Medium | |
| IND-1839 | Green Share | ESG | Clean energy revenue | Medium | |
| IND-1840 | ESG Score Trend | ESG | Rating momentum | Medium | |
| IND-1841 | Controversy Score | ESG | Negative events | Medium | |
| IND-1842 | SDG Alignment | ESG | UN goals | High | |
| IND-1843 | Impact Metrics | ESG | Outcome measurement | High | |

## Priority 171: Quant Screening

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1844 | Piotroski F-Score | Screening | Financial strength | Medium | |
| IND-1845 | Altman Z-Score | Screening | Bankruptcy risk | Medium | |
| IND-1846 | Beneish M-Score | Screening | Earnings manipulation | Medium | |
| IND-1847 | Ohlson O-Score | Screening | Bankruptcy probability | Medium | |
| IND-1848 | Merton Distance to Default | Screening | Credit risk | High | |
| IND-1849 | Graham Number | Screening | Intrinsic value | Low | |
| IND-1850 | Magic Formula | Screening | Greenblatt ranking | Medium | |
| IND-1851 | Acquirer's Multiple | Screening | EV/EBIT variant | Medium | |
| IND-1852 | NCAV/MV | Screening | Net-net | Low | |
| IND-1853 | Earnings Power Value | Screening | EPV | Medium | |

## Priority 172: Technical Screening

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1854 | 52-Week High Proximity | Screening | % from high | Low | |
| IND-1855 | 52-Week Low Proximity | Screening | % from low | Low | |
| IND-1856 | New High Count | Screening | Recent new highs | Low | |
| IND-1857 | Up/Down Volume Ratio | Screening | Volume strength | Low | |
| IND-1858 | Price vs MA Distance | Screening | Mean reversion | Low | |
| IND-1859 | Trend Strength Score | Screening | Composite trend | Medium | |
| IND-1860 | Momentum Rank | Screening | Relative momentum | Medium | |
| IND-1861 | Volatility Rank | Screening | Relative volatility | Medium | |
| IND-1862 | Liquidity Score | Screening | Trading ease | Medium | |
| IND-1863 | Pattern Score | Screening | Technical pattern | High | |

## Priority 173: Global Macro Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1864 | PMI Composite | Macro | Manufacturing + Services | Medium | |
| IND-1865 | Economic Surprise | Macro | Citi-style surprise | Medium | |
| IND-1866 | Recession Probability | Macro | Yield curve based | High | |
| IND-1867 | Nowcast GDP | Macro | Real-time estimate | High | |
| IND-1868 | Leading Indicator | Macro | Composite lead | High | |
| IND-1869 | Coincident Indicator | Macro | Current state | High | |
| IND-1870 | Lagging Indicator | Macro | Confirmation | Medium | |
| IND-1871 | Financial Conditions | Macro | FCI | High | |
| IND-1872 | Global Risk Appetite | Macro | Risk-on/off | High | |
| IND-1873 | Business Cycle Phase | Macro | Expansion/recession | High | |

## Priority 174: Central Bank Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1874 | Fed Funds Expectations | Central Bank | OIS-implied | Medium | |
| IND-1875 | Rate Hike Probability | Central Bank | Meeting odds | High | |
| IND-1876 | Balance Sheet Size | Central Bank | QE measure | Medium | |
| IND-1877 | Tapering Signal | Central Bank | QE reduction | High | |
| IND-1878 | Hawkish/Dovish Score | Central Bank | NLP of statements | High | |
| IND-1879 | Dot Plot Median | Central Bank | FOMC projections | Medium | |
| IND-1880 | Real Neutral Rate | Central Bank | r-star | High | |
| IND-1881 | Taylor Rule | Central Bank | Policy prescription | Medium | |
| IND-1882 | Policy Uncertainty | Central Bank | EPU index | High | |
| IND-1883 | Forward Guidance Score | Central Bank | Communication analysis | High | |

## Priority 175: Cross-Asset Momentum

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1884 | TSMOM (Time Series) | Momentum | Own-asset trend | Medium | |
| IND-1885 | XSMOM (Cross-Section) | Momentum | Relative strength | Medium | |
| IND-1886 | Dual Momentum | Momentum | Combined TS + XS | Medium | |
| IND-1887 | Momentum Crash Risk | Momentum | Reversal risk | High | |
| IND-1888 | Momentum Spread | Momentum | Winner-loser gap | Medium | |
| IND-1889 | Residual Momentum | Momentum | Factor-adjusted | High | |
| IND-1890 | Fundamental Momentum | Momentum | Earnings-based | Medium | |
| IND-1891 | Information Discreteness | Momentum | News vs drift | High | |
| IND-1892 | Momentum Duration | Momentum | Persistence | Medium | |
| IND-1893 | Momentum Quality | Momentum | Risk-adjusted mom | Medium | |

## Priority 176: Value Investing Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1894 | Price/Earnings | Value | P/E ratio | Low | |
| IND-1895 | Price/Book | Value | P/B ratio | Low | |
| IND-1896 | Price/Sales | Value | P/S ratio | Low | |
| IND-1897 | Price/Cash Flow | Value | P/CF ratio | Low | |
| IND-1898 | EV/EBITDA | Value | Enterprise value ratio | Medium | |
| IND-1899 | EV/Sales | Value | Revenue multiple | Medium | |
| IND-1900 | PEG Ratio | Value | Growth-adjusted P/E | Medium | |
| IND-1901 | Dividend Yield | Value | Annual yield | Low | |
| IND-1902 | Free Cash Flow Yield | Value | FCF/Price | Medium | |
| IND-1903 | Earnings Yield | Value | Inverse P/E | Low | |

## Priority 177: Growth Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1904 | Revenue Growth | Growth | Top-line growth | Low | |
| IND-1905 | EPS Growth | Growth | Earnings growth | Low | |
| IND-1906 | EBITDA Growth | Growth | Operating growth | Medium | |
| IND-1907 | Book Value Growth | Growth | Equity growth | Low | |
| IND-1908 | Dividend Growth | Growth | Payout growth | Low | |
| IND-1909 | FCF Growth | Growth | Cash flow growth | Medium | |
| IND-1910 | Sustainable Growth Rate | Growth | ROE × Retention | Medium | |
| IND-1911 | CAGR | Growth | Compound annual rate | Low | |
| IND-1912 | Growth Acceleration | Growth | YoY change in growth | Medium | |
| IND-1913 | Growth Consistency | Growth | Std dev of growth | Medium | |

## Priority 178: Quality Indicators

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1914 | Return on Equity | Quality | ROE | Low | |
| IND-1915 | Return on Assets | Quality | ROA | Low | |
| IND-1916 | Return on Invested Capital | Quality | ROIC | Medium | |
| IND-1917 | Gross Margin | Quality | Gross profit margin | Low | |
| IND-1918 | Operating Margin | Quality | EBIT margin | Low | |
| IND-1919 | Net Margin | Quality | Net income margin | Low | |
| IND-1920 | Asset Turnover | Quality | Sales/Assets | Low | |
| IND-1921 | Inventory Turnover | Quality | COGS/Inventory | Low | |
| IND-1922 | Receivables Turnover | Quality | Sales/Receivables | Low | |
| IND-1923 | Interest Coverage | Quality | EBIT/Interest | Low | |

## Priority 179: Leverage & Solvency

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1924 | Debt/Equity | Leverage | Financial leverage | Low | |
| IND-1925 | Debt/Assets | Leverage | Asset leverage | Low | |
| IND-1926 | Debt/EBITDA | Leverage | Debt burden | Medium | |
| IND-1927 | Net Debt/EBITDA | Leverage | Net debt burden | Medium | |
| IND-1928 | Current Ratio | Solvency | Short-term liquidity | Low | |
| IND-1929 | Quick Ratio | Solvency | Acid test | Low | |
| IND-1930 | Cash Ratio | Solvency | Immediate liquidity | Low | |
| IND-1931 | Working Capital | Solvency | Operational liquidity | Low | |
| IND-1932 | Financial Leverage Ratio | Leverage | Assets/Equity | Low | |
| IND-1933 | Equity Multiplier | Leverage | Total leverage | Low | |

## Priority 180: DuPont Analysis

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1934 | 3-Factor DuPont | DuPont | ROE decomposition | Medium | |
| IND-1935 | 5-Factor DuPont | DuPont | Extended decomposition | Medium | |
| IND-1936 | Tax Burden | DuPont | NI/EBT | Low | |
| IND-1937 | Interest Burden | DuPont | EBT/EBIT | Low | |
| IND-1938 | Operating Efficiency | DuPont | EBIT margin | Low | |
| IND-1939 | Asset Efficiency | DuPont | Turnover | Low | |
| IND-1940 | Financial Leverage | DuPont | Equity multiplier | Low | |
| IND-1941 | DuPont Trend | DuPont | Factor changes | Medium | |

## Priority 181: Accrual & Earnings Quality

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1942 | Sloan Accruals | Accrual | Total accruals | Medium | |
| IND-1943 | Operating Accruals | Accrual | Working capital accruals | Medium | |
| IND-1944 | Richardson Accruals | Accrual | Change in NOA | Medium | |
| IND-1945 | Cash Flow-NI Ratio | Accrual | CFO/NI | Medium | |
| IND-1946 | Earnings Persistence | Accrual | AR(1) coefficient | High | |
| IND-1947 | Earnings Smoothness | Accrual | Std dev ratio | Medium | |
| IND-1948 | Discretionary Accruals | Accrual | Jones model | High | |
| IND-1949 | Real Earnings Management | Accrual | Cash flow manipulation | High | |
| IND-1950 | Dechow F-Score | Accrual | Fraud probability | High | |
| IND-1951 | NOA/Assets | Accrual | Asset inflation | Medium | |

## Priority 182: Insider & Institutional

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1952 | Insider Ownership | Insider | % held by insiders | Low | |
| IND-1953 | Insider Buying | Insider | Net purchases | Medium | |
| IND-1954 | Insider Selling | Insider | Net sales | Medium | |
| IND-1955 | Insider Buy/Sell Ratio | Insider | Transaction ratio | Medium | |
| IND-1956 | Institutional Ownership | Inst | % held by institutions | Low | |
| IND-1957 | Institutional Change | Inst | QoQ ownership change | Medium | |
| IND-1958 | 13F Filing Changes | Inst | Hedge fund positions | Medium | |
| IND-1959 | Short Interest | Inst | Short %, days to cover | Medium | |
| IND-1960 | Short Squeeze Score | Inst | Squeeze potential | High | |
| IND-1961 | Ownership Concentration | Inst | Top holder % | Medium | |

## Priority 183: Analyst Coverage

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1962 | Analyst Count | Analyst | Number of analysts | Low | |
| IND-1963 | Consensus Rating | Analyst | Buy/Hold/Sell average | Medium | |
| IND-1964 | Rating Changes | Analyst | Upgrade/downgrade | Medium | |
| IND-1965 | Price Target | Analyst | Consensus target | Medium | |
| IND-1966 | Price Target Upside | Analyst | % to target | Low | |
| IND-1967 | EPS Estimate | Analyst | Consensus EPS | Medium | |
| IND-1968 | Revenue Estimate | Analyst | Consensus revenue | Medium | |
| IND-1969 | Estimate Accuracy | Analyst | Historical accuracy | High | |
| IND-1970 | Analyst Dispersion | Analyst | Estimate spread | Medium | |
| IND-1971 | Revision Trend | Analyst | Estimate momentum | Medium | |

## Priority 184: Corporate Actions

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1972 | Dividend History | Corp Action | Payout record | Medium | |
| IND-1973 | Dividend Streak | Corp Action | Consecutive increases | Low | |
| IND-1974 | Buyback Yield | Corp Action | Repurchase/Market cap | Medium | |
| IND-1975 | Net Payout Yield | Corp Action | Div + Buyback yield | Medium | |
| IND-1976 | Equity Issuance | Corp Action | Share dilution | Medium | |
| IND-1977 | M&A Activity | Corp Action | Acquisition indicator | High | |
| IND-1978 | Spin-off Indicator | Corp Action | Corporate restructuring | High | |
| IND-1979 | Split History | Corp Action | Stock split record | Low | |
| IND-1980 | Tender Offer | Corp Action | Buyback at premium | High | |
| IND-1981 | Rights Issue | Corp Action | Equity offering | Medium | |

## Priority 185: Event Studies

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1982 | Abnormal Return | Event | CAR calculation | High | |
| IND-1983 | Pre-Event Drift | Event | Pre-announcement | Medium | |
| IND-1984 | Post-Event Drift | Event | Post-announcement | Medium | |
| IND-1985 | Event Window Return | Event | Specific window | Medium | |
| IND-1986 | Announcement Day Vol | Event | Event volatility | Medium | |
| IND-1987 | Surprise Magnitude | Event | Deviation from expected | Medium | |
| IND-1988 | Event Clustering | Event | Multiple events | High | |
| IND-1989 | Information Leakage | Event | Pre-event movement | High | |
| IND-1990 | Market Reaction | Event | Immediate response | Medium | |
| IND-1991 | Longer-Term Drift | Event | Extended impact | High | |

## Priority 186: Pairs Trading

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-1992 | Pair Spread | Pairs | Price ratio spread | Medium | |
| IND-1993 | Pair Z-Score | Pairs | Spread standardized | Medium | |
| IND-1994 | Pair Correlation | Pairs | Rolling correlation | Medium | |
| IND-1995 | Pair Cointegration | Pairs | Engle-Granger test | High | |
| IND-1996 | Hedge Ratio (OLS) | Pairs | Beta hedge | Medium | |
| IND-1997 | Kalman Hedge Ratio | Pairs | Dynamic hedge | High | |
| IND-1998 | Mean Reversion Speed | Pairs | Half-life | High | |
| IND-1999 | Pair Entry Signal | Pairs | Trade trigger | Medium | |
| IND-2000 | Pair Exit Signal | Pairs | Close trigger | Medium | |
| IND-2001 | Pair PnL | Pairs | Strategy return | Medium | |

## Priority 187: Stat Arb Signals

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2002 | PCA Factor | Stat Arb | Principal component | High | |
| IND-2003 | PCA Residual | Stat Arb | Idiosyncratic return | High | |
| IND-2004 | Factor Neutral Return | Stat Arb | Market-neutral | High | |
| IND-2005 | Cross-Sectional Mean | Stat Arb | Universe average | Medium | |
| IND-2006 | Cross-Sectional Rank | Stat Arb | Relative position | Medium | |
| IND-2007 | Sector Neutral Return | Stat Arb | Industry-adjusted | High | |
| IND-2008 | Dollar Neutral Signal | Stat Arb | Long-short balance | Medium | |
| IND-2009 | Beta Neutral Signal | Stat Arb | Risk-adjusted | High | |
| IND-2010 | Portfolio Turnover | Stat Arb | Rebalance frequency | Medium | |
| IND-2011 | Signal Decay | Stat Arb | Alpha decay rate | High | |

## Priority 188: High-Frequency Signals

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2012 | Microprice | HF | Fair price estimate | High | |
| IND-2013 | Weighted Microprice | HF | Volume-weighted | High | |
| IND-2014 | Trade Flow Toxicity | HF | VPIN | High | |
| IND-2015 | Flash Crash Indicator | HF | Extreme volatility | High | |
| IND-2016 | Quote-to-Trade Ratio | HF | Noise measure | Medium | |
| IND-2017 | Message Intensity | HF | Order activity | Medium | |
| IND-2018 | Latency Arbitrage | HF | Speed advantage | High | |
| IND-2019 | Momentum Ignition | HF | Manipulation detection | High | |
| IND-2020 | Cross-Venue Spread | HF | Multi-exchange | High | |
| IND-2021 | Dark Pool Flow | HF | Non-displayed volume | High | |

## Priority 189: Market Quality

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2022 | Effective Spread | Quality | Actual trading cost | Medium | |
| IND-2023 | Realized Spread | Quality | Post-trade spread | Medium | |
| IND-2024 | Price Impact | Quality | Kyle's lambda | High | |
| IND-2025 | Information Share | Quality | Hasbrouck measure | High | |
| IND-2026 | Component Share | Quality | Mid-price contribution | High | |
| IND-2027 | Variance Ratio | Quality | Random walk test | Medium | |
| IND-2028 | Auto-Correlation (Returns) | Quality | Serial dependence | Medium | |
| IND-2029 | Price Efficiency | Quality | Delay measure | High | |
| IND-2030 | Price Discovery | Quality | Information speed | High | |
| IND-2031 | Market Resilience | Quality | Recovery after trade | High | |

## Priority 190: Intraday Patterns

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2032 | Opening Range | Intraday | First N-min range | Low | |
| IND-2033 | Closing Range | Intraday | Last N-min range | Low | |
| IND-2034 | Session High/Low | Intraday | Intraday extremes | Low | |
| IND-2035 | Power Hour | Intraday | Final hour behavior | Medium | |
| IND-2036 | Lunch Effect | Intraday | Midday pattern | Medium | |
| IND-2037 | Overnight Gap | Intraday | Open vs prior close | Low | |
| IND-2038 | Gap Fill Probability | Intraday | Gap closure rate | Medium | |
| IND-2039 | Intraday VWAP Bands | Intraday | Deviation from VWAP | Medium | |
| IND-2040 | Time-of-Day Return | Intraday | Hour-specific return | Medium | |
| IND-2041 | Session Momentum | Intraday | Intraday trend | Medium | |

## Priority 191: Calendar Effects

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2042 | Day-of-Week Effect | Calendar | Monday effect etc | Medium | |
| IND-2043 | Month-of-Year Effect | Calendar | January effect | Medium | |
| IND-2044 | Turn-of-Month | Calendar | Month boundary | Medium | |
| IND-2045 | Turn-of-Quarter | Calendar | Quarter boundary | Medium | |
| IND-2046 | Holiday Effect | Calendar | Pre/post holiday | Medium | |
| IND-2047 | Options Expiry Effect | Calendar | Expiry week | Medium | |
| IND-2048 | Triple Witching | Calendar | Quarterly expiry | Medium | |
| IND-2049 | FOMC Meeting Effect | Calendar | Fed meeting dates | Medium | |
| IND-2050 | Earnings Season | Calendar | Reporting period | Medium | |
| IND-2051 | Tax-Loss Selling | Calendar | Year-end effect | Medium | |

## Priority 192: Weather & Seasonal

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2052 | Temperature Deviation | Weather | vs Normal | Medium | |
| IND-2053 | Heating Degree Days | Weather | HDD | Low | |
| IND-2054 | Cooling Degree Days | Weather | CDD | Low | |
| IND-2055 | Hurricane Track | Weather | Storm path | High | |
| IND-2056 | Drought Index | Weather | PDSI | Medium | |
| IND-2057 | Crop Condition | Weather | USDA weekly | Medium | |
| IND-2058 | Snow Pack | Weather | Water supply | Medium | |
| IND-2059 | El Nino Index | Weather | ENSO | Medium | |
| IND-2060 | Seasonal Pattern | Seasonal | Historical seasonality | Medium | |
| IND-2061 | Seasonal Adjusted | Seasonal | Trend extraction | High | |

## Priority 193: Supply Chain

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2062 | Supplier Concentration | Supply | Key supplier risk | High | |
| IND-2063 | Customer Concentration | Supply | Key customer risk | High | |
| IND-2064 | Supply Chain Length | Supply | Tier depth | High | |
| IND-2065 | Geographic Exposure | Supply | Country risk | High | |
| IND-2066 | Shipping Index | Supply | Baltic Dry etc | Medium | |
| IND-2067 | Port Congestion | Supply | Logistics bottleneck | High | |
| IND-2068 | Container Rates | Supply | Freight cost | Medium | |
| IND-2069 | Lead Time Index | Supply | Order-to-delivery | High | |
| IND-2070 | Inventory/Sales Ratio | Supply | Channel inventory | Medium | |
| IND-2071 | Backlog Index | Supply | Order backlog | Medium | |

## Priority 194: Industry Specific - Technology

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2072 | SaaS Metrics (ARR) | Tech | Annual recurring revenue | Medium | |
| IND-2073 | Net Revenue Retention | Tech | Expansion/churn | Medium | |
| IND-2074 | CAC/LTV Ratio | Tech | Customer economics | Medium | |
| IND-2075 | Rule of 40 | Tech | Growth + margin | Medium | |
| IND-2076 | DAU/MAU Ratio | Tech | User engagement | Medium | |
| IND-2077 | Billings Growth | Tech | Cash collection | Medium | |
| IND-2078 | RPO Growth | Tech | Remaining performance | Medium | |
| IND-2079 | Magic Number | Tech | Sales efficiency | Medium | |
| IND-2080 | Burn Multiple | Tech | Cash efficiency | Medium | |
| IND-2081 | NDR Index | Tech | Net dollar retention | Medium | |

## Priority 195: Industry Specific - Banking

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2082 | Net Interest Margin | Bank | NIM | Medium | |
| IND-2083 | Loan/Deposit Ratio | Bank | LDR | Low | |
| IND-2084 | NPL Ratio | Bank | Non-performing loans | Medium | |
| IND-2085 | Provision Coverage | Bank | Loan loss reserves | Medium | |
| IND-2086 | CET1 Ratio | Bank | Capital adequacy | Medium | |
| IND-2087 | Tier 1 Leverage | Bank | Leverage ratio | Medium | |
| IND-2088 | Efficiency Ratio | Bank | Cost/Income | Low | |
| IND-2089 | Fee Income Ratio | Bank | Non-interest income | Medium | |
| IND-2090 | Credit Card Delinquency | Bank | Consumer credit | Medium | |
| IND-2091 | Mortgage Delinquency | Bank | Housing credit | Medium | |

## Priority 196: Industry Specific - Insurance

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2092 | Combined Ratio | Insurance | Underwriting profit | Medium | |
| IND-2093 | Loss Ratio | Insurance | Claims/Premiums | Medium | |
| IND-2094 | Expense Ratio | Insurance | Operating costs | Low | |
| IND-2095 | Embedded Value | Insurance | EV | High | |
| IND-2096 | New Business Value | Insurance | NBV | High | |
| IND-2097 | Persistency Ratio | Insurance | Policy retention | Medium | |
| IND-2098 | Reserve Adequacy | Insurance | Loss reserve | High | |
| IND-2099 | Investment Income Ratio | Insurance | Asset returns | Medium | |
| IND-2100 | Solvency Ratio | Insurance | Capital strength | Medium | |
| IND-2101 | Catastrophe Exposure | Insurance | Nat cat risk | High | |

## Priority 197: Industry Specific - REIT

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2102 | FFO | REIT | Funds from operations | Medium | |
| IND-2103 | AFFO | REIT | Adjusted FFO | Medium | |
| IND-2104 | NAV Premium/Discount | REIT | Price vs NAV | Medium | |
| IND-2105 | Cap Rate | REIT | NOI/Value | Medium | |
| IND-2106 | Occupancy Rate | REIT | Utilization | Low | |
| IND-2107 | Same-Store NOI Growth | REIT | Organic growth | Medium | |
| IND-2108 | Debt/Gross Asset Value | REIT | Leverage | Medium | |
| IND-2109 | Dividend Payout Ratio | REIT | FFO payout | Medium | |
| IND-2110 | Leasing Spread | REIT | Rent change | Medium | |
| IND-2111 | Weighted Avg Lease Term | REIT | WALT | Medium | |

## Priority 198: Industry Specific - Retail

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2112 | Same-Store Sales | Retail | Comp store growth | Medium | |
| IND-2113 | Sales per Square Foot | Retail | Productivity | Medium | |
| IND-2114 | Inventory Turn | Retail | Stock efficiency | Medium | |
| IND-2115 | Gross Margin ROI | Retail | GMROI | Medium | |
| IND-2116 | Sell-Through Rate | Retail | Units sold/received | Medium | |
| IND-2117 | Traffic Count | Retail | Store visits | Medium | |
| IND-2118 | Conversion Rate | Retail | Visitors to buyers | Medium | |
| IND-2119 | Average Transaction Value | Retail | ATV | Low | |
| IND-2120 | Units per Transaction | Retail | UPT | Low | |
| IND-2121 | E-Commerce Mix | Retail | Online share | Medium | |

## Priority 199: Industry Specific - Energy

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2122 | Reserve Replacement | Energy | Proven reserves | Medium | |
| IND-2123 | Finding Cost | Energy | Discovery cost | Medium | |
| IND-2124 | Lifting Cost | Energy | Production cost | Medium | |
| IND-2125 | Netback | Energy | Revenue - costs | Medium | |
| IND-2126 | Refining Margin | Energy | Crack spread | Medium | |
| IND-2127 | Utilization Rate | Energy | Refinery capacity | Low | |
| IND-2128 | Proved Reserves | Energy | 1P reserves | Medium | |
| IND-2129 | Production Growth | Energy | Volume growth | Medium | |
| IND-2130 | Decline Rate | Energy | Field depletion | Medium | |
| IND-2131 | Break-Even Oil Price | Energy | Project economics | High | |

## Priority 200: Comprehensive Statistical Tests

| ID | Indicator | Category | Description | Complexity | Status |
|----|-----------|----------|-------------|------------|--------|
| IND-2132 | ADF Test | Stationarity | Augmented Dickey-Fuller | High | |
| IND-2133 | KPSS Test | Stationarity | Stationarity test | High | |
| IND-2134 | Phillips-Perron | Stationarity | Unit root test | High | |
| IND-2135 | Ljung-Box | Autocorrelation | Serial correlation | Medium | |
| IND-2136 | Durbin-Watson | Autocorrelation | DW statistic | Medium | |
| IND-2137 | Breusch-Godfrey | Autocorrelation | Higher order | High | |
| IND-2138 | ARCH Test | Heteroskedasticity | Volatility clustering | High | |
| IND-2139 | White Test | Heteroskedasticity | General test | High | |
| IND-2140 | Jarque-Bera | Normality | JB test | Medium | |
| IND-2141 | Shapiro-Wilk | Normality | SW test | Medium | |

## Completed

| ID | Indicator | Completed | Notes | Status |
|----|-----------|-----------|-------|--------|
| - | Initial 33 indicators | 2025-01 | Migrated from quantlang |

---

## Summary by Category

| Category | Current | Planned | Total |
|----------|---------|---------|-------|--------|
| **Technical Analysis** | | | |
| Oscillators | 6 | 60 | 66 |
| Moving Averages | 8 | 40 | 48 |
| Trend | 19 | 16 | 35 |
| Volatility | 4 | 80 | 84 |
| Volume | 4 | 45 | 49 |
| Pattern | 0 | 80 | 80 |
| Candlestick | 0 | 75 | 75 |
| Support/Resistance | 2 | 15 | 17 |
| Filters | 3 | 35 | 38 |
| **Wave & Structure** | | | |
| Elliott Wave | 0 | 15 | 15 |
| Harmonic | 0 | 25 | 25 |
| Gann | 0 | 15 | 15 |
| Wyckoff | 0 | 12 | 12 |
| VSA | 0 | 15 | 15 |
| **Digital Signal Processing** | | | |
| DSP/Ehlers | 2 | 60 | 62 |
| Wavelet | 0 | 10 | 10 |
| Spectral/FFT | 0 | 10 | 10 |
| **Market Structure** | | | |
| Order Flow | 0 | 25 | 25 |
| Footprint | 0 | 10 | 10 |
| DOM | 0 | 10 | 10 |
| Profile | 0 | 15 | 15 |
| Microstructure | 0 | 25 | 25 |
| **Quantitative Finance** | | | |
| Factor Models | 0 | 25 | 25 |
| Risk Metrics | 0 | 30 | 30 |
| Portfolio | 0 | 20 | 20 |
| Stat Arb | 0 | 10 | 10 |
| Pairs Trading | 0 | 10 | 10 |
| Momentum | 0 | 25 | 25 |
| **Volatility Models** | | | |
| GARCH Family | 0 | 10 | 10 |
| Realized Vol | 0 | 10 | 10 |
| Jump Detection | 0 | 5 | 5 |
| **Options & Derivatives** | | | |
| Greeks | 0 | 15 | 15 |
| Vol Surface | 0 | 15 | 15 |
| IV Indicators | 0 | 10 | 10 |
| **Fixed Income** | | | |
| Rates/Duration | 0 | 25 | 25 |
| Credit | 0 | 15 | 15 |
| **Alternative Data** | | | |
| Sentiment/NLP | 0 | 25 | 25 |
| On-Chain/Crypto | 0 | 35 | 35 |
| Alt Data Sources | 0 | 20 | 20 |
| ESG | 0 | 15 | 15 |
| **Machine Learning** | | | |
| ML Signals | 0 | 25 | 25 |
| Regime Detection | 0 | 20 | 20 |
| Entropy/Complexity | 0 | 10 | 10 |
| **Fundamental Analysis** | | | |
| Value Investing | 0 | 15 | 15 |
| Growth Indicators | 0 | 15 | 15 |
| Quality Metrics | 0 | 20 | 20 |
| Accruals | 0 | 10 | 10 |
| **Macro & Economics** | | | |
| Economic Indicators | 0 | 15 | 15 |
| Central Bank | 0 | 10 | 10 |
| FX/Currency | 0 | 20 | 20 |
| Commodities | 0 | 15 | 15 |
| **Industry Specific** | | | |
| Technology/SaaS | 0 | 10 | 10 |
| Banking | 0 | 10 | 10 |
| Insurance | 0 | 10 | 10 |
| REIT | 0 | 10 | 10 |
| Retail | 0 | 10 | 10 |
| Energy | 0 | 10 | 10 |
| **Execution & Trading** | | | |
| Execution Analytics | 0 | 15 | 15 |
| HFT Signals | 0 | 15 | 15 |
| Intraday | 0 | 15 | 15 |
| Calendar Effects | 0 | 10 | 10 |
| **Statistical Tests** | | | |
| Stationarity | 0 | 5 | 5 |
| Correlation | 0 | 15 | 15 |
| Regression | 0 | 15 | 15 |
| Copula | 0 | 10 | 10 |
| **Platform-Specific** | | | |
| MetaTrader | 0 | 18 | 18 |
| TradingView | 0 | 18 | 18 |
| NinjaTrader | 0 | 11 | 11 |
| ThinkOrSwim | 0 | 12 | 12 |
| Bloomberg | 0 | 14 | 14 |
| **Practitioner Methods** | | | |
| Bill Williams | 0 | 12 | 12 |
| Tom DeMark | 0 | 25 | 25 |
| Larry Williams | 0 | 12 | 12 |
| Linda Raschke | 0 | 12 | 12 |
| Alexander Elder | 0 | 15 | 15 |
| Welles Wilder | 0 | 15 | 15 |
| Other Practitioners | 0 | 60 | 60 |
| **Chart Types** | | | |
| Point & Figure | 0 | 10 | 10 |
| Renko | 0 | 8 | 8 |
| Kagi | 0 | 8 | 8 |
| Three Line Break | 0 | 6 | 6 |
| **Other** | | | |
| Screening | 0 | 20 | 20 |
| Events | 0 | 15 | 15 |
| Weather/Seasonal | 0 | 10 | 10 |
| Supply Chain | 0 | 10 | 10 |
| **Total** | **44** | **2104** | **2148** |

---

## Implementation Notes

### Complexity Guide
- **Low**: Single calculation, simple formula
- **Medium**: Multiple components, state management
- **High**: Complex logic, multiple dependent calculations, DSP math

### Dependencies
- IND-008 (Accelerator) depends on IND-007 (Awesome Oscillator)
- IND-009 (Gator) depends on IND-010 (Alligator)
- Bill Williams indicators (007-011) should be implemented together
- IND-037 (Stochastic RSI) depends on existing RSI + Stochastic
- IND-060 (TTM Squeeze) depends on Bollinger + Keltner
- IND-061 (Elder Impulse) depends on existing MACD + EMA
- IND-062 (Schaff) depends on MACD + Stochastic
- Ehlers indicators (044-051) require DSP math utilities

### Implementation Groups
1. **Quick Wins (Low complexity)**: IND-001, 002, 004, 006, 053, 056, 057, 066, 071, 072, 080, 081, 099, 103, 104, 106, 107, 111, 117, 120, 121, 125, 132, 142, 144, 147, 151, 152, 153, 159, 162, 168, 172, 173, 188
2. **Bill Williams Suite**: IND-007 through IND-011
3. **Ehlers DSP Suite**: IND-044 through IND-051, 170, 171 (requires Hilbert Transform base)
4. **RSI Family**: IND-037 through IND-043
5. **Volume Suite**: IND-069 through IND-076, 159-166
6. **DeMark Suite**: IND-091 through IND-098
7. **Volatility Models**: IND-099 through IND-106
8. **Risk Metrics**: IND-107 through IND-116
9. **Intermarket Suite**: IND-117 through IND-124
10. **Breadth Suite**: IND-125 through IND-132, 067, 068
11. **Elder Suite**: IND-176 through IND-180
12. **Kase Suite**: IND-181 through IND-185
13. **Smart Money Concepts**: IND-136 through IND-140

### SIMD Optimization
All new indicators should follow the existing SIMD pattern in `indicator-core/src/simd/`.

### References

**Classic Technical Analysis:**
- Wilder: "New Concepts in Technical Trading Systems"
- Murphy: "Technical Analysis of the Financial Markets"
- Pring: "Technical Analysis Explained"
- Edwards & Magee: "Technical Analysis of Stock Trends"

**Candlesticks & Patterns:**
- Nison: "Japanese Candlestick Charting Techniques"
- Bulkowski: "Encyclopedia of Chart Patterns"
- Carney: "Harmonic Trading" (Gartley, Bat, Crab, Butterfly)

**Advanced Methods:**
- Ehlers: "Rocket Science for Traders", "Cybernetic Analysis for Stocks and Futures"
- Bill Williams: "Trading Chaos", "New Trading Dimensions"
- DeMark: "The New Science of Technical Analysis"
- Elder: "Trading for a Living", "Come Into My Trading Room"
- Kase: "Trading With The Odds"

**Wave Theory:**
- Prechter & Frost: "Elliott Wave Principle"
- Neely: "Mastering Elliott Wave"
- Gann: "45 Years in Wall Street"

**Market Structure:**
- Wyckoff: "The Richard D. Wyckoff Method"
- Smart Money Concepts: ICT methodology
- Order Flow: Dalton "Mind Over Markets", Steidlmayer "Markets and Market Logic"
- VSA: Tom Williams "Master the Markets"

**Quantitative:**
- Volatility models: Parkinson 1980, Garman-Klass 1980, Yang-Zhang 2000
- Options: Natenberg "Option Volatility and Pricing"
- Factor investing: Fama-French, AQR research
- Liquidity: Amihud 2002, Kyle 1985
- Credit: Merton 1974, CreditMetrics

**Alternative Data:**
- Crypto on-chain: Glassnode, CryptoQuant, Willy Woo
- Sentiment: AAII, COT, social media NLP
- ESG: MSCI, Sustainalytics methodologies

**Machine Learning:**
- scikit-learn, TensorFlow, PyTorch documentation
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Machine Learning for Asset Managers" - Marcos López de Prado

**Asset Classes:**
- Fixed Income: Fabozzi "Bond Markets, Analysis and Strategies"
- Real Estate: Geltner "Commercial Real Estate Analysis"
- Commodities: Geman "Commodities and Commodity Derivatives"
- Seasonality: Hirsch "Stock Trader's Almanac"
