# Time Series Prediction Algorithms

This guide explains the time series prediction algorithms available in rustful-ts, including when to use each one and how they work.

## Table of Contents

1. [ARIMA](#arima)
2. [Exponential Smoothing](#exponential-smoothing)
3. [Moving Averages](#moving-averages)
4. [Linear Regression](#linear-regression)
5. [Algorithm Comparison](#algorithm-comparison)

---

## ARIMA

**AutoRegressive Integrated Moving Average**

ARIMA is one of the most versatile and widely used time series forecasting methods. It combines three components:

### Components

| Component | Parameter | Description |
|-----------|-----------|-------------|
| **AR** (AutoRegressive) | p | Uses past values to predict future values |
| **I** (Integrated) | d | Differencing to achieve stationarity |
| **MA** (Moving Average) | q | Uses past forecast errors |

### Mathematical Model

```
ARIMA(p, d, q):
  Y'_t = c + φ₁Y'_{t-1} + ... + φ_pY'_{t-p} + θ₁ε_{t-1} + ... + θ_qε_{t-q} + ε_t

Where:
  Y'_t = differenced series (d times)
  φ   = AR coefficients
  θ   = MA coefficients
  ε_t = white noise error
```

### When to Use

- General-purpose forecasting
- Data that can be made stationary through differencing
- When you need interpretable parameters
- Short to medium-term forecasts

### Parameter Selection

| Parameter | Typical Range | How to Choose |
|-----------|---------------|---------------|
| p | 0-5 | ACF cuts off, PACF tails off → use PACF |
| d | 0-2 | Number of differences to achieve stationarity |
| q | 0-5 | PACF cuts off, ACF tails off → use ACF |

### Example

```typescript
import { initWasm, Arima } from 'rustful-ts';

await initWasm();

// ARIMA(1, 1, 1) - common starting point
const model = new Arima(1, 1, 1);

const data = [/* your time series */];
await model.fit(data);

const forecast = await model.predict(10);
```

---

## Exponential Smoothing

Exponential smoothing methods assign exponentially decreasing weights to past observations. Three variants are available:

### Simple Exponential Smoothing (SES)

**Best for**: Data without trend or seasonality

```
S_t = α × Y_t + (1 - α) × S_{t-1}
```

| Parameter | Range | Effect |
|-----------|-------|--------|
| α (alpha) | 0-1 | Higher = more responsive to recent changes |

**Produces flat forecasts** - the forecast for all future periods is the current level.

```typescript
import { SimpleExponentialSmoothing } from 'rustful-ts';

const model = new SimpleExponentialSmoothing(0.3);
await model.fit(data);
const forecast = await model.predict(5); // All values will be the same
```

### Double Exponential Smoothing (Holt's Method)

**Best for**: Data with trend but no seasonality

```
Level:  L_t = α × Y_t + (1 - α) × (L_{t-1} + T_{t-1})
Trend:  T_t = β × (L_t - L_{t-1}) + (1 - β) × T_{t-1}
Forecast: F_{t+h} = L_t + h × T_t
```

| Parameter | Range | Effect |
|-----------|-------|--------|
| α (alpha) | 0-1 | Level smoothing |
| β (beta) | 0-1 | Trend smoothing |

```typescript
import { Holt } from 'rustful-ts';

const model = new Holt(0.3, 0.1);
await model.fit(data);
const forecast = await model.predict(5); // Values follow the trend
```

### Triple Exponential Smoothing (Holt-Winters)

**Best for**: Data with both trend and seasonality

Two variants:
- **Additive**: Seasonal effect is constant (e.g., +10 units in December)
- **Multiplicative**: Seasonal effect is proportional (e.g., +10% in December)

```
Additive:
  Level:    L_t = α(Y_t - S_{t-m}) + (1-α)(L_{t-1} + T_{t-1})
  Trend:    T_t = β(L_t - L_{t-1}) + (1-β)T_{t-1}
  Seasonal: S_t = γ(Y_t - L_t) + (1-γ)S_{t-m}
  Forecast: F_{t+h} = L_t + h×T_t + S_{t+h-m}
```

| Parameter | Range | Effect |
|-----------|-------|--------|
| α (alpha) | 0-1 | Level smoothing |
| β (beta) | 0-1 | Trend smoothing |
| γ (gamma) | 0-1 | Seasonal smoothing |
| period | ≥2 | Seasonal period (12 for monthly, 4 for quarterly) |

```typescript
import { HoltWinters, SeasonalType } from 'rustful-ts';

// Monthly data with yearly seasonality
const model = new HoltWinters(0.3, 0.1, 0.2, 12, SeasonalType.Additive);
await model.fit(monthlyData); // Need at least 2 full seasons (24 months)
const forecast = await model.predict(12); // Forecast next year
```

---

## Moving Averages

### Simple Moving Average (SMA)

Computes the unweighted mean of the previous `n` observations.

```
SMA_t = (Y_{t} + Y_{t-1} + ... + Y_{t-n+1}) / n
```

**Best for**:
- Smoothing noisy data
- Identifying trends
- Quick baseline forecasts

```typescript
import { SimpleMovingAverage } from 'rustful-ts';

const sma = new SimpleMovingAverage(5); // 5-period SMA
await sma.fit(data);

// Get smoothed values
const smoothed = sma.getSmoothedValues();

// Forecast (produces flat values)
const forecast = await sma.predict(3);
```

### Weighted Moving Average (WMA)

Like SMA but with custom weights. Typically, more recent observations get higher weights.

```
WMA_t = Σ(w_i × Y_{t-i+1}) / Σw_i
```

```typescript
import { WeightedMovingAverage } from 'rustful-ts';

// Linear weights: [1, 2, 3] - most recent has weight 3
const wma = WeightedMovingAverage.linear(3);
await wma.fit(data);

// Or with custom weights
const customWma = new WeightedMovingAverage([0.1, 0.2, 0.3, 0.4]);
```

---

## Linear Regression

### Basic Linear Regression

Fits a straight line through time series data.

```
Y_t = a + b × t + ε_t
```

**Best for**:
- Data with clear linear trend
- Quick trend estimation
- Baseline model

```typescript
import { LinearRegression } from 'rustful-ts';

const model = new LinearRegression();
await model.fit(data);

console.log('Slope:', model.getSlope());
console.log('Intercept:', model.getIntercept());
console.log('R²:', model.getRSquared());

const forecast = await model.predict(5);
```

### Seasonal Linear Regression

Extends linear regression with seasonal dummy variables.

```
Y_t = a + b × t + Σ(s_i × D_i) + ε_t
```

Where `D_i` are seasonal dummy variables.

```typescript
import { SeasonalLinearRegression } from 'rustful-ts';

// Quarterly data with yearly seasonality
const model = new SeasonalLinearRegression(4);
await model.fit(quarterlyData);

// Get seasonal factors
const factors = model.getSeasonalFactors();
console.log('Q1 effect:', factors[0]);
console.log('Q2 effect:', factors[1]);
// ...
```

---

## Algorithm Comparison

### By Data Characteristics

| Data Pattern | Recommended Algorithm |
|--------------|----------------------|
| No trend, no seasonality | SES, SMA |
| Trend only | Holt, Linear Regression |
| Seasonality only | Seasonal Linear Regression |
| Trend + Seasonality | Holt-Winters |
| Complex patterns | ARIMA |
| Noisy data | SMA, WMA |

### By Use Case

| Use Case | Best Choice |
|----------|-------------|
| Quick baseline | SMA, Linear Regression |
| Production forecasting | ARIMA, Holt-Winters |
| Real-time updates | SES (fast updates) |
| Interpretability needed | Linear Regression |
| Seasonal business data | Holt-Winters |

### Computational Complexity

| Algorithm | Fit Complexity | Predict Complexity |
|-----------|---------------|-------------------|
| SMA | O(n) | O(1) |
| SES | O(n) | O(1) |
| Holt | O(n) | O(h) |
| Holt-Winters | O(n) | O(h) |
| Linear Regression | O(n) | O(h) |
| ARIMA | O(n²) | O(h) |

Where n = data length, h = forecast horizon

---

## See Also

- [API Reference](../api/README.md) - Detailed API documentation
- [Examples](../../ts/examples/) - Code examples
