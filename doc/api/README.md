# API Reference

Complete API documentation for rustful-ts.

## Table of Contents

- [Initialization](#initialization)
- [Predictor Interface](#predictor-interface)
- [Algorithms](#algorithms)
  - [Arima](#arima)
  - [SimpleExponentialSmoothing](#simpleexponentialsmoothing)
  - [Holt](#holt)
  - [HoltWinters](#holtwinters)
  - [SimpleMovingAverage](#simplemovingaverage)
  - [WeightedMovingAverage](#weightedmovingaverage)
  - [LinearRegression](#linearregression)
  - [SeasonalLinearRegression](#seasonallinearregression)
  - [TimeSeriesKNN](#timeseriesknn)
- [Utilities](#utilities)
  - [Metrics](#metrics)
  - [Preprocessing](#preprocessing)
- [Types](#types)

---

## Initialization

### `initWasm()`

Initialize the WebAssembly module. Must be called once before using any algorithms.

```typescript
import { initWasm, isWasmReady } from 'rustful-ts';

await initWasm();
console.log(isWasmReady()); // true
```

### `isWasmReady()`

Check if WASM is initialized.

```typescript
function isWasmReady(): boolean;
```

---

## Predictor Interface

All forecasting models implement the `Predictor` interface. This provides a consistent API across all algorithms, enabling plug-and-play model swapping.

### Interface Definition

```typescript
interface Predictor {
  /**
   * Fit the model to historical data
   * @param data - Time series observations
   */
  fit(data: number[]): Promise<void>;

  /**
   * Generate predictions for future time steps
   * @param steps - Number of steps to forecast
   * @returns Array of predicted values
   */
  predict(steps: number): Promise<number[]>;

  /**
   * Check if the model has been fitted
   */
  isFitted(): boolean;
}
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **Consistency** | Same API for all models |
| **Swappable** | Easy to compare different algorithms |
| **Testable** | Mock any model with the interface |
| **Extensible** | Add custom models that implement `Predictor` |

### Usage Pattern

```typescript
import { Predictor, Arima, Holt, HoltWinters } from 'rustful-ts';

// Generic forecasting function works with ANY model
async function forecast(
  model: Predictor,
  data: number[],
  horizon: number
): Promise<number[]> {
  await model.fit(data);
  return model.predict(horizon);
}

// Swap models easily
const data = [10, 12, 14, 16, 18, 20];

const arimaResult = await forecast(new Arima(1, 1, 0), data, 5);
const holtResult = await forecast(new Holt(0.3, 0.1), data, 5);

// Compare results
console.log('ARIMA:', arimaResult);
console.log('Holt:', holtResult);
```

### Model Comparison Example

```typescript
import { Predictor, Arima, Holt, LinearRegression, mae } from 'rustful-ts';

async function compareModels(
  models: { name: string; model: Predictor }[],
  trainData: number[],
  testData: number[]
): Promise<void> {
  for (const { name, model } of models) {
    await model.fit(trainData);
    const predictions = await model.predict(testData.length);
    const error = mae(testData, predictions);
    console.log(`${name}: MAE = ${error.toFixed(4)}`);
  }
}

// Usage
await compareModels(
  [
    { name: 'ARIMA(1,1,0)', model: new Arima(1, 1, 0) },
    { name: 'Holt', model: new Holt(0.3, 0.1) },
    { name: 'Linear', model: new LinearRegression() },
  ],
  trainData,
  testData
);
```

### Custom Model Implementation

```typescript
import { Predictor } from 'rustful-ts';

// Create your own model
class NaiveForecast implements Predictor {
  private lastValue: number = 0;
  private fitted: boolean = false;

  async fit(data: number[]): Promise<void> {
    this.lastValue = data[data.length - 1];
    this.fitted = true;
  }

  async predict(steps: number): Promise<number[]> {
    return Array(steps).fill(this.lastValue);
  }

  isFitted(): boolean {
    return this.fitted;
  }
}

// Works with existing infrastructure
const naive = new NaiveForecast();
await forecast(naive, data, 5);
```

---

## Algorithms

All algorithms implement the `Predictor` interface:

```typescript
interface Predictor {
  fit(data: number[]): Promise<void>;
  predict(steps: number): Promise<number[]>;
  isFitted(): boolean;
}
```

---

### Arima

ARIMA(p, d, q) model for time series forecasting.

#### Constructor

```typescript
new Arima(p: number, d: number, q: number)
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| p | number | 0-10 | AR order (autoregressive) |
| d | number | 0-2 | Differencing order |
| q | number | 0-10 | MA order (moving average) |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(data)` | `Promise<void>` | Fit model to data |
| `predict(steps)` | `Promise<number[]>` | Generate forecasts |
| `isFitted()` | `boolean` | Check if fitted |
| `getParams()` | `ArimaParams` | Get model parameters |

#### Example

```typescript
import { initWasm, Arima } from 'rustful-ts';

await initWasm();

const model = new Arima(1, 1, 1);
await model.fit([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]);

const forecast = await model.predict(5);
console.log(forecast); // [~34, ~36, ~38, ~40, ~42]

console.log(model.getParams()); // { p: 1, d: 1, q: 1 }
```

---

### SimpleExponentialSmoothing

Simple exponential smoothing for stationary data.

#### Constructor

```typescript
new SimpleExponentialSmoothing(alpha: number)
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| alpha | number | (0, 1) | Smoothing parameter |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(data)` | `Promise<void>` | Fit model |
| `predict(steps)` | `Promise<number[]>` | Forecast (flat) |
| `getLevel()` | `number` | Current level estimate |

#### Example

```typescript
import { SimpleExponentialSmoothing } from 'rustful-ts';

const model = new SimpleExponentialSmoothing(0.3);
await model.fit([10, 12, 11, 13, 12, 14]);

const forecast = await model.predict(3);
console.log(forecast); // [12.8, 12.8, 12.8] (flat forecast)
console.log(model.getLevel()); // 12.8
```

---

### Holt

Double exponential smoothing (Holt's linear trend method).

#### Constructor

```typescript
new Holt(alpha: number, beta: number)
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| alpha | number | (0, 1) | Level smoothing |
| beta | number | (0, 1) | Trend smoothing |

#### Example

```typescript
import { Holt } from 'rustful-ts';

const model = new Holt(0.3, 0.1);
await model.fit([10, 12, 14, 16, 18, 20]);

const forecast = await model.predict(3);
console.log(forecast); // [~22, ~24, ~26]
```

---

### HoltWinters

Triple exponential smoothing for seasonal data.

#### Constructor

```typescript
new HoltWinters(
  alpha: number,
  beta: number,
  gamma: number,
  period: number,
  seasonalType?: SeasonalType
)
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| alpha | number | (0, 1) | Level smoothing |
| beta | number | (0, 1) | Trend smoothing |
| gamma | number | (0, 1) | Seasonal smoothing |
| period | number | ≥2 | Seasonal period |
| seasonalType | SeasonalType | - | Additive or Multiplicative |

#### SeasonalType Enum

```typescript
enum SeasonalType {
  Additive = 'additive',
  Multiplicative = 'multiplicative'
}
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(data)` | `Promise<void>` | Fit (needs ≥2 periods) |
| `predict(steps)` | `Promise<number[]>` | Seasonal forecast |
| `getSeasonalComponents()` | `number[]` | Seasonal factors |

#### Example

```typescript
import { HoltWinters, SeasonalType } from 'rustful-ts';

// Monthly data with yearly seasonality
const model = new HoltWinters(0.3, 0.1, 0.2, 12, SeasonalType.Additive);
await model.fit(monthlyData); // At least 24 points

const forecast = await model.predict(12);
const seasonal = model.getSeasonalComponents(); // 12 factors
```

---

### SimpleMovingAverage

Simple moving average for smoothing.

#### Constructor

```typescript
new SimpleMovingAverage(window: number)
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| window | number | ≥2 | Window size |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(data)` | `Promise<void>` | Compute SMA |
| `predict(steps)` | `Promise<number[]>` | Flat forecast |
| `getSmoothedValues()` | `number[]` | Smoothed series |
| `getWindowSize()` | `number` | Window size |

#### Example

```typescript
import { SimpleMovingAverage } from 'rustful-ts';

const sma = new SimpleMovingAverage(3);
await sma.fit([1, 2, 3, 4, 5, 6, 7]);

console.log(sma.getSmoothedValues()); // [2, 3, 4, 5, 6]
console.log(await sma.predict(2));     // [6, 6]
```

---

### WeightedMovingAverage

Weighted moving average with custom weights.

#### Constructor

```typescript
new WeightedMovingAverage(weights: number[])
```

#### Static Methods

```typescript
static linear(window: number): WeightedMovingAverage
static exponential(window: number, decay?: number): WeightedMovingAverage
```

#### Example

```typescript
import { WeightedMovingAverage } from 'rustful-ts';

// Linear weights [1, 2, 3]
const wma = WeightedMovingAverage.linear(3);
await wma.fit([10, 12, 14, 16, 18, 20]);

console.log(wma.getSmoothedValues());
console.log(wma.getWeights()); // [1, 2, 3]
```

---

### LinearRegression

Simple linear trend model.

#### Constructor

```typescript
new LinearRegression()
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(data)` | `Promise<void>` | Fit trend line |
| `predict(steps)` | `Promise<number[]>` | Trend forecast |
| `getSlope()` | `number` | Trend per time unit |
| `getIntercept()` | `number` | Y-intercept |
| `getRSquared()` | `number` | Coefficient of determination |

#### Example

```typescript
import { LinearRegression } from 'rustful-ts';

const model = new LinearRegression();
await model.fit([10, 12, 14, 16, 18, 20]);

console.log(model.getSlope());     // 2.0
console.log(model.getIntercept()); // 10.0
console.log(model.getRSquared());  // 1.0

const forecast = await model.predict(3);
console.log(forecast); // [22, 24, 26]
```

---

### SeasonalLinearRegression

Linear regression with seasonal factors.

#### Constructor

```typescript
new SeasonalLinearRegression(period: number)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `getSeasonalFactors()` | `number[]` | Seasonal adjustments |
| `getSlope()` | `number` | Trend |
| `getIntercept()` | `number` | Base level |

---

### TimeSeriesKNN

K-Nearest Neighbors for time series prediction. Finds similar historical patterns and predicts based on what followed those patterns.

#### Constructor

```typescript
new TimeSeriesKNN(k: number, windowSize: number, metric?: DistanceMetric)
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| k | number | 1-50 | Number of neighbors |
| windowSize | number | 2-100 | Pattern window size |
| metric | DistanceMetric | - | Euclidean (default) or Manhattan |

#### DistanceMetric Enum

```typescript
enum DistanceMetric {
  Euclidean = 'euclidean',
  Manhattan = 'manhattan'
}
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(data)` | `Promise<void>` | Extract patterns from data |
| `predict(steps)` | `Promise<number[]>` | Predict based on similar patterns |
| `isFitted()` | `boolean` | Check if fitted |
| `getPatternCount()` | `number` | Number of stored patterns |
| `getWindowSize()` | `number` | Window size |
| `getK()` | `number` | Number of neighbors |

#### Example

```typescript
import { initWasm, TimeSeriesKNN, DistanceMetric } from 'rustful-ts';

await initWasm();

// Create KNN with k=5 neighbors, window size 10
const model = new TimeSeriesKNN(5, 10);

// Fit to periodic data
const data = Array.from({ length: 100 }, (_, i) =>
  Math.sin(i * 0.2) * 10 + 50
);
await model.fit(data);

// Predict next 5 values
const forecast = await model.predict(5);
console.log(forecast);

// Use Manhattan distance
const modelManhattan = new TimeSeriesKNN(5, 10, DistanceMetric.Manhattan);
```

---

## Utilities

### Metrics

```typescript
import { mae, mse, rmse, mape, smape, rSquared, computeMetrics } from 'rustful-ts';
```

| Function | Description |
|----------|-------------|
| `mae(actual, predicted)` | Mean Absolute Error |
| `mse(actual, predicted)` | Mean Squared Error |
| `rmse(actual, predicted)` | Root Mean Squared Error |
| `mape(actual, predicted)` | Mean Absolute Percentage Error |
| `smape(actual, predicted)` | Symmetric MAPE |
| `rSquared(actual, predicted)` | R-squared |
| `computeMetrics(actual, predicted)` | All metrics at once |

#### Example

```typescript
import { mae, rmse, computeMetrics } from 'rustful-ts';

const actual = [10, 20, 30, 40, 50];
const predicted = [12, 18, 33, 42, 48];

console.log(mae(actual, predicted));  // 2.8
console.log(rmse(actual, predicted)); // 3.03

const all = computeMetrics(actual, predicted);
console.log(all.rSquared); // 0.97
```

---

### Preprocessing

```typescript
import {
  normalize, denormalize,
  standardize, destandardize,
  difference, seasonalDifference,
  trainTestSplit, createLagFeatures,
  detectOutliersIQR, interpolateLinear
} from 'rustful-ts';
```

| Function | Description |
|----------|-------------|
| `normalize(data)` | Scale to [0, 1] |
| `denormalize(data, min, max)` | Reverse normalization |
| `standardize(data)` | Z-score standardization |
| `destandardize(data, mean, std)` | Reverse standardization |
| `difference(data, order)` | First-order differences |
| `seasonalDifference(data, period)` | Seasonal differences |
| `trainTestSplit(data, testRatio)` | Time-respecting split |
| `createLagFeatures(data, lags)` | Create ML features |
| `detectOutliersIQR(data, multiplier)` | Find outlier indices |
| `interpolateLinear(data)` | Fill NaN values |

#### Example

```typescript
import { normalize, difference, trainTestSplit } from 'rustful-ts';

// Normalize
const { normalized, min, max } = normalize([10, 20, 30, 40, 50]);
console.log(normalized); // [0, 0.25, 0.5, 0.75, 1]

// Difference
const diff = difference([1, 3, 6, 10], 1);
console.log(diff); // [2, 3, 4]

// Split
const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const { train, test } = trainTestSplit(data, 0.2);
console.log(train); // [1, 2, 3, 4, 5, 6, 7, 8]
console.log(test);  // [9, 10]
```

---

## Types

```typescript
// Time series data
type TimeSeriesData = number[];

// Predictor interface
interface Predictor {
  fit(data: TimeSeriesData): Promise<void>;
  predict(steps: number): Promise<number[]>;
  isFitted(): boolean;
}

// ARIMA parameters
interface ArimaParams {
  p: number;
  d: number;
  q: number;
}

// Metrics summary
interface MetricsSummary {
  mae: number;
  mse: number;
  rmse: number;
  mape: number;
  smape: number;
  rSquared: number;
}

// Cross-validation results
interface CrossValidationResults {
  maeScores: number[];
  rmseScores: number[];
  meanMae: number;
  meanRmse: number;
  stdMae: number;
  nFolds: number;
}
```
