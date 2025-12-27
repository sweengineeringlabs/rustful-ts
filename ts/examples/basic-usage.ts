/**
 * Basic usage examples for rustful-ts
 *
 * Run with: npx ts-node examples/basic-usage.ts
 */

import {
  initWasm,
  Arima,
  SimpleExponentialSmoothing,
  Holt,
  HoltWinters,
  SeasonalType,
  SimpleMovingAverage,
  LinearRegression,
  mae,
  rmse,
  trainTestSplit,
  normalize,
} from '../src';

async function main() {
  // Initialize WASM (required once)
  await initWasm();
  console.log('WASM initialized!\n');

  // ==========================================================================
  // Example 1: ARIMA for general forecasting
  // ==========================================================================
  console.log('=== ARIMA Example ===');

  const trendData = Array.from({ length: 50 }, (_, i) =>
    10 + i * 2 + Math.random() * 3
  );

  const arima = new Arima(1, 1, 0);
  await arima.fit(trendData);
  const arimaForecast = await arima.predict(5);

  console.log('ARIMA(1,1,0) forecast:', arimaForecast.map((x) => x.toFixed(2)));
  console.log();

  // ==========================================================================
  // Example 2: Holt-Winters for seasonal data
  // ==========================================================================
  console.log('=== Holt-Winters Example ===');

  // Generate monthly data with yearly seasonality
  const seasonalData = Array.from({ length: 48 }, (_, i) => {
    const trend = 100 + i * 0.5;
    const season = 20 * Math.sin((i * Math.PI * 2) / 12);
    const noise = Math.random() * 5;
    return trend + season + noise;
  });

  const hw = new HoltWinters(0.3, 0.1, 0.2, 12, SeasonalType.Additive);
  await hw.fit(seasonalData);
  const hwForecast = await hw.predict(12);

  console.log('Holt-Winters forecast (next 12 months):');
  console.log(hwForecast.map((x) => x.toFixed(2)).join(', '));
  console.log('Seasonal components:', hw.getSeasonalComponents().map((x) => x.toFixed(2)));
  console.log();

  // ==========================================================================
  // Example 3: Model comparison
  // ==========================================================================
  console.log('=== Model Comparison ===');

  const linearData = Array.from({ length: 30 }, (_, i) => 10 + i * 2);
  const { train, test } = trainTestSplit(linearData, 0.2);

  // Try different models
  const models = [
    { name: 'SES', model: new SimpleExponentialSmoothing(0.3) },
    { name: 'Holt', model: new Holt(0.3, 0.1) },
    { name: 'Linear', model: new LinearRegression() },
  ];

  for (const { name, model } of models) {
    await model.fit(train);
    const predictions = await model.predict(test.length);

    console.log(`${name}:`);
    console.log(`  MAE:  ${mae(test, predictions).toFixed(4)}`);
    console.log(`  RMSE: ${rmse(test, predictions).toFixed(4)}`);
  }
  console.log();

  // ==========================================================================
  // Example 4: Data preprocessing
  // ==========================================================================
  console.log('=== Preprocessing Example ===');

  const rawData = [100, 200, 300, 400, 500];
  const { normalized, min, max } = normalize(rawData);

  console.log('Original:', rawData);
  console.log('Normalized:', normalized);
  console.log(`Min: ${min}, Max: ${max}`);
  console.log();

  // ==========================================================================
  // Example 5: Moving Average smoothing
  // ==========================================================================
  console.log('=== Moving Average Example ===');

  const noisyData = Array.from({ length: 20 }, (_, i) =>
    50 + Math.random() * 20
  );

  const sma = new SimpleMovingAverage(5);
  await sma.fit(noisyData);
  const smoothed = sma.getSmoothedValues();

  console.log('Original variance:', variance(noisyData).toFixed(2));
  console.log('Smoothed variance:', variance(smoothed).toFixed(2));
  console.log('Noise reduced by:', ((1 - variance(smoothed) / variance(noisyData)) * 100).toFixed(1) + '%');
}

function variance(data: number[]): number {
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  return data.reduce((acc, x) => acc + (x - mean) ** 2, 0) / data.length;
}

main().catch(console.error);
