/**
 * Example: Fetch stock data and forecast prices
 *
 * Run with:
 *   npx ts-node examples/stock-forecast.ts
 *
 * Requires: npm install yahoo-finance2
 */

import {
  initWasm,
  Arima,
  Holt,
  LinearRegression,
  mae,
  rmse,
  mape,
  trainTestSplit,
} from '../src';

// Simple Yahoo Finance fetcher using the public API
async function fetchYahooData(
  symbol: string,
  startDate: string,
  endDate: string
): Promise<{ dates: string[]; prices: number[] }> {
  const start = Math.floor(new Date(startDate).getTime() / 1000);
  const end = Math.floor(new Date(endDate).getTime() / 1000);

  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=${start}&period2=${end}&interval=1d`;

  const response = await fetch(url);
  const data = await response.json();

  if (data.chart.error) {
    throw new Error(data.chart.error.description);
  }

  const result = data.chart.result[0];
  const timestamps: number[] = result.timestamp;
  const closes: number[] = result.indicators.quote[0].close;

  const dates: string[] = [];
  const prices: number[] = [];

  for (let i = 0; i < timestamps.length; i++) {
    if (closes[i] != null) {
      dates.push(new Date(timestamps[i] * 1000).toISOString().split('T')[0]);
      prices.push(closes[i]);
    }
  }

  return { dates, prices };
}

async function main() {
  console.log('=== Stock Price Forecasting Demo ===\n');

  // Initialize WASM
  await initWasm();

  // Fetch data
  const symbol = 'AAPL';
  const startDate = '2024-01-01';
  const endDate = '2024-12-01';

  console.log(`Fetching ${symbol} data from ${startDate} to ${endDate}...`);

  const { dates, prices } = await fetchYahooData(symbol, startDate, endDate);
  console.log(`Retrieved ${prices.length} data points\n`);

  // Split train/test
  const { train, test } = trainTestSplit(prices, 0.2);
  const forecastHorizon = test.length;

  console.log(`Training set: ${train.length} points`);
  console.log(`Test set: ${test.length} points\n`);

  // Price stats
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  console.log(`Price range: $${minPrice.toFixed(2)} - $${maxPrice.toFixed(2)}\n`);

  console.log('=== Model Comparison ===\n');

  // 1. Linear Regression
  const linear = new LinearRegression();
  await linear.fit(train);
  const linearForecast = await linear.predict(forecastHorizon);

  console.log('Linear Regression:');
  console.log(`  Slope: ${linear.getSlope().toFixed(4)} ($ per day)`);
  console.log(`  R²: ${linear.getRSquared().toFixed(4)}`);
  printMetrics(test, linearForecast);

  // 2. Holt
  const holt = new Holt(0.3, 0.1);
  await holt.fit(train);
  const holtForecast = await holt.predict(forecastHorizon);

  console.log('Holt (Double Exp. Smoothing):');
  printMetrics(test, holtForecast);

  // 3. ARIMA
  const arima = new Arima(1, 1, 0);
  await arima.fit(train);
  const arimaForecast = await arima.predict(forecastHorizon);

  console.log('ARIMA(1,1,0):');
  printMetrics(test, arimaForecast);

  // Sample predictions
  console.log('\n=== Sample Predictions (last 5 days) ===\n');
  console.log('Date        |    Actual |     ARIMA');
  console.log('------------|-----------|----------');

  const startIdx = Math.max(0, test.length - 5);
  for (let i = startIdx; i < test.length; i++) {
    const dateIdx = train.length + i;
    const date = dates[dateIdx] || `Day ${dateIdx + 1}`;
    console.log(
      `${date} | $${test[i].toFixed(2).padStart(8)} | $${arimaForecast[i].toFixed(2).padStart(8)}`
    );
  }

  // Returns analysis
  console.log('\n=== Returns Analysis ===\n');
  const returns = prices.slice(1).map((p, i) => (p - prices[i]) / prices[i]);
  const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const volatility = Math.sqrt(
    returns.reduce((acc, r) => acc + (r - avgReturn) ** 2, 0) / returns.length
  );

  console.log(`Average daily return: ${(avgReturn * 100).toFixed(4)}%`);
  console.log(`Daily volatility: ${(volatility * 100).toFixed(4)}%`);
  console.log(`Annualized volatility: ${(volatility * Math.sqrt(252) * 100).toFixed(2)}%`);
}

function printMetrics(actual: number[], predicted: number[]) {
  console.log(`  MAE:  $${mae(actual, predicted).toFixed(2)}`);
  console.log(`  RMSE: $${rmse(actual, predicted).toFixed(2)}`);
  console.log(`  MAPE: ${(mape(actual, predicted) * 100).toFixed(2)}%\n`);
}

main().catch(console.error);
