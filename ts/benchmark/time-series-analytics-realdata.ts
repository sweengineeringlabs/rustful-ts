/**
 * Performance benchmark: Real-world datasets
 * Compares WASM-backed implementations using actual time series data
 */
import * as fs from 'fs';
import * as path from 'path';

// CSV parsing helper
function parseCSV(filepath: string): { headers: string[]; data: number[] } {
  const content = fs.readFileSync(filepath, 'utf-8');
  const lines = content.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.replace(/"/g, ''));
  const data = lines.slice(1).map(line => {
    const parts = line.split(',');
    return parseFloat(parts[parts.length - 1].replace(/"/g, ''));
  });
  return { headers, data };
}

// Compute returns from prices
function computeReturns(prices: number[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  return returns;
}

// Timing helper
async function benchmark(
  name: string,
  fn: () => Promise<void> | void,
  iterations: number = 100
): Promise<{ name: string; avgMs: number; totalMs: number; iterations: number }> {
  // Warmup
  for (let i = 0; i < 5; i++) {
    await fn();
  }

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    await fn();
  }
  const totalMs = performance.now() - start;
  const avgMs = totalMs / iterations;

  return { name, avgMs, totalMs, iterations };
}

async function runBenchmarks() {
  console.log('='.repeat(70));
  console.log('Performance Benchmark: rustful-ts (Real-World Datasets)');
  console.log('='.repeat(70));
  console.log();

  // Load datasets
  const datasetsDir = path.join(__dirname, 'datasets');

  console.log('Loading datasets...');
  const stockData = parseCSV(path.join(datasetsDir, 'stock-spy-daily.csv'));
  const weatherData = parseCSV(path.join(datasetsDir, 'weather-melbourne-daily-temp.csv'));
  const sunspotsData = parseCSV(path.join(datasetsDir, 'sensor-sunspots.csv'));
  const airlineData = parseCSV(path.join(datasetsDir, 'sensor-airline-passengers.csv'));

  console.log(`  Stock (SPY): ${stockData.data.length} points`);
  console.log(`  Weather (Melbourne): ${weatherData.data.length} points`);
  console.log(`  Sunspots: ${sunspotsData.data.length} points`);
  console.log(`  Airline: ${airlineData.data.length} points`);
  console.log();

  // Initialize WASM
  const { initWasm, isWasmReady } = await import('../src/wasm-loader');
  console.log('Initializing WASM...');
  await initWasm();
  console.log(`WASM ready: ${isWasmReady()}`);
  console.log();

  const results: Array<{ name: string; avgMs: number; totalMs: number; iterations: number; dataset: string }> = [];

  // Compute returns for financial metrics
  const stockReturns = computeReturns(stockData.data);

  // ============================================
  // Financial Risk Metrics (Stock Data)
  // ============================================
  console.log('--- Financial Risk Metrics (SPY Stock Data) ---');

  const risk = await import('../src/financial/risk');

  results.push({
    ...await benchmark('sharpeRatio (SPY)', async () => {
      await risk.sharpeRatio(stockReturns, 0.02);
    }, 500),
    dataset: 'stock'
  });

  results.push({
    ...await benchmark('sortinoRatio (SPY)', async () => {
      await risk.sortinoRatio(stockReturns, 0.02);
    }, 500),
    dataset: 'stock'
  });

  results.push({
    ...await benchmark('maxDrawdown (SPY)', async () => {
      await risk.maxDrawdown(stockData.data);
    }, 500),
    dataset: 'stock'
  });

  results.push({
    ...await benchmark('varHistorical (SPY)', async () => {
      await risk.varHistorical(stockReturns, 0.95);
    }, 500),
    dataset: 'stock'
  });

  console.log();

  // ============================================
  // Anomaly Detection (Weather Data)
  // ============================================
  console.log('--- Anomaly Detection (Melbourne Temperature) ---');

  const { ZScoreDetector, IQRDetector } = await import('../src/anomaly/detectors');

  const zscoreDetector = new ZScoreDetector(3.0);
  results.push({
    ...await benchmark('ZScoreDetector.fit (Weather)', async () => {
      await zscoreDetector.fit(weatherData.data);
    }, 100),
    dataset: 'weather'
  });

  await zscoreDetector.fit(weatherData.data);
  results.push({
    ...await benchmark('ZScoreDetector.detect (Weather)', async () => {
      await zscoreDetector.detect(weatherData.data);
    }, 100),
    dataset: 'weather'
  });

  const iqrDetector = new IQRDetector(1.5);
  results.push({
    ...await benchmark('IQRDetector.fit (Weather)', async () => {
      await iqrDetector.fit(weatherData.data);
    }, 100),
    dataset: 'weather'
  });

  await iqrDetector.fit(weatherData.data);
  results.push({
    ...await benchmark('IQRDetector.detect (Weather)', async () => {
      await iqrDetector.detect(weatherData.data);
    }, 100),
    dataset: 'weather'
  });

  console.log();

  // ============================================
  // Core Algorithms (Sunspots - Long Cyclical)
  // ============================================
  console.log('--- Core Algorithms (Sunspots Data) ---');

  const { Arima } = await import('../src/algorithms/arima');
  const { SimpleExponentialSmoothing } = await import('../src/algorithms/exponential-smoothing');

  const arima = new Arima(1, 1, 1);
  results.push({
    ...await benchmark('Arima.fit (Sunspots)', async () => {
      await arima.fit(sunspotsData.data);
    }, 20),
    dataset: 'sunspots'
  });

  await arima.fit(sunspotsData.data);
  results.push({
    ...await benchmark('Arima.predict (Sunspots)', async () => {
      await arima.predict(12);
    }, 100),
    dataset: 'sunspots'
  });

  const ses = new SimpleExponentialSmoothing(0.3);
  results.push({
    ...await benchmark('SES.fit (Sunspots)', async () => {
      await ses.fit(sunspotsData.data);
    }, 50),
    dataset: 'sunspots'
  });

  await ses.fit(sunspotsData.data);
  results.push({
    ...await benchmark('SES.predict (Sunspots)', async () => {
      await ses.predict(12);
    }, 500),
    dataset: 'sunspots'
  });

  console.log();

  // ============================================
  // Core Algorithms (Airline - Short Seasonal)
  // ============================================
  console.log('--- Core Algorithms (Airline Data) ---');

  const arimaAirline = new Arima(1, 1, 1);
  results.push({
    ...await benchmark('Arima.fit (Airline)', async () => {
      await arimaAirline.fit(airlineData.data);
    }, 50),
    dataset: 'airline'
  });

  await arimaAirline.fit(airlineData.data);
  results.push({
    ...await benchmark('Arima.predict (Airline)', async () => {
      await arimaAirline.predict(12);
    }, 100),
    dataset: 'airline'
  });

  const sesAirline = new SimpleExponentialSmoothing(0.3);
  results.push({
    ...await benchmark('SES.fit (Airline)', async () => {
      await sesAirline.fit(airlineData.data);
    }, 100),
    dataset: 'airline'
  });

  console.log();

  // ============================================
  // Ensemble Methods (Multiple Predictions)
  // ============================================
  console.log('--- Ensemble Methods ---');

  const { combinePredictions, EnsembleMethod } = await import('../src/automl/ensemble');

  // Generate predictions from different models for ensemble
  const predictions: number[][] = [];
  for (let i = 0; i < 5; i++) {
    const model = new SimpleExponentialSmoothing(0.2 + i * 0.1);
    await model.fit(airlineData.data);
    predictions.push(await model.predict(24));
  }

  results.push({
    ...await benchmark('combinePredictions Average (5 models)', async () => {
      await combinePredictions(predictions, EnsembleMethod.Average);
    }, 500),
    dataset: 'ensemble'
  });

  results.push({
    ...await benchmark('combinePredictions Median (5 models)', async () => {
      await combinePredictions(predictions, EnsembleMethod.Median);
    }, 500),
    dataset: 'ensemble'
  });

  results.push({
    ...await benchmark('combinePredictions Weighted (5 models)', async () => {
      const weights = [0.3, 0.25, 0.2, 0.15, 0.1];
      await combinePredictions(predictions, EnsembleMethod.WeightedAverage, weights);
    }, 500),
    dataset: 'ensemble'
  });

  console.log();

  // ============================================
  // Print Results
  // ============================================
  console.log('='.repeat(70));
  console.log('RESULTS (Real-World Datasets)');
  console.log('='.repeat(70));
  console.log();
  console.log('| Benchmark | Dataset | Avg (ms) | Total (ms) | Iterations |');
  console.log('|-----------|---------|----------|------------|------------|');

  for (const r of results) {
    console.log(`| ${r.name.padEnd(35)} | ${r.dataset.padEnd(7)} | ${r.avgMs.toFixed(4).padStart(8)} | ${r.totalMs.toFixed(2).padStart(10)} | ${r.iterations.toString().padStart(10)} |`);
  }

  console.log();
  console.log('Benchmark complete.');

  // Output as JSON for easy comparison
  const jsonOutput = {
    timestamp: new Date().toISOString(),
    wasmReady: isWasmReady(),
    datasets: {
      stock: { name: 'SPY Daily Prices', points: stockData.data.length, source: 'Yahoo Finance' },
      weather: { name: 'Melbourne Daily Temp', points: weatherData.data.length, source: 'BoM' },
      sunspots: { name: 'Monthly Sunspots', points: sunspotsData.data.length, source: 'SIDC' },
      airline: { name: 'Monthly Passengers', points: airlineData.data.length, source: 'Box-Jenkins' }
    },
    results: results.map(r => ({
      name: r.name,
      dataset: r.dataset,
      avgMs: r.avgMs,
      totalMs: r.totalMs,
      iterations: r.iterations
    }))
  };

  console.log('\n--- JSON Output ---');
  console.log(JSON.stringify(jsonOutput, null, 2));
}

runBenchmarks().catch(console.error);
