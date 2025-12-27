/**
 * Performance benchmark: Pure TypeScript vs WASM-backed implementations
 */

// Generate test data
function generateData(size: number): number[] {
  const data: number[] = [];
  let value = 100;
  for (let i = 0; i < size; i++) {
    value += (Math.random() - 0.5) * 10;
    data.push(value);
  }
  return data;
}

function generateReturns(size: number): number[] {
  return Array.from({ length: size }, () => (Math.random() - 0.5) * 0.1);
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
  console.log('='.repeat(60));
  console.log('Performance Benchmark: rustful-ts');
  console.log('='.repeat(60));
  console.log();

  // Import dynamically to work with both versions
  const { initWasm, isWasmReady } = await import('./src/wasm-loader');

  // Initialize WASM
  console.log('Initializing WASM...');
  await initWasm();
  console.log(`WASM ready: ${isWasmReady()}`);
  console.log();

  const results: Array<{ name: string; avgMs: number; totalMs: number; iterations: number }> = [];

  // Test data sizes
  const SMALL = 100;
  const MEDIUM = 1000;
  const LARGE = 10000;

  // ============================================
  // Anomaly Detection Benchmarks
  // ============================================
  console.log('--- Anomaly Detection ---');

  const { ZScoreDetector, IQRDetector } = await import('./src/anomaly/detectors');

  const anomalyData = generateData(MEDIUM);

  // Z-Score Detector
  const zscoreDetector = new ZScoreDetector(3.0);
  results.push(await benchmark('ZScoreDetector.fit (1000 pts)', async () => {
    await zscoreDetector.fit(anomalyData);
  }, 100));

  await zscoreDetector.fit(anomalyData);
  results.push(await benchmark('ZScoreDetector.detect (1000 pts)', async () => {
    await zscoreDetector.detect(anomalyData);
  }, 100));

  // IQR Detector
  const iqrDetector = new IQRDetector(1.5);
  results.push(await benchmark('IQRDetector.fit (1000 pts)', async () => {
    await iqrDetector.fit(anomalyData);
  }, 100));

  await iqrDetector.fit(anomalyData);
  results.push(await benchmark('IQRDetector.detect (1000 pts)', async () => {
    await iqrDetector.detect(anomalyData);
  }, 100));

  console.log();

  // ============================================
  // Financial Risk Metrics Benchmarks
  // ============================================
  console.log('--- Financial Risk Metrics ---');

  const risk = await import('./src/financial/risk');

  const returnsSmall = generateReturns(SMALL);
  const returnsMedium = generateReturns(MEDIUM);
  const returnsLarge = generateReturns(LARGE);
  const equityCurve = generateData(MEDIUM);

  // Sharpe Ratio
  results.push(await benchmark('sharpeRatio (100 pts)', async () => {
    await risk.sharpeRatio(returnsSmall, 0.02);
  }, 500));

  results.push(await benchmark('sharpeRatio (1000 pts)', async () => {
    await risk.sharpeRatio(returnsMedium, 0.02);
  }, 500));

  results.push(await benchmark('sharpeRatio (10000 pts)', async () => {
    await risk.sharpeRatio(returnsLarge, 0.02);
  }, 100));

  // Sortino Ratio
  results.push(await benchmark('sortinoRatio (1000 pts)', async () => {
    await risk.sortinoRatio(returnsMedium, 0.02);
  }, 500));

  // Max Drawdown
  results.push(await benchmark('maxDrawdown (1000 pts)', async () => {
    await risk.maxDrawdown(equityCurve);
  }, 500));

  results.push(await benchmark('maxDrawdown (10000 pts)', async () => {
    await risk.maxDrawdown(generateData(LARGE));
  }, 100));

  // VaR
  results.push(await benchmark('varHistorical (1000 pts)', async () => {
    await risk.varHistorical(returnsMedium, 0.95);
  }, 500));

  console.log();

  // ============================================
  // Ensemble Benchmarks
  // ============================================
  console.log('--- Ensemble Methods ---');

  const { combinePredictions, EnsembleMethod } = await import('./src/automl/ensemble');

  const predictions = Array.from({ length: 10 }, () => generateData(100));

  results.push(await benchmark('combinePredictions Average (10x100)', async () => {
    await combinePredictions(predictions, EnsembleMethod.Average);
  }, 500));

  results.push(await benchmark('combinePredictions Median (10x100)', async () => {
    await combinePredictions(predictions, EnsembleMethod.Median);
  }, 500));

  results.push(await benchmark('combinePredictions Weighted (10x100)', async () => {
    const weights = Array.from({ length: 10 }, () => 0.1);
    await combinePredictions(predictions, EnsembleMethod.WeightedAverage, weights);
  }, 500));

  console.log();

  // ============================================
  // Core Algorithm Benchmarks (always WASM)
  // ============================================
  console.log('--- Core Algorithms (WASM) ---');

  const { Arima } = await import('./src/algorithms/arima');
  const { SimpleExponentialSmoothing } = await import('./src/algorithms/exponential-smoothing');

  const forecastData = generateData(MEDIUM);

  // ARIMA
  const arima = new Arima(1, 1, 1);
  results.push(await benchmark('Arima.fit (1000 pts)', async () => {
    await arima.fit(forecastData);
  }, 50));

  await arima.fit(forecastData);
  results.push(await benchmark('Arima.predict (10 steps)', async () => {
    await arima.predict(10);
  }, 100));

  // SES
  const ses = new SimpleExponentialSmoothing(0.3);
  results.push(await benchmark('SES.fit (1000 pts)', async () => {
    await ses.fit(forecastData);
  }, 100));

  await ses.fit(forecastData);
  results.push(await benchmark('SES.predict (10 steps)', async () => {
    await ses.predict(10);
  }, 500));

  console.log();

  // ============================================
  // Print Results
  // ============================================
  console.log('='.repeat(60));
  console.log('RESULTS');
  console.log('='.repeat(60));
  console.log();
  console.log('| Benchmark | Avg (ms) | Total (ms) | Iterations |');
  console.log('|-----------|----------|------------|------------|');

  for (const r of results) {
    console.log(`| ${r.name.padEnd(35)} | ${r.avgMs.toFixed(4).padStart(8)} | ${r.totalMs.toFixed(2).padStart(10)} | ${r.iterations.toString().padStart(10)} |`);
  }

  console.log();
  console.log('Benchmark complete.');

  // Output as JSON for easy comparison
  const jsonOutput = {
    timestamp: new Date().toISOString(),
    wasmReady: isWasmReady(),
    results: results.map(r => ({
      name: r.name,
      avgMs: r.avgMs,
      totalMs: r.totalMs,
      iterations: r.iterations
    }))
  };

  console.log('\n--- JSON Output ---');
  console.log(JSON.stringify(jsonOutput, null, 2));
}

runBenchmarks().catch(console.error);
