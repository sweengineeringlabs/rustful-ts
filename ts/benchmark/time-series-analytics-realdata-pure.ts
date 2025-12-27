/**
 * Performance benchmark: Real-world datasets (Pure TypeScript)
 * Run this at commit e73696b for comparison with WASM version
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

// Timing helper (sync version for pure TS)
function benchmark(
  name: string,
  fn: () => void,
  iterations: number = 100
): { name: string; avgMs: number; totalMs: number; iterations: number } {
  // Warmup
  for (let i = 0; i < 5; i++) {
    fn();
  }

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const totalMs = performance.now() - start;
  const avgMs = totalMs / iterations;

  return { name, avgMs, totalMs, iterations };
}

async function runBenchmarks() {
  console.log('='.repeat(70));
  console.log('Performance Benchmark: rustful-ts (Real-World Datasets - Pure TS)');
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

  const results: Array<{ name: string; avgMs: number; totalMs: number; iterations: number; dataset: string }> = [];

  // Compute returns for financial metrics
  const stockReturns = computeReturns(stockData.data);

  // ============================================
  // Financial Risk Metrics (Stock Data)
  // ============================================
  console.log('--- Financial Risk Metrics (SPY Stock Data) ---');

  const risk = await import('../src/financial/risk');

  results.push({
    ...benchmark('sharpeRatio (SPY)', () => {
      risk.sharpeRatio(stockReturns, 0.02);
    }, 500),
    dataset: 'stock'
  });

  results.push({
    ...benchmark('sortinoRatio (SPY)', () => {
      risk.sortinoRatio(stockReturns, 0.02);
    }, 500),
    dataset: 'stock'
  });

  results.push({
    ...benchmark('maxDrawdown (SPY)', () => {
      risk.maxDrawdown(stockData.data);
    }, 500),
    dataset: 'stock'
  });

  results.push({
    ...benchmark('varHistorical (SPY)', () => {
      risk.varHistorical(stockReturns, 0.95);
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
    ...benchmark('ZScoreDetector.fit (Weather)', () => {
      zscoreDetector.fit(weatherData.data);
    }, 100),
    dataset: 'weather'
  });

  zscoreDetector.fit(weatherData.data);
  results.push({
    ...benchmark('ZScoreDetector.detect (Weather)', () => {
      zscoreDetector.detect(weatherData.data);
    }, 100),
    dataset: 'weather'
  });

  const iqrDetector = new IQRDetector(1.5);
  results.push({
    ...benchmark('IQRDetector.fit (Weather)', () => {
      iqrDetector.fit(weatherData.data);
    }, 100),
    dataset: 'weather'
  });

  iqrDetector.fit(weatherData.data);
  results.push({
    ...benchmark('IQRDetector.detect (Weather)', () => {
      iqrDetector.detect(weatherData.data);
    }, 100),
    dataset: 'weather'
  });

  console.log();

  // ============================================
  // Ensemble Methods (Pure TS implementations)
  // ============================================
  console.log('--- Ensemble Methods ---');

  // Generate sample predictions for ensemble
  const predictions: number[][] = [];
  for (let i = 0; i < 5; i++) {
    // Simple forecast: last value + small increments
    const pred = Array.from({ length: 24 }, (_, j) =>
      airlineData.data[airlineData.data.length - 1] * (1 + (i * 0.01) + (j * 0.005))
    );
    predictions.push(pred);
  }

  // Pure TS ensemble implementations
  const combineAverage = (preds: number[][]) => {
    const nSteps = preds[0].length;
    return Array.from({ length: nSteps }, (_, i) =>
      preds.reduce((sum, p) => sum + p[i], 0) / preds.length
    );
  };

  const combineMedian = (preds: number[][]) => {
    const nSteps = preds[0].length;
    return Array.from({ length: nSteps }, (_, i) => {
      const vals = preds.map(p => p[i]).sort((a, b) => a - b);
      const mid = Math.floor(vals.length / 2);
      return vals.length % 2 ? vals[mid] : (vals[mid-1] + vals[mid]) / 2;
    });
  };

  const combineWeighted = (preds: number[][], weights: number[]) => {
    const nSteps = preds[0].length;
    return Array.from({ length: nSteps }, (_, i) =>
      preds.reduce((sum, p, j) => sum + p[i] * weights[j], 0)
    );
  };

  results.push({
    ...benchmark('combinePredictions Average (5 models)', () => {
      combineAverage(predictions);
    }, 500),
    dataset: 'ensemble'
  });

  results.push({
    ...benchmark('combinePredictions Median (5 models)', () => {
      combineMedian(predictions);
    }, 500),
    dataset: 'ensemble'
  });

  results.push({
    ...benchmark('combinePredictions Weighted (5 models)', () => {
      const weights = [0.3, 0.25, 0.2, 0.15, 0.1];
      combineWeighted(predictions, weights);
    }, 500),
    dataset: 'ensemble'
  });

  console.log();

  // ============================================
  // Print Results
  // ============================================
  console.log('='.repeat(70));
  console.log('RESULTS (Real-World Datasets - Pure TypeScript)');
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
    implementation: 'Pure TypeScript',
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
