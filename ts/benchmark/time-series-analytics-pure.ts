/**
 * Performance benchmark: Pure TypeScript implementation (sync methods)
 * This script is for the previous commit where methods were synchronous
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

// Timing helper (sync version)
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
  console.log('='.repeat(60));
  console.log('Performance Benchmark: rustful-ts (Pure TypeScript)');
  console.log('='.repeat(60));
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

  const { ZScoreDetector, IQRDetector } = await import('../src/anomaly/detectors');

  const anomalyData = generateData(MEDIUM);

  // Z-Score Detector
  const zscoreDetector = new ZScoreDetector(3.0);
  results.push(benchmark('ZScoreDetector.fit (1000 pts)', () => {
    zscoreDetector.fit(anomalyData);
  }, 100));

  zscoreDetector.fit(anomalyData);
  results.push(benchmark('ZScoreDetector.detect (1000 pts)', () => {
    zscoreDetector.detect(anomalyData);
  }, 100));

  // IQR Detector
  const iqrDetector = new IQRDetector(1.5);
  results.push(benchmark('IQRDetector.fit (1000 pts)', () => {
    iqrDetector.fit(anomalyData);
  }, 100));

  iqrDetector.fit(anomalyData);
  results.push(benchmark('IQRDetector.detect (1000 pts)', () => {
    iqrDetector.detect(anomalyData);
  }, 100));

  console.log();

  // ============================================
  // Financial Risk Metrics Benchmarks
  // ============================================
  console.log('--- Financial Risk Metrics ---');

  const risk = await import('../src/financial/risk');

  const returnsSmall = generateReturns(SMALL);
  const returnsMedium = generateReturns(MEDIUM);
  const returnsLarge = generateReturns(LARGE);
  const equityCurve = generateData(MEDIUM);

  // Sharpe Ratio
  results.push(benchmark('sharpeRatio (100 pts)', () => {
    risk.sharpeRatio(returnsSmall, 0.02);
  }, 500));

  results.push(benchmark('sharpeRatio (1000 pts)', () => {
    risk.sharpeRatio(returnsMedium, 0.02);
  }, 500));

  results.push(benchmark('sharpeRatio (10000 pts)', () => {
    risk.sharpeRatio(returnsLarge, 0.02);
  }, 100));

  // Sortino Ratio
  results.push(benchmark('sortinoRatio (1000 pts)', () => {
    risk.sortinoRatio(returnsMedium, 0.02);
  }, 500));

  // Max Drawdown
  results.push(benchmark('maxDrawdown (1000 pts)', () => {
    risk.maxDrawdown(equityCurve);
  }, 500));

  results.push(benchmark('maxDrawdown (10000 pts)', () => {
    risk.maxDrawdown(generateData(LARGE));
  }, 100));

  // VaR
  results.push(benchmark('varHistorical (1000 pts)', () => {
    risk.varHistorical(returnsMedium, 0.95);
  }, 500));

  console.log();

  // ============================================
  // Ensemble Benchmarks
  // ============================================
  console.log('--- Ensemble Methods ---');

  // Note: The pure TS version might have different API
  try {
    const { EnsembleMethod } = await import('../src/automl/ensemble');

    // Pure TS version had combine_predictions or similar
    // This is a placeholder - adjust based on actual API
    const predictions = Array.from({ length: 10 }, () => generateData(100));

    // Simple average implementation for comparison
    const combineAverage = (preds: number[][]) => {
      const nSteps = preds[0].length;
      return Array.from({ length: nSteps }, (_, i) =>
        preds.reduce((sum, p) => sum + p[i], 0) / preds.length
      );
    };

    results.push(benchmark('combinePredictions Average (10x100)', () => {
      combineAverage(predictions);
    }, 500));

    const combineMedian = (preds: number[][]) => {
      const nSteps = preds[0].length;
      return Array.from({ length: nSteps }, (_, i) => {
        const vals = preds.map(p => p[i]).sort((a, b) => a - b);
        const mid = Math.floor(vals.length / 2);
        return vals.length % 2 ? vals[mid] : (vals[mid-1] + vals[mid]) / 2;
      });
    };

    results.push(benchmark('combinePredictions Median (10x100)', () => {
      combineMedian(predictions);
    }, 500));

    results.push(benchmark('combinePredictions Weighted (10x100)', () => {
      const weights = Array.from({ length: 10 }, () => 0.1);
      const nSteps = predictions[0].length;
      Array.from({ length: nSteps }, (_, i) =>
        predictions.reduce((sum, p, j) => sum + p[i] * weights[j], 0)
      );
    }, 500));
  } catch (e) {
    console.log('Ensemble benchmarks skipped (different API)');
  }

  console.log();

  // ============================================
  // Print Results
  // ============================================
  console.log('='.repeat(60));
  console.log('RESULTS (Pure TypeScript)');
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
    implementation: 'Pure TypeScript',
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
