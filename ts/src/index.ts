/**
 * rustful-ts: High-performance time series prediction framework
 *
 * This library provides TypeScript bindings for time series forecasting
 * algorithms implemented in Rust and compiled to WebAssembly, plus
 * additional modules for pipelines, financial analytics, anomaly detection,
 * and automatic model selection.
 *
 * @packageDocumentation
 */

// ============================================
// Core Algorithms
// ============================================
export { Arima } from './algorithms/arima';
export {
  SimpleExponentialSmoothing,
  Holt,
  HoltWinters,
  SeasonalType,
} from './algorithms/exponential-smoothing';
export { SimpleMovingAverage, WeightedMovingAverage } from './algorithms/moving-average';
export { LinearRegression, SeasonalLinearRegression } from './algorithms/linear-regression';
export { TimeSeriesKNN, DistanceMetric } from './algorithms/knn';

// ============================================
// Pipeline API
// ============================================
export { Pipeline } from './pipeline';
export {
  PipelineStep,
  NormalizeStep,
  StandardizeStep,
  DifferenceStep,
  LogTransformStep,
  ClipOutliersStep,
} from './pipeline/steps';

// ============================================
// Financial Analytics
// ============================================
export { Portfolio, Position } from './financial/portfolio';
export { BacktestResult, backtest, SimpleStrategy, Trade } from './financial/backtesting';
export {
  sharpeRatio,
  sharpeRatioSync,
  sortinoRatio,
  sortinoRatioSync,
  maxDrawdown,
  maxDrawdownSync,
  drawdownSeries,
  varHistorical,
  varHistoricalSync,
  cvar,
  dailyReturns,
  cumulativeReturns,
  annualizedReturn,
  annualizedVolatility,
} from './financial/risk';
export { Signal, SignalGenerator, SMACrossover, RSIStrategy } from './financial/signals';

// ============================================
// Anomaly Detection
// ============================================
export {
  AnomalyDetector,
  AnomalyResult,
  ZScoreDetector,
  IQRDetector,
  MADDetector,
} from './anomaly/detectors';
export { Monitor, Alert, AlertSeverity } from './anomaly/monitor';

// ============================================
// AutoML
// ============================================
export {
  AutoSelector,
  AutoMLConfig,
  ModelSelectionResult,
  ModelType,
} from './automl/selector';
export { EnsembleForecaster, EnsembleMethod, combinePredictions } from './automl/ensemble';

// ============================================
// Utilities
// ============================================
export * from './utils/metrics';
export * from './utils/preprocessing';

// ============================================
// Types
// ============================================
export * from './types';

// ============================================
// WASM Initialization
// ============================================
export { initWasm, initWasmFromPath, isWasmReady, getRuntime } from './wasm-loader';
