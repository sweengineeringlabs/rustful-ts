/**
 * Moving Average methods for time series smoothing and forecasting
 *
 * @module algorithms/moving-average
 */

import { getWasmModule, ensureWasm } from '../wasm-loader';
import type { Predictor, TimeSeriesData } from '../types';

/**
 * Simple Moving Average (SMA)
 *
 * Computes the unweighted mean of the previous `window` observations.
 * Useful for smoothing noisy data and identifying trends.
 *
 * @example
 * ```typescript
 * import { initWasm, SimpleMovingAverage } from 'rustful-ts';
 *
 * await initWasm();
 *
 * const sma = new SimpleMovingAverage(3);
 * await sma.fit([10, 12, 11, 13, 15, 14, 16, 18]);
 *
 * // Get smoothed values
 * const smoothed = sma.getSmoothedValues();
 *
 * // Forecast (produces flat values)
 * const forecast = await sma.predict(3);
 * ```
 */
export class SimpleMovingAverage implements Predictor {
  private inner: unknown = null;
  private fitted = false;
  private readonly window: number;

  /**
   * Create a new SMA
   *
   * @param window - Number of observations to average (must be >= 2)
   */
  constructor(window: number) {
    if (window < 2) {
      throw new Error('Window size must be at least 2');
    }
    this.window = window;
  }

  async fit(data: TimeSeriesData): Promise<void> {
    await ensureWasm();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = getWasmModule() as any;

    this.inner = new wasm.WasmSMA(this.window);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (this.inner as any).fit(new Float64Array(data));
    this.fitted = true;
  }

  async predict(steps: number): Promise<number[]> {
    if (!this.fitted || !this.inner) {
      throw new Error('Model must be fitted before prediction');
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const result = (this.inner as any).predict(steps);
    return Array.from(result as Float64Array);
  }

  isFitted(): boolean {
    return this.fitted;
  }

  /**
   * Get the smoothed time series
   *
   * The smoothed series is shorter than the original by (window - 1) points.
   */
  getSmoothedValues(): number[] {
    if (!this.fitted || !this.inner) {
      throw new Error('Model must be fitted first');
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return Array.from((this.inner as any).smoothed_values() as Float64Array);
  }

  /**
   * Get window size
   */
  getWindowSize(): number {
    return this.window;
  }
}

/**
 * Weighted Moving Average (WMA)
 *
 * Like SMA but with custom weights for each position in the window.
 * Typically used with linearly decreasing weights (most recent has highest weight).
 *
 * @example
 * ```typescript
 * import { initWasm, WeightedMovingAverage } from 'rustful-ts';
 *
 * await initWasm();
 *
 * // Linear weights: oldest=1, middle=2, newest=3
 * const wma = WeightedMovingAverage.linear(3);
 * await wma.fit([10, 12, 11, 13, 15, 14, 16, 18]);
 *
 * const forecast = await wma.predict(3);
 * ```
 */
export class WeightedMovingAverage implements Predictor {
  private inner: unknown = null;
  private fitted = false;
  private readonly weights: number[];
  private smoothedValues: number[] = [];

  /**
   * Create a new WMA with specified weights
   *
   * Weights are applied in order: first weight to oldest observation.
   * Weights will be normalized to sum to 1 internally.
   *
   * @param weights - Weights for each position in window (must have at least 2 elements)
   */
  constructor(weights: number[]) {
    if (weights.length < 2) {
      throw new Error('Must have at least 2 weights');
    }
    if (weights.some((w) => w < 0)) {
      throw new Error('All weights must be non-negative');
    }
    const sum = weights.reduce((a, b) => a + b, 0);
    if (sum <= 0) {
      throw new Error('Weights must sum to a positive value');
    }

    this.weights = [...weights];
  }

  /**
   * Create WMA with linear weights (1, 2, 3, ..., n)
   *
   * @param window - Size of the window
   */
  static linear(window: number): WeightedMovingAverage {
    const weights = Array.from({ length: window }, (_, i) => i + 1);
    return new WeightedMovingAverage(weights);
  }

  /**
   * Create WMA with exponential weights
   *
   * @param window - Size of the window
   * @param decay - Decay factor (0 < decay < 1)
   */
  static exponential(window: number, decay: number = 0.5): WeightedMovingAverage {
    const weights = Array.from({ length: window }, (_, i) =>
      Math.pow(decay, window - i - 1)
    );
    return new WeightedMovingAverage(weights);
  }

  async fit(data: TimeSeriesData): Promise<void> {
    await ensureWasm();

    // Compute WMA manually since WASM binding may not be complete
    const window = this.weights.length;
    if (data.length < window) {
      throw new Error(`Need at least ${window} observations`);
    }

    // Normalize weights
    const sum = this.weights.reduce((a, b) => a + b, 0);
    const normalizedWeights = this.weights.map((w) => w / sum);

    this.smoothedValues = [];
    for (let i = 0; i <= data.length - window; i++) {
      let weightedSum = 0;
      for (let j = 0; j < window; j++) {
        weightedSum += normalizedWeights[j] * data[i + j];
      }
      this.smoothedValues.push(weightedSum);
    }

    this.fitted = true;
  }

  async predict(steps: number): Promise<number[]> {
    if (!this.fitted) {
      throw new Error('Model must be fitted before prediction');
    }

    // WMA produces flat forecasts at the last smoothed value
    const lastValue = this.smoothedValues[this.smoothedValues.length - 1];
    return Array(steps).fill(lastValue);
  }

  isFitted(): boolean {
    return this.fitted;
  }

  /**
   * Get the smoothed time series
   */
  getSmoothedValues(): number[] {
    if (!this.fitted) {
      throw new Error('Model must be fitted first');
    }
    return [...this.smoothedValues];
  }

  /**
   * Get weights
   */
  getWeights(): number[] {
    return [...this.weights];
  }
}
