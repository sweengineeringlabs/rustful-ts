/**
 * K-Nearest Neighbors for time series prediction
 *
 * @module algorithms/knn
 */

import { getWasmModule, ensureWasm } from '../wasm-loader';
import type { Predictor, TimeSeriesData } from '../types';

/**
 * Distance metric for comparing time series patterns
 */
export enum DistanceMetric {
  /** Standard Euclidean distance */
  Euclidean = 'euclidean',
  /** Manhattan (L1) distance */
  Manhattan = 'manhattan',
}

/**
 * K-Nearest Neighbors for time series prediction
 *
 * Finds similar historical patterns and uses them to predict future values.
 *
 * **How it works:**
 * 1. Extract sliding windows (subsequences) from historical data
 * 2. For a new prediction, find the K most similar historical windows
 * 3. Predict based on what happened after those similar windows
 *
 * @example
 * ```typescript
 * import { initWasm, TimeSeriesKNN, DistanceMetric } from 'rustful-ts';
 *
 * await initWasm();
 *
 * // Create KNN with k=5 neighbors, window size of 10
 * const model = new TimeSeriesKNN(5, 10);
 *
 * // Fit to periodic data
 * const data = Array.from({ length: 100 }, (_, i) =>
 *   Math.sin(i * 0.2) * 10 + 50
 * );
 * await model.fit(data);
 *
 * // Predict next 5 values
 * const forecast = await model.predict(5);
 * ```
 */
export class TimeSeriesKNN implements Predictor {
  private inner: unknown = null;
  private fitted = false;
  private readonly k: number;
  private readonly windowSize: number;
  private readonly metric: DistanceMetric;

  /**
   * Create a new KNN time series predictor
   *
   * @param k - Number of neighbors to consider (1-50)
   * @param windowSize - Size of the pattern window (2-100)
   * @param metric - Distance metric (default: Euclidean)
   */
  constructor(
    k: number,
    windowSize: number,
    metric: DistanceMetric = DistanceMetric.Euclidean
  ) {
    if (k < 1 || k > 50) {
      throw new Error('k must be between 1 and 50');
    }
    if (windowSize < 2 || windowSize > 100) {
      throw new Error('windowSize must be between 2 and 100');
    }

    this.k = k;
    this.windowSize = windowSize;
    this.metric = metric;
  }

  /**
   * Fit the model to historical data
   *
   * @param data - Time series observations (minimum: k + windowSize + 1)
   */
  async fit(data: TimeSeriesData): Promise<void> {
    await ensureWasm();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = getWasmModule() as any;

    if (this.metric === DistanceMetric.Manhattan) {
      this.inner = wasm.WasmKNN.new_manhattan(this.k, this.windowSize);
    } else {
      this.inner = new wasm.WasmKNN(this.k, this.windowSize);
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (this.inner as any).fit(new Float64Array(data));
    this.fitted = true;
  }

  /**
   * Predict future values based on similar historical patterns
   *
   * @param steps - Number of steps to forecast
   * @returns Array of predicted values
   */
  async predict(steps: number): Promise<number[]> {
    if (!this.fitted || !this.inner) {
      throw new Error('Model must be fitted before prediction');
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const result = (this.inner as any).predict(steps);
    return Array.from(result as Float64Array);
  }

  /**
   * Check if the model has been fitted
   */
  isFitted(): boolean {
    return this.fitted;
  }

  /**
   * Get the number of stored patterns
   */
  getPatternCount(): number {
    if (!this.fitted || !this.inner) {
      return 0;
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (this.inner as any).n_patterns();
  }

  /**
   * Get the window size
   */
  getWindowSize(): number {
    return this.windowSize;
  }

  /**
   * Get k (number of neighbors)
   */
  getK(): number {
    return this.k;
  }

  /**
   * Get the distance metric
   */
  getMetric(): DistanceMetric {
    return this.metric;
  }
}
