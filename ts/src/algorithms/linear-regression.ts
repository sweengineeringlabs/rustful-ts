/**
 * Linear Regression for time series forecasting
 *
 * @module algorithms/linear-regression
 */

import { getWasmModule, ensureWasm } from '../wasm-loader';
import type { Predictor, TimeSeriesData } from '../types';

/**
 * Linear Regression model for time series
 *
 * Fits y = intercept + slope * t where t is the time index.
 * Good for data with a clear linear trend.
 *
 * @example
 * ```typescript
 * import { initWasm, LinearRegression } from 'rustful-ts';
 *
 * await initWasm();
 *
 * const model = new LinearRegression();
 * await model.fit([10, 12, 14, 16, 18, 20]);
 *
 * console.log(model.getSlope());     // ~2.0
 * console.log(model.getIntercept()); // ~10.0
 * console.log(model.getRSquared());  // ~1.0
 *
 * const forecast = await model.predict(3);
 * // Returns [22, 24, 26]
 * ```
 */
export class LinearRegression implements Predictor {
  private inner: unknown = null;
  private fitted = false;

  constructor() {}

  async fit(data: TimeSeriesData): Promise<void> {
    await ensureWasm();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = getWasmModule() as any;

    this.inner = new wasm.WasmLinearRegression();
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
   * Get the slope (trend per time unit)
   */
  getSlope(): number {
    if (!this.fitted || !this.inner) {
      throw new Error('Model must be fitted first');
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (this.inner as any).slope();
  }

  /**
   * Get the y-intercept
   */
  getIntercept(): number {
    if (!this.fitted || !this.inner) {
      throw new Error('Model must be fitted first');
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (this.inner as any).intercept();
  }

  /**
   * Get R-squared (coefficient of determination)
   *
   * Values close to 1.0 indicate a good fit.
   */
  getRSquared(): number {
    if (!this.fitted || !this.inner) {
      throw new Error('Model must be fitted first');
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (this.inner as any).r_squared();
  }
}

/**
 * Linear Regression with seasonal dummy variables
 *
 * Extends basic linear regression to capture seasonal patterns.
 *
 * @example
 * ```typescript
 * import { initWasm, SeasonalLinearRegression } from 'rustful-ts';
 *
 * await initWasm();
 *
 * // Monthly data with yearly seasonality
 * const model = new SeasonalLinearRegression(12);
 * await model.fit(monthlyData); // At least 24 observations
 *
 * const forecast = await model.predict(12);
 * ```
 */
export class SeasonalLinearRegression implements Predictor {
  private fitted = false;
  private readonly period: number;
  private slope = 0;
  private intercept = 0;
  private seasonalFactors: number[] = [];
  private nObservations = 0;

  /**
   * Create a new seasonal linear regression model
   *
   * @param period - Number of observations per seasonal cycle
   */
  constructor(period: number) {
    if (period < 2) {
      throw new Error('Period must be at least 2');
    }
    this.period = period;
  }

  async fit(data: TimeSeriesData): Promise<void> {
    const minRequired = this.period * 2;
    if (data.length < minRequired) {
      throw new Error(`Need at least ${minRequired} observations`);
    }

    this.nObservations = data.length;

    // Step 1: Fit linear trend
    const n = data.length;
    const sumT = (n * (n - 1)) / 2;
    const sumY = data.reduce((a, b) => a + b, 0);
    const sumT2 = (n * (n - 1) * (2 * n - 1)) / 6;
    const sumTY = data.reduce((acc, y, t) => acc + t * y, 0);

    const denominator = n * sumT2 - sumT * sumT;
    this.slope = (n * sumTY - sumT * sumY) / denominator;
    this.intercept = (sumY - this.slope * sumT) / n;

    // Step 2: Detrend
    const detrended = data.map((y, t) => y - (this.intercept + this.slope * t));

    // Step 3: Compute seasonal factors
    this.seasonalFactors = new Array(this.period).fill(0);
    const counts = new Array(this.period).fill(0);

    detrended.forEach((val, i) => {
      const seasonIdx = i % this.period;
      this.seasonalFactors[seasonIdx] += val;
      counts[seasonIdx]++;
    });

    for (let i = 0; i < this.period; i++) {
      if (counts[i] > 0) {
        this.seasonalFactors[i] /= counts[i];
      }
    }

    // Normalize to sum to 0
    const meanFactor =
      this.seasonalFactors.reduce((a, b) => a + b, 0) / this.period;
    this.seasonalFactors = this.seasonalFactors.map((f) => f - meanFactor);

    this.fitted = true;
  }

  async predict(steps: number): Promise<number[]> {
    if (!this.fitted) {
      throw new Error('Model must be fitted before prediction');
    }

    const forecasts: number[] = [];
    for (let i = 0; i < steps; i++) {
      const t = this.nObservations + i;
      const seasonIdx = t % this.period;
      const trend = this.intercept + this.slope * t;
      forecasts.push(trend + this.seasonalFactors[seasonIdx]);
    }

    return forecasts;
  }

  isFitted(): boolean {
    return this.fitted;
  }

  /**
   * Get seasonal factors
   */
  getSeasonalFactors(): number[] {
    if (!this.fitted) {
      throw new Error('Model must be fitted first');
    }
    return [...this.seasonalFactors];
  }

  /**
   * Get the slope
   */
  getSlope(): number {
    return this.slope;
  }

  /**
   * Get the intercept
   */
  getIntercept(): number {
    return this.intercept;
  }
}
