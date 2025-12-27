/**
 * Exponential Smoothing methods for time series forecasting
 *
 * @module algorithms/exponential-smoothing
 */

import { getWasmModule, ensureWasm } from '../wasm-loader';
import type { Predictor, TimeSeriesData } from '../types';

/**
 * Seasonal type for Holt-Winters
 */
export enum SeasonalType {
  /** Additive seasonality: Y = Level + Trend + Season + Error */
  Additive = 'additive',
  /** Multiplicative seasonality: Y = (Level + Trend) * Season * Error */
  Multiplicative = 'multiplicative',
}

/**
 * Simple Exponential Smoothing (SES)
 *
 * Best for: Data without clear trend or seasonal pattern.
 * Produces flat forecasts at the current level.
 *
 * @example
 * ```typescript
 * import { initWasm, SimpleExponentialSmoothing } from 'rustful-ts';
 *
 * await initWasm();
 *
 * // Create model with alpha = 0.3
 * const model = new SimpleExponentialSmoothing(0.3);
 *
 * await model.fit([10, 12, 11, 13, 12, 14, 13, 15]);
 * const forecast = await model.predict(3);
 * ```
 */
export class SimpleExponentialSmoothing implements Predictor {
  private inner: unknown = null;
  private fitted = false;
  private readonly alpha: number;

  /**
   * Create a new SES model
   *
   * @param alpha - Smoothing parameter (0 < alpha < 1).
   *                Higher values give more weight to recent observations.
   *                Typical range: 0.1 - 0.3
   */
  constructor(alpha: number) {
    if (alpha <= 0 || alpha >= 1) {
      throw new Error('Alpha must be between 0 and 1 (exclusive)');
    }
    this.alpha = alpha;
  }

  async fit(data: TimeSeriesData): Promise<void> {
    await ensureWasm();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = getWasmModule() as any;

    this.inner = new wasm.WasmSES(this.alpha);
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
   * Get the current level estimate
   */
  getLevel(): number {
    if (!this.fitted || !this.inner) {
      throw new Error('Model must be fitted first');
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (this.inner as any).level();
  }
}

/**
 * Double Exponential Smoothing (Holt's Linear Trend Method)
 *
 * Extends SES to capture linear trends in the data.
 * Best for: Data with trend but no seasonality.
 *
 * @example
 * ```typescript
 * import { initWasm, Holt } from 'rustful-ts';
 *
 * await initWasm();
 *
 * const model = new Holt(0.3, 0.1);
 * await model.fit([10, 12, 14, 16, 18, 20, 22, 24]);
 * const forecast = await model.predict(3);
 * // Returns increasing values following the trend
 * ```
 */
export class Holt implements Predictor {
  private inner: unknown = null;
  private fitted = false;
  private readonly alpha: number;
  private readonly beta: number;

  /**
   * Create a new Holt's method model
   *
   * @param alpha - Level smoothing (0 < alpha < 1)
   * @param beta - Trend smoothing (0 < beta < 1). Typical: 0.1 - 0.2
   */
  constructor(alpha: number, beta: number) {
    if (alpha <= 0 || alpha >= 1) {
      throw new Error('Alpha must be between 0 and 1 (exclusive)');
    }
    if (beta <= 0 || beta >= 1) {
      throw new Error('Beta must be between 0 and 1 (exclusive)');
    }
    this.alpha = alpha;
    this.beta = beta;
  }

  async fit(data: TimeSeriesData): Promise<void> {
    await ensureWasm();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = getWasmModule() as any;

    this.inner = new wasm.WasmHolt(this.alpha, this.beta);
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
}

/**
 * Triple Exponential Smoothing (Holt-Winters Method)
 *
 * Extends double exponential smoothing to capture seasonality.
 * Best for: Data with both trend and seasonal patterns.
 *
 * @example
 * ```typescript
 * import { initWasm, HoltWinters, SeasonalType } from 'rustful-ts';
 *
 * await initWasm();
 *
 * // Monthly data with yearly seasonality
 * const model = new HoltWinters(0.3, 0.1, 0.2, 12, SeasonalType.Additive);
 *
 * // Fit with at least 2 complete seasonal cycles
 * await model.fit(monthlyData); // At least 24 observations
 *
 * // Forecast next year
 * const forecast = await model.predict(12);
 * ```
 */
export class HoltWinters implements Predictor {
  private inner: unknown = null;
  private fitted = false;
  private readonly alpha: number;
  private readonly beta: number;
  private readonly gamma: number;
  private readonly period: number;
  private readonly seasonalType: SeasonalType;

  /**
   * Create a new Holt-Winters model
   *
   * @param alpha - Level smoothing (0 < alpha < 1)
   * @param beta - Trend smoothing (0 < beta < 1)
   * @param gamma - Seasonal smoothing (0 < gamma < 1)
   * @param period - Number of observations per seasonal cycle (e.g., 12 for monthly with yearly seasonality)
   * @param seasonalType - Additive or Multiplicative seasonality
   */
  constructor(
    alpha: number,
    beta: number,
    gamma: number,
    period: number,
    seasonalType: SeasonalType = SeasonalType.Additive
  ) {
    if (alpha <= 0 || alpha >= 1) {
      throw new Error('Alpha must be between 0 and 1 (exclusive)');
    }
    if (beta <= 0 || beta >= 1) {
      throw new Error('Beta must be between 0 and 1 (exclusive)');
    }
    if (gamma <= 0 || gamma >= 1) {
      throw new Error('Gamma must be between 0 and 1 (exclusive)');
    }
    if (period < 2) {
      throw new Error('Period must be at least 2');
    }

    this.alpha = alpha;
    this.beta = beta;
    this.gamma = gamma;
    this.period = period;
    this.seasonalType = seasonalType;
  }

  async fit(data: TimeSeriesData): Promise<void> {
    await ensureWasm();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = getWasmModule() as any;

    if (this.seasonalType === SeasonalType.Additive) {
      this.inner = wasm.WasmHoltWinters.new_additive(
        this.alpha,
        this.beta,
        this.gamma,
        this.period
      );
    } else {
      this.inner = wasm.WasmHoltWinters.new_multiplicative(
        this.alpha,
        this.beta,
        this.gamma,
        this.period
      );
    }

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
   * Get the seasonal components
   */
  getSeasonalComponents(): number[] {
    if (!this.fitted || !this.inner) {
      throw new Error('Model must be fitted first');
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return Array.from((this.inner as any).seasonal_components() as Float64Array);
  }
}
