/**
 * ARIMA (AutoRegressive Integrated Moving Average) model
 *
 * @module algorithms/arima
 */

import { getWasmModule, ensureWasm } from '../wasm-loader';
import type { Predictor, TimeSeriesData, ArimaParams } from '../types';

/**
 * ARIMA model for time series forecasting
 *
 * ARIMA combines three components:
 * - **AR (AutoRegressive)**: Uses past values to predict future values
 * - **I (Integrated)**: Differencing to achieve stationarity
 * - **MA (Moving Average)**: Uses past forecast errors
 *
 * @example
 * ```typescript
 * import { initWasm, Arima } from 'rustful-ts';
 *
 * await initWasm();
 *
 * // Create ARIMA(1, 1, 1) model
 * const model = new Arima(1, 1, 1);
 *
 * // Fit to historical data
 * const data = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28];
 * await model.fit(data);
 *
 * // Forecast next 5 values
 * const forecast = await model.predict(5);
 * console.log(forecast); // [30, 32, 34, 36, 38]
 * ```
 */
export class Arima implements Predictor {
  private inner: unknown = null;
  private fitted = false;
  private readonly params: ArimaParams;

  /**
   * Create a new ARIMA model
   *
   * @param p - AR order (autoregressive). Number of lag observations. Range: 0-10.
   * @param d - Differencing order. Number of times to difference the data. Range: 0-2.
   * @param q - MA order (moving average). Size of moving average window. Range: 0-10.
   *
   * @throws Error if parameters are out of valid range
   */
  constructor(p: number, d: number, q: number) {
    if (p < 0 || p > 10) {
      throw new Error('AR order (p) must be between 0 and 10');
    }
    if (d < 0 || d > 2) {
      throw new Error('Differencing order (d) must be between 0 and 2');
    }
    if (q < 0 || q > 10) {
      throw new Error('MA order (q) must be between 0 and 10');
    }

    this.params = { p, d, q };
  }

  /**
   * Fit the ARIMA model to historical data
   *
   * @param data - Time series observations (minimum length: p + d + q + 10)
   * @throws Error if data is too short or contains invalid values
   */
  async fit(data: TimeSeriesData): Promise<void> {
    await ensureWasm();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = getWasmModule() as any;

    this.inner = new wasm.WasmArima(this.params.p, this.params.d, this.params.q);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (this.inner as any).fit(new Float64Array(data));
    this.fitted = true;
  }

  /**
   * Generate predictions for future time steps
   *
   * @param steps - Number of steps to forecast ahead
   * @returns Array of predicted values
   * @throws Error if model hasn't been fitted
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
   * Get model parameters
   */
  getParams(): ArimaParams {
    return { ...this.params };
  }
}
