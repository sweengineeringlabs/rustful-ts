/**
 * Forecast accuracy metrics
 *
 * @module utils/metrics
 */

import { getWasmModule, ensureWasm } from '../wasm-loader';

/**
 * Mean Absolute Error (MAE)
 *
 * Average of absolute differences between predictions and actual values.
 * Lower is better. Same scale as the data.
 *
 * @param actual - Actual observed values
 * @param predicted - Predicted values
 * @returns MAE value
 *
 * @example
 * ```typescript
 * import { mae } from 'rustful-ts';
 *
 * const actual = [10, 20, 30, 40, 50];
 * const predicted = [12, 18, 33, 42, 48];
 * console.log(mae(actual, predicted)); // 2.8
 * ```
 */
export function mae(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length || actual.length === 0) {
    return NaN;
  }

  let sum = 0;
  for (let i = 0; i < actual.length; i++) {
    sum += Math.abs(actual[i] - predicted[i]);
  }

  return sum / actual.length;
}

/**
 * Mean Squared Error (MSE)
 *
 * Average of squared differences. Penalizes large errors more heavily.
 * Lower is better.
 */
export function mse(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length || actual.length === 0) {
    return NaN;
  }

  let sum = 0;
  for (let i = 0; i < actual.length; i++) {
    const diff = actual[i] - predicted[i];
    sum += diff * diff;
  }

  return sum / actual.length;
}

/**
 * Root Mean Squared Error (RMSE)
 *
 * Square root of MSE. Same scale as the data.
 * Lower is better.
 */
export function rmse(actual: number[], predicted: number[]): number {
  return Math.sqrt(mse(actual, predicted));
}

/**
 * Mean Absolute Percentage Error (MAPE)
 *
 * Average of absolute percentage errors. Scale-independent.
 * Returns value as a decimal (0.1 = 10%).
 * Undefined when actual values are zero.
 */
export function mape(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length || actual.length === 0) {
    return NaN;
  }

  let sum = 0;
  let count = 0;

  for (let i = 0; i < actual.length; i++) {
    if (Math.abs(actual[i]) > 1e-10) {
      sum += Math.abs((actual[i] - predicted[i]) / actual[i]);
      count++;
    }
  }

  return count > 0 ? sum / count : NaN;
}

/**
 * Symmetric Mean Absolute Percentage Error (sMAPE)
 *
 * Symmetric version of MAPE that handles zero values better.
 * Returns value between 0 and 2 (as a decimal).
 */
export function smape(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length || actual.length === 0) {
    return NaN;
  }

  let sum = 0;
  for (let i = 0; i < actual.length; i++) {
    const denom = Math.abs(actual[i]) + Math.abs(predicted[i]);
    if (denom > 1e-10) {
      sum += (2 * Math.abs(actual[i] - predicted[i])) / denom;
    }
  }

  return sum / actual.length;
}

/**
 * R-squared (Coefficient of Determination)
 *
 * Measures how well predictions explain variance in actual values.
 * - 1.0 = perfect predictions
 * - 0.0 = predictions are as good as mean
 * - negative = predictions are worse than mean
 */
export function rSquared(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length || actual.length === 0) {
    return NaN;
  }

  const mean = actual.reduce((a, b) => a + b, 0) / actual.length;

  let ssTot = 0;
  let ssRes = 0;

  for (let i = 0; i < actual.length; i++) {
    ssTot += (actual[i] - mean) ** 2;
    ssRes += (actual[i] - predicted[i]) ** 2;
  }

  if (ssTot < 1e-10) {
    return 1;
  }

  return 1 - ssRes / ssTot;
}

/**
 * Summary of all common metrics
 */
export interface MetricsSummary {
  mae: number;
  mse: number;
  rmse: number;
  mape: number;
  smape: number;
  rSquared: number;
}

/**
 * Compute all metrics at once
 *
 * @example
 * ```typescript
 * import { computeMetrics } from 'rustful-ts';
 *
 * const actual = [10, 20, 30, 40, 50];
 * const predicted = [12, 18, 33, 42, 48];
 *
 * const metrics = computeMetrics(actual, predicted);
 * console.log(metrics.mae);  // 2.8
 * console.log(metrics.rmse); // 3.03...
 * ```
 */
export function computeMetrics(
  actual: number[],
  predicted: number[]
): MetricsSummary {
  return {
    mae: mae(actual, predicted),
    mse: mse(actual, predicted),
    rmse: rmse(actual, predicted),
    mape: mape(actual, predicted),
    smape: smape(actual, predicted),
    rSquared: rSquared(actual, predicted),
  };
}

/**
 * WASM-accelerated MAE (for large datasets)
 */
export async function maeWasm(
  actual: number[],
  predicted: number[]
): Promise<number> {
  await ensureWasm();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasm = getWasmModule() as any;
  return wasm.compute_mae(new Float64Array(actual), new Float64Array(predicted));
}

/**
 * WASM-accelerated RMSE (for large datasets)
 */
export async function rmseWasm(
  actual: number[],
  predicted: number[]
): Promise<number> {
  await ensureWasm();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasm = getWasmModule() as any;
  return wasm.compute_rmse(new Float64Array(actual), new Float64Array(predicted));
}
