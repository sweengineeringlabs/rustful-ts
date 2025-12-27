/**
 * Ensemble forecasting methods - WASM-backed implementations
 */

import { getWasmModule, ensureWasm } from '../wasm-loader';
import type { Predictor, TimeSeriesData } from '../types';

/**
 * Ensemble combination methods
 */
export enum EnsembleMethod {
  Average = 'AVERAGE',
  WeightedAverage = 'WEIGHTED_AVERAGE',
  Median = 'MEDIAN',
}

/**
 * Combine predictions using WASM (internal helper)
 */
async function combineWithWasm(
  predictions: number[][],
  method: EnsembleMethod,
  weights?: number[]
): Promise<number[]> {
  await ensureWasm();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasm = getWasmModule() as any;

  switch (method) {
    case EnsembleMethod.Average:
      return Array.from(wasm.ensemble_average(predictions) as Float64Array);

    case EnsembleMethod.WeightedAverage:
      if (!weights) {
        weights = predictions.map(() => 1 / predictions.length);
      }
      return Array.from(
        wasm.ensemble_weighted_average(predictions, new Float64Array(weights)) as Float64Array
      );

    case EnsembleMethod.Median:
      return Array.from(wasm.ensemble_median(predictions) as Float64Array);

    default:
      return Array.from(wasm.ensemble_average(predictions) as Float64Array);
  }
}

/**
 * Ensemble forecaster combining multiple models (WASM-backed)
 *
 * Uses WASM for both model predictions (if models are WASM-backed)
 * and for the combination step.
 *
 * @example
 * ```typescript
 * import { initWasm, Arima, SimpleExponentialSmoothing, EnsembleForecaster, EnsembleMethod } from 'rustful-ts';
 *
 * await initWasm();
 *
 * const ensemble = new EnsembleForecaster(
 *   [new Arima(1, 1, 1), new SimpleExponentialSmoothing(0.3)],
 *   EnsembleMethod.Average
 * );
 *
 * await ensemble.fit(data);
 * const forecast = await ensemble.predict(10);
 * ```
 */
export class EnsembleForecaster implements Predictor {
  private models: Predictor[];
  private weights: number[];
  private method: EnsembleMethod;
  private fitted = false;

  /**
   * Create an ensemble forecaster
   * @param models - Array of predictor models
   * @param method - Combination method
   * @param weights - Optional weights for weighted average (must sum to 1)
   */
  constructor(
    models: Predictor[],
    method: EnsembleMethod = EnsembleMethod.Average,
    weights?: number[]
  ) {
    if (models.length === 0) {
      throw new Error('At least one model is required');
    }

    this.models = models;
    this.method = method;

    if (weights) {
      if (weights.length !== models.length) {
        throw new Error('Weights must have same length as models');
      }
      this.weights = weights;
    } else {
      // Equal weights by default
      this.weights = models.map(() => 1 / models.length);
    }
  }

  /**
   * Fit all models to data
   */
  async fit(data: TimeSeriesData): Promise<void> {
    await Promise.all(this.models.map((model) => model.fit(data)));
    this.fitted = true;
  }

  /**
   * Generate ensemble predictions (WASM-backed combination)
   */
  async predict(steps: number): Promise<number[]> {
    if (!this.fitted) {
      throw new Error('Ensemble must be fitted before prediction');
    }

    // Get predictions from all models (each model uses WASM internally)
    const allPredictions = await Promise.all(
      this.models.map((model) => model.predict(steps))
    );

    // Combine using WASM
    return combineWithWasm(allPredictions, this.method, this.weights);
  }

  isFitted(): boolean {
    return this.fitted;
  }

  /**
   * Get individual model predictions (for analysis)
   */
  async getIndividualPredictions(steps: number): Promise<number[][]> {
    if (!this.fitted) {
      throw new Error('Ensemble must be fitted before prediction');
    }
    return Promise.all(this.models.map((model) => model.predict(steps)));
  }
}

/**
 * Standalone function to combine predictions (WASM-backed)
 *
 * @example
 * ```typescript
 * const predictions = [
 *   [10, 11, 12],
 *   [12, 13, 14],
 *   [11, 12, 13],
 * ];
 * const combined = await combinePredictions(predictions, EnsembleMethod.Average);
 * // combined = [11, 12, 13]
 * ```
 */
export async function combinePredictions(
  predictions: number[][],
  method: EnsembleMethod,
  weights?: number[]
): Promise<number[]> {
  if (predictions.length === 0) {
    return [];
  }
  return combineWithWasm(predictions, method, weights);
}
