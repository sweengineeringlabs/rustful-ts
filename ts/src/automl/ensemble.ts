/**
 * Ensemble forecasting methods
 */

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
 * Ensemble forecaster combining multiple models
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
   * Generate ensemble predictions
   */
  async predict(steps: number): Promise<number[]> {
    if (!this.fitted) {
      throw new Error('Ensemble must be fitted before prediction');
    }

    // Get predictions from all models
    const allPredictions = await Promise.all(
      this.models.map((model) => model.predict(steps))
    );

    // Combine predictions
    return this.combine(allPredictions);
  }

  /**
   * Combine predictions using the specified method
   */
  private combine(predictions: number[][]): number[] {
    const nSteps = predictions[0].length;
    const result: number[] = [];

    for (let i = 0; i < nSteps; i++) {
      const values = predictions.map((p) => p[i]);

      switch (this.method) {
        case EnsembleMethod.Average:
          result.push(values.reduce((a, b) => a + b, 0) / values.length);
          break;

        case EnsembleMethod.WeightedAverage:
          result.push(
            values.reduce((sum, v, j) => sum + v * this.weights[j], 0)
          );
          break;

        case EnsembleMethod.Median: {
          const sorted = [...values].sort((a, b) => a - b);
          const mid = Math.floor(sorted.length / 2);
          result.push(
            sorted.length % 2 !== 0
              ? sorted[mid]
              : (sorted[mid - 1] + sorted[mid]) / 2
          );
          break;
        }
      }
    }

    return result;
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
