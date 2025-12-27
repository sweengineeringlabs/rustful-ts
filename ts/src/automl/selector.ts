/**
 * Automatic model selection
 */

import type { Predictor, TimeSeriesData } from '../types';
import { mae, rmse } from '../utils/metrics';
import { trainTestSplit } from '../utils/preprocessing';

/**
 * Model types for AutoML
 */
export enum ModelType {
  Arima = 'ARIMA',
  SES = 'SES',
  Holt = 'HOLT',
  HoltWinters = 'HOLT_WINTERS',
  LinearRegression = 'LINEAR_REGRESSION',
  SMA = 'SMA',
  KNN = 'KNN',
}

/**
 * AutoML configuration
 */
export interface AutoMLConfig {
  /** Models to consider */
  models?: ModelType[];
  /** Metric to optimize ('mae' or 'rmse') */
  metric?: 'mae' | 'rmse';
  /** Test set ratio for validation */
  testRatio?: number;
  /** Forecast horizon for validation */
  horizon?: number;
}

/**
 * Result of model selection
 */
export interface ModelSelectionResult {
  /** Best model type */
  bestModel: ModelType;
  /** Best model instance */
  model: Predictor;
  /** Best score achieved */
  score: number;
  /** All model scores */
  allScores: Array<{ model: ModelType; score: number; params?: Record<string, unknown> }>;
}

/**
 * Automatic model selector
 */
export class AutoSelector implements Predictor {
  private config: Required<AutoMLConfig>;
  private bestModel: Predictor | null = null;
  private fitted = false;

  constructor(config: AutoMLConfig = {}) {
    this.config = {
      models: config.models || [
        ModelType.Arima,
        ModelType.SES,
        ModelType.Holt,
        ModelType.LinearRegression,
        ModelType.SMA,
      ],
      metric: config.metric || 'mae',
      testRatio: config.testRatio || 0.2,
      horizon: config.horizon || 1,
    };
  }

  /**
   * Create a model instance based on type
   */
  private async createModel(
    type: ModelType,
    params?: Record<string, unknown>
  ): Promise<Predictor> {
    switch (type) {
      case ModelType.Arima: {
        const { Arima } = await import('../algorithms/arima');
        const p = (params?.p as number) ?? 1;
        const d = (params?.d as number) ?? 1;
        const q = (params?.q as number) ?? 0;
        return new Arima(p, d, q);
      }
      case ModelType.SES: {
        const { SimpleExponentialSmoothing } = await import('../algorithms/exponential-smoothing');
        const alpha = (params?.alpha as number) ?? 0.3;
        return new SimpleExponentialSmoothing(alpha);
      }
      case ModelType.Holt: {
        const { Holt } = await import('../algorithms/exponential-smoothing');
        const alpha = (params?.alpha as number) ?? 0.3;
        const beta = (params?.beta as number) ?? 0.1;
        return new Holt(alpha, beta);
      }
      case ModelType.HoltWinters: {
        const { HoltWinters, SeasonalType } = await import('../algorithms/exponential-smoothing');
        const alpha = (params?.alpha as number) ?? 0.3;
        const beta = (params?.beta as number) ?? 0.1;
        const gamma = (params?.gamma as number) ?? 0.1;
        const period = (params?.period as number) ?? 12;
        return new HoltWinters(alpha, beta, gamma, period, SeasonalType.Additive);
      }
      case ModelType.LinearRegression: {
        const { LinearRegression } = await import('../algorithms/linear-regression');
        return new LinearRegression();
      }
      case ModelType.SMA: {
        const { SimpleMovingAverage } = await import('../algorithms/moving-average');
        const window = (params?.window as number) ?? 5;
        return new SimpleMovingAverage(window);
      }
      case ModelType.KNN: {
        const { TimeSeriesKNN } = await import('../algorithms/knn');
        const k = (params?.k as number) ?? 3;
        const windowSize = (params?.windowSize as number) ?? 5;
        return new TimeSeriesKNN(k, windowSize);
      }
      default:
        throw new Error(`Unknown model type: ${type}`);
    }
  }

  /**
   * Evaluate a model on validation data
   */
  private async evaluateModel(
    model: Predictor,
    trainData: TimeSeriesData,
    testData: TimeSeriesData
  ): Promise<number> {
    try {
      await model.fit(trainData);
      const predictions = await model.predict(testData.length);

      const metricFn = this.config.metric === 'rmse' ? rmse : mae;
      return metricFn(testData, predictions);
    } catch {
      return Infinity;
    }
  }

  /**
   * Perform model selection
   */
  async select(data: TimeSeriesData): Promise<ModelSelectionResult> {
    const { train, test } = trainTestSplit(data, this.config.testRatio);

    const results: ModelSelectionResult['allScores'] = [];
    let bestScore = Infinity;
    let bestModelType = this.config.models[0];
    let bestModelInstance: Predictor | null = null;

    for (const modelType of this.config.models) {
      const model = await this.createModel(modelType);
      const score = await this.evaluateModel(model, train, test);

      results.push({ model: modelType, score });

      if (score < bestScore) {
        bestScore = score;
        bestModelType = modelType;
        bestModelInstance = model;
      }
    }

    // Refit best model on full data
    if (bestModelInstance) {
      await bestModelInstance.fit(data);
    }

    return {
      bestModel: bestModelType,
      model: bestModelInstance!,
      score: bestScore,
      allScores: results.sort((a, b) => a.score - b.score),
    };
  }

  /**
   * Fit using automatic model selection
   */
  async fit(data: TimeSeriesData): Promise<void> {
    const result = await this.select(data);
    this.bestModel = result.model;
    this.fitted = true;
  }

  /**
   * Predict using the best model
   */
  async predict(steps: number): Promise<number[]> {
    if (!this.fitted || !this.bestModel) {
      throw new Error('AutoSelector must be fitted before prediction');
    }
    return this.bestModel.predict(steps);
  }

  isFitted(): boolean {
    return this.fitted;
  }
}
