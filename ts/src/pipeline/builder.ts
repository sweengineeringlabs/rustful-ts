/**
 * Pipeline builder for composable time series forecasting
 */

import type { Predictor, TimeSeriesData } from '../types';
import {
  PipelineStep,
  NormalizeStep,
  StandardizeStep,
  DifferenceStep,
  LogTransformStep,
  ClipOutliersStep,
} from './steps';

/**
 * Fluent API for building forecasting pipelines
 *
 * @example
 * ```typescript
 * const forecast = await Pipeline.create()
 *   .normalize()
 *   .difference(1)
 *   .withArima(1, 0, 1)
 *   .fitPredict(data, 10);
 * ```
 */
export class Pipeline implements Predictor {
  private steps: PipelineStep[] = [];
  private model: Predictor | null = null;
  private fitted = false;
  private originalData: TimeSeriesData = [];

  private constructor() {}

  /**
   * Create a new pipeline builder
   */
  static create(): Pipeline {
    return new Pipeline();
  }

  // ============================================
  // Preprocessing Steps
  // ============================================

  /**
   * Add normalization step (scale to [0, 1])
   */
  normalize(): this {
    this.steps.push(new NormalizeStep());
    return this;
  }

  /**
   * Add standardization step (zero mean, unit variance)
   */
  standardize(): this {
    this.steps.push(new StandardizeStep());
    return this;
  }

  /**
   * Add differencing step
   * @param order - Differencing order (default: 1)
   */
  difference(order: number = 1): this {
    this.steps.push(new DifferenceStep(order));
    return this;
  }

  /**
   * Add log transformation
   */
  logTransform(): this {
    this.steps.push(new LogTransformStep());
    return this;
  }

  /**
   * Clip outliers using IQR method
   * @param multiplier - IQR multiplier (default: 1.5)
   */
  clipOutliers(multiplier: number = 1.5): this {
    this.steps.push(new ClipOutliersStep(multiplier));
    return this;
  }

  /**
   * Add a custom pipeline step
   */
  addStep(step: PipelineStep): this {
    this.steps.push(step);
    return this;
  }

  // ============================================
  // Model Selection
  // ============================================

  /**
   * Use ARIMA model
   */
  withArima(p: number, d: number, q: number): this {
    // Lazy import to avoid circular dependencies
    const { Arima } = require('../algorithms/arima');
    this.model = new Arima(p, d, q);
    return this;
  }

  /**
   * Use Simple Exponential Smoothing
   */
  withSES(alpha: number): this {
    const { SimpleExponentialSmoothing } = require('../algorithms/exponential-smoothing');
    this.model = new SimpleExponentialSmoothing(alpha);
    return this;
  }

  /**
   * Use Holt's method (double exponential smoothing)
   */
  withHolt(alpha: number, beta: number): this {
    const { Holt } = require('../algorithms/exponential-smoothing');
    this.model = new Holt(alpha, beta);
    return this;
  }

  /**
   * Use Holt-Winters (triple exponential smoothing)
   */
  withHoltWinters(
    alpha: number,
    beta: number,
    gamma: number,
    period: number,
    seasonal: 'additive' | 'multiplicative' = 'additive'
  ): this {
    const { HoltWinters, SeasonalType } = require('../algorithms/exponential-smoothing');
    const type = seasonal === 'additive' ? SeasonalType.Additive : SeasonalType.Multiplicative;
    this.model = new HoltWinters(alpha, beta, gamma, period, type);
    return this;
  }

  /**
   * Use Simple Moving Average
   */
  withSMA(window: number): this {
    const { SimpleMovingAverage } = require('../algorithms/moving-average');
    this.model = new SimpleMovingAverage(window);
    return this;
  }

  /**
   * Use Linear Regression
   */
  withLinearRegression(): this {
    const { LinearRegression } = require('../algorithms/linear-regression');
    this.model = new LinearRegression();
    return this;
  }

  /**
   * Use K-Nearest Neighbors
   */
  withKNN(k: number, windowSize: number): this {
    const { TimeSeriesKNN } = require('../algorithms/knn');
    this.model = new TimeSeriesKNN(k, windowSize);
    return this;
  }

  /**
   * Use a custom model
   */
  withModel(model: Predictor): this {
    this.model = model;
    return this;
  }

  // ============================================
  // Execution
  // ============================================

  /**
   * Fit the pipeline to data
   */
  async fit(data: TimeSeriesData): Promise<void> {
    if (!this.model) {
      throw new Error('No model specified. Use withArima(), withSES(), etc. to add a model.');
    }

    this.originalData = [...data];
    let transformed = [...data];

    // Fit and transform through each step
    for (const step of this.steps) {
      await step.fit(transformed);
      transformed = await step.transform(transformed);
    }

    // Fit the model on transformed data
    await this.model.fit(transformed);
    this.fitted = true;
  }

  /**
   * Generate predictions
   */
  async predict(steps: number): Promise<number[]> {
    if (!this.fitted || !this.model) {
      throw new Error('Pipeline must be fitted before prediction');
    }

    // Get predictions from model
    let predictions = await this.model.predict(steps);

    // Apply inverse transforms in reverse order
    for (let i = this.steps.length - 1; i >= 0; i--) {
      predictions = await this.steps[i].inverseTransform(predictions);
    }

    return predictions;
  }

  /**
   * Fit the pipeline and generate predictions in one call
   */
  async fitPredict(data: TimeSeriesData, steps: number): Promise<number[]> {
    await this.fit(data);
    return this.predict(steps);
  }

  /**
   * Check if the pipeline is fitted
   */
  isFitted(): boolean {
    return this.fitted;
  }

  /**
   * Get the pipeline steps
   */
  getSteps(): readonly PipelineStep[] {
    return this.steps;
  }

  /**
   * Get information about the pipeline
   */
  describe(): string {
    const stepNames = this.steps.map((s) => s.name).join(' -> ');
    const modelName = this.model?.constructor.name || 'None';
    return `Pipeline: ${stepNames || 'No preprocessing'} -> ${modelName}`;
  }
}
