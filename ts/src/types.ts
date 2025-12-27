/**
 * Common types for rustful-ts
 */

/**
 * Time series data as an array of numbers
 */
export type TimeSeriesData = number[];

/**
 * Forecast result with predictions and optional confidence intervals
 */
export interface ForecastResult {
  /** Point predictions */
  predictions: number[];
  /** Lower confidence bound (if available) */
  lowerBound?: number[];
  /** Upper confidence bound (if available) */
  upperBound?: number[];
  /** Confidence level (e.g., 0.95 for 95%) */
  confidenceLevel?: number;
}

/**
 * Model fit statistics
 */
export interface FitStatistics {
  /** Mean Absolute Error on training data */
  mae?: number;
  /** Root Mean Squared Error on training data */
  rmse?: number;
  /** Mean Absolute Percentage Error */
  mape?: number;
  /** R-squared (coefficient of determination) */
  rSquared?: number;
  /** Akaike Information Criterion */
  aic?: number;
  /** Bayesian Information Criterion */
  bic?: number;
}

/**
 * Common interface for all time series predictors
 */
export interface Predictor {
  /**
   * Fit the model to historical data
   * @param data - Time series observations
   */
  fit(data: TimeSeriesData): Promise<void>;

  /**
   * Generate predictions for future time steps
   * @param steps - Number of steps to forecast
   * @returns Array of predicted values
   */
  predict(steps: number): Promise<number[]>;

  /**
   * Check if the model has been fitted
   */
  isFitted(): boolean;
}

/**
 * ARIMA model parameters
 */
export interface ArimaParams {
  /** AR order (p) */
  p: number;
  /** Differencing order (d) */
  d: number;
  /** MA order (q) */
  q: number;
}

/**
 * Exponential smoothing parameters
 */
export interface ExpSmoothingParams {
  /** Level smoothing (alpha) */
  alpha: number;
  /** Trend smoothing (beta) - for Holt and Holt-Winters */
  beta?: number;
  /** Seasonal smoothing (gamma) - for Holt-Winters */
  gamma?: number;
  /** Seasonal period - for Holt-Winters */
  period?: number;
}

/**
 * Cross-validation configuration
 */
export interface CrossValidationConfig {
  /** Minimum training set size */
  minTrainSize: number;
  /** Forecast horizon */
  horizon: number;
  /** Step size between folds */
  step: number;
  /** Validation strategy */
  strategy: 'expanding' | 'sliding';
}

/**
 * Cross-validation results
 */
export interface CrossValidationResults {
  /** MAE scores for each fold */
  maeScores: number[];
  /** RMSE scores for each fold */
  rmseScores: number[];
  /** Mean MAE across folds */
  meanMae: number;
  /** Mean RMSE across folds */
  meanRmse: number;
  /** Standard deviation of MAE */
  stdMae: number;
  /** Number of folds */
  nFolds: number;
}
