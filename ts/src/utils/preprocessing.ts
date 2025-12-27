/**
 * Data preprocessing utilities for time series
 *
 * @module utils/preprocessing
 */

/**
 * Normalize data to [0, 1] range (min-max scaling)
 *
 * @param data - Input data
 * @returns Object with normalized data, min, and max values
 *
 * @example
 * ```typescript
 * import { normalize, denormalize } from 'rustful-ts';
 *
 * const data = [10, 20, 30, 40, 50];
 * const { normalized, min, max } = normalize(data);
 * // normalized: [0, 0.25, 0.5, 0.75, 1]
 *
 * // Later, convert back
 * const original = denormalize(normalized, min, max);
 * ```
 */
export function normalize(data: number[]): {
  normalized: number[];
  min: number;
  max: number;
} {
  if (data.length === 0) {
    return { normalized: [], min: 0, max: 1 };
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min;

  if (Math.abs(range) < 1e-10) {
    return { normalized: data.map(() => 0.5), min, max };
  }

  const normalized = data.map((x) => (x - min) / range);
  return { normalized, min, max };
}

/**
 * Denormalize data from [0, 1] range
 */
export function denormalize(data: number[], min: number, max: number): number[] {
  const range = max - min;
  return data.map((x) => x * range + min);
}

/**
 * Standardize data to zero mean and unit variance (z-score)
 *
 * @param data - Input data
 * @returns Object with standardized data, mean, and standard deviation
 */
export function standardize(data: number[]): {
  standardized: number[];
  mean: number;
  stdDev: number;
} {
  if (data.length === 0) {
    return { standardized: [], mean: 0, stdDev: 1 };
  }

  const n = data.length;
  const mean = data.reduce((a, b) => a + b, 0) / n;
  const variance = data.reduce((acc, x) => acc + (x - mean) ** 2, 0) / n;
  const stdDev = Math.sqrt(variance);

  if (stdDev < 1e-10) {
    return { standardized: data.map(() => 0), mean, stdDev: 1 };
  }

  const standardized = data.map((x) => (x - mean) / stdDev);
  return { standardized, mean, stdDev };
}

/**
 * Destandardize data
 */
export function destandardize(
  data: number[],
  mean: number,
  stdDev: number
): number[] {
  return data.map((x) => x * stdDev + mean);
}

/**
 * Apply log transformation (log1p for handling zeros)
 */
export function logTransform(data: number[]): number[] {
  return data.map((x) => Math.log1p(x));
}

/**
 * Inverse log transformation
 */
export function inverseLogTransform(data: number[]): number[] {
  return data.map((x) => Math.expm1(x));
}

/**
 * Compute first-order differences
 *
 * @param data - Input data
 * @param order - Number of times to difference (default: 1)
 * @returns Differenced series (length reduced by `order`)
 *
 * @example
 * ```typescript
 * import { difference } from 'rustful-ts';
 *
 * const data = [1, 3, 6, 10];
 * console.log(difference(data, 1)); // [2, 3, 4]
 * console.log(difference(data, 2)); // [1, 1]
 * ```
 */
export function difference(data: number[], order: number = 1): number[] {
  let result = [...data];
  for (let i = 0; i < order; i++) {
    if (result.length <= 1) {
      return [];
    }
    const differenced: number[] = [];
    for (let j = 1; j < result.length; j++) {
      differenced.push(result[j] - result[j - 1]);
    }
    result = differenced;
  }
  return result;
}

/**
 * Compute seasonal differences
 *
 * @param data - Input data
 * @param period - Seasonal period
 * @returns Seasonally differenced series
 */
export function seasonalDifference(data: number[], period: number): number[] {
  if (data.length <= period) {
    return [];
  }

  const result: number[] = [];
  for (let i = period; i < data.length; i++) {
    result.push(data[i] - data[i - period]);
  }
  return result;
}

/**
 * Remove NaN and infinite values
 */
export function cleanData(data: number[]): number[] {
  return data.filter((x) => Number.isFinite(x));
}

/**
 * Linear interpolation for missing values (NaN)
 */
export function interpolateLinear(data: number[]): number[] {
  const result = [...data];
  const n = result.length;

  for (let i = 0; i < n; i++) {
    if (Number.isNaN(result[i])) {
      // Find previous valid value
      let prevIdx = -1;
      for (let j = i - 1; j >= 0; j--) {
        if (!Number.isNaN(result[j])) {
          prevIdx = j;
          break;
        }
      }

      // Find next valid value
      let nextIdx = -1;
      for (let j = i + 1; j < n; j++) {
        if (!Number.isNaN(result[j])) {
          nextIdx = j;
          break;
        }
      }

      if (prevIdx >= 0 && nextIdx >= 0) {
        const ratio = (i - prevIdx) / (nextIdx - prevIdx);
        result[i] = result[prevIdx] + ratio * (result[nextIdx] - result[prevIdx]);
      } else if (prevIdx >= 0) {
        result[i] = result[prevIdx];
      } else if (nextIdx >= 0) {
        result[i] = result[nextIdx];
      } else {
        result[i] = 0;
      }
    }
  }

  return result;
}

/**
 * Detect outliers using IQR method
 *
 * @param data - Input data
 * @param multiplier - IQR multiplier (default: 1.5)
 * @returns Indices of outlier points
 */
export function detectOutliersIQR(
  data: number[],
  multiplier: number = 1.5
): number[] {
  const sorted = [...data].filter(Number.isFinite).sort((a, b) => a - b);

  if (sorted.length < 4) {
    return [];
  }

  const q1 = sorted[Math.floor(sorted.length / 4)];
  const q3 = sorted[Math.floor((3 * sorted.length) / 4)];
  const iqr = q3 - q1;

  const lowerBound = q1 - multiplier * iqr;
  const upperBound = q3 + multiplier * iqr;

  return data
    .map((x, i) => (x < lowerBound || x > upperBound ? i : -1))
    .filter((i) => i >= 0);
}

/**
 * Create lagged features for machine learning
 *
 * @param data - Input time series
 * @param lags - Number of lag features to create
 * @returns Array of feature vectors (each row is [lag1, lag2, ..., lagN, target])
 */
export function createLagFeatures(
  data: number[],
  lags: number
): { features: number[][]; target: number[] } {
  if (data.length <= lags) {
    return { features: [], target: [] };
  }

  const features: number[][] = [];
  const target: number[] = [];

  for (let i = lags; i < data.length; i++) {
    const row: number[] = [];
    for (let j = lags; j >= 1; j--) {
      row.push(data[i - j]);
    }
    features.push(row);
    target.push(data[i]);
  }

  return { features, target };
}

/**
 * Split data into train and test sets (respecting time order)
 *
 * @param data - Input data
 * @param testRatio - Fraction of data for testing (0-1)
 * @returns Object with train and test arrays
 */
export function trainTestSplit(
  data: number[],
  testRatio: number = 0.2
): { train: number[]; test: number[] } {
  const ratio = Math.max(0.1, Math.min(0.9, testRatio));
  const splitIdx = Math.floor((1 - ratio) * data.length);
  const boundedSplitIdx = Math.max(1, Math.min(data.length - 1, splitIdx));

  return {
    train: data.slice(0, boundedSplitIdx),
    test: data.slice(boundedSplitIdx),
  };
}
