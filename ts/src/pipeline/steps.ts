/**
 * Pipeline step implementations
 */

import type { TimeSeriesData } from '../types';

/**
 * Interface for pipeline transformation steps
 */
export interface PipelineStep {
  /** Name of this step */
  readonly name: string;

  /** Transform data forward */
  transform(data: TimeSeriesData): Promise<TimeSeriesData>;

  /** Inverse transform (undo the transformation) */
  inverseTransform(data: TimeSeriesData): Promise<TimeSeriesData>;

  /** Fit the step to data (learn parameters) */
  fit(data: TimeSeriesData): Promise<void>;
}

/**
 * Normalize data to [0, 1] range
 */
export class NormalizeStep implements PipelineStep {
  readonly name = 'normalize';
  private min = 0;
  private max = 1;
  private fitted = false;

  async fit(data: TimeSeriesData): Promise<void> {
    this.min = Math.min(...data);
    this.max = Math.max(...data);
    this.fitted = true;
  }

  async transform(data: TimeSeriesData): Promise<TimeSeriesData> {
    if (!this.fitted) {
      await this.fit(data);
    }
    const range = this.max - this.min;
    if (range === 0) {
      return data.map(() => 0.5);
    }
    return data.map((x) => (x - this.min) / range);
  }

  async inverseTransform(data: TimeSeriesData): Promise<TimeSeriesData> {
    const range = this.max - this.min;
    return data.map((x) => x * range + this.min);
  }
}

/**
 * Standardize data to zero mean and unit variance
 */
export class StandardizeStep implements PipelineStep {
  readonly name = 'standardize';
  private mean = 0;
  private std = 1;
  private fitted = false;

  async fit(data: TimeSeriesData): Promise<void> {
    const n = data.length;
    this.mean = data.reduce((a, b) => a + b, 0) / n;
    const variance = data.reduce((sum, x) => sum + (x - this.mean) ** 2, 0) / n;
    this.std = Math.sqrt(variance);
    this.fitted = true;
  }

  async transform(data: TimeSeriesData): Promise<TimeSeriesData> {
    if (!this.fitted) {
      await this.fit(data);
    }
    if (this.std === 0) {
      return data.map(() => 0);
    }
    return data.map((x) => (x - this.mean) / this.std);
  }

  async inverseTransform(data: TimeSeriesData): Promise<TimeSeriesData> {
    return data.map((x) => x * this.std + this.mean);
  }
}

/**
 * Compute differences of order d
 */
export class DifferenceStep implements PipelineStep {
  readonly name = 'difference';
  private order: number;
  private initialValues: number[] = [];

  constructor(order: number = 1) {
    this.order = order;
  }

  async fit(data: TimeSeriesData): Promise<void> {
    // Store initial values for inverse transform
    this.initialValues = data.slice(0, this.order);
  }

  async transform(data: TimeSeriesData): Promise<TimeSeriesData> {
    let result = [...data];
    for (let d = 0; d < this.order; d++) {
      const diff: number[] = [];
      for (let i = 1; i < result.length; i++) {
        diff.push(result[i] - result[i - 1]);
      }
      if (d === 0) {
        this.initialValues = [result[0]];
      } else {
        this.initialValues.push(result[0]);
      }
      result = diff;
    }
    return result;
  }

  async inverseTransform(data: TimeSeriesData): Promise<TimeSeriesData> {
    let result = [...data];
    for (let d = this.order - 1; d >= 0; d--) {
      const undiff: number[] = [this.initialValues[d] || 0];
      for (const val of result) {
        undiff.push(undiff[undiff.length - 1] + val);
      }
      result = undiff;
    }
    return result;
  }
}

/**
 * Apply log transformation (for positive data)
 */
export class LogTransformStep implements PipelineStep {
  readonly name = 'log';
  private shift = 0;

  async fit(data: TimeSeriesData): Promise<void> {
    const minVal = Math.min(...data);
    // Shift data if needed to ensure all values are positive
    this.shift = minVal <= 0 ? Math.abs(minVal) + 1 : 0;
  }

  async transform(data: TimeSeriesData): Promise<TimeSeriesData> {
    return data.map((x) => Math.log(x + this.shift));
  }

  async inverseTransform(data: TimeSeriesData): Promise<TimeSeriesData> {
    return data.map((x) => Math.exp(x) - this.shift);
  }
}

/**
 * Clip outliers using IQR method
 */
export class ClipOutliersStep implements PipelineStep {
  readonly name = 'clip_outliers';
  private multiplier: number;
  private lower = -Infinity;
  private upper = Infinity;

  constructor(multiplier: number = 1.5) {
    this.multiplier = multiplier;
  }

  async fit(data: TimeSeriesData): Promise<void> {
    const sorted = [...data].sort((a, b) => a - b);
    const n = sorted.length;
    const q1 = sorted[Math.floor(n * 0.25)];
    const q3 = sorted[Math.floor(n * 0.75)];
    const iqr = q3 - q1;
    this.lower = q1 - this.multiplier * iqr;
    this.upper = q3 + this.multiplier * iqr;
  }

  async transform(data: TimeSeriesData): Promise<TimeSeriesData> {
    return data.map((x) => Math.max(this.lower, Math.min(this.upper, x)));
  }

  async inverseTransform(data: TimeSeriesData): Promise<TimeSeriesData> {
    // Clipping is not reversible
    return data;
  }
}
