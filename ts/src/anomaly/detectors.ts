/**
 * Anomaly detectors - WASM-backed implementations
 */

import { getWasmModule, ensureWasm } from '../wasm-loader';

/**
 * Result of anomaly detection
 */
export interface AnomalyResult {
  /** Boolean array indicating anomalies */
  isAnomaly: boolean[];
  /** Anomaly scores for each point */
  scores: number[];
  /** Threshold used for detection */
  threshold: number;
  /** Indices of anomalies */
  anomalyIndices: number[];
}

/**
 * Interface for anomaly detectors
 */
export interface AnomalyDetector {
  /** Name of the detector */
  readonly name: string;

  /** Fit the detector to training data */
  fit(data: number[]): Promise<void>;

  /** Detect anomalies in data */
  detect(data: number[]): Promise<AnomalyResult>;

  /** Get anomaly scores (higher = more anomalous) */
  score(data: number[]): Promise<number[]>;
}

/**
 * Z-Score based anomaly detector (WASM-backed)
 *
 * Detects anomalies by computing how many standard deviations
 * each point is from the mean.
 *
 * @example
 * ```typescript
 * const detector = new ZScoreDetector(3.0);
 * await detector.fit(trainingData);
 * const result = await detector.detect(newData);
 * console.log(result.anomalyIndices);
 * ```
 */
export class ZScoreDetector implements AnomalyDetector {
  readonly name = 'Z-Score';
  private inner: unknown = null;
  private threshold: number;
  private fitted = false;

  constructor(threshold: number = 3.0) {
    this.threshold = threshold;
  }

  async fit(data: number[]): Promise<void> {
    await ensureWasm();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = getWasmModule() as any;

    this.inner = new wasm.WasmZScoreDetector(this.threshold);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (this.inner as any).fit(new Float64Array(data));
    this.fitted = true;
  }

  async score(data: number[]): Promise<number[]> {
    if (!this.fitted || !this.inner) {
      throw new Error('Detector must be fitted before scoring');
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const result = (this.inner as any).score(new Float64Array(data));
    return Array.from(result as Float64Array);
  }

  async detect(data: number[]): Promise<AnomalyResult> {
    if (!this.fitted || !this.inner) {
      await this.fit(data);
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = this.inner as any;
    const scores = Array.from(wasm.score(new Float64Array(data)) as Float64Array);
    const detectResult = Array.from(wasm.detect(new Float64Array(data)) as Uint8Array);
    const isAnomaly = detectResult.map((v: number) => v === 1);
    const anomalyIndices = isAnomaly
      .map((a, i) => (a ? i : -1))
      .filter((i) => i >= 0);

    return {
      isAnomaly,
      scores,
      threshold: this.threshold,
      anomalyIndices,
    };
  }
}

/**
 * Interquartile Range (IQR) based anomaly detector (WASM-backed)
 *
 * Detects anomalies using the IQR method. Points outside
 * [Q1 - multiplier*IQR, Q3 + multiplier*IQR] are anomalies.
 *
 * @example
 * ```typescript
 * const detector = new IQRDetector(1.5);
 * await detector.fit(trainingData);
 * const result = await detector.detect(newData);
 * ```
 */
export class IQRDetector implements AnomalyDetector {
  readonly name = 'IQR';
  private inner: unknown = null;
  private multiplier: number;
  private fitted = false;

  constructor(multiplier: number = 1.5) {
    this.multiplier = multiplier;
  }

  async fit(data: number[]): Promise<void> {
    await ensureWasm();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = getWasmModule() as any;

    this.inner = new wasm.WasmIQRDetector(this.multiplier);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (this.inner as any).fit(new Float64Array(data));
    this.fitted = true;
  }

  async score(data: number[]): Promise<number[]> {
    if (!this.fitted || !this.inner) {
      throw new Error('Detector must be fitted before scoring');
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const result = (this.inner as any).score(new Float64Array(data));
    return Array.from(result as Float64Array);
  }

  async detect(data: number[]): Promise<AnomalyResult> {
    if (!this.fitted || !this.inner) {
      await this.fit(data);
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = this.inner as any;
    const scores = Array.from(wasm.score(new Float64Array(data)) as Float64Array);
    const detectResult = Array.from(wasm.detect(new Float64Array(data)) as Uint8Array);
    const isAnomaly = detectResult.map((v: number) => v === 1);
    const anomalyIndices = isAnomaly
      .map((a, i) => (a ? i : -1))
      .filter((i) => i >= 0);

    return {
      isAnomaly,
      scores,
      threshold: this.multiplier,
      anomalyIndices,
    };
  }
}

/**
 * Median Absolute Deviation (MAD) based anomaly detector
 *
 * More robust to outliers than Z-score. Uses median instead of mean.
 * Note: This is a pure TypeScript implementation (no WASM binding yet).
 *
 * @example
 * ```typescript
 * const detector = new MADDetector(3.5);
 * await detector.fit(trainingData);
 * const result = await detector.detect(newData);
 * ```
 */
export class MADDetector implements AnomalyDetector {
  readonly name = 'MAD';
  private threshold: number;
  private median = 0;
  private mad = 0;
  private fitted = false;

  constructor(threshold: number = 3.5) {
    this.threshold = threshold;
  }

  private calculateMedian(data: number[]): number {
    const sorted = [...data].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0
      ? sorted[mid]
      : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  async fit(data: number[]): Promise<void> {
    this.median = this.calculateMedian(data);
    const deviations = data.map((x) => Math.abs(x - this.median));
    this.mad = this.calculateMedian(deviations);
    this.fitted = true;
  }

  async score(data: number[]): Promise<number[]> {
    // Modified Z-score using MAD
    // Constant 0.6745 makes MAD consistent with standard deviation for normal distribution
    const k = 0.6745;
    if (this.mad === 0) {
      return data.map(() => 0);
    }
    return data.map((x) => Math.abs((x - this.median) / (this.mad / k)));
  }

  async detect(data: number[]): Promise<AnomalyResult> {
    if (!this.fitted) {
      await this.fit(data);
    }

    const scores = await this.score(data);
    const isAnomaly = scores.map((s) => s > this.threshold);
    const anomalyIndices = isAnomaly
      .map((a, i) => (a ? i : -1))
      .filter((i) => i >= 0);

    return {
      isAnomaly,
      scores,
      threshold: this.threshold,
      anomalyIndices,
    };
  }
}
