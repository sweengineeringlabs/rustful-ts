/**
 * Anomaly detectors
 */

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
  fit(data: number[]): void;

  /** Detect anomalies in data */
  detect(data: number[]): AnomalyResult;

  /** Get anomaly scores (higher = more anomalous) */
  score(data: number[]): number[];
}

/**
 * Z-Score based anomaly detector
 */
export class ZScoreDetector implements AnomalyDetector {
  readonly name = 'Z-Score';
  private threshold: number;
  private mean = 0;
  private std = 1;
  private fitted = false;

  constructor(threshold: number = 3.0) {
    this.threshold = threshold;
  }

  fit(data: number[]): void {
    const n = data.length;
    this.mean = data.reduce((a, b) => a + b, 0) / n;
    const variance = data.reduce((sum, x) => sum + (x - this.mean) ** 2, 0) / n;
    this.std = Math.sqrt(variance);
    this.fitted = true;
  }

  score(data: number[]): number[] {
    if (this.std === 0) {
      return data.map(() => 0);
    }
    return data.map((x) => Math.abs((x - this.mean) / this.std));
  }

  detect(data: number[]): AnomalyResult {
    if (!this.fitted) {
      this.fit(data);
    }

    const scores = this.score(data);
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

/**
 * Interquartile Range (IQR) based anomaly detector
 */
export class IQRDetector implements AnomalyDetector {
  readonly name = 'IQR';
  private multiplier: number;
  private q1 = 0;
  private q3 = 0;
  private iqr = 0;
  private fitted = false;

  constructor(multiplier: number = 1.5) {
    this.multiplier = multiplier;
  }

  fit(data: number[]): void {
    const sorted = [...data].sort((a, b) => a - b);
    const n = sorted.length;
    this.q1 = sorted[Math.floor(n * 0.25)];
    this.q3 = sorted[Math.floor(n * 0.75)];
    this.iqr = this.q3 - this.q1;
    this.fitted = true;
  }

  score(data: number[]): number[] {
    if (this.iqr === 0) {
      return data.map(() => 0);
    }
    const lower = this.q1 - this.multiplier * this.iqr;
    const upper = this.q3 + this.multiplier * this.iqr;

    return data.map((x) => {
      if (x < lower) return (lower - x) / this.iqr;
      if (x > upper) return (x - upper) / this.iqr;
      return 0;
    });
  }

  detect(data: number[]): AnomalyResult {
    if (!this.fitted) {
      this.fit(data);
    }

    const lower = this.q1 - this.multiplier * this.iqr;
    const upper = this.q3 + this.multiplier * this.iqr;

    const isAnomaly = data.map((x) => x < lower || x > upper);
    const scores = this.score(data);
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
 * More robust to outliers than Z-score
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

  fit(data: number[]): void {
    this.median = this.calculateMedian(data);
    const deviations = data.map((x) => Math.abs(x - this.median));
    this.mad = this.calculateMedian(deviations);
    this.fitted = true;
  }

  score(data: number[]): number[] {
    // Modified Z-score using MAD
    // Constant 0.6745 makes MAD consistent with standard deviation for normal distribution
    const k = 0.6745;
    if (this.mad === 0) {
      return data.map(() => 0);
    }
    return data.map((x) => Math.abs((x - this.median) / (this.mad / k)));
  }

  detect(data: number[]): AnomalyResult {
    if (!this.fitted) {
      this.fit(data);
    }

    const scores = this.score(data);
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
