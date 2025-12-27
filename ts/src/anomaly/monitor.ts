/**
 * Real-time monitoring and alerting
 */

import { AnomalyDetector, ZScoreDetector } from './detectors';

/**
 * Alert severity levels
 */
export enum AlertSeverity {
  Warning = 'WARNING',
  Critical = 'CRITICAL',
}

/**
 * An alert triggered by anomaly detection
 */
export interface Alert {
  /** Timestamp of the alert */
  timestamp: number;
  /** The anomalous value */
  value: number;
  /** Anomaly score */
  score: number;
  /** Alert severity */
  severity: AlertSeverity;
  /** Human-readable message */
  message: string;
}

/**
 * Callback for handling alerts
 */
export type AlertHandler = (alert: Alert) => void;

/**
 * Real-time anomaly monitor for streaming data
 */
export class Monitor {
  private detector: AnomalyDetector;
  private bufferSize: number;
  private buffer: number[] = [];
  private criticalThreshold: number;
  private handlers: AlertHandler[] = [];

  /**
   * Create a new monitor
   * @param detector - Anomaly detector to use
   * @param bufferSize - Number of points to keep in buffer
   * @param criticalThreshold - Score above which alerts are critical (default: 5)
   */
  constructor(
    detector: AnomalyDetector = new ZScoreDetector(),
    bufferSize: number = 100,
    criticalThreshold: number = 5
  ) {
    this.detector = detector;
    this.bufferSize = bufferSize;
    this.criticalThreshold = criticalThreshold;
  }

  /**
   * Register an alert handler
   */
  onAlert(handler: AlertHandler): this {
    this.handlers.push(handler);
    return this;
  }

  /**
   * Push a new value and check for anomalies
   * @returns Alert if anomaly detected, null otherwise
   */
  push(value: number): Alert | null {
    this.buffer.push(value);

    // Trim buffer if needed
    if (this.buffer.length > this.bufferSize) {
      this.buffer.shift();
    }

    // Need enough data to detect anomalies
    if (this.buffer.length < Math.min(10, this.bufferSize)) {
      return null;
    }

    // Refit detector on current buffer
    this.detector.fit(this.buffer);

    // Check if latest value is anomalous
    const result = this.detector.detect([value]);

    if (result.isAnomaly[0]) {
      const score = result.scores[0];
      const severity =
        score > this.criticalThreshold ? AlertSeverity.Critical : AlertSeverity.Warning;

      const alert: Alert = {
        timestamp: Date.now(),
        value,
        score,
        severity,
        message: `Anomaly detected: value=${value.toFixed(4)}, score=${score.toFixed(2)}`,
      };

      // Notify handlers
      for (const handler of this.handlers) {
        handler(alert);
      }

      return alert;
    }

    return null;
  }

  /**
   * Push multiple values and get all alerts
   */
  pushBatch(values: number[]): Alert[] {
    const alerts: Alert[] = [];
    for (const value of values) {
      const alert = this.push(value);
      if (alert) {
        alerts.push(alert);
      }
    }
    return alerts;
  }

  /**
   * Get current buffer contents
   */
  getBuffer(): number[] {
    return [...this.buffer];
  }

  /**
   * Clear the buffer
   */
  reset(): void {
    this.buffer = [];
  }

  /**
   * Get buffer statistics
   */
  getStats(): { mean: number; std: number; min: number; max: number; count: number } {
    if (this.buffer.length === 0) {
      return { mean: 0, std: 0, min: 0, max: 0, count: 0 };
    }

    const n = this.buffer.length;
    const mean = this.buffer.reduce((a, b) => a + b, 0) / n;
    const variance = this.buffer.reduce((sum, x) => sum + (x - mean) ** 2, 0) / n;

    return {
      mean,
      std: Math.sqrt(variance),
      min: Math.min(...this.buffer),
      max: Math.max(...this.buffer),
      count: n,
    };
  }
}
