import { describe, it, expect } from 'vitest';
import {
  mae,
  mse,
  rmse,
  mape,
  smape,
  rSquared,
  computeMetrics,
} from '../src/utils/metrics';

describe('Metrics', () => {
  // ==================== MAE Tests ====================

  describe('mae', () => {
    it('calculates MAE correctly', () => {
      const actual = [10, 20, 30, 40, 50];
      const predicted = [12, 18, 33, 42, 48];
      // |10-12| + |20-18| + |30-33| + |40-42| + |50-48| = 2+2+3+2+2 = 11
      // 11 / 5 = 2.2
      expect(mae(actual, predicted)).toBeCloseTo(2.2, 5);
    });

    it('returns 0 for perfect predictions', () => {
      const data = [1, 2, 3, 4, 5];
      expect(mae(data, data)).toBe(0);
    });

    it('returns NaN for empty arrays', () => {
      expect(mae([], [])).toBeNaN();
    });

    it('returns NaN for mismatched lengths', () => {
      expect(mae([1, 2, 3], [1, 2])).toBeNaN();
    });

    it('handles negative values', () => {
      const actual = [-10, -5, 0, 5, 10];
      const predicted = [-8, -7, 2, 3, 8];
      // 2 + 2 + 2 + 2 + 2 = 10 / 5 = 2
      expect(mae(actual, predicted)).toBeCloseTo(2, 5);
    });
  });

  // ==================== MSE Tests ====================

  describe('mse', () => {
    it('calculates MSE correctly', () => {
      const actual = [1, 2, 3, 4, 5];
      const predicted = [1.5, 2.5, 3.5, 4.5, 5.5];
      // Each diff is 0.5, squared = 0.25
      // Sum = 1.25, mean = 0.25
      expect(mse(actual, predicted)).toBeCloseTo(0.25, 5);
    });

    it('returns 0 for perfect predictions', () => {
      const data = [1, 2, 3, 4, 5];
      expect(mse(data, data)).toBe(0);
    });

    it('returns NaN for empty arrays', () => {
      expect(mse([], [])).toBeNaN();
    });

    it('penalizes large errors more heavily', () => {
      // Same MAE, different MSE
      const actual = [10, 10, 10, 10];
      const pred1 = [11, 11, 11, 11]; // errors: 1,1,1,1 → MSE = 1
      const pred2 = [14, 10, 10, 10]; // errors: 4,0,0,0 → MSE = 4

      expect(mse(actual, pred1)).toBeLessThan(mse(actual, pred2));
    });
  });

  // ==================== RMSE Tests ====================

  describe('rmse', () => {
    it('calculates RMSE as sqrt of MSE', () => {
      const actual = [1, 2, 3, 4, 5];
      const predicted = [2, 3, 4, 5, 6];
      // MSE = 1, RMSE = 1
      expect(rmse(actual, predicted)).toBeCloseTo(1, 5);
    });

    it('returns 0 for perfect predictions', () => {
      const data = [1, 2, 3];
      expect(rmse(data, data)).toBe(0);
    });

    it('is in same scale as data', () => {
      const actual = [100, 200, 300];
      const predicted = [110, 210, 310];
      // RMSE should be around 10, same scale as data
      expect(rmse(actual, predicted)).toBeCloseTo(10, 5);
    });
  });

  // ==================== MAPE Tests ====================

  describe('mape', () => {
    it('calculates MAPE correctly', () => {
      const actual = [100, 100, 100];
      const predicted = [110, 90, 100];
      // |10/100| + |10/100| + |0| = 0.1 + 0.1 + 0 = 0.2 / 3
      expect(mape(actual, predicted)).toBeCloseTo(0.2 / 3, 5);
    });

    it('returns 0 for perfect predictions', () => {
      const data = [10, 20, 30];
      expect(mape(data, data)).toBe(0);
    });

    it('returns NaN for empty arrays', () => {
      expect(mape([], [])).toBeNaN();
    });

    it('handles zero actuals by excluding them', () => {
      const actual = [0, 10, 20];
      const predicted = [5, 10, 20];
      // Only 10 and 20 are counted
      expect(mape(actual, predicted)).toBe(0);
    });

    it('returns 0.1 for 10% error', () => {
      const actual = [100, 200, 300];
      const predicted = [110, 220, 330];
      // All 10% off
      expect(mape(actual, predicted)).toBeCloseTo(0.1, 5);
    });
  });

  // ==================== sMAPE Tests ====================

  describe('smape', () => {
    it('calculates sMAPE correctly', () => {
      const actual = [100, 200];
      const predicted = [110, 210];
      // 2*10/(100+110) + 2*10/(200+210) = 20/210 + 20/410
      const expected = (20 / 210 + 20 / 410) / 2;
      expect(smape(actual, predicted)).toBeCloseTo(expected, 5);
    });

    it('returns 0 for perfect predictions', () => {
      const data = [10, 20, 30];
      expect(smape(data, data)).toBe(0);
    });

    it('handles zeros better than MAPE', () => {
      const actual = [0, 10];
      const predicted = [1, 10];
      // First: 2*1/(0+1) = 2, second: 0
      // (2 + 0) / 2 = 1
      expect(smape(actual, predicted)).toBeCloseTo(1, 5);
    });
  });

  // ==================== R-Squared Tests ====================

  describe('rSquared', () => {
    it('returns 1 for perfect predictions', () => {
      const actual = [1, 2, 3, 4, 5];
      expect(rSquared(actual, actual)).toBe(1);
    });

    it('returns high value for good predictions', () => {
      const actual = [1, 2, 3, 4, 5];
      const predicted = [1.1, 1.9, 3.1, 3.9, 5.1];
      expect(rSquared(actual, predicted)).toBeGreaterThan(0.95);
    });

    it('returns near 0 for mean-level predictions', () => {
      const actual = [1, 2, 3, 4, 5];
      const mean = 3;
      const predicted = [mean, mean, mean, mean, mean];
      expect(rSquared(actual, predicted)).toBeCloseTo(0, 5);
    });

    it('can be negative for bad predictions', () => {
      const actual = [1, 2, 3, 4, 5];
      const predicted = [5, 4, 3, 2, 1]; // Inverted
      expect(rSquared(actual, predicted)).toBeLessThan(0);
    });

    it('returns NaN for empty arrays', () => {
      expect(rSquared([], [])).toBeNaN();
    });

    it('returns 1 for constant data', () => {
      const actual = [5, 5, 5, 5];
      expect(rSquared(actual, actual)).toBe(1);
    });
  });

  // ==================== computeMetrics Tests ====================

  describe('computeMetrics', () => {
    it('returns all metrics', () => {
      const actual = [10, 20, 30, 40, 50];
      const predicted = [12, 18, 33, 42, 48];

      const metrics = computeMetrics(actual, predicted);

      expect(metrics).toHaveProperty('mae');
      expect(metrics).toHaveProperty('mse');
      expect(metrics).toHaveProperty('rmse');
      expect(metrics).toHaveProperty('mape');
      expect(metrics).toHaveProperty('smape');
      expect(metrics).toHaveProperty('rSquared');

      expect(metrics.mae).toBeCloseTo(2.2, 5);
      expect(metrics.rmse).toBeCloseTo(Math.sqrt(metrics.mse), 5);
    });

    it('all metrics are 0 or 1 for perfect predictions', () => {
      const data = [10, 20, 30, 40, 50];
      const metrics = computeMetrics(data, data);

      expect(metrics.mae).toBe(0);
      expect(metrics.mse).toBe(0);
      expect(metrics.rmse).toBe(0);
      expect(metrics.mape).toBe(0);
      expect(metrics.smape).toBe(0);
      expect(metrics.rSquared).toBe(1);
    });
  });
});
