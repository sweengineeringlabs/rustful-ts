import { describe, it, expect } from 'vitest';
import {
  normalize,
  denormalize,
  standardize,
  destandardize,
  logTransform,
  inverseLogTransform,
  difference,
  seasonalDifference,
  cleanData,
  interpolateLinear,
  detectOutliersIQR,
  createLagFeatures,
  trainTestSplit,
} from '../src/utils/preprocessing';

describe('Preprocessing', () => {
  // ==================== Normalize/Denormalize Tests ====================

  describe('normalize', () => {
    it('normalizes data to [0, 1] range', () => {
      const data = [10, 20, 30, 40, 50];
      const { normalized, min, max } = normalize(data);

      expect(min).toBe(10);
      expect(max).toBe(50);
      expect(normalized[0]).toBe(0);
      expect(normalized[4]).toBe(1);
      expect(normalized[2]).toBeCloseTo(0.5, 5);
    });

    it('handles empty array', () => {
      const { normalized, min, max } = normalize([]);
      expect(normalized).toEqual([]);
      expect(min).toBe(0);
      expect(max).toBe(1);
    });

    it('handles constant data', () => {
      const { normalized } = normalize([5, 5, 5, 5]);
      expect(normalized).toEqual([0.5, 0.5, 0.5, 0.5]);
    });

    it('handles negative values', () => {
      const { normalized, min, max } = normalize([-10, 0, 10]);
      expect(min).toBe(-10);
      expect(max).toBe(10);
      expect(normalized).toEqual([0, 0.5, 1]);
    });
  });

  describe('denormalize', () => {
    it('reverses normalization', () => {
      const original = [10, 20, 30, 40, 50];
      const { normalized, min, max } = normalize(original);
      const restored = denormalize(normalized, min, max);

      for (let i = 0; i < original.length; i++) {
        expect(restored[i]).toBeCloseTo(original[i], 5);
      }
    });
  });

  // ==================== Standardize/Destandardize Tests ====================

  describe('standardize', () => {
    it('produces zero mean', () => {
      const { standardized } = standardize([1, 2, 3, 4, 5]);
      const mean = standardized.reduce((a, b) => a + b, 0) / standardized.length;
      expect(mean).toBeCloseTo(0, 10);
    });

    it('produces unit variance', () => {
      const { standardized } = standardize([1, 2, 3, 4, 5]);
      const mean = standardized.reduce((a, b) => a + b, 0) / standardized.length;
      const variance =
        standardized.reduce((acc, x) => acc + (x - mean) ** 2, 0) / standardized.length;
      expect(variance).toBeCloseTo(1, 5);
    });

    it('handles empty array', () => {
      const { standardized, mean, stdDev } = standardize([]);
      expect(standardized).toEqual([]);
      expect(mean).toBe(0);
      expect(stdDev).toBe(1);
    });

    it('handles constant data', () => {
      const { standardized } = standardize([5, 5, 5, 5]);
      expect(standardized).toEqual([0, 0, 0, 0]);
    });
  });

  describe('destandardize', () => {
    it('reverses standardization', () => {
      const original = [10, 20, 30, 40, 50];
      const { standardized, mean, stdDev } = standardize(original);
      const restored = destandardize(standardized, mean, stdDev);

      for (let i = 0; i < original.length; i++) {
        expect(restored[i]).toBeCloseTo(original[i], 5);
      }
    });
  });

  // ==================== Log Transform Tests ====================

  describe('logTransform / inverseLogTransform', () => {
    it('applies log1p transformation', () => {
      const data = [0, 1, 2, 3];
      const transformed = logTransform(data);

      expect(transformed[0]).toBe(0); // log1p(0) = 0
      expect(transformed[1]).toBeCloseTo(Math.log(2), 5); // log1p(1) = log(2)
    });

    it('inverse reverses the transformation', () => {
      const original = [0, 1, 10, 100];
      const transformed = logTransform(original);
      const restored = inverseLogTransform(transformed);

      for (let i = 0; i < original.length; i++) {
        expect(restored[i]).toBeCloseTo(original[i], 5);
      }
    });
  });

  // ==================== Difference Tests ====================

  describe('difference', () => {
    it('computes first-order differences', () => {
      const data = [1, 3, 6, 10];
      const diff = difference(data, 1);
      expect(diff).toEqual([2, 3, 4]);
    });

    it('computes second-order differences', () => {
      const data = [1, 3, 6, 10];
      // First diff: [2, 3, 4]
      // Second diff: [1, 1]
      const diff = difference(data, 2);
      expect(diff).toEqual([1, 1]);
    });

    it('returns empty for insufficient data', () => {
      expect(difference([1], 1)).toEqual([]);
      expect(difference([1, 2], 2)).toEqual([]);
    });

    it('defaults to order 1', () => {
      const data = [1, 4, 9, 16];
      expect(difference(data)).toEqual([3, 5, 7]);
    });
  });

  describe('seasonalDifference', () => {
    it('computes seasonal differences', () => {
      // Monthly data with period 3
      const data = [10, 20, 30, 15, 25, 35];
      const diff = seasonalDifference(data, 3);
      // [15-10, 25-20, 35-30] = [5, 5, 5]
      expect(diff).toEqual([5, 5, 5]);
    });

    it('returns empty for insufficient data', () => {
      expect(seasonalDifference([1, 2, 3], 3)).toEqual([]);
      expect(seasonalDifference([1, 2], 4)).toEqual([]);
    });
  });

  // ==================== Data Cleaning Tests ====================

  describe('cleanData', () => {
    it('removes NaN values', () => {
      const data = [1, NaN, 3, NaN, 5];
      expect(cleanData(data)).toEqual([1, 3, 5]);
    });

    it('removes Infinity values', () => {
      const data = [1, Infinity, 3, -Infinity, 5];
      expect(cleanData(data)).toEqual([1, 3, 5]);
    });

    it('keeps valid numbers', () => {
      const data = [1, 2, 3, 4, 5];
      expect(cleanData(data)).toEqual([1, 2, 3, 4, 5]);
    });

    it('handles empty array', () => {
      expect(cleanData([])).toEqual([]);
    });
  });

  describe('interpolateLinear', () => {
    it('interpolates NaN values', () => {
      const data = [1, NaN, 3];
      const interpolated = interpolateLinear(data);
      expect(interpolated).toEqual([1, 2, 3]);
    });

    it('handles multiple consecutive NaNs', () => {
      const data = [0, NaN, NaN, 6];
      const interpolated = interpolateLinear(data);
      expect(interpolated).toEqual([0, 2, 4, 6]);
    });

    it('handles NaN at start', () => {
      const data = [NaN, NaN, 3, 4];
      const interpolated = interpolateLinear(data);
      expect(interpolated[0]).toBe(3);
      expect(interpolated[1]).toBe(3);
    });

    it('handles NaN at end', () => {
      const data = [1, 2, NaN, NaN];
      const interpolated = interpolateLinear(data);
      expect(interpolated[2]).toBe(2);
      expect(interpolated[3]).toBe(2);
    });

    it('handles all valid data', () => {
      const data = [1, 2, 3, 4];
      expect(interpolateLinear(data)).toEqual([1, 2, 3, 4]);
    });
  });

  // ==================== Outlier Detection Tests ====================

  describe('detectOutliersIQR', () => {
    it('detects outliers', () => {
      // Normal data centered around 10-15 with extreme outlier at index 0
      const data = [1000, 10, 11, 12, 13, 14, 15, 10, 11, 12];
      const outliers = detectOutliersIQR(data);
      expect(outliers).toContain(0);
    });

    it('returns empty for normal data', () => {
      const data = [10, 11, 12, 13, 14, 15, 16];
      const outliers = detectOutliersIQR(data);
      expect(outliers).toEqual([]);
    });

    it('returns empty for insufficient data', () => {
      expect(detectOutliersIQR([1, 2, 3])).toEqual([]);
    });

    it('respects multiplier parameter', () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20];
      // With default 1.5x IQR, 20 might be outlier
      // With higher multiplier, it might not be
      const outliers1 = detectOutliersIQR(data, 1.5);
      const outliers3 = detectOutliersIQR(data, 3.0);
      expect(outliers1.length).toBeGreaterThanOrEqual(outliers3.length);
    });
  });

  // ==================== Feature Engineering Tests ====================

  describe('createLagFeatures', () => {
    it('creates lag features', () => {
      const data = [1, 2, 3, 4, 5];
      const { features, target } = createLagFeatures(data, 2);

      // With 2 lags, first usable point is index 2
      expect(features.length).toBe(3);
      expect(target.length).toBe(3);

      // features[0] should be [1, 2], target[0] should be 3
      expect(features[0]).toEqual([1, 2]);
      expect(target[0]).toBe(3);

      // features[1] should be [2, 3], target[1] should be 4
      expect(features[1]).toEqual([2, 3]);
      expect(target[1]).toBe(4);
    });

    it('returns empty for insufficient data', () => {
      const { features, target } = createLagFeatures([1, 2], 3);
      expect(features).toEqual([]);
      expect(target).toEqual([]);
    });
  });

  describe('trainTestSplit', () => {
    it('splits data with default ratio', () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const { train, test } = trainTestSplit(data);

      expect(train.length).toBe(8); // 80%
      expect(test.length).toBe(2); // 20%
      expect([...train, ...test]).toEqual(data);
    });

    it('respects custom ratio', () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const { train, test } = trainTestSplit(data, 0.3);

      expect(train.length).toBe(7);
      expect(test.length).toBe(3);
    });

    it('bounds ratio to valid range', () => {
      const data = [1, 2, 3, 4, 5];

      // Very small ratio - bounded to 0.1
      const split1 = trainTestSplit(data, 0.01);
      expect(split1.test.length).toBeGreaterThan(0);

      // Very large ratio - bounded to 0.9
      const split2 = trainTestSplit(data, 0.99);
      expect(split2.train.length).toBeGreaterThan(0);
    });

    it('maintains time order', () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const { train, test } = trainTestSplit(data, 0.2);

      // Train should be first 8, test should be last 2
      expect(train).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
      expect(test).toEqual([9, 10]);
    });
  });
});
