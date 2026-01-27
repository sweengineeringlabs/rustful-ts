//! Elastic Volume Weighted Moving Average (EVWMA) implementation.
//!
//! A volume-adaptive moving average that uses volume to dynamically adjust its smoothing.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Elastic Volume Weighted Moving Average (EVWMA).
///
/// EVWMA is a moving average that adapts its smoothing based on trading volume.
/// The indicator treats volume as a measure of importance - periods with higher
/// volume have more influence on the average.
///
/// Unlike traditional volume-weighted averages that simply weight by volume,
/// EVWMA uses cumulative volume over a period to create an elastic smoothing
/// factor. This makes the average more responsive when volume is high and
/// more stable when volume is low.
///
/// # Formula
///
/// The EVWMA uses a dynamic smoothing factor based on volume:
///
/// ```text
/// k = Volume[i] / Sum(Volume, period)
/// EVWMA[i] = (1 - k) * EVWMA[i-1] + k * Price[i]
/// ```
///
/// Where:
/// - `k` is the volume-based smoothing factor
/// - `Sum(Volume, period)` is the sum of volume over the lookback period
///
/// # Interpretation
///
/// - High volume periods cause larger price moves in the average
/// - Low volume periods have minimal effect on the average
/// - Useful for identifying significant price moves backed by volume
/// - Can filter out low-volume noise while capturing high-volume trends
#[derive(Debug, Clone)]
pub struct EVWMA {
    period: usize,
}

impl EVWMA {
    /// Create a new EVWMA with the specified period.
    ///
    /// # Arguments
    /// - `period`: The lookback period for calculating the volume sum
    pub fn new(period: usize) -> Self {
        Self {
            period: period.max(1),
        }
    }

    /// Calculate EVWMA values for the given price and volume data.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n == 0 || n != volume.len() {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        if n < self.period {
            return result;
        }

        // Calculate initial volume sum for the first period
        let mut vol_sum: f64 = volume[0..self.period].iter().sum();

        // Initialize EVWMA with first price at period index
        // Use VWMA for initialization for better starting point
        let mut price_vol_sum = 0.0;
        for i in 0..self.period {
            price_vol_sum += close[i] * volume[i];
        }
        let mut evwma = if vol_sum > 0.0 {
            price_vol_sum / vol_sum
        } else {
            close[self.period - 1]
        };

        result[self.period - 1] = evwma;

        // Calculate EVWMA for remaining bars
        for i in self.period..n {
            // Update rolling volume sum
            vol_sum = vol_sum - volume[i - self.period] + volume[i];

            // Calculate smoothing factor based on volume
            let k = if vol_sum > 0.0 {
                (volume[i] / vol_sum).min(1.0)
            } else {
                // Fallback to equal weighting if no volume
                1.0 / self.period as f64
            };

            // Update EVWMA
            evwma = (1.0 - k) * evwma + k * close[i];
            result[i] = evwma;
        }

        result
    }

    /// Calculate EVWMA with custom volume normalization.
    ///
    /// This variant allows specifying a custom volume cap to prevent
    /// extreme volume spikes from dominating the average.
    pub fn calculate_with_cap(&self, close: &[f64], volume: &[f64], volume_cap: f64) -> Vec<f64> {
        let n = close.len();
        if n == 0 || n != volume.len() {
            return vec![f64::NAN; n];
        }

        // Cap volume values
        let capped_volume: Vec<f64> = volume.iter().map(|&v| v.min(volume_cap)).collect();

        self.calculate(close, &capped_volume)
    }
}

impl Default for EVWMA {
    fn default() -> Self {
        Self { period: 20 }
    }
}

impl TechnicalIndicator for EVWMA {
    fn name(&self) -> &str {
        "EVWMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evwma_basic() {
        let evwma = EVWMA::new(5);
        let close = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
        ];
        let volume = vec![
            1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
        ];

        let result = evwma.calculate(&close, &volume);

        assert_eq!(result.len(), 10);

        // First period-1 values should be NaN
        for i in 0..4 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }

        // Subsequent values should be valid
        for i in 4..10 {
            assert!(!result[i].is_nan(), "Expected valid value at index {}", i);
        }
    }

    #[test]
    fn test_evwma_equal_volume() {
        let evwma = EVWMA::new(3);
        let close = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let volume = vec![1000.0, 1000.0, 1000.0, 1000.0, 1000.0];

        let result = evwma.calculate(&close, &volume);

        // With equal volume, each bar contributes 1/3 of the period volume
        // So k = 1/3 for each bar after initialization
        assert!(!result[4].is_nan());
    }

    #[test]
    fn test_evwma_high_volume_impact() {
        let evwma = EVWMA::new(3);
        let close = vec![100.0, 100.0, 100.0, 110.0, 110.0];

        // Low volume for first bars, high volume when price jumps
        let volume_high = vec![100.0, 100.0, 100.0, 1000.0, 100.0];
        let volume_low = vec![100.0, 100.0, 100.0, 100.0, 100.0];

        let result_high = evwma.calculate(&close, &volume_high);
        let result_low = evwma.calculate(&close, &volume_low);

        // High volume version should react more to the price jump
        // at index 3 because the high volume gives that bar more weight
        assert!(
            result_high[3] > result_low[3],
            "High volume EVWMA ({}) should be higher than low volume ({}) after price jump",
            result_high[3],
            result_low[3]
        );
    }

    #[test]
    fn test_evwma_low_volume_stability() {
        let evwma = EVWMA::new(5);

        // Price spike with very low volume - should have minimal effect
        let close = vec![
            100.0, 100.0, 100.0, 100.0, 100.0, 150.0, 100.0, 100.0, 100.0, 100.0,
        ];
        let volume = vec![
            1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0, 1000.0, 1000.0,
        ];

        let result = evwma.calculate(&close, &volume);

        // The spike at index 5 should have minimal impact due to low volume
        // EVWMA should stay relatively close to 100
        assert!(
            result[5] < 110.0,
            "Low volume spike should have minimal impact: got {}",
            result[5]
        );
    }

    #[test]
    fn test_evwma_trending_with_volume() {
        let evwma = EVWMA::new(10);

        // Uptrend with increasing volume
        let close: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let volume: Vec<f64> = (0..30).map(|i| 1000.0 + i as f64 * 100.0).collect();

        let result = evwma.calculate(&close, &volume);

        // EVWMA should follow the trend
        assert!(result[29] > result[15], "EVWMA should follow uptrend");
        // Should be below current price (lagging)
        assert!(
            result[29] < close[29],
            "EVWMA should lag behind price in uptrend"
        );
    }

    #[test]
    fn test_evwma_zero_volume_handling() {
        let evwma = EVWMA::new(3);
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let volume = vec![0.0, 0.0, 0.0, 0.0, 0.0];

        let result = evwma.calculate(&close, &volume);

        // Should handle zero volume gracefully
        assert_eq!(result.len(), 5);
        // With zero volume sum, should use fallback weighting
        for i in 2..5 {
            assert!(
                !result[i].is_nan(),
                "Should produce valid values at index {}",
                i
            );
        }
    }

    #[test]
    fn test_evwma_insufficient_data() {
        let evwma = EVWMA::new(10);
        let close = vec![100.0, 101.0, 102.0];
        let volume = vec![1000.0, 1000.0, 1000.0];

        let result = evwma.calculate(&close, &volume);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_evwma_mismatched_lengths() {
        let evwma = EVWMA::new(3);
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let volume = vec![1000.0, 1000.0, 1000.0]; // Shorter

        let result = evwma.calculate(&close, &volume);

        // Should return NaN for mismatched data
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_evwma_empty_data() {
        let evwma = EVWMA::default();
        let result = evwma.calculate(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_evwma_constant_price() {
        let evwma = EVWMA::new(5);
        let close = vec![100.0; 20];
        let volume = vec![1000.0; 20];

        let result = evwma.calculate(&close, &volume);

        // EVWMA of constant price should equal that price
        for i in 4..20 {
            assert!(
                (result[i] - 100.0).abs() < 1e-10,
                "EVWMA of constant price should equal price at index {}: got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_evwma_default() {
        let evwma = EVWMA::default();
        assert_eq!(evwma.period, 20);
    }

    #[test]
    fn test_evwma_period_minimum() {
        let evwma = EVWMA::new(0);
        assert_eq!(evwma.period, 1, "Period should be at least 1");
    }

    #[test]
    fn test_evwma_technical_indicator_trait() {
        let evwma = EVWMA::new(10);
        assert_eq!(evwma.name(), "EVWMA");
        assert_eq!(evwma.min_periods(), 10);
        assert_eq!(evwma.output_features(), 1);
    }

    #[test]
    fn test_evwma_compute() {
        let evwma = EVWMA::new(5);
        let data = OHLCVSeries {
            open: vec![100.0; 20],
            high: vec![101.0; 20],
            low: vec![99.0; 20],
            close: (0..20).map(|i| 100.0 + i as f64).collect(),
            volume: vec![1000.0; 20],
        };

        let result = evwma.compute(&data).unwrap();
        assert_eq!(result.primary.len(), 20);
    }

    #[test]
    fn test_evwma_compute_insufficient_data() {
        let evwma = EVWMA::new(10);
        let data = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![101.0; 5],
            low: vec![99.0; 5],
            close: vec![100.0; 5],
            volume: vec![1000.0; 5],
        };

        let result = evwma.compute(&data);
        assert!(result.is_err());
        if let Err(IndicatorError::InsufficientData { required, got }) = result {
            assert_eq!(required, 10);
            assert_eq!(got, 5);
        }
    }

    #[test]
    fn test_evwma_with_cap() {
        let evwma = EVWMA::new(3);
        let close = vec![100.0, 100.0, 100.0, 110.0, 110.0];

        // Extreme volume spike
        let volume = vec![100.0, 100.0, 100.0, 10000.0, 100.0];

        let result_uncapped = evwma.calculate(&close, &volume);
        let result_capped = evwma.calculate_with_cap(&close, &volume, 500.0);

        // Capped version should be more stable (less affected by spike)
        // The jump at index 3 should be more moderate with capping
        assert!(
            (result_capped[3] - result_capped[2]).abs()
                < (result_uncapped[3] - result_uncapped[2]).abs(),
            "Capped EVWMA should be more stable against volume spikes"
        );
    }

    #[test]
    fn test_evwma_volume_proportion() {
        let evwma = EVWMA::new(3);
        let close = vec![100.0, 100.0, 100.0, 200.0];
        // Volume weights: 1:1:1 for first period, then bar 3 has 50% of period volume
        let volume = vec![100.0, 100.0, 100.0, 200.0];

        let result = evwma.calculate(&close, &volume);

        // At index 3:
        // vol_sum = 100 + 100 + 200 = 400
        // k = 200/400 = 0.5
        // EVWMA = 0.5 * previous + 0.5 * 200
        // Previous EVWMA was around 100, so new should be around 150
        assert!(
            result[3] > 140.0 && result[3] < 160.0,
            "EVWMA should reflect 50% weight: got {}",
            result[3]
        );
    }

    #[test]
    fn test_evwma_vs_vwma_responsiveness() {
        // EVWMA should be more responsive than traditional VWMA
        // because it uses cumulative volume as the denominator
        let evwma = EVWMA::new(5);

        // Price jumps with consistent volume
        let close = vec![
            100.0, 100.0, 100.0, 100.0, 100.0, 120.0, 120.0, 120.0, 120.0, 120.0,
        ];
        let volume = vec![
            1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
        ];

        let result = evwma.calculate(&close, &volume);

        // EVWMA should start moving toward 120 immediately after the jump
        assert!(result[5] > 100.0, "EVWMA should respond to price jump");
        assert!(
            result[9] > result[5],
            "EVWMA should continue moving toward new level"
        );
    }
}
