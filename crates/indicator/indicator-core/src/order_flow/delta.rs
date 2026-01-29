//! Delta (IND-216) - Buy volume minus sell volume
//!
//! Delta measures the difference between buying and selling pressure
//! by approximating buy/sell volume from price action within each bar.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Delta Output with buy/sell breakdown
#[derive(Debug, Clone)]
pub struct DeltaOutput {
    /// Net delta (buy volume - sell volume)
    pub delta: Vec<f64>,
    /// Estimated buy volume
    pub buy_volume: Vec<f64>,
    /// Estimated sell volume
    pub sell_volume: Vec<f64>,
}

/// Delta Configuration
#[derive(Debug, Clone)]
pub struct DeltaConfig {
    /// Smoothing period (1 = no smoothing)
    pub period: usize,
    /// Use typical price for position calculation
    pub use_typical_price: bool,
}

impl Default for DeltaConfig {
    fn default() -> Self {
        Self {
            period: 1,
            use_typical_price: false,
        }
    }
}

/// Delta (IND-216)
///
/// Calculates the difference between estimated buy and sell volume.
/// Uses the close position within the bar to estimate the buy/sell split:
/// - Close near high = more buying pressure
/// - Close near low = more selling pressure
///
/// Formula:
/// - Position = (Close - Low) / (High - Low)
/// - Buy Volume = Volume * Position
/// - Sell Volume = Volume * (1 - Position)
/// - Delta = Buy Volume - Sell Volume
#[derive(Debug, Clone)]
pub struct Delta {
    config: DeltaConfig,
}

impl Delta {
    pub fn new(config: DeltaConfig) -> Result<Self> {
        if config.period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self {
            config: DeltaConfig::default(),
        }
    }

    /// Calculate delta with full output breakdown
    pub fn calculate_full(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> DeltaOutput {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut delta = vec![0.0; n];
        let mut buy_volume = vec![0.0; n];
        let mut sell_volume = vec![0.0; n];

        for i in 0..n {
            let range = high[i] - low[i];
            if range > 0.0 {
                // Calculate close position within bar (0 = low, 1 = high)
                let position = if self.config.use_typical_price {
                    let typical = (high[i] + low[i] + close[i]) / 3.0;
                    (typical - low[i]) / range
                } else {
                    (close[i] - low[i]) / range
                };

                buy_volume[i] = volume[i] * position;
                sell_volume[i] = volume[i] * (1.0 - position);
                delta[i] = buy_volume[i] - sell_volume[i];
            }
        }

        // Apply smoothing if period > 1
        if self.config.period > 1 {
            delta = self.smooth(&delta);
            buy_volume = self.smooth(&buy_volume);
            sell_volume = self.smooth(&sell_volume);
        }

        DeltaOutput {
            delta,
            buy_volume,
            sell_volume,
        }
    }

    /// Calculate delta values only
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        self.calculate_full(high, low, close, volume).delta
    }

    /// Apply simple moving average smoothing
    fn smooth(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(self.config.period - 1);
            let count = i - start + 1;
            result[i] = data[start..=i].iter().sum::<f64>() / count as f64;
        }

        result
    }
}

impl TechnicalIndicator for Delta {
    fn name(&self) -> &str {
        "Delta"
    }

    fn min_periods(&self) -> usize {
        self.config.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.period {
            return Err(IndicatorError::InsufficientData {
                required: self.config.period,
                got: data.close.len(),
            });
        }

        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(output.delta, output.buy_volume, output.sell_volume))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // Uptrending data with closes near highs (bullish)
        let high = vec![105.0, 107.0, 109.0, 111.0, 113.0, 115.0, 117.0, 119.0, 121.0, 123.0];
        let low = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0];
        let close = vec![104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0, 122.0];
        let volume = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0];
        (high, low, close, volume)
    }

    #[test]
    fn test_delta_basic() {
        let delta = Delta::default_config();
        let (high, low, close, volume) = make_test_data();

        let result = delta.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 10);

        // Closes are near highs (position = 0.8), so delta should be positive
        // Delta = volume * (2 * position - 1) = volume * (2 * 0.8 - 1) = volume * 0.6
        for (i, &d) in result.iter().enumerate() {
            assert!(d > 0.0, "Delta at index {} should be positive, got {}", i, d);
        }
    }

    #[test]
    fn test_delta_with_smoothing() {
        let config = DeltaConfig {
            period: 3,
            use_typical_price: false,
        };
        let delta = Delta::new(config).unwrap();
        let (high, low, close, volume) = make_test_data();

        let result = delta.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 10);

        // Smoothed values should be non-zero
        assert!(result[5].abs() > 0.0);
    }

    #[test]
    fn test_delta_full_output() {
        let delta = Delta::default_config();
        let (high, low, close, volume) = make_test_data();

        let output = delta.calculate_full(&high, &low, &close, &volume);

        assert_eq!(output.delta.len(), 10);
        assert_eq!(output.buy_volume.len(), 10);
        assert_eq!(output.sell_volume.len(), 10);

        // Buy volume + sell volume should equal total volume
        for i in 0..10 {
            let total = output.buy_volume[i] + output.sell_volume[i];
            assert!((total - volume[i]).abs() < 1e-10, "Volumes don't sum correctly at {}", i);
        }

        // Delta should equal buy - sell
        for i in 0..10 {
            let expected = output.buy_volume[i] - output.sell_volume[i];
            assert!((output.delta[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_delta_bearish() {
        let delta = Delta::default_config();
        // Close near lows (bearish)
        let high = vec![105.0, 107.0, 109.0, 111.0, 113.0];
        let low = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let close = vec![101.0, 103.0, 105.0, 107.0, 109.0]; // Close near lows
        let volume = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0];

        let result = delta.calculate(&high, &low, &close, &volume);

        // Delta should be negative when closes are near lows
        for &d in &result {
            assert!(d < 0.0, "Delta should be negative for closes near lows");
        }
    }

    #[test]
    fn test_delta_zero_range() {
        let delta = Delta::default_config();
        // Zero range bars (high == low)
        let high = vec![100.0, 100.0, 100.0];
        let low = vec![100.0, 100.0, 100.0];
        let close = vec![100.0, 100.0, 100.0];
        let volume = vec![1000.0, 1100.0, 1200.0];

        let result = delta.calculate(&high, &low, &close, &volume);

        // Delta should be zero when range is zero
        for &d in &result {
            assert_eq!(d, 0.0, "Delta should be zero for zero-range bars");
        }
    }

    #[test]
    fn test_delta_invalid_period() {
        let config = DeltaConfig {
            period: 0,
            use_typical_price: false,
        };
        let result = Delta::new(config);
        assert!(result.is_err());
    }
}
