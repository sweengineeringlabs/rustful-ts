//! Commitment of Traders Indicators
//!
//! Indicators for analyzing futures positioning based on COT-like patterns.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Commitment of Traders (IND-282) - Futures positioning proxy
///
/// This indicator creates a proxy for COT data using price and volume
/// patterns that often reflect institutional vs retail positioning.
#[derive(Debug, Clone)]
pub struct CommitmentOfTraders {
    period: usize,
    lookback: usize,
}

/// Configuration for CommitmentOfTraders
#[derive(Debug, Clone)]
pub struct CommitmentOfTradersConfig {
    pub period: usize,
    pub lookback: usize,
}

impl Default for CommitmentOfTradersConfig {
    fn default() -> Self {
        Self {
            period: 14,
            lookback: 52,
        }
    }
}

impl CommitmentOfTraders {
    pub fn new(period: usize, lookback: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if lookback < period {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least equal to period".to_string(),
            });
        }
        Ok(Self { period, lookback })
    }

    pub fn with_config(config: CommitmentOfTradersConfig) -> Result<Self> {
        Self::new(config.period, config.lookback)
    }

    /// Calculate commercial (hedger) positioning proxy (-100 to 100)
    ///
    /// Positive values suggest commercials are net long (bullish for prices)
    /// Negative values suggest commercials are net short (bearish for prices)
    pub fn calculate_commercial(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Commercials tend to fade extremes - they sell into strength and buy weakness
            // Calculate price position within range
            let max_high = high[start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_low = low[start..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let range = max_high - min_low;

            let price_position = if range > 0.0 {
                (close[i] - min_low) / range
            } else {
                0.5
            };

            // Commercials are contrarian - inverse of price position
            let commercial_position = 1.0 - price_position;

            // Volume-weighted adjustment (commercials accumulate on low volume)
            let recent_start = i.saturating_sub(self.period);
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / (i - start).max(1) as f64;
            let recent_avg_volume: f64 = volume[recent_start..=i].iter().sum::<f64>() / (i - recent_start + 1) as f64;

            let volume_factor = if avg_volume > 0.0 {
                (avg_volume / recent_avg_volume.max(0.001)).min(2.0).max(0.5)
            } else {
                1.0
            };

            // Combine: commercial position with volume adjustment
            let raw_position = (commercial_position * 2.0 - 1.0) * volume_factor;
            result[i] = (raw_position * 100.0).clamp(-100.0, 100.0);
        }
        result
    }

    /// Calculate speculator (large trader) positioning proxy (-100 to 100)
    ///
    /// Positive values suggest speculators are net long
    /// Negative values suggest speculators are net short
    pub fn calculate_speculator(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Speculators are trend followers - they go with momentum
            // Calculate momentum
            let momentum = if close[start] > 0.0 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Calculate trend strength
            let mut up_days = 0;
            let mut down_days = 0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_days += 1;
                } else if close[j] < close[j - 1] {
                    down_days += 1;
                }
            }

            let trend_bias = if up_days + down_days > 0 {
                (up_days as f64 - down_days as f64) / (up_days + down_days) as f64
            } else {
                0.0
            };

            // Speculators pile in on high volume moves
            let recent_start = i.saturating_sub(self.period);
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / (i - start).max(1) as f64;
            let recent_avg_volume: f64 = volume[recent_start..=i].iter().sum::<f64>() / (i - recent_start + 1) as f64;

            let volume_factor = if avg_volume > 0.0 {
                (recent_avg_volume / avg_volume).min(2.0).max(0.5)
            } else {
                1.0
            };

            // Combine momentum and trend with volume
            let speculator_position = (momentum.signum() * trend_bias.abs() + trend_bias) / 2.0 * volume_factor;
            result[i] = (speculator_position * 100.0).clamp(-100.0, 100.0);
        }
        result
    }

    /// Calculate net positioning index (-100 to 100)
    ///
    /// Combines commercial and speculator positioning for overall market sentiment
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let commercial = self.calculate_commercial(high, low, close, volume);
        let speculator = self.calculate_speculator(high, low, close, volume);

        let n = commercial.len().min(speculator.len());
        let mut result = vec![0.0; n];

        for i in 0..n {
            // When commercials and speculators disagree, favor commercials (smart money)
            result[i] = (commercial[i] * 0.6 + speculator[i] * 0.4).clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for CommitmentOfTraders {
    fn name(&self) -> &str {
        "Commitment of Traders"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let commercial = self.calculate_commercial(&data.high, &data.low, &data.close, &data.volume);
        let speculator = self.calculate_speculator(&data.high, &data.low, &data.close, &data.volume);
        let net = self.calculate(&data.high, &data.low, &data.close, &data.volume);

        Ok(IndicatorOutput::triple(commercial, speculator, net))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create longer test data for lookback period
        let mut high = Vec::with_capacity(60);
        let mut low = Vec::with_capacity(60);
        let mut close = Vec::with_capacity(60);
        let mut volume = Vec::with_capacity(60);

        for i in 0..60 {
            let base = 100.0 + (i as f64 * 0.5);
            high.push(base + 5.0);
            low.push(base);
            close.push(base + 3.0);
            volume.push(1000.0 + (i as f64 * 50.0));
        }

        (high, low, close, volume)
    }

    #[test]
    fn test_commitment_of_traders_creation() {
        let indicator = CommitmentOfTraders::new(14, 52);
        assert!(indicator.is_ok());

        let indicator = CommitmentOfTraders::new(3, 52);
        assert!(indicator.is_err());

        let indicator = CommitmentOfTraders::new(14, 10);
        assert!(indicator.is_err());
    }

    #[test]
    fn test_commitment_of_traders_commercial() {
        let (high, low, close, volume) = make_test_data();
        let indicator = CommitmentOfTraders::new(14, 52).unwrap();
        let result = indicator.calculate_commercial(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 53..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_commitment_of_traders_speculator() {
        let (high, low, close, volume) = make_test_data();
        let indicator = CommitmentOfTraders::new(14, 52).unwrap();
        let result = indicator.calculate_speculator(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 53..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_commitment_of_traders_net() {
        let (high, low, close, volume) = make_test_data();
        let indicator = CommitmentOfTraders::new(14, 52).unwrap();
        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 53..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_commitment_of_traders_min_periods() {
        let indicator = CommitmentOfTraders::new(14, 52).unwrap();
        assert_eq!(indicator.min_periods(), 53);
    }
}
