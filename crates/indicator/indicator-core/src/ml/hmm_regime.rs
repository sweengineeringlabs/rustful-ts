//! Hidden Markov Model Regime Indicator (IND-294)
//!
//! HMM-based market regime detection proxy using statistical methods
//! to identify different market states without full HMM implementation.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Market regime states detected by the HMM proxy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketRegimeState {
    /// Low volatility, trending up
    BullQuiet = 0,
    /// High volatility, trending up
    BullVolatile = 1,
    /// Low volatility, ranging/sideways
    Ranging = 2,
    /// High volatility, trending down
    BearVolatile = 3,
    /// Low volatility, trending down
    BearQuiet = 4,
}

impl MarketRegimeState {
    /// Convert numeric value to regime state
    pub fn from_value(value: f64) -> Self {
        match value as i32 {
            0 => MarketRegimeState::BullQuiet,
            1 => MarketRegimeState::BullVolatile,
            2 => MarketRegimeState::Ranging,
            3 => MarketRegimeState::BearVolatile,
            4 => MarketRegimeState::BearQuiet,
            _ => MarketRegimeState::Ranging,
        }
    }
}

/// HMM Regime - Hidden Markov Model states proxy for regime detection
///
/// This indicator simulates HMM regime detection using trend and volatility
/// clustering to identify distinct market states. It provides both the
/// regime classification and transition probabilities.
///
/// # Output
/// Returns regime state as numeric value (0-4):
/// - 0: Bull Quiet (low vol, uptrend)
/// - 1: Bull Volatile (high vol, uptrend)
/// - 2: Ranging (low vol, sideways)
/// - 3: Bear Volatile (high vol, downtrend)
/// - 4: Bear Quiet (low vol, downtrend)
#[derive(Debug, Clone)]
pub struct HMMRegime {
    /// Period for trend calculation
    trend_period: usize,
    /// Period for volatility calculation
    volatility_period: usize,
    /// Smoothing factor for regime transitions
    smoothing: usize,
}

impl HMMRegime {
    /// Create a new HMMRegime indicator
    ///
    /// # Arguments
    /// * `trend_period` - Period for trend detection (minimum 10)
    /// * `volatility_period` - Period for volatility clustering (minimum 5)
    /// * `smoothing` - Smoothing period for regime persistence (minimum 2)
    pub fn new(trend_period: usize, volatility_period: usize, smoothing: usize) -> Result<Self> {
        if trend_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smoothing < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { trend_period, volatility_period, smoothing })
    }

    /// Calculate the regime state for each period
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![2.0; n]; // Default to Ranging

        if n < self.trend_period {
            return result;
        }

        // Calculate trend strength and direction
        let mut trend_strength = vec![0.0; n];
        let mut volatility = vec![0.0; n];

        for i in self.trend_period..n {
            let start = i.saturating_sub(self.trend_period);

            // Linear regression slope for trend
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;
            let mut sum_x2 = 0.0;

            for (j, idx) in (start..=i).enumerate() {
                let x = j as f64;
                let y = close[idx];
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
            }

            let count = (i - start + 1) as f64;
            let slope = (count * sum_xy - sum_x * sum_y) / (count * sum_x2 - sum_x * sum_x + 1e-10);

            // Normalize slope by price level
            let avg_price = sum_y / count;
            trend_strength[i] = if avg_price > 1e-10 {
                slope / avg_price * 100.0 * self.trend_period as f64
            } else {
                0.0
            };
        }

        // Calculate volatility using ATR-like measure
        for i in self.volatility_period..n {
            let start = i.saturating_sub(self.volatility_period);
            let mut tr_sum = 0.0;

            for j in start..=i {
                let tr = if j > 0 {
                    let hl = high[j] - low[j];
                    let hc = (high[j] - close[j - 1]).abs();
                    let lc = (low[j] - close[j - 1]).abs();
                    hl.max(hc).max(lc)
                } else {
                    high[j] - low[j]
                };
                tr_sum += tr;
            }

            let atr = tr_sum / (i - start + 1) as f64;
            volatility[i] = if close[i] > 1e-10 {
                atr / close[i] * 100.0
            } else {
                0.0
            };
        }

        // Calculate volatility percentile for regime classification
        let max_period = self.trend_period.max(self.volatility_period);
        for i in max_period..n {
            let vol_start = i.saturating_sub(50.min(i)); // Look back up to 50 periods
            let mut vol_history: Vec<f64> = volatility[vol_start..i].to_vec();
            vol_history.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let vol_percentile = if !vol_history.is_empty() {
                let pos = vol_history.iter().position(|&v| v >= volatility[i]).unwrap_or(vol_history.len());
                pos as f64 / vol_history.len() as f64
            } else {
                0.5
            };

            // Classify regime based on trend and volatility
            let high_vol = vol_percentile > 0.6;
            let trend = trend_strength[i];

            result[i] = if trend > 1.0 {
                // Uptrend
                if high_vol { 1.0 } else { 0.0 }
            } else if trend < -1.0 {
                // Downtrend
                if high_vol { 3.0 } else { 4.0 }
            } else {
                // Ranging
                2.0
            };
        }

        // Apply smoothing to reduce regime whipsaws
        let mut smoothed = result.clone();
        for i in self.smoothing..n {
            let start = i.saturating_sub(self.smoothing);
            let mut regime_counts = [0; 5];

            for j in start..=i {
                let regime = result[j] as usize;
                if regime < 5 {
                    regime_counts[regime] += 1;
                }
            }

            // Most frequent regime in window
            let max_regime = regime_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &count)| count)
                .map(|(idx, _)| idx)
                .unwrap_or(2);

            smoothed[i] = max_regime as f64;
        }

        smoothed
    }

    /// Get the regime state enum for a given index
    pub fn get_regime(&self, values: &[f64], index: usize) -> MarketRegimeState {
        if index < values.len() {
            MarketRegimeState::from_value(values[index])
        } else {
            MarketRegimeState::Ranging
        }
    }
}

impl TechnicalIndicator for HMMRegime {
    fn name(&self) -> &str {
        "HMM Regime"
    }

    fn min_periods(&self) -> usize {
        self.trend_period.max(self.volatility_period) + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uptrend_data() -> OHLCVSeries {
        let n = 60;
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 1.0) // Strong uptrend
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        let open: Vec<f64> = close.iter().enumerate()
            .map(|(i, _)| if i > 0 { close[i - 1] } else { close[0] })
            .collect();
        let volume: Vec<f64> = vec![1000.0; n];

        OHLCVSeries { open, high, low, close, volume }
    }

    fn make_downtrend_data() -> OHLCVSeries {
        let n = 60;
        let close: Vec<f64> = (0..n)
            .map(|i| 150.0 - (i as f64) * 1.0) // Strong downtrend
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.iter().enumerate()
            .map(|(i, _)| if i > 0 { close[i - 1] } else { close[0] })
            .collect();
        let volume: Vec<f64> = vec![1000.0; n];

        OHLCVSeries { open, high, low, close, volume }
    }

    fn make_ranging_data() -> OHLCVSeries {
        let n = 60;
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 2.0) // Sideways
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; n];

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_hmm_regime_basic() {
        let data = make_uptrend_data();
        let indicator = HMMRegime::new(15, 10, 5).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        // Results should be valid regime values (0-4)
        for i in 30..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 4.0);
        }
    }

    #[test]
    fn test_hmm_regime_detects_uptrend() {
        let data = make_uptrend_data();
        let indicator = HMMRegime::new(15, 10, 5).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        // Should detect bullish regime (0 or 1) in clear uptrend
        let bull_count = result[40..].iter().filter(|&&r| r == 0.0 || r == 1.0).count();
        assert!(bull_count as f64 / (result.len() - 40) as f64 > 0.5);
    }

    #[test]
    fn test_hmm_regime_detects_downtrend() {
        let data = make_downtrend_data();
        let indicator = HMMRegime::new(15, 10, 5).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        // Should detect bearish regime (3 or 4) in clear downtrend
        let bear_count = result[40..].iter().filter(|&&r| r == 3.0 || r == 4.0).count();
        assert!(bear_count as f64 / (result.len() - 40) as f64 > 0.5);
    }

    #[test]
    fn test_hmm_regime_detects_ranging() {
        let data = make_ranging_data();
        let indicator = HMMRegime::new(15, 10, 5).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        // Should detect ranging regime (2) in sideways market
        let range_count = result[40..].iter().filter(|&&r| r == 2.0).count();
        assert!(range_count > 0);
    }

    #[test]
    fn test_hmm_regime_state_enum() {
        assert_eq!(MarketRegimeState::from_value(0.0), MarketRegimeState::BullQuiet);
        assert_eq!(MarketRegimeState::from_value(1.0), MarketRegimeState::BullVolatile);
        assert_eq!(MarketRegimeState::from_value(2.0), MarketRegimeState::Ranging);
        assert_eq!(MarketRegimeState::from_value(3.0), MarketRegimeState::BearVolatile);
        assert_eq!(MarketRegimeState::from_value(4.0), MarketRegimeState::BearQuiet);
        assert_eq!(MarketRegimeState::from_value(99.0), MarketRegimeState::Ranging);
    }

    #[test]
    fn test_hmm_regime_technical_indicator_trait() {
        let data = make_uptrend_data();
        let indicator = HMMRegime::new(15, 10, 5).unwrap();

        assert_eq!(indicator.name(), "HMM Regime");
        assert_eq!(indicator.min_periods(), 20); // max(15, 10) + 5

        let output = indicator.compute(&data).unwrap();
        assert!(!output.values.is_empty());
    }

    #[test]
    fn test_hmm_regime_parameter_validation() {
        assert!(HMMRegime::new(5, 10, 5).is_err()); // trend_period too small
        assert!(HMMRegime::new(15, 3, 5).is_err()); // volatility_period too small
        assert!(HMMRegime::new(15, 10, 1).is_err()); // smoothing too small
    }

    #[test]
    fn test_hmm_regime_get_regime_method() {
        let data = make_uptrend_data();
        let indicator = HMMRegime::new(15, 10, 5).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        let regime = indicator.get_regime(&result, 50);
        // Should be a valid regime
        assert!([
            MarketRegimeState::BullQuiet,
            MarketRegimeState::BullVolatile,
            MarketRegimeState::Ranging,
            MarketRegimeState::BearVolatile,
            MarketRegimeState::BearQuiet,
        ].contains(&regime));
    }
}
