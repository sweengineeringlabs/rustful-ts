//! ImbalanceDetector (IND-220) - Bid/ask volume imbalance detection
//!
//! Detects significant imbalances between buying and selling pressure
//! that may indicate institutional activity or potential price moves.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator, SignalIndicator, IndicatorSignal,
};

/// Type of imbalance detected
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImbalanceType {
    /// No significant imbalance
    None,
    /// Buying imbalance - significant excess buying pressure
    BuyImbalance,
    /// Selling imbalance - significant excess selling pressure
    SellImbalance,
    /// Stacked buying - multiple consecutive buy imbalances
    StackedBuy,
    /// Stacked selling - multiple consecutive sell imbalances
    StackedSell,
}

/// Imbalance Detector Output
#[derive(Debug, Clone)]
pub struct ImbalanceDetectorOutput {
    /// Imbalance type at each bar
    pub imbalance_type: Vec<ImbalanceType>,
    /// Imbalance ratio (buy/sell or sell/buy)
    pub imbalance_ratio: Vec<f64>,
    /// Stacked imbalance count (positive = buy, negative = sell)
    pub stacked_count: Vec<i32>,
    /// Signal values: 1 = buy imbalance, -1 = sell imbalance
    pub signal: Vec<f64>,
}

/// Imbalance Detector Configuration
#[derive(Debug, Clone)]
pub struct ImbalanceDetectorConfig {
    /// Minimum ratio for imbalance detection (e.g., 3.0 = 3:1 ratio)
    pub imbalance_threshold: f64,
    /// Minimum stacked imbalances for strong signal
    pub min_stacked: usize,
    /// Lookback period for average volume calculation
    pub volume_lookback: usize,
    /// Minimum volume threshold (as multiplier of average)
    pub min_volume_multiplier: f64,
}

impl Default for ImbalanceDetectorConfig {
    fn default() -> Self {
        Self {
            imbalance_threshold: 3.0,
            min_stacked: 3,
            volume_lookback: 20,
            min_volume_multiplier: 0.8,
        }
    }
}

/// ImbalanceDetector (IND-220)
///
/// Detects bid/ask volume imbalances that may signal institutional activity.
///
/// An imbalance occurs when the ratio of buying to selling volume
/// (or vice versa) exceeds a threshold, suggesting aggressive
/// participation on one side of the market.
///
/// Features:
/// - Single bar imbalance detection
/// - Stacked imbalance tracking (consecutive imbalances)
/// - Volume filter to avoid low-volume false signals
///
/// Interpretation:
/// - Buy imbalance: Aggressive buying, potential support
/// - Sell imbalance: Aggressive selling, potential resistance
/// - Stacked imbalances: Strong institutional interest
#[derive(Debug, Clone)]
pub struct ImbalanceDetector {
    config: ImbalanceDetectorConfig,
}

impl ImbalanceDetector {
    pub fn new(config: ImbalanceDetectorConfig) -> Result<Self> {
        if config.imbalance_threshold <= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "imbalance_threshold".to_string(),
                reason: "must be greater than 1.0".to_string(),
            });
        }
        if config.min_stacked < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_stacked".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if config.volume_lookback < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_lookback".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self {
            config: ImbalanceDetectorConfig::default(),
        }
    }

    /// Calculate imbalance detection with full output
    pub fn calculate_full(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> ImbalanceDetectorOutput {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut imbalance_type = vec![ImbalanceType::None; n];
        let mut imbalance_ratio = vec![1.0; n];
        let mut stacked_count = vec![0i32; n];
        let mut signal = vec![0.0; n];

        if n == 0 {
            return ImbalanceDetectorOutput {
                imbalance_type,
                imbalance_ratio,
                stacked_count,
                signal,
            };
        }

        // Calculate running average volume for filter
        let mut avg_volume = vec![0.0; n];
        let mut vol_sum = 0.0;
        for i in 0..n {
            vol_sum += volume[i];
            let count = (i + 1).min(self.config.volume_lookback);
            if i >= self.config.volume_lookback {
                vol_sum -= volume[i - self.config.volume_lookback];
            }
            avg_volume[i] = vol_sum / count as f64;
        }

        // Track consecutive imbalances
        let mut consecutive_buy = 0i32;
        let mut consecutive_sell = 0i32;

        for i in 0..n {
            let range = high[i] - low[i];

            // Volume filter
            let min_vol = avg_volume[i] * self.config.min_volume_multiplier;
            if range <= 0.0 || volume[i] < min_vol {
                consecutive_buy = 0;
                consecutive_sell = 0;
                continue;
            }

            // Estimate buy/sell volume based on close position
            let position = (close[i] - low[i]) / range;
            let buy_volume = volume[i] * position;
            let sell_volume = volume[i] * (1.0 - position);

            // Calculate imbalance ratio
            let ratio = if sell_volume > 1e-10 {
                buy_volume / sell_volume
            } else if buy_volume > 1e-10 {
                f64::INFINITY
            } else {
                1.0
            };

            let inverse_ratio = if buy_volume > 1e-10 {
                sell_volume / buy_volume
            } else if sell_volume > 1e-10 {
                f64::INFINITY
            } else {
                1.0
            };

            // Detect imbalances
            if ratio >= self.config.imbalance_threshold {
                // Buy imbalance
                consecutive_buy += 1;
                consecutive_sell = 0;
                imbalance_ratio[i] = ratio;

                if consecutive_buy >= self.config.min_stacked as i32 {
                    imbalance_type[i] = ImbalanceType::StackedBuy;
                    signal[i] = 2.0;
                } else {
                    imbalance_type[i] = ImbalanceType::BuyImbalance;
                    signal[i] = 1.0;
                }
            } else if inverse_ratio >= self.config.imbalance_threshold {
                // Sell imbalance
                consecutive_sell += 1;
                consecutive_buy = 0;
                imbalance_ratio[i] = -inverse_ratio;

                if consecutive_sell >= self.config.min_stacked as i32 {
                    imbalance_type[i] = ImbalanceType::StackedSell;
                    signal[i] = -2.0;
                } else {
                    imbalance_type[i] = ImbalanceType::SellImbalance;
                    signal[i] = -1.0;
                }
            } else {
                // No significant imbalance
                consecutive_buy = 0;
                consecutive_sell = 0;
                imbalance_ratio[i] = ratio;
            }

            stacked_count[i] = consecutive_buy - consecutive_sell;
        }

        ImbalanceDetectorOutput {
            imbalance_type,
            imbalance_ratio,
            stacked_count,
            signal,
        }
    }

    /// Calculate imbalance signal only
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        self.calculate_full(high, low, close, volume).signal
    }
}

impl TechnicalIndicator for ImbalanceDetector {
    fn name(&self) -> &str {
        "Imbalance Detector"
    }

    fn min_periods(&self) -> usize {
        self.config.volume_lookback
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.volume_lookback {
            return Err(IndicatorError::InsufficientData {
                required: self.config.volume_lookback,
                got: data.close.len(),
            });
        }

        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            output.signal,
            output.imbalance_ratio,
            output.stacked_count.iter().map(|&x| x as f64).collect(),
        ))
    }
}

impl SignalIndicator for ImbalanceDetector {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);

        if let Some(&last) = output.signal.last() {
            if last >= 2.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if last <= -2.0 {
                return Ok(IndicatorSignal::Bearish);
            } else if last > 0.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if last < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);

        Ok(output
            .signal
            .iter()
            .map(|&s| {
                if s > 0.0 {
                    IndicatorSignal::Bullish
                } else if s < 0.0 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

impl Default for ImbalanceDetector {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imbalance_detector_basic() {
        let detector = ImbalanceDetector::default_config();

        // Create data with strong buy imbalance (closes very near highs)
        let high = vec![105.0; 25];
        let low = vec![100.0; 25];
        let close = vec![104.8; 25]; // 96% position = strong buy
        let volume = vec![1000.0; 25];

        let result = detector.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 25);

        // With 96% buy and 4% sell, ratio is 24:1, should detect buy imbalance
        // After min_stacked consecutive, should see stacked signals
        assert!(result[22] > 0.0, "Should detect buy imbalance");
    }

    #[test]
    fn test_imbalance_detector_sell_imbalance() {
        let detector = ImbalanceDetector::default_config();

        // Create data with strong sell imbalance (closes very near lows)
        let high = vec![105.0; 25];
        let low = vec![100.0; 25];
        let close = vec![100.2; 25]; // 4% position = strong sell
        let volume = vec![1000.0; 25];

        let result = detector.calculate(&high, &low, &close, &volume);

        // Should detect sell imbalance
        assert!(result[22] < 0.0, "Should detect sell imbalance");
    }

    #[test]
    fn test_imbalance_detector_no_imbalance() {
        let detector = ImbalanceDetector::default_config();

        // Create data with balanced volume (closes at middle)
        let high = vec![110.0; 25];
        let low = vec![100.0; 25];
        let close = vec![105.0; 25]; // 50% position = balanced
        let volume = vec![1000.0; 25];

        let result = detector.calculate(&high, &low, &close, &volume);

        // Should not detect significant imbalance
        for &s in &result[20..] {
            assert_eq!(s, 0.0, "Should not detect imbalance at middle");
        }
    }

    #[test]
    fn test_imbalance_detector_full_output() {
        let detector = ImbalanceDetector::default_config();
        let high = vec![105.0; 25];
        let low = vec![100.0; 25];
        let close = vec![104.0; 25];
        let volume = vec![1000.0; 25];

        let output = detector.calculate_full(&high, &low, &close, &volume);

        assert_eq!(output.imbalance_type.len(), 25);
        assert_eq!(output.imbalance_ratio.len(), 25);
        assert_eq!(output.stacked_count.len(), 25);
        assert_eq!(output.signal.len(), 25);
    }

    #[test]
    fn test_imbalance_detector_stacked() {
        let config = ImbalanceDetectorConfig {
            imbalance_threshold: 2.0, // Lower threshold
            min_stacked: 3,
            volume_lookback: 10,
            min_volume_multiplier: 0.5,
        };
        let detector = ImbalanceDetector::new(config).unwrap();

        // Create data with strong buy imbalance
        let high = vec![105.0; 15];
        let low = vec![100.0; 15];
        let close = vec![104.5; 15]; // 90% = 9:1 ratio
        let volume = vec![1000.0; 15];

        let output = detector.calculate_full(&high, &low, &close, &volume);

        // After 3+ consecutive, should have stacked count
        let last_stacked = output.stacked_count[14];
        assert!(last_stacked >= 3, "Should have stacked buy imbalances");

        // Should have StackedBuy type
        assert_eq!(output.imbalance_type[14], ImbalanceType::StackedBuy);
    }

    #[test]
    fn test_imbalance_detector_invalid_config() {
        let config = ImbalanceDetectorConfig {
            imbalance_threshold: 0.5, // Invalid
            min_stacked: 3,
            volume_lookback: 20,
            min_volume_multiplier: 0.8,
        };
        assert!(ImbalanceDetector::new(config).is_err());
    }

    #[test]
    fn test_imbalance_type_enum() {
        let imb = ImbalanceType::BuyImbalance;
        assert_eq!(imb, ImbalanceType::BuyImbalance);
        assert_ne!(imb, ImbalanceType::SellImbalance);
    }

    #[test]
    fn test_imbalance_detector_low_volume_filter() {
        let config = ImbalanceDetectorConfig {
            imbalance_threshold: 3.0,
            min_stacked: 3,
            volume_lookback: 10,
            min_volume_multiplier: 2.0, // Very high filter
        };
        let detector = ImbalanceDetector::new(config).unwrap();

        // Normal volume data
        let high = vec![105.0; 15];
        let low = vec![100.0; 15];
        let close = vec![104.8; 15];
        let volume = vec![1000.0; 15];

        let result = detector.calculate(&high, &low, &close, &volume);

        // High volume filter should prevent detection
        // (average is 1000, threshold is 2000)
        for &s in &result {
            assert_eq!(s, 0.0, "High volume filter should prevent signals");
        }
    }
}
