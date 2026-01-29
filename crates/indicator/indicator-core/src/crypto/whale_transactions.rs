//! Whale Transactions (Large Transfers Count) - IND-269
//!
//! Tracks the number and volume of large cryptocurrency transfers.
//!
//! Whale transactions are large value transfers that can indicate:
//! - Institutional movements
//! - Exchange deposits/withdrawals
//! - OTC trades
//! - Market manipulation attempts
//!
//! Interpretation:
//! - High whale activity with price increase: Accumulation
//! - High whale activity with price decrease: Distribution
//! - Low whale activity: Consolidation phase

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for whale transaction detection.
#[derive(Debug, Clone)]
pub struct WhaleTransactionsConfig {
    /// Threshold for whale transaction detection (as multiple of average).
    pub whale_threshold: f64,
    /// Period for calculating average transaction size.
    pub lookback_period: usize,
    /// Smoothing period for whale activity index.
    pub smooth_period: usize,
}

impl Default for WhaleTransactionsConfig {
    fn default() -> Self {
        Self {
            whale_threshold: 3.0, // 3x average = whale
            lookback_period: 30,
            smooth_period: 7,
        }
    }
}

/// Whale Transactions output.
#[derive(Debug, Clone)]
pub struct WhaleTransactionsOutput {
    /// Whale activity index (0-100 scale).
    pub whale_index: Vec<f64>,
    /// Estimated whale count (normalized).
    pub whale_count: Vec<f64>,
    /// Whale volume ratio (whale volume / total volume).
    pub whale_volume_ratio: Vec<f64>,
    /// Net whale direction: positive = accumulation, negative = distribution.
    pub net_whale_flow: Vec<f64>,
}

/// Whale transaction signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhaleSignal {
    /// Heavy whale accumulation detected.
    HeavyAccumulation,
    /// Moderate whale accumulation.
    Accumulation,
    /// Normal whale activity.
    Neutral,
    /// Moderate whale distribution.
    Distribution,
    /// Heavy whale distribution detected.
    HeavyDistribution,
}

/// Whale Transactions (Large Transfers Count) - IND-269
///
/// Monitors large cryptocurrency transfers to detect whale activity.
///
/// # Formula
/// ```text
/// Whale Index = EMA(Whale Volume / Total Volume, smooth_period) * 100
/// Whale Count = Count(Volume > Threshold * Average Volume)
/// Net Flow = Σ(Whale Volume * Price Direction)
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::{WhaleTransactions, WhaleTransactionsConfig};
///
/// let config = WhaleTransactionsConfig::default();
/// let whale = WhaleTransactions::new(config).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct WhaleTransactions {
    config: WhaleTransactionsConfig,
}

impl WhaleTransactions {
    /// Create a new Whale Transactions indicator.
    pub fn new(config: WhaleTransactionsConfig) -> Result<Self> {
        if config.whale_threshold < 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "whale_threshold".to_string(),
                reason: "must be at least 1.0".to_string(),
            });
        }
        if config.lookback_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if config.smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration.
    pub fn default_config() -> Result<Self> {
        Self::new(WhaleTransactionsConfig::default())
    }

    /// Calculate whale transaction metrics.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> WhaleTransactionsOutput {
        let n = close.len().min(volume.len());

        if n < self.config.lookback_period {
            return WhaleTransactionsOutput {
                whale_index: vec![0.0; n],
                whale_count: vec![0.0; n],
                whale_volume_ratio: vec![0.0; n],
                net_whale_flow: vec![0.0; n],
            };
        }

        let mut whale_index = vec![0.0; n];
        let mut whale_count = vec![0.0; n];
        let mut whale_volume_ratio = vec![0.0; n];
        let mut net_whale_flow = vec![0.0; n];

        for i in self.config.lookback_period..n {
            let start = i - self.config.lookback_period;

            // Calculate average volume over lookback period
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>()
                / self.config.lookback_period as f64;

            // Whale threshold
            let whale_thresh = avg_volume * self.config.whale_threshold;

            // Count whale transactions and volume
            let mut w_count = 0.0;
            let mut w_volume = 0.0;
            let mut total_vol = 0.0;
            let mut w_flow = 0.0;

            for j in start..=i {
                total_vol += volume[j];

                if volume[j] > whale_thresh {
                    w_count += 1.0;
                    w_volume += volume[j];

                    // Determine direction based on price change
                    let direction = if j > 0 {
                        (close[j] - close[j - 1]).signum()
                    } else {
                        0.0
                    };
                    w_flow += volume[j] * direction;
                }
            }

            // Whale volume ratio
            if total_vol > 1e-10 {
                whale_volume_ratio[i] = w_volume / total_vol;
            }

            // Whale count (normalized by period)
            whale_count[i] = w_count / self.config.lookback_period as f64 * 100.0;

            // Net whale flow (normalized)
            if w_volume > 1e-10 {
                net_whale_flow[i] = w_flow / w_volume * 100.0;
            }
        }

        // Calculate whale index with smoothing
        whale_index = self.apply_ema(&whale_volume_ratio, self.config.smooth_period);
        for i in 0..n {
            whale_index[i] *= 100.0;
        }

        WhaleTransactionsOutput {
            whale_index,
            whale_count,
            whale_volume_ratio,
            net_whale_flow,
        }
    }

    /// Apply EMA smoothing.
    fn apply_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        if n == 0 || period == 0 {
            return result;
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        result[0] = data[0];

        for i in 1..n {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Interpret whale activity.
    pub fn interpret(&self, whale_index: f64, net_flow: f64) -> WhaleSignal {
        if whale_index.is_nan() || net_flow.is_nan() {
            return WhaleSignal::Neutral;
        }

        // High whale activity
        if whale_index > 30.0 {
            if net_flow > 30.0 {
                WhaleSignal::HeavyAccumulation
            } else if net_flow > 10.0 {
                WhaleSignal::Accumulation
            } else if net_flow < -30.0 {
                WhaleSignal::HeavyDistribution
            } else if net_flow < -10.0 {
                WhaleSignal::Distribution
            } else {
                WhaleSignal::Neutral
            }
        } else if whale_index > 15.0 {
            if net_flow > 20.0 {
                WhaleSignal::Accumulation
            } else if net_flow < -20.0 {
                WhaleSignal::Distribution
            } else {
                WhaleSignal::Neutral
            }
        } else {
            WhaleSignal::Neutral
        }
    }

    /// Convert whale signal to indicator signal.
    pub fn to_indicator_signal(&self, signal: WhaleSignal) -> IndicatorSignal {
        match signal {
            WhaleSignal::HeavyAccumulation => IndicatorSignal::Bullish,
            WhaleSignal::Accumulation => IndicatorSignal::Bullish,
            WhaleSignal::Neutral => IndicatorSignal::Neutral,
            WhaleSignal::Distribution => IndicatorSignal::Bearish,
            WhaleSignal::HeavyDistribution => IndicatorSignal::Bearish,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &WhaleTransactionsConfig {
        &self.config
    }
}

impl TechnicalIndicator for WhaleTransactions {
    fn name(&self) -> &str {
        "Whale Transactions"
    }

    fn min_periods(&self) -> usize {
        self.config.lookback_period + self.config.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(output.whale_index, output.whale_count, output.whale_volume_ratio))
    }
}

impl Default for WhaleTransactions {
    fn default() -> Self {
        Self::new(WhaleTransactionsConfig::default()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        // Simulate price and volume data with some whale-like spikes
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.2).sin() * 3.0)
            .collect();

        let mut volume: Vec<f64> = vec![1000.0; 50];
        // Add whale spikes
        volume[15] = 5000.0; // Whale transaction
        volume[25] = 6000.0; // Whale transaction
        volume[35] = 4500.0; // Whale transaction

        (close, volume)
    }

    #[test]
    fn test_whale_transactions_basic() {
        let whale = WhaleTransactions::default();
        let (close, volume) = make_test_data();

        let output = whale.calculate(&close, &volume);

        assert_eq!(output.whale_index.len(), close.len());
        assert_eq!(output.whale_count.len(), close.len());
        assert_eq!(output.whale_volume_ratio.len(), close.len());
        assert_eq!(output.net_whale_flow.len(), close.len());
    }

    #[test]
    fn test_whale_detection() {
        let whale = WhaleTransactions::default();
        let (close, volume) = make_test_data();

        let output = whale.calculate(&close, &volume);

        // After whale transactions, whale index should increase
        // Check that whale activity is detected
        let post_whale = output.whale_index[36..40].iter().sum::<f64>();
        assert!(post_whale > 0.0);
    }

    #[test]
    fn test_whale_interpretation() {
        let whale = WhaleTransactions::default();

        assert_eq!(
            whale.interpret(50.0, 50.0),
            WhaleSignal::HeavyAccumulation
        );
        assert_eq!(
            whale.interpret(50.0, -50.0),
            WhaleSignal::HeavyDistribution
        );
        assert_eq!(whale.interpret(10.0, 0.0), WhaleSignal::Neutral);
    }

    #[test]
    fn test_whale_signal_conversion() {
        let whale = WhaleTransactions::default();

        assert_eq!(
            whale.to_indicator_signal(WhaleSignal::HeavyAccumulation),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            whale.to_indicator_signal(WhaleSignal::HeavyDistribution),
            IndicatorSignal::Bearish
        );
        assert_eq!(
            whale.to_indicator_signal(WhaleSignal::Neutral),
            IndicatorSignal::Neutral
        );
    }

    #[test]
    fn test_whale_validation() {
        assert!(WhaleTransactions::new(WhaleTransactionsConfig {
            whale_threshold: 0.5, // Invalid
            lookback_period: 30,
            smooth_period: 7,
        })
        .is_err());

        assert!(WhaleTransactions::new(WhaleTransactionsConfig {
            whale_threshold: 3.0,
            lookback_period: 2, // Invalid
            smooth_period: 7,
        })
        .is_err());
    }

    #[test]
    fn test_whale_empty_input() {
        let whale = WhaleTransactions::default();
        let output = whale.calculate(&[], &[]);

        assert!(output.whale_index.is_empty());
    }

    #[test]
    fn test_whale_technical_indicator_trait() {
        let whale = WhaleTransactions::default();
        assert_eq!(whale.name(), "Whale Transactions");
        assert!(whale.min_periods() > 0);
    }
}
