//! Thermocap Ratio (Market Cap / Miner Revenue) - IND-272
//!
//! Compares market capitalization to cumulative miner revenue (thermocap).
//!
//! Thermocap = All-time sum of mining rewards (in USD)
//! This represents the "cost of security" or total investment by miners.
//!
//! The ratio indicates how many multiples of miner investment the market is valuing.
//!
//! Interpretation:
//! - High ratio (>30): Market overvalued relative to security spend
//! - Low ratio (<10): Market undervalued relative to security spend
//! - Rising ratio: Speculative phase
//! - Falling ratio: Value compression

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Thermocap Ratio calculation.
#[derive(Debug, Clone)]
pub struct ThermocapRatioConfig {
    /// Lookback period for miner revenue estimation.
    pub lookback_period: usize,
    /// Smoothing period for ratio.
    pub smooth_period: usize,
    /// Lower threshold (undervalued).
    pub lower_threshold: f64,
    /// Upper threshold (overvalued).
    pub upper_threshold: f64,
}

impl Default for ThermocapRatioConfig {
    fn default() -> Self {
        Self {
            lookback_period: 30,
            smooth_period: 14,
            lower_threshold: 10.0,
            upper_threshold: 30.0,
        }
    }
}

/// Thermocap Ratio output.
#[derive(Debug, Clone)]
pub struct ThermocapRatioOutput {
    /// Thermocap (cumulative miner revenue) proxy.
    pub thermocap: Vec<f64>,
    /// Market cap proxy.
    pub market_cap: Vec<f64>,
    /// Thermocap ratio.
    pub ratio: Vec<f64>,
    /// Smoothed ratio.
    pub ratio_smoothed: Vec<f64>,
    /// Ratio momentum.
    pub ratio_momentum: Vec<f64>,
}

/// Thermocap Ratio signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermocapSignal {
    /// Extreme overvaluation (very high ratio).
    ExtremelyOvervalued,
    /// Overvalued.
    Overvalued,
    /// Fair value.
    FairValue,
    /// Undervalued.
    Undervalued,
    /// Extreme undervaluation (very low ratio).
    ExtremelyUndervalued,
}

/// Thermocap Ratio (Market Cap / Miner Revenue) - IND-272
///
/// Measures market valuation relative to cumulative miner/validator revenue.
///
/// # Formula
/// ```text
/// Thermocap = Σ(Block Reward * Price at time of mining)
/// Thermocap Ratio = Market Cap / Thermocap
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::{ThermocapRatio, ThermocapRatioConfig};
///
/// let config = ThermocapRatioConfig::default();
/// let thermocap = ThermocapRatio::new(config).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ThermocapRatio {
    config: ThermocapRatioConfig,
}

impl ThermocapRatio {
    /// Create a new Thermocap Ratio indicator.
    pub fn new(config: ThermocapRatioConfig) -> Result<Self> {
        if config.lookback_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if config.smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if config.lower_threshold >= config.upper_threshold {
            return Err(IndicatorError::InvalidParameter {
                name: "lower_threshold".to_string(),
                reason: "must be less than upper_threshold".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration.
    pub fn default_config() -> Result<Self> {
        Self::new(ThermocapRatioConfig::default())
    }

    /// Calculate Thermocap Ratio metrics.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> ThermocapRatioOutput {
        let n = close.len().min(volume.len());

        if n < self.config.lookback_period {
            return ThermocapRatioOutput {
                thermocap: vec![0.0; n],
                market_cap: vec![0.0; n],
                ratio: vec![0.0; n],
                ratio_smoothed: vec![0.0; n],
                ratio_momentum: vec![0.0; n],
            };
        }

        let mut thermocap = vec![0.0; n];
        let mut market_cap = vec![0.0; n];
        let mut ratio = vec![0.0; n];
        let mut ratio_smoothed = vec![0.0; n];
        let mut ratio_momentum = vec![0.0; n];

        // Estimate miner revenue proxy using volume (as a proxy for network activity)
        // In practice, this would use actual block reward data
        let avg_volume: f64 = volume.iter().sum::<f64>() / n as f64;

        // Build cumulative thermocap
        let mut cumulative_thermocap = 0.0;
        for i in 0..n {
            // Miner revenue proxy: fraction of daily volume at daily price
            // This simulates block rewards being sold
            let daily_miner_revenue = (volume[i] / avg_volume * 0.01) * close[i] * avg_volume;
            cumulative_thermocap += daily_miner_revenue;
            thermocap[i] = cumulative_thermocap;

            // Market cap proxy
            market_cap[i] = close[i] * avg_volume * 1000.0; // Scale factor

            // Calculate ratio
            if thermocap[i] > 1e-10 {
                ratio[i] = market_cap[i] / thermocap[i];
            }
        }

        // Apply smoothing to ratio
        ratio_smoothed = self.apply_ema(&ratio, self.config.smooth_period);

        // Calculate momentum
        for i in self.config.smooth_period..n {
            let prev = ratio_smoothed[i - self.config.smooth_period];
            if prev > 1e-10 {
                ratio_momentum[i] = (ratio_smoothed[i] / prev - 1.0) * 100.0;
            }
        }

        ThermocapRatioOutput {
            thermocap,
            market_cap,
            ratio,
            ratio_smoothed,
            ratio_momentum,
        }
    }

    /// Apply EMA smoothing.
    fn apply_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = data.to_vec();

        if n == 0 || period <= 1 {
            return result;
        }

        let alpha = 2.0 / (period as f64 + 1.0);

        for i in 1..n {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Interpret signal based on ratio value.
    pub fn interpret(&self, ratio: f64) -> ThermocapSignal {
        if ratio.is_nan() || ratio <= 0.0 {
            return ThermocapSignal::FairValue;
        }

        let extreme_upper = self.config.upper_threshold * 1.5;
        let extreme_lower = self.config.lower_threshold * 0.5;

        if ratio > extreme_upper {
            ThermocapSignal::ExtremelyOvervalued
        } else if ratio > self.config.upper_threshold {
            ThermocapSignal::Overvalued
        } else if ratio < extreme_lower {
            ThermocapSignal::ExtremelyUndervalued
        } else if ratio < self.config.lower_threshold {
            ThermocapSignal::Undervalued
        } else {
            ThermocapSignal::FairValue
        }
    }

    /// Convert signal to indicator signal.
    pub fn to_indicator_signal(&self, signal: ThermocapSignal) -> IndicatorSignal {
        match signal {
            ThermocapSignal::ExtremelyOvervalued => IndicatorSignal::Bearish,
            ThermocapSignal::Overvalued => IndicatorSignal::Bearish,
            ThermocapSignal::FairValue => IndicatorSignal::Neutral,
            ThermocapSignal::Undervalued => IndicatorSignal::Bullish,
            ThermocapSignal::ExtremelyUndervalued => IndicatorSignal::Bullish,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &ThermocapRatioConfig {
        &self.config
    }
}

impl TechnicalIndicator for ThermocapRatio {
    fn name(&self) -> &str {
        "Thermocap Ratio"
    }

    fn min_periods(&self) -> usize {
        self.config.lookback_period + self.config.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(output.thermocap, output.market_cap, output.ratio))
    }
}

impl Default for ThermocapRatio {
    fn default() -> Self {
        Self::new(ThermocapRatioConfig::default()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.1).sin() * 5.0)
            .collect();

        let volume: Vec<f64> = (0..100)
            .map(|i| 1000.0 + (i as f64 * 0.15).sin() * 300.0)
            .collect();

        (close, volume)
    }

    #[test]
    fn test_thermocap_ratio_basic() {
        let thermocap = ThermocapRatio::default();
        let (close, volume) = make_test_data();

        let output = thermocap.calculate(&close, &volume);

        assert_eq!(output.thermocap.len(), close.len());
        assert_eq!(output.market_cap.len(), close.len());
        assert_eq!(output.ratio.len(), close.len());
        assert_eq!(output.ratio_smoothed.len(), close.len());
    }

    #[test]
    fn test_thermocap_is_cumulative() {
        let thermocap = ThermocapRatio::default();
        let (close, volume) = make_test_data();

        let output = thermocap.calculate(&close, &volume);

        // Thermocap should be monotonically increasing
        for i in 1..output.thermocap.len() {
            assert!(
                output.thermocap[i] >= output.thermocap[i - 1],
                "Thermocap should be cumulative"
            );
        }
    }

    #[test]
    fn test_thermocap_signal_interpretation() {
        let thermocap = ThermocapRatio::default();

        assert_eq!(
            thermocap.interpret(50.0),
            ThermocapSignal::ExtremelyOvervalued
        );
        assert_eq!(thermocap.interpret(35.0), ThermocapSignal::Overvalued);
        assert_eq!(thermocap.interpret(20.0), ThermocapSignal::FairValue);
        assert_eq!(thermocap.interpret(8.0), ThermocapSignal::Undervalued);
        assert_eq!(
            thermocap.interpret(3.0),
            ThermocapSignal::ExtremelyUndervalued
        );
    }

    #[test]
    fn test_thermocap_signal_conversion() {
        let thermocap = ThermocapRatio::default();

        assert_eq!(
            thermocap.to_indicator_signal(ThermocapSignal::ExtremelyOvervalued),
            IndicatorSignal::Bearish
        );
        assert_eq!(
            thermocap.to_indicator_signal(ThermocapSignal::ExtremelyUndervalued),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            thermocap.to_indicator_signal(ThermocapSignal::FairValue),
            IndicatorSignal::Neutral
        );
    }

    #[test]
    fn test_thermocap_validation() {
        assert!(ThermocapRatio::new(ThermocapRatioConfig {
            lookback_period: 2, // Invalid
            smooth_period: 14,
            lower_threshold: 10.0,
            upper_threshold: 30.0,
        })
        .is_err());

        assert!(ThermocapRatio::new(ThermocapRatioConfig {
            lookback_period: 30,
            smooth_period: 14,
            lower_threshold: 40.0, // Invalid: >= upper
            upper_threshold: 30.0,
        })
        .is_err());
    }

    #[test]
    fn test_thermocap_empty_input() {
        let thermocap = ThermocapRatio::default();
        let output = thermocap.calculate(&[], &[]);

        assert!(output.thermocap.is_empty());
    }

    #[test]
    fn test_thermocap_technical_indicator_trait() {
        let thermocap = ThermocapRatio::default();
        assert_eq!(thermocap.name(), "Thermocap Ratio");
        assert!(thermocap.min_periods() > 0);
    }

    #[test]
    fn test_thermocap_ratio_momentum() {
        let thermocap = ThermocapRatio::default();
        let (close, volume) = make_test_data();

        let output = thermocap.calculate(&close, &volume);

        // Momentum should be calculated after warmup
        let momentum_sum: f64 = output.ratio_momentum[50..].iter().sum();
        assert!(!momentum_sum.is_nan());
    }
}
