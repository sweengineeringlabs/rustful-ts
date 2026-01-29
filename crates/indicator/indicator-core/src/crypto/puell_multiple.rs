//! Puell Multiple (Daily Issuance Value / 365-day MA) - IND-273
//!
//! Compares daily mining/staking revenue to its yearly average.
//!
//! The Puell Multiple looks at the supply side of Bitcoin's economy - miners.
//! It measures the ratio of daily issuance value to the 365-day moving average.
//!
//! Interpretation:
//! - High values (>4): Miners earning well above normal, potential market top
//! - Low values (<0.5): Miners under stress, potential market bottom
//! - Normal range (0.5-4): Healthy miner economics

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Puell Multiple calculation.
#[derive(Debug, Clone)]
pub struct PuellMultipleConfig {
    /// Period for moving average (typically 365 days).
    pub ma_period: usize,
    /// Lower threshold (stressed miners).
    pub lower_threshold: f64,
    /// Upper threshold (euphoric miners).
    pub upper_threshold: f64,
    /// Smoothing period.
    pub smooth_period: usize,
}

impl Default for PuellMultipleConfig {
    fn default() -> Self {
        Self {
            ma_period: 365,
            lower_threshold: 0.5,
            upper_threshold: 4.0,
            smooth_period: 7,
        }
    }
}

/// Puell Multiple output.
#[derive(Debug, Clone)]
pub struct PuellMultipleOutput {
    /// Daily issuance value proxy.
    pub daily_issuance: Vec<f64>,
    /// Moving average of issuance.
    pub ma_issuance: Vec<f64>,
    /// Puell Multiple values.
    pub puell: Vec<f64>,
    /// Smoothed Puell Multiple.
    pub puell_smoothed: Vec<f64>,
    /// Z-score of Puell Multiple.
    pub puell_zscore: Vec<f64>,
}

/// Puell Multiple signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PuellSignal {
    /// Extreme low - miners capitulating, potential bottom.
    ExtremeLow,
    /// Low - miners under stress.
    Low,
    /// Normal - healthy miner economics.
    Normal,
    /// High - miners profitable.
    High,
    /// Extreme high - potential market top.
    ExtremeHigh,
}

/// Puell Multiple (Daily Issuance Value / 365-day MA) - IND-273
///
/// Measures miner profitability relative to historical norms.
///
/// # Formula
/// ```text
/// Daily Issuance Value = Block Rewards * Price
/// Puell Multiple = Daily Issuance Value / SMA(Daily Issuance Value, 365)
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::{PuellMultiple, PuellMultipleConfig};
///
/// let config = PuellMultipleConfig {
///     ma_period: 90, // Use 90 days for smaller dataset
///     ..Default::default()
/// };
/// let puell = PuellMultiple::new(config).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PuellMultiple {
    config: PuellMultipleConfig,
}

impl PuellMultiple {
    /// Create a new Puell Multiple indicator.
    pub fn new(config: PuellMultipleConfig) -> Result<Self> {
        if config.ma_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "ma_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if config.lower_threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "lower_threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if config.upper_threshold <= config.lower_threshold {
            return Err(IndicatorError::InvalidParameter {
                name: "upper_threshold".to_string(),
                reason: "must be greater than lower_threshold".to_string(),
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
        Self::new(PuellMultipleConfig::default())
    }

    /// Calculate Puell Multiple metrics.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> PuellMultipleOutput {
        let n = close.len().min(volume.len());

        if n < self.config.ma_period {
            return PuellMultipleOutput {
                daily_issuance: vec![0.0; n],
                ma_issuance: vec![0.0; n],
                puell: vec![1.0; n],
                puell_smoothed: vec![1.0; n],
                puell_zscore: vec![0.0; n],
            };
        }

        let mut daily_issuance = vec![0.0; n];
        let mut ma_issuance = vec![0.0; n];
        let mut puell = vec![1.0; n];
        let mut puell_smoothed = vec![1.0; n];
        let mut puell_zscore = vec![0.0; n];

        // Calculate daily issuance proxy (volume * price as revenue proxy)
        let avg_volume: f64 = volume.iter().sum::<f64>() / n as f64;
        for i in 0..n {
            // Issuance proxy: fraction of volume as "new coins" * price
            daily_issuance[i] = (volume[i] / avg_volume * 0.005) * close[i] * avg_volume;
        }

        // Calculate moving average of issuance
        for i in (self.config.ma_period - 1)..n {
            let start = i + 1 - self.config.ma_period;
            let sum: f64 = daily_issuance[start..=i].iter().sum();
            ma_issuance[i] = sum / self.config.ma_period as f64;

            // Calculate Puell Multiple
            if ma_issuance[i] > 1e-10 {
                puell[i] = daily_issuance[i] / ma_issuance[i];
            }
        }

        // Apply smoothing
        puell_smoothed = self.apply_ema(&puell, self.config.smooth_period);

        // Calculate Z-score
        puell_zscore = self.calculate_zscore(&puell);

        PuellMultipleOutput {
            daily_issuance,
            ma_issuance,
            puell,
            puell_smoothed,
            puell_zscore,
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

    /// Calculate rolling Z-score.
    fn calculate_zscore(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];
        let period = self.config.ma_period;

        if n < period {
            return result;
        }

        for i in (period - 1)..n {
            let start = i + 1 - period;
            let window: Vec<f64> = data[start..=i]
                .iter()
                .filter(|&&x| x > 0.0 && !x.is_nan())
                .copied()
                .collect();

            if window.len() < 2 {
                continue;
            }

            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 && data[i] > 0.0 {
                result[i] = (data[i] - mean) / std_dev;
            }
        }

        result
    }

    /// Interpret Puell Multiple value.
    pub fn interpret(&self, puell_value: f64) -> PuellSignal {
        if puell_value.is_nan() || puell_value <= 0.0 {
            return PuellSignal::Normal;
        }

        let extreme_upper = self.config.upper_threshold * 1.5;
        let extreme_lower = self.config.lower_threshold * 0.5;

        if puell_value > extreme_upper {
            PuellSignal::ExtremeHigh
        } else if puell_value > self.config.upper_threshold {
            PuellSignal::High
        } else if puell_value < extreme_lower {
            PuellSignal::ExtremeLow
        } else if puell_value < self.config.lower_threshold {
            PuellSignal::Low
        } else {
            PuellSignal::Normal
        }
    }

    /// Convert signal to indicator signal.
    pub fn to_indicator_signal(&self, signal: PuellSignal) -> IndicatorSignal {
        match signal {
            PuellSignal::ExtremeHigh => IndicatorSignal::Bearish, // Sell signal
            PuellSignal::High => IndicatorSignal::Bearish,
            PuellSignal::Normal => IndicatorSignal::Neutral,
            PuellSignal::Low => IndicatorSignal::Bullish,
            PuellSignal::ExtremeLow => IndicatorSignal::Bullish, // Buy signal
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &PuellMultipleConfig {
        &self.config
    }
}

impl TechnicalIndicator for PuellMultiple {
    fn name(&self) -> &str {
        "Puell Multiple"
    }

    fn min_periods(&self) -> usize {
        self.config.ma_period + self.config.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(output.daily_issuance, output.ma_issuance, output.puell))
    }
}

impl Default for PuellMultiple {
    fn default() -> Self {
        Self::new(PuellMultipleConfig::default()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data(len: usize) -> (Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..len)
            .map(|i| 100.0 + (i as f64) * 0.2 + (i as f64 * 0.05).sin() * 10.0)
            .collect();

        let volume: Vec<f64> = (0..len)
            .map(|i| 1000.0 + (i as f64 * 0.1).sin() * 400.0)
            .collect();

        (close, volume)
    }

    #[test]
    fn test_puell_multiple_basic() {
        let config = PuellMultipleConfig {
            ma_period: 50, // Shorter for test
            ..Default::default()
        };
        let puell = PuellMultiple::new(config).unwrap();
        let (close, volume) = make_test_data(100);

        let output = puell.calculate(&close, &volume);

        assert_eq!(output.daily_issuance.len(), close.len());
        assert_eq!(output.ma_issuance.len(), close.len());
        assert_eq!(output.puell.len(), close.len());
        assert_eq!(output.puell_smoothed.len(), close.len());
    }

    #[test]
    fn test_puell_centers_around_one() {
        let config = PuellMultipleConfig {
            ma_period: 30,
            ..Default::default()
        };
        let puell = PuellMultiple::new(config).unwrap();

        // Use constant price and volume for stable test
        let close = vec![100.0; 100];
        let volume = vec![1000.0; 100];

        let output = puell.calculate(&close, &volume);

        // With constant inputs, Puell should be close to 1.0
        let puell_avg: f64 = output.puell[50..].iter().sum::<f64>()
            / (output.puell.len() - 50) as f64;
        assert!((puell_avg - 1.0).abs() < 0.1, "Puell avg: {}", puell_avg);
    }

    #[test]
    fn test_puell_signal_interpretation() {
        let puell = PuellMultiple::default();

        assert_eq!(puell.interpret(8.0), PuellSignal::ExtremeHigh);
        assert_eq!(puell.interpret(5.0), PuellSignal::High);
        assert_eq!(puell.interpret(2.0), PuellSignal::Normal);
        assert_eq!(puell.interpret(0.4), PuellSignal::Low);
        assert_eq!(puell.interpret(0.2), PuellSignal::ExtremeLow);
    }

    #[test]
    fn test_puell_signal_conversion() {
        let puell = PuellMultiple::default();

        assert_eq!(
            puell.to_indicator_signal(PuellSignal::ExtremeHigh),
            IndicatorSignal::Bearish
        );
        assert_eq!(
            puell.to_indicator_signal(PuellSignal::ExtremeLow),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            puell.to_indicator_signal(PuellSignal::Normal),
            IndicatorSignal::Neutral
        );
    }

    #[test]
    fn test_puell_validation() {
        assert!(PuellMultiple::new(PuellMultipleConfig {
            ma_period: 5, // Invalid
            lower_threshold: 0.5,
            upper_threshold: 4.0,
            smooth_period: 7,
        })
        .is_err());

        assert!(PuellMultiple::new(PuellMultipleConfig {
            ma_period: 365,
            lower_threshold: 0.5,
            upper_threshold: 0.3, // Invalid: < lower
            smooth_period: 7,
        })
        .is_err());

        assert!(PuellMultiple::new(PuellMultipleConfig {
            ma_period: 365,
            lower_threshold: -0.5, // Invalid: <= 0
            upper_threshold: 4.0,
            smooth_period: 7,
        })
        .is_err());
    }

    #[test]
    fn test_puell_empty_input() {
        let puell = PuellMultiple::default();
        let output = puell.calculate(&[], &[]);

        assert!(output.puell.is_empty());
    }

    #[test]
    fn test_puell_technical_indicator_trait() {
        let puell = PuellMultiple::default();
        assert_eq!(puell.name(), "Puell Multiple");
        assert!(puell.min_periods() > 0);
    }

    #[test]
    fn test_puell_zscore() {
        let config = PuellMultipleConfig {
            ma_period: 30,
            ..Default::default()
        };
        let puell = PuellMultiple::new(config).unwrap();
        let (close, volume) = make_test_data(100);

        let output = puell.calculate(&close, &volume);

        // Z-score should be calculated for valid periods
        let zscore_valid: Vec<&f64> = output.puell_zscore[50..]
            .iter()
            .filter(|x| !x.is_nan())
            .collect();
        assert!(!zscore_valid.is_empty());
    }
}
