//! HODL Waves (Coin Age Distribution) - IND-270
//!
//! Tracks the distribution of coins by their age (time since last movement).
//!
//! HODL Waves visualize the percentage of circulating supply grouped by age bands:
//! - Young coins (< 1 month): Recently moved, active traders
//! - Medium coins (1-6 months): Medium-term holders
//! - Old coins (6+ months): Long-term holders (HODLers)
//!
//! Interpretation:
//! - Increasing young coins: Distribution/profit-taking phase
//! - Increasing old coins: Accumulation/conviction phase
//! - Old coins decreasing rapidly: Long-term holders capitulating

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for HODL Waves analysis.
#[derive(Debug, Clone)]
pub struct HODLWavesConfig {
    /// Short-term age boundary (in periods, represents young coins).
    pub short_term_periods: usize,
    /// Medium-term age boundary (in periods).
    pub medium_term_periods: usize,
    /// Lookback for trend detection.
    pub trend_lookback: usize,
}

impl Default for HODLWavesConfig {
    fn default() -> Self {
        Self {
            short_term_periods: 30,   // ~1 month
            medium_term_periods: 180, // ~6 months
            trend_lookback: 14,
        }
    }
}

/// HODL Waves output.
#[derive(Debug, Clone)]
pub struct HODLWavesOutput {
    /// Young coins percentage proxy (0-100).
    pub young_coins: Vec<f64>,
    /// Medium-term coins percentage proxy (0-100).
    pub medium_coins: Vec<f64>,
    /// Old coins percentage proxy (0-100).
    pub old_coins: Vec<f64>,
    /// HODL wave score: positive = accumulation, negative = distribution.
    pub hodl_score: Vec<f64>,
    /// Supply stability index (0-100).
    pub stability_index: Vec<f64>,
}

/// HODL Waves phase interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HODLPhase {
    /// Strong accumulation - old coins increasing.
    StrongAccumulation,
    /// Moderate accumulation.
    Accumulation,
    /// Consolidation - stable distribution.
    Consolidation,
    /// Distribution - young coins increasing.
    Distribution,
    /// Capitulation - old coins decreasing rapidly.
    Capitulation,
}

/// HODL Waves (Coin Age Distribution) - IND-270
///
/// Estimates coin age distribution using price and volume patterns.
///
/// # Formula
/// ```text
/// Young Coins Proxy = High Turnover Periods / Total
/// Old Coins Proxy = Low Turnover Periods / Total
/// HODL Score = Old Coins Change - Young Coins Change
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::{HODLWaves, HODLWavesConfig};
///
/// let config = HODLWavesConfig::default();
/// let hodl = HODLWaves::new(config).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct HODLWaves {
    config: HODLWavesConfig,
}

impl HODLWaves {
    /// Create a new HODL Waves indicator.
    pub fn new(config: HODLWavesConfig) -> Result<Self> {
        if config.short_term_periods < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_term_periods".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if config.medium_term_periods <= config.short_term_periods {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_term_periods".to_string(),
                reason: "must be greater than short_term_periods".to_string(),
            });
        }
        if config.trend_lookback < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_lookback".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration.
    pub fn default_config() -> Result<Self> {
        Self::new(HODLWavesConfig::default())
    }

    /// Calculate HODL Waves metrics.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> HODLWavesOutput {
        let n = close.len().min(volume.len());

        if n < self.config.medium_term_periods {
            return HODLWavesOutput {
                young_coins: vec![0.0; n],
                medium_coins: vec![0.0; n],
                old_coins: vec![0.0; n],
                hodl_score: vec![0.0; n],
                stability_index: vec![50.0; n],
            };
        }

        let mut young_coins = vec![0.0; n];
        let mut medium_coins = vec![0.0; n];
        let mut old_coins = vec![0.0; n];
        let mut hodl_score = vec![0.0; n];
        let mut stability_index = vec![50.0; n];

        // Calculate turnover rates to proxy coin age
        let turnover = self.calculate_turnover(close, volume);

        for i in self.config.medium_term_periods..n {
            // Calculate age distribution proxy based on turnover patterns
            let short_start = i - self.config.short_term_periods;
            let medium_start = i - self.config.medium_term_periods;

            // Average turnover in different periods
            let short_turnover: f64 = turnover[short_start..=i].iter().sum::<f64>()
                / self.config.short_term_periods as f64;

            let medium_turnover: f64 = turnover[medium_start..short_start].iter().sum::<f64>()
                / (self.config.medium_term_periods - self.config.short_term_periods) as f64;

            // Higher turnover = more young coins (recently moved)
            // Lower turnover = more old coins (not moved)
            let total_activity = short_turnover + medium_turnover + 1e-10;

            // Young coins proxy: high recent turnover indicates active trading
            young_coins[i] = (short_turnover / total_activity * 60.0).min(100.0);

            // Medium coins: moderate activity
            medium_coins[i] = (medium_turnover / total_activity * 30.0).min(100.0);

            // Old coins: inverse of activity (what's not being traded)
            old_coins[i] = (100.0 - young_coins[i] - medium_coins[i]).max(0.0);

            // Normalize to 100%
            let total = young_coins[i] + medium_coins[i] + old_coins[i];
            if total > 1e-10 {
                young_coins[i] = young_coins[i] / total * 100.0;
                medium_coins[i] = medium_coins[i] / total * 100.0;
                old_coins[i] = old_coins[i] / total * 100.0;
            }

            // HODL Score: positive when old coins increasing
            if i >= self.config.medium_term_periods + self.config.trend_lookback {
                let old_change = old_coins[i] - old_coins[i - self.config.trend_lookback];
                let young_change = young_coins[i] - young_coins[i - self.config.trend_lookback];
                hodl_score[i] = old_change - young_change;
            }

            // Stability index: lower volatility in distribution = more stable
            let dist_vol = self.calculate_distribution_volatility(
                &young_coins,
                &medium_coins,
                &old_coins,
                i,
            );
            stability_index[i] = (100.0 - dist_vol * 10.0).max(0.0).min(100.0);
        }

        HODLWavesOutput {
            young_coins,
            medium_coins,
            old_coins,
            hodl_score,
            stability_index,
        }
    }

    /// Calculate turnover rate proxy.
    fn calculate_turnover(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut turnover = vec![0.0; n];

        for i in 1..n {
            // Turnover proxy: volume relative to price change
            let price_range = (close[i] - close[i - 1]).abs() / close[i - 1];
            turnover[i] = volume[i] * (1.0 + price_range);
        }

        // Normalize turnover
        let max_turnover = turnover.iter().cloned().fold(1e-10, f64::max);
        for t in turnover.iter_mut() {
            *t /= max_turnover;
        }

        turnover
    }

    /// Calculate volatility in distribution.
    fn calculate_distribution_volatility(
        &self,
        young: &[f64],
        medium: &[f64],
        old: &[f64],
        index: usize,
    ) -> f64 {
        let lookback = self.config.trend_lookback.min(index);
        if lookback < 2 {
            return 0.0;
        }

        let start = index - lookback;

        // Calculate variance of each category
        let young_var = self.variance(&young[start..=index]);
        let medium_var = self.variance(&medium[start..=index]);
        let old_var = self.variance(&old[start..=index]);

        (young_var + medium_var + old_var).sqrt()
    }

    /// Calculate variance.
    fn variance(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
    }

    /// Interpret HODL phase.
    pub fn interpret(&self, hodl_score: f64, stability: f64, old_coins: f64) -> HODLPhase {
        if hodl_score.is_nan() {
            return HODLPhase::Consolidation;
        }

        if hodl_score > 5.0 && old_coins > 40.0 {
            HODLPhase::StrongAccumulation
        } else if hodl_score > 2.0 {
            HODLPhase::Accumulation
        } else if hodl_score < -5.0 && stability < 30.0 {
            HODLPhase::Capitulation
        } else if hodl_score < -2.0 {
            HODLPhase::Distribution
        } else {
            HODLPhase::Consolidation
        }
    }

    /// Convert HODL phase to indicator signal.
    pub fn to_indicator_signal(&self, phase: HODLPhase) -> IndicatorSignal {
        match phase {
            HODLPhase::StrongAccumulation => IndicatorSignal::Bullish,
            HODLPhase::Accumulation => IndicatorSignal::Bullish,
            HODLPhase::Consolidation => IndicatorSignal::Neutral,
            HODLPhase::Distribution => IndicatorSignal::Bearish,
            HODLPhase::Capitulation => IndicatorSignal::Bearish,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &HODLWavesConfig {
        &self.config
    }
}

impl TechnicalIndicator for HODLWaves {
    fn name(&self) -> &str {
        "HODL Waves"
    }

    fn min_periods(&self) -> usize {
        self.config.medium_term_periods + self.config.trend_lookback
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(output.young_coins, output.medium_coins, output.old_coins))
    }
}

impl Default for HODLWaves {
    fn default() -> Self {
        Self::new(HODLWavesConfig::default()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..250)
            .map(|i| 100.0 + (i as f64) * 0.3 + (i as f64 * 0.1).sin() * 5.0)
            .collect();

        let volume: Vec<f64> = (0..250)
            .map(|i| 1000.0 + (i as f64 * 0.15).sin() * 500.0)
            .collect();

        (close, volume)
    }

    #[test]
    fn test_hodl_waves_basic() {
        let hodl = HODLWaves::default();
        let (close, volume) = make_test_data();

        let output = hodl.calculate(&close, &volume);

        assert_eq!(output.young_coins.len(), close.len());
        assert_eq!(output.medium_coins.len(), close.len());
        assert_eq!(output.old_coins.len(), close.len());
        assert_eq!(output.hodl_score.len(), close.len());
    }

    #[test]
    fn test_hodl_distribution_sums_to_100() {
        let hodl = HODLWaves::default();
        let (close, volume) = make_test_data();

        let output = hodl.calculate(&close, &volume);

        // Check that distribution sums to 100% (with tolerance)
        for i in 200..output.young_coins.len() {
            let total = output.young_coins[i] + output.medium_coins[i] + output.old_coins[i];
            assert!((total - 100.0).abs() < 1.0, "Total: {}", total);
        }
    }

    #[test]
    fn test_hodl_phase_interpretation() {
        let hodl = HODLWaves::default();

        assert_eq!(
            hodl.interpret(10.0, 80.0, 50.0),
            HODLPhase::StrongAccumulation
        );
        assert_eq!(
            hodl.interpret(3.0, 60.0, 30.0),
            HODLPhase::Accumulation
        );
        assert_eq!(hodl.interpret(0.0, 50.0, 33.0), HODLPhase::Consolidation);
        assert_eq!(hodl.interpret(-3.0, 50.0, 25.0), HODLPhase::Distribution);
        assert_eq!(hodl.interpret(-10.0, 20.0, 20.0), HODLPhase::Capitulation);
    }

    #[test]
    fn test_hodl_signal_conversion() {
        let hodl = HODLWaves::default();

        assert_eq!(
            hodl.to_indicator_signal(HODLPhase::StrongAccumulation),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            hodl.to_indicator_signal(HODLPhase::Capitulation),
            IndicatorSignal::Bearish
        );
        assert_eq!(
            hodl.to_indicator_signal(HODLPhase::Consolidation),
            IndicatorSignal::Neutral
        );
    }

    #[test]
    fn test_hodl_validation() {
        assert!(HODLWaves::new(HODLWavesConfig {
            short_term_periods: 2, // Invalid
            medium_term_periods: 180,
            trend_lookback: 14,
        })
        .is_err());

        assert!(HODLWaves::new(HODLWavesConfig {
            short_term_periods: 30,
            medium_term_periods: 20, // Invalid: <= short_term
            trend_lookback: 14,
        })
        .is_err());
    }

    #[test]
    fn test_hodl_empty_input() {
        let hodl = HODLWaves::default();
        let output = hodl.calculate(&[], &[]);

        assert!(output.young_coins.is_empty());
    }

    #[test]
    fn test_hodl_technical_indicator_trait() {
        let hodl = HODLWaves::default();
        assert_eq!(hodl.name(), "HODL Waves");
        assert!(hodl.min_periods() > 0);
    }
}
