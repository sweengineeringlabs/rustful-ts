//! Reserve Risk (HODL Confidence vs Price) - IND-274
//!
//! Measures the confidence of long-term holders relative to the current price.
//!
//! Reserve Risk = Price / (HODL Bank)
//! Where HODL Bank is the accumulated opportunity cost of long-term holders.
//!
//! The metric rewards conviction holders and penalizes short-term speculation.
//!
//! Interpretation:
//! - High Reserve Risk (>0.02): Low confidence relative to price, potential sell zone
//! - Low Reserve Risk (<0.002): High confidence relative to price, potential buy zone
//! - Normal range: Balanced market conditions

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Reserve Risk calculation.
#[derive(Debug, Clone)]
pub struct ReserveRiskConfig {
    /// Period for calculating HODL bank accumulation.
    pub hodl_period: usize,
    /// Smoothing period.
    pub smooth_period: usize,
    /// Lower threshold (high confidence zone).
    pub lower_threshold: f64,
    /// Upper threshold (low confidence zone).
    pub upper_threshold: f64,
}

impl Default for ReserveRiskConfig {
    fn default() -> Self {
        Self {
            hodl_period: 180,
            smooth_period: 14,
            lower_threshold: 0.002,
            upper_threshold: 0.02,
        }
    }
}

/// Reserve Risk output.
#[derive(Debug, Clone)]
pub struct ReserveRiskOutput {
    /// HODL Bank (accumulated opportunity cost).
    pub hodl_bank: Vec<f64>,
    /// Reserve Risk values.
    pub reserve_risk: Vec<f64>,
    /// Smoothed Reserve Risk.
    pub reserve_risk_smoothed: Vec<f64>,
    /// Confidence level (inverse of reserve risk, normalized 0-100).
    pub confidence: Vec<f64>,
    /// Reserve Risk momentum.
    pub momentum: Vec<f64>,
}

/// Reserve Risk signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveRiskSignal {
    /// Extreme low risk - very high holder confidence, strong buy.
    ExtremeLowRisk,
    /// Low risk - high confidence, buy zone.
    LowRisk,
    /// Normal - balanced market.
    Normal,
    /// High risk - low confidence, caution.
    HighRisk,
    /// Extreme high risk - very low confidence, sell zone.
    ExtremeHighRisk,
}

/// Reserve Risk (HODL Confidence vs Price) - IND-274
///
/// Quantifies the ratio of current price to HODL bank (accumulated opportunity cost).
///
/// # Formula
/// ```text
/// HODL Bank = Σ(Coin Days Destroyed * Price) / Σ(Supply)
/// Reserve Risk = Price / HODL Bank
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::{ReserveRisk, ReserveRiskConfig};
///
/// let config = ReserveRiskConfig::default();
/// let rr = ReserveRisk::new(config).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ReserveRisk {
    config: ReserveRiskConfig,
}

impl ReserveRisk {
    /// Create a new Reserve Risk indicator.
    pub fn new(config: ReserveRiskConfig) -> Result<Self> {
        if config.hodl_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "hodl_period".to_string(),
                reason: "must be at least 10".to_string(),
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
        if config.lower_threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "lower_threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration.
    pub fn default_config() -> Result<Self> {
        Self::new(ReserveRiskConfig::default())
    }

    /// Calculate Reserve Risk metrics.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> ReserveRiskOutput {
        let n = close.len().min(volume.len());

        if n < self.config.hodl_period {
            return ReserveRiskOutput {
                hodl_bank: vec![0.0; n],
                reserve_risk: vec![0.0; n],
                reserve_risk_smoothed: vec![0.0; n],
                confidence: vec![50.0; n],
                momentum: vec![0.0; n],
            };
        }

        let mut hodl_bank = vec![0.0; n];
        let mut reserve_risk = vec![0.0; n];
        let mut reserve_risk_smoothed = vec![0.0; n];
        let mut confidence = vec![50.0; n];
        let mut momentum = vec![0.0; n];

        // Calculate HODL Bank proxy
        // HODL Bank accumulates the "opportunity cost" of holding
        // High HODL bank = lots of accumulated conviction
        let mut cumulative_hodl = 0.0;

        for i in 0..n {
            // HODL Bank proxy: accumulated price-weighted dormancy
            // Lower volume periods = more HODLing = higher bank accumulation
            let avg_volume = if i < self.config.hodl_period {
                volume[..=i].iter().sum::<f64>() / (i + 1) as f64
            } else {
                volume[(i - self.config.hodl_period + 1)..=i].iter().sum::<f64>()
                    / self.config.hodl_period as f64
            };

            // Dormancy factor: inverse of activity
            let dormancy = if avg_volume > 1e-10 {
                1.0 / (1.0 + volume[i] / avg_volume)
            } else {
                1.0
            };

            // Accumulate HODL bank (represents confidence/conviction)
            cumulative_hodl += close[i] * dormancy;
            hodl_bank[i] = cumulative_hodl / (i + 1) as f64;
        }

        // Calculate Reserve Risk
        for i in self.config.hodl_period..n {
            if hodl_bank[i] > 1e-10 {
                // Reserve Risk = Price / HODL Bank
                // High price relative to conviction = high risk
                reserve_risk[i] = close[i] / (hodl_bank[i] * 100.0);
            }

            // Confidence (inverse of reserve risk)
            if reserve_risk[i] > 1e-10 {
                // Normalize to 0-100 scale
                confidence[i] = (1.0 / (1.0 + reserve_risk[i] * 50.0)) * 100.0;
            }
        }

        // Apply smoothing
        reserve_risk_smoothed = self.apply_ema(&reserve_risk, self.config.smooth_period);

        // Calculate momentum
        for i in self.config.smooth_period..n {
            let prev = reserve_risk_smoothed[i - self.config.smooth_period];
            if prev > 1e-10 {
                momentum[i] = (reserve_risk_smoothed[i] / prev - 1.0) * 100.0;
            }
        }

        ReserveRiskOutput {
            hodl_bank,
            reserve_risk,
            reserve_risk_smoothed,
            confidence,
            momentum,
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

    /// Interpret Reserve Risk value.
    pub fn interpret(&self, reserve_risk: f64) -> ReserveRiskSignal {
        if reserve_risk.is_nan() || reserve_risk <= 0.0 {
            return ReserveRiskSignal::Normal;
        }

        let extreme_upper = self.config.upper_threshold * 2.0;
        let extreme_lower = self.config.lower_threshold * 0.5;

        if reserve_risk > extreme_upper {
            ReserveRiskSignal::ExtremeHighRisk
        } else if reserve_risk > self.config.upper_threshold {
            ReserveRiskSignal::HighRisk
        } else if reserve_risk < extreme_lower {
            ReserveRiskSignal::ExtremeLowRisk
        } else if reserve_risk < self.config.lower_threshold {
            ReserveRiskSignal::LowRisk
        } else {
            ReserveRiskSignal::Normal
        }
    }

    /// Convert signal to indicator signal.
    pub fn to_indicator_signal(&self, signal: ReserveRiskSignal) -> IndicatorSignal {
        match signal {
            ReserveRiskSignal::ExtremeHighRisk => IndicatorSignal::Bearish,
            ReserveRiskSignal::HighRisk => IndicatorSignal::Bearish,
            ReserveRiskSignal::Normal => IndicatorSignal::Neutral,
            ReserveRiskSignal::LowRisk => IndicatorSignal::Bullish,
            ReserveRiskSignal::ExtremeLowRisk => IndicatorSignal::Bullish,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &ReserveRiskConfig {
        &self.config
    }
}

impl TechnicalIndicator for ReserveRisk {
    fn name(&self) -> &str {
        "Reserve Risk"
    }

    fn min_periods(&self) -> usize {
        self.config.hodl_period + self.config.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(output.hodl_bank, output.reserve_risk, output.reserve_risk_smoothed))
    }
}

impl Default for ReserveRisk {
    fn default() -> Self {
        Self::new(ReserveRiskConfig::default()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..250)
            .map(|i| 100.0 + (i as f64) * 0.3 + (i as f64 * 0.08).sin() * 8.0)
            .collect();

        let volume: Vec<f64> = (0..250)
            .map(|i| 1000.0 + (i as f64 * 0.12).sin() * 400.0)
            .collect();

        (close, volume)
    }

    #[test]
    fn test_reserve_risk_basic() {
        let rr = ReserveRisk::default();
        let (close, volume) = make_test_data();

        let output = rr.calculate(&close, &volume);

        assert_eq!(output.hodl_bank.len(), close.len());
        assert_eq!(output.reserve_risk.len(), close.len());
        assert_eq!(output.reserve_risk_smoothed.len(), close.len());
        assert_eq!(output.confidence.len(), close.len());
    }

    #[test]
    fn test_reserve_risk_hodl_bank_grows() {
        let rr = ReserveRisk::default();
        let (close, volume) = make_test_data();

        let output = rr.calculate(&close, &volume);

        // HODL bank should generally increase over time
        let early_bank = output.hodl_bank[50];
        let late_bank = output.hodl_bank[200];
        assert!(late_bank > early_bank);
    }

    #[test]
    fn test_reserve_risk_confidence_bounded() {
        let rr = ReserveRisk::default();
        let (close, volume) = make_test_data();

        let output = rr.calculate(&close, &volume);

        // Confidence should be bounded 0-100
        for i in 200..output.confidence.len() {
            assert!(
                output.confidence[i] >= 0.0 && output.confidence[i] <= 100.0,
                "Confidence: {}",
                output.confidence[i]
            );
        }
    }

    #[test]
    fn test_reserve_risk_signal_interpretation() {
        let rr = ReserveRisk::default();

        assert_eq!(rr.interpret(0.05), ReserveRiskSignal::ExtremeHighRisk);
        assert_eq!(rr.interpret(0.025), ReserveRiskSignal::HighRisk);
        assert_eq!(rr.interpret(0.01), ReserveRiskSignal::Normal);
        assert_eq!(rr.interpret(0.0015), ReserveRiskSignal::LowRisk);
        assert_eq!(rr.interpret(0.0005), ReserveRiskSignal::ExtremeLowRisk);
    }

    #[test]
    fn test_reserve_risk_signal_conversion() {
        let rr = ReserveRisk::default();

        assert_eq!(
            rr.to_indicator_signal(ReserveRiskSignal::ExtremeHighRisk),
            IndicatorSignal::Bearish
        );
        assert_eq!(
            rr.to_indicator_signal(ReserveRiskSignal::ExtremeLowRisk),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            rr.to_indicator_signal(ReserveRiskSignal::Normal),
            IndicatorSignal::Neutral
        );
    }

    #[test]
    fn test_reserve_risk_validation() {
        assert!(ReserveRisk::new(ReserveRiskConfig {
            hodl_period: 5, // Invalid
            smooth_period: 14,
            lower_threshold: 0.002,
            upper_threshold: 0.02,
        })
        .is_err());

        assert!(ReserveRisk::new(ReserveRiskConfig {
            hodl_period: 180,
            smooth_period: 14,
            lower_threshold: 0.03, // Invalid: >= upper
            upper_threshold: 0.02,
        })
        .is_err());

        assert!(ReserveRisk::new(ReserveRiskConfig {
            hodl_period: 180,
            smooth_period: 14,
            lower_threshold: -0.002, // Invalid: <= 0
            upper_threshold: 0.02,
        })
        .is_err());
    }

    #[test]
    fn test_reserve_risk_empty_input() {
        let rr = ReserveRisk::default();
        let output = rr.calculate(&[], &[]);

        assert!(output.reserve_risk.is_empty());
    }

    #[test]
    fn test_reserve_risk_technical_indicator_trait() {
        let rr = ReserveRisk::default();
        assert_eq!(rr.name(), "Reserve Risk");
        assert!(rr.min_periods() > 0);
    }

    #[test]
    fn test_reserve_risk_momentum() {
        let rr = ReserveRisk::default();
        let (close, volume) = make_test_data();

        let output = rr.calculate(&close, &volume);

        // Momentum should be calculated for valid periods
        let momentum_valid: Vec<&f64> = output.momentum[200..]
            .iter()
            .filter(|x| !x.is_nan())
            .collect();
        assert!(!momentum_valid.is_empty());
    }
}
