//! Central Bank Policy Indicator (IND-312)
//!
//! Hawkish/dovish score based on price action and momentum analysis.
//! Provides a proxy for central bank policy stance based on market behavior.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Central Bank Policy - Hawkish/Dovish score proxy (IND-312)
///
/// This indicator estimates central bank policy stance based on
/// currency strength, momentum, and volatility patterns.
///
/// # Interpretation
/// - Positive values (0 to 100) indicate hawkish policy expectations
/// - Negative values (-100 to 0) indicate dovish policy expectations
/// - Higher absolute values indicate stronger policy conviction
///
/// # Example
/// ```ignore
/// use indicator_core::forex::CentralBankPolicy;
///
/// let cbp = CentralBankPolicy::new(20, 10).unwrap();
/// let policy_score = cbp.calculate(&close);
/// ```
#[derive(Debug, Clone)]
pub struct CentralBankPolicy {
    /// Long-term lookback for trend analysis
    long_period: usize,
    /// Short-term lookback for momentum
    short_period: usize,
}

impl CentralBankPolicy {
    /// Create a new CentralBankPolicy indicator.
    ///
    /// # Arguments
    /// * `long_period` - Long-term lookback period (minimum 10)
    /// * `short_period` - Short-term lookback period (minimum 3)
    pub fn new(long_period: usize, short_period: usize) -> Result<Self> {
        if long_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if short_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if short_period >= long_period {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be less than long_period".to_string(),
            });
        }
        Ok(Self { long_period, short_period })
    }

    /// Calculate hawkish/dovish policy score.
    ///
    /// # Arguments
    /// * `close` - Closing prices
    ///
    /// # Returns
    /// Vector of policy scores (-100 to 100, positive = hawkish)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.long_period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Calculate long-term trend (rate hike expectations)
            let long_start = i.saturating_sub(self.long_period);
            let long_return = if close[long_start] > 0.0 {
                (close[i] / close[long_start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Calculate short-term momentum
            let short_start = i.saturating_sub(self.short_period);
            let short_return = if close[short_start] > 0.0 {
                (close[i] / close[short_start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Calculate volatility (uncertainty in policy)
            let returns: Vec<f64> = ((long_start + 1)..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();

            let vol = if !returns.is_empty() {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                variance.sqrt() * (252.0_f64).sqrt() * 100.0 // Annualized
            } else {
                10.0 // Default volatility
            };

            // Calculate momentum consistency
            let momentum_alignment = if long_return.signum() == short_return.signum() {
                1.0
            } else {
                0.5
            };

            // Policy score combines trend, momentum, and volatility
            // Strong trend + aligned momentum = strong policy conviction
            // High volatility reduces conviction
            let trend_score = long_return.tanh() * 50.0; // Normalize to -50 to 50
            let momentum_score = short_return.tanh() * 30.0; // Normalize to -30 to 30
            let vol_adjustment = if vol > 0.0 {
                (20.0 / vol).min(1.0).max(0.5)
            } else {
                1.0
            };

            result[i] = (trend_score + momentum_score * momentum_alignment) * vol_adjustment;

            // Clamp to -100 to 100
            result[i] = result[i].max(-100.0).min(100.0);
        }

        result
    }

    /// Calculate with extended output including component scores.
    pub fn calculate_extended(&self, close: &[f64]) -> CentralBankPolicyOutput {
        let n = close.len();
        if n < self.long_period + 1 {
            return CentralBankPolicyOutput {
                policy_score: vec![0.0; n],
                trend_component: vec![0.0; n],
                momentum_component: vec![0.0; n],
                volatility_factor: vec![0.0; n],
                policy_regime: vec![PolicyRegime::Neutral; n],
            };
        }

        let mut policy_score = vec![0.0; n];
        let mut trend_component = vec![0.0; n];
        let mut momentum_component = vec![0.0; n];
        let mut volatility_factor = vec![0.0; n];
        let mut policy_regime = vec![PolicyRegime::Neutral; n];

        for i in self.long_period..n {
            let long_start = i.saturating_sub(self.long_period);
            let long_return = if close[long_start] > 0.0 {
                (close[i] / close[long_start] - 1.0) * 100.0
            } else {
                0.0
            };

            let short_start = i.saturating_sub(self.short_period);
            let short_return = if close[short_start] > 0.0 {
                (close[i] / close[short_start] - 1.0) * 100.0
            } else {
                0.0
            };

            let returns: Vec<f64> = ((long_start + 1)..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();

            let vol = if !returns.is_empty() {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                variance.sqrt() * (252.0_f64).sqrt() * 100.0
            } else {
                10.0
            };

            trend_component[i] = long_return.tanh() * 50.0;
            momentum_component[i] = short_return.tanh() * 30.0;
            volatility_factor[i] = if vol > 0.0 {
                (20.0 / vol).min(1.0).max(0.5)
            } else {
                1.0
            };

            let momentum_alignment = if long_return.signum() == short_return.signum() {
                1.0
            } else {
                0.5
            };

            policy_score[i] = (trend_component[i] + momentum_component[i] * momentum_alignment)
                * volatility_factor[i];
            policy_score[i] = policy_score[i].max(-100.0).min(100.0);

            // Determine policy regime
            policy_regime[i] = if policy_score[i] > 30.0 {
                PolicyRegime::Hawkish
            } else if policy_score[i] > 10.0 {
                PolicyRegime::MildlyHawkish
            } else if policy_score[i] < -30.0 {
                PolicyRegime::Dovish
            } else if policy_score[i] < -10.0 {
                PolicyRegime::MildlyDovish
            } else {
                PolicyRegime::Neutral
            };
        }

        CentralBankPolicyOutput {
            policy_score,
            trend_component,
            momentum_component,
            volatility_factor,
            policy_regime,
        }
    }
}

/// Policy regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyRegime {
    /// Strong hawkish policy (rate hikes expected)
    Hawkish,
    /// Mildly hawkish
    MildlyHawkish,
    /// Neutral policy stance
    Neutral,
    /// Mildly dovish
    MildlyDovish,
    /// Strong dovish policy (rate cuts expected)
    Dovish,
}

/// Extended output for CentralBankPolicy indicator.
#[derive(Debug, Clone)]
pub struct CentralBankPolicyOutput {
    /// Overall policy score (-100 to 100)
    pub policy_score: Vec<f64>,
    /// Trend component of the score
    pub trend_component: Vec<f64>,
    /// Momentum component of the score
    pub momentum_component: Vec<f64>,
    /// Volatility adjustment factor
    pub volatility_factor: Vec<f64>,
    /// Classified policy regime
    pub policy_regime: Vec<PolicyRegime>,
}

impl TechnicalIndicator for CentralBankPolicy {
    fn name(&self) -> &str {
        "Central Bank Policy"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        vec![
            1.1000, 1.1020, 1.1040, 1.1060, 1.1080, 1.1100, 1.1120, 1.1140, 1.1160, 1.1180,
            1.1200, 1.1220, 1.1240, 1.1260, 1.1280, 1.1300, 1.1320, 1.1340, 1.1360, 1.1380,
            1.1400, 1.1420, 1.1440, 1.1460, 1.1480, 1.1500, 1.1520, 1.1540, 1.1560, 1.1580,
        ]
    }

    #[test]
    fn test_central_bank_policy_new() {
        assert!(CentralBankPolicy::new(20, 5).is_ok());
        assert!(CentralBankPolicy::new(9, 5).is_err()); // long_period too small
        assert!(CentralBankPolicy::new(20, 2).is_err()); // short_period too small
        assert!(CentralBankPolicy::new(10, 10).is_err()); // short >= long
    }

    #[test]
    fn test_central_bank_policy_calculate() {
        let close = make_test_data();
        let cbp = CentralBankPolicy::new(15, 5).unwrap();
        let result = cbp.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Consistent uptrend should show hawkish (positive) score
        assert!(result[25] > 0.0);
        // Values should be bounded
        assert!(result[25] >= -100.0 && result[25] <= 100.0);
    }

    #[test]
    fn test_central_bank_policy_extended() {
        let close = make_test_data();
        let cbp = CentralBankPolicy::new(15, 5).unwrap();
        let output = cbp.calculate_extended(&close);

        assert_eq!(output.policy_score.len(), close.len());
        assert_eq!(output.trend_component.len(), close.len());
        assert_eq!(output.momentum_component.len(), close.len());
        assert_eq!(output.volatility_factor.len(), close.len());
        assert_eq!(output.policy_regime.len(), close.len());

        // Uptrend should be hawkish
        assert!(matches!(output.policy_regime[25], PolicyRegime::Hawkish | PolicyRegime::MildlyHawkish));
    }

    #[test]
    fn test_central_bank_policy_downtrend() {
        let close: Vec<f64> = (0..30)
            .map(|i| 1.20 - (i as f64 * 0.002))
            .collect();

        let cbp = CentralBankPolicy::new(15, 5).unwrap();
        let output = cbp.calculate_extended(&close);

        // Downtrend should be dovish
        assert!(output.policy_score[25] < 0.0);
        assert!(matches!(output.policy_regime[25], PolicyRegime::Dovish | PolicyRegime::MildlyDovish));
    }

    #[test]
    fn test_central_bank_policy_technical_indicator() {
        let close = make_test_data();
        let cbp = CentralBankPolicy::new(15, 5).unwrap();

        assert_eq!(cbp.name(), "Central Bank Policy");
        assert_eq!(cbp.min_periods(), 16);

        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x * 1.01).collect(),
            low: close.iter().map(|x| x * 0.99).collect(),
            close: close.clone(),
            volume: vec![1000.0; close.len()],
        };

        let output = cbp.compute(&data).unwrap();
        assert!(output.primary.len() == close.len());
    }
}
