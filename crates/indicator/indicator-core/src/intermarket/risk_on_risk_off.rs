//! Risk-On/Risk-Off Indicator (IND-402)
//!
//! Cross-asset risk measure that determines market risk appetite
//! by analyzing relative performance across risk-on and risk-off assets.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Risk appetite regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskRegime {
    /// Strong risk-on environment.
    StrongRiskOn,
    /// Moderate risk-on environment.
    ModerateRiskOn,
    /// Neutral risk environment.
    Neutral,
    /// Moderate risk-off environment.
    ModerateRiskOff,
    /// Strong risk-off environment.
    StrongRiskOff,
}

impl RiskRegime {
    /// Convert to numeric score (-2 to +2).
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::StrongRiskOn => 2.0,
            Self::ModerateRiskOn => 1.0,
            Self::Neutral => 0.0,
            Self::ModerateRiskOff => -1.0,
            Self::StrongRiskOff => -2.0,
        }
    }

    /// Create from score.
    pub fn from_score(score: f64) -> Self {
        if score > 1.5 {
            Self::StrongRiskOn
        } else if score > 0.5 {
            Self::ModerateRiskOn
        } else if score > -0.5 {
            Self::Neutral
        } else if score > -1.5 {
            Self::ModerateRiskOff
        } else {
            Self::StrongRiskOff
        }
    }
}

/// Risk-On/Risk-Off output.
#[derive(Debug, Clone)]
pub struct RiskOnRiskOffOutput {
    /// Risk score (-2 to +2, positive = risk-on).
    pub risk_score: f64,
    /// Current risk regime.
    pub regime: RiskRegime,
    /// Risk-on asset momentum.
    pub risk_on_momentum: f64,
    /// Risk-off asset momentum.
    pub risk_off_momentum: f64,
    /// Confidence in regime determination (0-1).
    pub confidence: f64,
    /// Z-score of current reading.
    pub z_score: f64,
}

/// Risk-On/Risk-Off Indicator (IND-402)
///
/// Measures market risk appetite by comparing the performance of
/// risk-on assets (equities, high yield, EM) versus risk-off assets
/// (treasuries, gold, VIX, USD).
///
/// # Methodology
/// 1. Calculate momentum for risk-on assets (stocks, credit, etc.)
/// 2. Calculate momentum for risk-off assets (bonds, gold, VIX)
/// 3. Compute relative strength ratio
/// 4. Smooth and normalize to create risk score
/// 5. Classify into risk regime
///
/// # Interpretation
/// - Positive score: Risk-on environment (favor equities, credit)
/// - Negative score: Risk-off environment (favor bonds, gold, cash)
/// - Transitions signal regime changes
#[derive(Debug, Clone)]
pub struct RiskOnRiskOff {
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Period for smoothing.
    smooth_period: usize,
    /// Period for z-score calculation.
    zscore_period: usize,
    /// Risk-on asset prices (e.g., SPY, HYG, EEM).
    risk_on_assets: Vec<Vec<f64>>,
    /// Risk-off asset prices (e.g., TLT, GLD, VIX inverse).
    risk_off_assets: Vec<Vec<f64>>,
    /// Asset weights for risk-on.
    risk_on_weights: Vec<f64>,
    /// Asset weights for risk-off.
    risk_off_weights: Vec<f64>,
}

impl RiskOnRiskOff {
    /// Create a new Risk-On/Risk-Off indicator.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (e.g., 20)
    /// * `smooth_period` - Smoothing period (e.g., 5)
    /// * `zscore_period` - Period for z-score normalization (e.g., 252)
    pub fn new(momentum_period: usize, smooth_period: usize, zscore_period: usize) -> Result<Self> {
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if zscore_period < momentum_period {
            return Err(IndicatorError::InvalidParameter {
                name: "zscore_period".to_string(),
                reason: "must be at least momentum_period".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            smooth_period,
            zscore_period,
            risk_on_assets: Vec::new(),
            risk_off_assets: Vec::new(),
            risk_on_weights: Vec::new(),
            risk_off_weights: Vec::new(),
        })
    }

    /// Create with default parameters (20, 5, 252).
    pub fn default_params() -> Result<Self> {
        Self::new(20, 5, 252)
    }

    /// Add a risk-on asset with weight.
    pub fn add_risk_on_asset(&mut self, prices: Vec<f64>, weight: f64) {
        self.risk_on_assets.push(prices);
        self.risk_on_weights.push(weight);
    }

    /// Add a risk-off asset with weight.
    pub fn add_risk_off_asset(&mut self, prices: Vec<f64>, weight: f64) {
        self.risk_off_assets.push(prices);
        self.risk_off_weights.push(weight);
    }

    /// Set risk-on assets with equal weights.
    pub fn with_risk_on_assets(mut self, assets: Vec<Vec<f64>>) -> Self {
        let weight = 1.0 / assets.len().max(1) as f64;
        self.risk_on_weights = vec![weight; assets.len()];
        self.risk_on_assets = assets;
        self
    }

    /// Set risk-off assets with equal weights.
    pub fn with_risk_off_assets(mut self, assets: Vec<Vec<f64>>) -> Self {
        let weight = 1.0 / assets.len().max(1) as f64;
        self.risk_off_weights = vec![weight; assets.len()];
        self.risk_off_assets = assets;
        self
    }

    /// Calculate rate of change.
    fn calculate_roc(prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        if n < period + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period];
        for i in period..n {
            let prev = prices[i - period];
            if prev.abs() > 1e-10 {
                result.push((prices[i] - prev) / prev * 100.0);
            } else {
                result.push(f64::NAN);
            }
        }
        result
    }

    /// Calculate exponential moving average.
    fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return Vec::new();
        }
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];
        let alpha = 2.0 / (period as f64 + 1.0);

        // Initialize with SMA
        let valid_init: Vec<f64> = data[..period]
            .iter()
            .filter(|x| !x.is_nan())
            .copied()
            .collect();
        if valid_init.is_empty() {
            return vec![f64::NAN; n];
        }
        let sma = valid_init.iter().sum::<f64>() / valid_init.len() as f64;
        result[period - 1] = sma;

        for i in period..n {
            if data[i].is_nan() {
                result[i] = result[i - 1];
            } else {
                let prev = result[i - 1];
                if prev.is_nan() {
                    result[i] = data[i];
                } else {
                    result[i] = alpha * data[i] + (1.0 - alpha) * prev;
                }
            }
        }

        result
    }

    /// Calculate rolling z-score.
    fn rolling_zscore(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..n {
            let window: Vec<f64> = data[(i + 1 - period)..=i]
                .iter()
                .filter(|x| !x.is_nan())
                .copied()
                .collect();

            if window.len() < 2 {
                result.push(f64::NAN);
                continue;
            }

            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / window.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                let current = data[i];
                if current.is_nan() {
                    result.push(f64::NAN);
                } else {
                    result.push((current - mean) / std_dev);
                }
            } else {
                result.push(0.0);
            }
        }

        result
    }

    /// Calculate weighted average momentum.
    fn weighted_momentum(&self, assets: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
        if assets.is_empty() {
            return Vec::new();
        }

        let n = assets.iter().map(|a| a.len()).min().unwrap_or(0);
        let mut result = vec![0.0; n];

        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        let norm_weights: Vec<f64> = if weight_sum > 1e-10 {
            weights.iter().map(|w| w / weight_sum).collect()
        } else {
            vec![1.0 / assets.len() as f64; assets.len()]
        };

        for (asset, weight) in assets.iter().zip(norm_weights.iter()) {
            let roc = Self::calculate_roc(asset, self.momentum_period);
            for i in 0..n.min(roc.len()) {
                if !roc[i].is_nan() {
                    result[i] += roc[i] * weight;
                }
            }
        }

        result
    }

    /// Calculate Risk-On/Risk-Off outputs.
    pub fn calculate(&self) -> Vec<RiskOnRiskOffOutput> {
        if self.risk_on_assets.is_empty() || self.risk_off_assets.is_empty() {
            return Vec::new();
        }

        let n = self
            .risk_on_assets
            .iter()
            .chain(self.risk_off_assets.iter())
            .map(|a| a.len())
            .min()
            .unwrap_or(0);

        if n < self.momentum_period + 1 {
            return Vec::new();
        }

        // Calculate weighted momentums
        let risk_on_mom = self.weighted_momentum(&self.risk_on_assets, &self.risk_on_weights);
        let risk_off_mom = self.weighted_momentum(&self.risk_off_assets, &self.risk_off_weights);

        // Calculate relative strength (risk-on minus risk-off)
        let mut raw_score: Vec<f64> = risk_on_mom
            .iter()
            .zip(risk_off_mom.iter())
            .map(|(on, off)| on - off)
            .collect();

        // Smooth the raw score
        let smoothed = Self::ema(&raw_score, self.smooth_period);

        // Calculate z-score for normalization
        let z_scores = Self::rolling_zscore(&smoothed, self.zscore_period);

        // Build outputs
        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            let risk_score = smoothed.get(i).copied().unwrap_or(0.0);
            let z_score = z_scores.get(i).copied().unwrap_or(0.0);

            // Normalize score to -2 to +2 range using z-score
            let normalized_score = z_score.max(-2.0).min(2.0);
            let regime = RiskRegime::from_score(normalized_score);

            // Calculate confidence based on signal clarity
            let confidence = if z_score.is_nan() {
                0.0
            } else {
                (z_score.abs() / 2.0).min(1.0)
            };

            results.push(RiskOnRiskOffOutput {
                risk_score,
                regime,
                risk_on_momentum: risk_on_mom.get(i).copied().unwrap_or(0.0),
                risk_off_momentum: risk_off_mom.get(i).copied().unwrap_or(0.0),
                confidence,
                z_score,
            });
        }

        results
    }

    /// Get risk score series.
    pub fn risk_scores(&self) -> Vec<f64> {
        self.calculate().iter().map(|o| o.risk_score).collect()
    }

    /// Get z-score series.
    pub fn z_scores(&self) -> Vec<f64> {
        self.calculate().iter().map(|o| o.z_score).collect()
    }

    /// Get regime series as numeric values.
    pub fn regime_scores(&self) -> Vec<f64> {
        self.calculate().iter().map(|o| o.regime.as_f64()).collect()
    }
}

impl TechnicalIndicator for RiskOnRiskOff {
    fn name(&self) -> &str {
        "Risk On Risk Off"
    }

    fn min_periods(&self) -> usize {
        self.zscore_period
    }

    fn compute(&self, _data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.risk_on_assets.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "risk_on_assets".to_string(),
                reason: "Risk-on assets must be set before computing".to_string(),
            });
        }
        if self.risk_off_assets.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "risk_off_assets".to_string(),
                reason: "Risk-off assets must be set before computing".to_string(),
            });
        }

        let min_len = self
            .risk_on_assets
            .iter()
            .chain(self.risk_off_assets.iter())
            .map(|a| a.len())
            .min()
            .unwrap_or(0);

        if min_len < self.zscore_period {
            return Err(IndicatorError::InsufficientData {
                required: self.zscore_period,
                got: min_len,
            });
        }

        let z_scores = self.z_scores();
        Ok(IndicatorOutput::single(z_scores))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_indicator(n: usize) -> RiskOnRiskOff {
        // Create synthetic risk-on assets (generally trending up)
        let equities: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.3 + ((i as f64) * 0.1).sin() * 3.0)
            .collect();

        let high_yield: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.2 + ((i as f64) * 0.15).sin() * 2.0)
            .collect();

        // Create synthetic risk-off assets (inverse relationship)
        let treasuries: Vec<f64> = (0..n)
            .map(|i| 100.0 - (i as f64) * 0.1 + ((i as f64) * 0.1).cos() * 2.0)
            .collect();

        let gold: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.05 + ((i as f64) * 0.08).cos() * 1.5)
            .collect();

        let mut indicator = RiskOnRiskOff::new(20, 5, 60).unwrap();
        indicator.add_risk_on_asset(equities, 0.6);
        indicator.add_risk_on_asset(high_yield, 0.4);
        indicator.add_risk_off_asset(treasuries, 0.5);
        indicator.add_risk_off_asset(gold, 0.5);

        indicator
    }

    #[test]
    fn test_risk_on_risk_off_basic() {
        let indicator = create_test_indicator(100);
        let outputs = indicator.calculate();

        assert_eq!(outputs.len(), 100);

        // Check last output has valid regime
        let last = outputs.last().unwrap();
        assert!(last.risk_score.is_finite());
    }

    #[test]
    fn test_regime_classification() {
        let indicator = create_test_indicator(100);
        let outputs = indicator.calculate();

        // Check regimes are valid
        for output in &outputs {
            match output.regime {
                RiskRegime::StrongRiskOn
                | RiskRegime::ModerateRiskOn
                | RiskRegime::Neutral
                | RiskRegime::ModerateRiskOff
                | RiskRegime::StrongRiskOff => {}
            }
        }
    }

    #[test]
    fn test_z_score_range() {
        let indicator = create_test_indicator(100);
        let z_scores = indicator.z_scores();

        // After warmup, z-scores should be reasonable
        for z in z_scores.iter().skip(70) {
            if !z.is_nan() {
                // Z-scores should typically be within -4 to +4
                assert!(z.abs() < 10.0);
            }
        }
    }

    #[test]
    fn test_confidence_range() {
        let indicator = create_test_indicator(100);
        let outputs = indicator.calculate();

        for output in &outputs {
            assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
        }
    }

    #[test]
    fn test_momentum_calculation() {
        let indicator = create_test_indicator(100);
        let outputs = indicator.calculate();

        // After warmup, should have momentum values
        let last = outputs.last().unwrap();
        assert!(last.risk_on_momentum.is_finite());
        assert!(last.risk_off_momentum.is_finite());
    }

    #[test]
    fn test_with_builder_pattern() {
        let indicator = RiskOnRiskOff::new(20, 5, 60)
            .unwrap()
            .with_risk_on_assets(vec![
                (0..100).map(|i| 100.0 + i as f64).collect(),
                (0..100).map(|i| 100.0 + i as f64 * 0.5).collect(),
            ])
            .with_risk_off_assets(vec![
                (0..100).map(|i| 100.0 - i as f64 * 0.1).collect(),
            ]);

        let outputs = indicator.calculate();
        assert!(!outputs.is_empty());
    }

    #[test]
    fn test_technical_indicator_impl() {
        let indicator = create_test_indicator(100);
        let data = OHLCVSeries::from_close(vec![100.0; 100]);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
    }

    #[test]
    fn test_empty_assets_error() {
        let indicator = RiskOnRiskOff::new(20, 5, 60).unwrap();
        let data = OHLCVSeries::from_close(vec![100.0; 100]);
        let result = indicator.compute(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_parameters() {
        let result = RiskOnRiskOff::new(2, 5, 60);
        assert!(result.is_err());

        let result = RiskOnRiskOff::new(20, 0, 60);
        assert!(result.is_err());

        let result = RiskOnRiskOff::new(20, 5, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_regime_from_score() {
        assert_eq!(RiskRegime::from_score(2.0), RiskRegime::StrongRiskOn);
        assert_eq!(RiskRegime::from_score(1.0), RiskRegime::ModerateRiskOn);
        assert_eq!(RiskRegime::from_score(0.0), RiskRegime::Neutral);
        assert_eq!(RiskRegime::from_score(-1.0), RiskRegime::ModerateRiskOff);
        assert_eq!(RiskRegime::from_score(-2.0), RiskRegime::StrongRiskOff);
    }
}
