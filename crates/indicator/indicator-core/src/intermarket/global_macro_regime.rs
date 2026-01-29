//! Global Macro Regime Indicator (IND-498)
//!
//! Growth/inflation matrix for macro regime classification.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

// ============================================================================
// MacroRegime Enum
// ============================================================================

/// Market macro regime classifications.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MacroRegime {
    /// Rising growth, falling inflation - Risk-on equities
    Goldilocks = 1,
    /// Rising growth, rising inflation - Commodities, value
    Reflation = 2,
    /// Falling growth, rising inflation - Stagflation, defensives
    Stagflation = 3,
    /// Falling growth, falling inflation - Bonds, quality
    Deflation = 4,
}

impl MacroRegime {
    /// Get numeric value for the regime.
    pub fn as_f64(&self) -> f64 {
        match self {
            MacroRegime::Goldilocks => 1.0,
            MacroRegime::Reflation => 2.0,
            MacroRegime::Stagflation => 3.0,
            MacroRegime::Deflation => 4.0,
        }
    }

    /// Convert from numeric value.
    pub fn from_f64(value: f64) -> Option<Self> {
        match value as i32 {
            1 => Some(MacroRegime::Goldilocks),
            2 => Some(MacroRegime::Reflation),
            3 => Some(MacroRegime::Stagflation),
            4 => Some(MacroRegime::Deflation),
            _ => None,
        }
    }
}

// ============================================================================
// GlobalMacroRegime
// ============================================================================

/// Global Macro Regime - Growth/inflation matrix classifier.
///
/// This indicator classifies the macro environment into one of four
/// regimes based on growth and inflation proxies derived from price data.
///
/// # Theory
/// Different asset classes perform differently across macro regimes:
/// - Goldilocks: Equities, growth stocks
/// - Reflation: Commodities, cyclicals, value
/// - Stagflation: Commodities, defensives, TIPS
/// - Deflation: Bonds, quality, low vol
///
/// # Interpretation
/// - `regime = 1 (Goldilocks)`: Risk-on, favor equities
/// - `regime = 2 (Reflation)`: Favor commodities and cyclicals
/// - `regime = 3 (Stagflation)`: Defensive positioning
/// - `regime = 4 (Deflation)`: Favor bonds and quality
///
/// # Additional Outputs
/// - `growth_score`: Growth momentum (-1 to 1)
/// - `inflation_score`: Inflation proxy (-1 to 1)
/// - `regime_confidence`: Confidence in regime classification
#[derive(Debug, Clone)]
pub struct GlobalMacroRegime {
    /// Period for growth momentum calculation.
    growth_period: usize,
    /// Period for inflation proxy calculation.
    inflation_period: usize,
    /// Smoothing period for regime transitions.
    smoothing: usize,
    /// Optional commodity series for inflation proxy.
    commodity_series: Vec<f64>,
    /// Optional bond series for growth proxy.
    bond_series: Vec<f64>,
}

impl GlobalMacroRegime {
    /// Create a new GlobalMacroRegime indicator.
    ///
    /// # Arguments
    /// * `growth_period` - Period for growth calculation (min: 10)
    /// * `inflation_period` - Period for inflation proxy (min: 10)
    /// * `smoothing` - Smoothing for regime transitions (min: 1)
    pub fn new(growth_period: usize, inflation_period: usize, smoothing: usize) -> Result<Self> {
        if growth_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "growth_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if inflation_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "inflation_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if smoothing < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            growth_period,
            inflation_period,
            smoothing,
            commodity_series: Vec::new(),
            bond_series: Vec::new(),
        })
    }

    /// Create with default parameters.
    pub fn default_params() -> Result<Self> {
        Self::new(20, 20, 5)
    }

    /// Set commodity series for inflation proxy.
    pub fn with_commodity(mut self, series: &[f64]) -> Self {
        self.commodity_series = series.to_vec();
        self
    }

    /// Set bond series for growth proxy.
    pub fn with_bonds(mut self, series: &[f64]) -> Self {
        self.bond_series = series.to_vec();
        self
    }

    /// Calculate growth score from equity prices.
    fn calculate_growth_score(&self, prices: &[f64], idx: usize) -> f64 {
        if idx < self.growth_period {
            return 0.0;
        }

        let start = idx.saturating_sub(self.growth_period);
        let returns: Vec<f64> = ((start + 1)..=idx)
            .filter(|&j| j < prices.len() && j > 0)
            .map(|j| (prices[j] / prices[j - 1]).ln())
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let vol = variance.sqrt();

        // Growth score: risk-adjusted return
        if vol > 0.0 {
            (mean_return / vol).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Calculate inflation score from price volatility and momentum.
    fn calculate_inflation_score(&self, prices: &[f64], idx: usize) -> f64 {
        if idx < self.inflation_period {
            return 0.0;
        }

        let start = idx.saturating_sub(self.inflation_period);

        // Use realized volatility acceleration as inflation proxy
        let half_period = self.inflation_period / 2;
        let recent_start = idx.saturating_sub(half_period);

        let recent_returns: Vec<f64> = ((recent_start + 1)..=idx)
            .filter(|&j| j < prices.len() && j > 0)
            .map(|j| (prices[j] / prices[j - 1]).ln())
            .collect();

        let old_returns: Vec<f64> = ((start + 1)..=recent_start)
            .filter(|&j| j < prices.len() && j > 0)
            .map(|j| (prices[j] / prices[j - 1]).ln())
            .collect();

        if recent_returns.is_empty() || old_returns.is_empty() {
            return 0.0;
        }

        let calc_vol = |rets: &[f64]| -> f64 {
            let mean = rets.iter().sum::<f64>() / rets.len() as f64;
            let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
            var.sqrt()
        };

        let recent_vol = calc_vol(&recent_returns);
        let old_vol = calc_vol(&old_returns);

        // Rising volatility suggests rising inflation
        if old_vol > 0.0 {
            ((recent_vol / old_vol) - 1.0).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Calculate inflation score from commodity prices.
    fn calculate_commodity_inflation(&self, commodities: &[f64], idx: usize) -> f64 {
        if idx < self.inflation_period {
            return 0.0;
        }

        let start = idx.saturating_sub(self.inflation_period);
        let returns: Vec<f64> = ((start + 1)..=idx)
            .filter(|&j| j < commodities.len() && j > 0)
            .map(|j| (commodities[j] / commodities[j - 1]).ln())
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        // Commodity momentum as inflation proxy
        (mean_return * 252.0 * 100.0 / 20.0).clamp(-1.0, 1.0)
    }

    /// Determine regime from growth and inflation scores.
    fn classify_regime(&self, growth: f64, inflation: f64) -> MacroRegime {
        if growth >= 0.0 && inflation < 0.0 {
            MacroRegime::Goldilocks
        } else if growth >= 0.0 && inflation >= 0.0 {
            MacroRegime::Reflation
        } else if growth < 0.0 && inflation >= 0.0 {
            MacroRegime::Stagflation
        } else {
            MacroRegime::Deflation
        }
    }

    /// Calculate macro regime from a single equity series.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let max_period = self.growth_period.max(self.inflation_period);
        let mut result = vec![0.0; n];

        if n < max_period + self.smoothing {
            return result;
        }

        let mut regime_history: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            if i < max_period {
                regime_history.push(0.0);
                continue;
            }

            let growth_score = self.calculate_growth_score(close, i);
            let inflation_score = if !self.commodity_series.is_empty() && i < self.commodity_series.len() {
                self.calculate_commodity_inflation(&self.commodity_series, i)
            } else {
                self.calculate_inflation_score(close, i)
            };

            let regime = self.classify_regime(growth_score, inflation_score);
            regime_history.push(regime.as_f64());
        }

        // Apply smoothing (mode over smoothing period)
        for i in 0..n {
            if i < max_period + self.smoothing - 1 {
                result[i] = regime_history[i];
            } else {
                let start = i + 1 - self.smoothing;
                let window = &regime_history[start..=i];

                // Find mode
                let mut counts = [0i32; 5];
                for &r in window {
                    let idx = r as usize;
                    if idx > 0 && idx < 5 {
                        counts[idx] += 1;
                    }
                }
                let mode = counts.iter()
                    .enumerate()
                    .skip(1)
                    .max_by_key(|(_, &c)| c)
                    .map(|(i, _)| i as f64)
                    .unwrap_or(0.0);
                result[i] = mode;
            }
        }

        result
    }

    /// Calculate detailed regime output with scores.
    pub fn calculate_detailed(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let max_period = self.growth_period.max(self.inflation_period);
        let mut regime = vec![0.0; n];
        let mut growth = vec![0.0; n];
        let mut inflation = vec![0.0; n];
        let mut confidence = vec![0.0; n];

        if n < max_period + self.smoothing {
            return (regime, growth, inflation, confidence);
        }

        for i in max_period..n {
            let growth_score = self.calculate_growth_score(close, i);
            let inflation_score = if !self.commodity_series.is_empty() && i < self.commodity_series.len() {
                self.calculate_commodity_inflation(&self.commodity_series, i)
            } else {
                self.calculate_inflation_score(close, i)
            };

            let macro_regime = self.classify_regime(growth_score, inflation_score);

            regime[i] = macro_regime.as_f64();
            growth[i] = growth_score;
            inflation[i] = inflation_score;
            // Confidence based on distance from decision boundaries
            confidence[i] = (growth_score.abs() + inflation_score.abs()) / 2.0;
        }

        (regime, growth, inflation, confidence)
    }
}

impl TechnicalIndicator for GlobalMacroRegime {
    fn name(&self) -> &str {
        "Global Macro Regime"
    }

    fn min_periods(&self) -> usize {
        self.growth_period.max(self.inflation_period) + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (regime, growth, inflation, confidence) = self.calculate_detailed(&data.close);
        Ok(IndicatorOutput::triple(regime, growth, inflation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_growth_data() -> Vec<f64> {
        // Simulated growth environment (uptrend with low vol)
        (0..50).map(|i| 100.0 * (1.0 + 0.001 * i as f64)).collect()
    }

    fn make_deflation_data() -> Vec<f64> {
        // Simulated deflation (downtrend)
        (0..50).map(|i| 100.0 * (1.0 - 0.001 * i as f64)).collect()
    }

    #[test]
    fn test_macro_regime_creation() {
        let gmr = GlobalMacroRegime::new(20, 20, 5);
        assert!(gmr.is_ok());

        let gmr_err = GlobalMacroRegime::new(5, 20, 5);
        assert!(gmr_err.is_err());
    }

    #[test]
    fn test_macro_regime_enum() {
        assert_eq!(MacroRegime::Goldilocks.as_f64(), 1.0);
        assert_eq!(MacroRegime::Reflation.as_f64(), 2.0);
        assert_eq!(MacroRegime::Stagflation.as_f64(), 3.0);
        assert_eq!(MacroRegime::Deflation.as_f64(), 4.0);

        assert_eq!(MacroRegime::from_f64(1.0), Some(MacroRegime::Goldilocks));
        assert_eq!(MacroRegime::from_f64(5.0), None);
    }

    #[test]
    fn test_macro_regime_calculation() {
        let data = make_growth_data();
        let gmr = GlobalMacroRegime::default_params().unwrap();
        let result = gmr.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Should classify as some regime in valid region
        assert!(result[40] >= 0.0 && result[40] <= 4.0);
    }

    #[test]
    fn test_macro_regime_detailed() {
        let data = make_growth_data();
        let gmr = GlobalMacroRegime::new(15, 15, 3).unwrap();
        let (regime, growth, inflation, confidence) = gmr.calculate_detailed(&data);

        assert_eq!(regime.len(), data.len());
        assert_eq!(growth.len(), data.len());
        assert_eq!(inflation.len(), data.len());
        assert_eq!(confidence.len(), data.len());

        // Growth should be positive in uptrend
        assert!(growth[40] >= 0.0);
    }

    #[test]
    fn test_macro_regime_trait() {
        let data = make_growth_data();
        let gmr = GlobalMacroRegime::default_params().unwrap();

        assert_eq!(gmr.name(), "Global Macro Regime");
        assert!(gmr.min_periods() > 0);

        let series = OHLCVSeries {
            open: data.clone(),
            high: data.clone(),
            low: data.clone(),
            close: data.clone(),
            volume: vec![1000.0; data.len()],
        };

        let output = gmr.compute(&series);
        assert!(output.is_ok());
    }
}
