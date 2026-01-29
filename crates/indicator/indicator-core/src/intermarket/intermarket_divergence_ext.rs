//! Intermarket Divergence Extended (IND-403)
//!
//! Extended asset class divergence indicator that measures divergences
//! across multiple asset classes to identify regime changes and
//! potential market turning points.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Divergence type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceType {
    /// Strong bullish divergence (risk assets diverging up).
    StrongBullish,
    /// Moderate bullish divergence.
    ModerateBullish,
    /// No significant divergence.
    None,
    /// Moderate bearish divergence.
    ModerateBearish,
    /// Strong bearish divergence (risk assets diverging down).
    StrongBearish,
}

impl DivergenceType {
    /// Convert to numeric score (-2 to +2).
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::StrongBullish => 2.0,
            Self::ModerateBullish => 1.0,
            Self::None => 0.0,
            Self::ModerateBearish => -1.0,
            Self::StrongBearish => -2.0,
        }
    }

    /// Create from score.
    pub fn from_score(score: f64) -> Self {
        if score > 1.5 {
            Self::StrongBullish
        } else if score > 0.5 {
            Self::ModerateBullish
        } else if score > -0.5 {
            Self::None
        } else if score > -1.5 {
            Self::ModerateBearish
        } else {
            Self::StrongBearish
        }
    }
}

/// Asset class divergence data.
#[derive(Debug, Clone)]
pub struct AssetClassDivergence {
    /// Asset class name.
    pub name: String,
    /// Current momentum.
    pub momentum: f64,
    /// Deviation from group average.
    pub deviation: f64,
    /// Z-score of deviation.
    pub z_score: f64,
    /// Is diverging from group.
    pub is_diverging: bool,
}

/// Intermarket Divergence Extended output.
#[derive(Debug, Clone)]
pub struct IntermarketDivergenceExtOutput {
    /// Overall divergence score.
    pub divergence_score: f64,
    /// Divergence type classification.
    pub divergence_type: DivergenceType,
    /// Cross-asset correlation.
    pub cross_correlation: f64,
    /// Divergence breadth (% of assets diverging).
    pub divergence_breadth: f64,
    /// Per-asset divergence data.
    pub asset_divergences: Vec<AssetClassDivergence>,
    /// Regime change signal strength.
    pub regime_change_signal: f64,
}

/// Intermarket Divergence Extended (IND-403)
///
/// Measures divergences across multiple asset classes (equities, bonds,
/// commodities, currencies, volatility) to identify potential regime
/// changes and market turning points.
///
/// # Methodology
/// 1. Calculate momentum for each asset class
/// 2. Compute cross-asset correlations
/// 3. Identify assets diverging from historical correlations
/// 4. Calculate divergence breadth and intensity
/// 5. Generate regime change signals
///
/// # Interpretation
/// - High divergence breadth: Potential regime change
/// - Correlation breakdown: Market stress or transition
/// - Persistent divergence: Trend change likely
#[derive(Debug, Clone)]
pub struct IntermarketDivergenceExt {
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Period for correlation calculation.
    correlation_period: usize,
    /// Period for z-score normalization.
    zscore_period: usize,
    /// Divergence threshold (standard deviations).
    divergence_threshold: f64,
    /// Asset class names.
    asset_names: Vec<String>,
    /// Asset class price series.
    asset_prices: Vec<Vec<f64>>,
}

impl IntermarketDivergenceExt {
    /// Create a new Intermarket Divergence Extended indicator.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (e.g., 20)
    /// * `correlation_period` - Period for correlation calculation (e.g., 60)
    /// * `zscore_period` - Period for z-score normalization (e.g., 252)
    /// * `divergence_threshold` - Threshold in standard deviations (e.g., 1.5)
    pub fn new(
        momentum_period: usize,
        correlation_period: usize,
        zscore_period: usize,
        divergence_threshold: f64,
    ) -> Result<Self> {
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if correlation_period < momentum_period {
            return Err(IndicatorError::InvalidParameter {
                name: "correlation_period".to_string(),
                reason: "must be at least momentum_period".to_string(),
            });
        }
        if divergence_threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "divergence_threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            correlation_period,
            zscore_period,
            divergence_threshold,
            asset_names: Vec::new(),
            asset_prices: Vec::new(),
        })
    }

    /// Create with default parameters (20, 60, 252, 1.5).
    pub fn default_params() -> Result<Self> {
        Self::new(20, 60, 252, 1.5)
    }

    /// Add an asset class.
    pub fn add_asset(&mut self, name: &str, prices: Vec<f64>) {
        self.asset_names.push(name.to_string());
        self.asset_prices.push(prices);
    }

    /// Set assets from vectors.
    pub fn with_assets(mut self, names: Vec<String>, prices: Vec<Vec<f64>>) -> Self {
        self.asset_names = names;
        self.asset_prices = prices;
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

    /// Calculate correlation between two series.
    fn correlation(series1: &[f64], series2: &[f64]) -> f64 {
        if series1.len() != series2.len() || series1.len() < 2 {
            return f64::NAN;
        }

        let valid: Vec<(f64, f64)> = series1
            .iter()
            .zip(series2.iter())
            .filter(|(a, b)| !a.is_nan() && !b.is_nan())
            .map(|(a, b)| (*a, *b))
            .collect();

        if valid.len() < 2 {
            return f64::NAN;
        }

        let n = valid.len() as f64;
        let mean1: f64 = valid.iter().map(|(a, _)| a).sum::<f64>() / n;
        let mean2: f64 = valid.iter().map(|(_, b)| b).sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (a, b) in &valid {
            let d1 = a - mean1;
            let d2 = b - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate average cross-correlation for a window.
    fn average_cross_correlation(momentums: &[Vec<f64>], index: usize, period: usize) -> f64 {
        let n_assets = momentums.len();
        if n_assets < 2 || index < period {
            return f64::NAN;
        }

        let start = index + 1 - period;
        let mut correlations = Vec::new();

        for i in 0..n_assets {
            for j in (i + 1)..n_assets {
                let window1: Vec<f64> = momentums[i][start..=index].to_vec();
                let window2: Vec<f64> = momentums[j][start..=index].to_vec();
                let corr = Self::correlation(&window1, &window2);
                if !corr.is_nan() {
                    correlations.push(corr);
                }
            }
        }

        if correlations.is_empty() {
            f64::NAN
        } else {
            correlations.iter().sum::<f64>() / correlations.len() as f64
        }
    }

    /// Calculate rolling z-score.
    fn rolling_zscore(data: &[f64], current: f64, period: usize, index: usize) -> f64 {
        if index < period {
            return f64::NAN;
        }

        let window: Vec<f64> = data[(index + 1 - period)..=index]
            .iter()
            .filter(|x| !x.is_nan())
            .copied()
            .collect();

        if window.len() < 2 {
            return f64::NAN;
        }

        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / window.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev > 1e-10 {
            (current - mean) / std_dev
        } else {
            0.0
        }
    }

    /// Calculate Intermarket Divergence Extended outputs.
    pub fn calculate(&self) -> Vec<IntermarketDivergenceExtOutput> {
        if self.asset_prices.len() < 2 {
            return Vec::new();
        }

        let n = self.asset_prices.iter().map(|p| p.len()).min().unwrap_or(0);
        if n < self.correlation_period {
            return Vec::new();
        }

        // Calculate momentums for all assets
        let momentums: Vec<Vec<f64>> = self
            .asset_prices
            .iter()
            .map(|p| Self::calculate_roc(p, self.momentum_period))
            .collect();

        let mut results = Vec::with_capacity(n);
        let mut prev_correlations: Vec<f64> = Vec::new();

        for i in 0..n {
            // Calculate average momentum
            let valid_moms: Vec<f64> = momentums
                .iter()
                .filter_map(|m| {
                    let v = m[i];
                    if v.is_nan() { None } else { Some(v) }
                })
                .collect();

            let avg_momentum = if valid_moms.is_empty() {
                0.0
            } else {
                valid_moms.iter().sum::<f64>() / valid_moms.len() as f64
            };

            // Calculate cross-correlation
            let cross_corr = Self::average_cross_correlation(
                &momentums,
                i,
                self.correlation_period.min(i + 1),
            );
            prev_correlations.push(cross_corr);

            // Calculate per-asset divergences
            let mut asset_divergences = Vec::new();
            let mut diverging_count = 0;
            let mut total_deviation = 0.0;

            for (idx, name) in self.asset_names.iter().enumerate() {
                let momentum = momentums[idx][i];
                let deviation = if momentum.is_nan() {
                    f64::NAN
                } else {
                    momentum - avg_momentum
                };

                let z_score = if !deviation.is_nan() && i >= self.zscore_period {
                    // Calculate z-score of this asset's deviation
                    let deviations: Vec<f64> = (0..=i)
                        .map(|j| {
                            let m = momentums[idx][j];
                            let valid: Vec<f64> = momentums
                                .iter()
                                .filter_map(|moms| {
                                    let v = moms[j];
                                    if v.is_nan() { None } else { Some(v) }
                                })
                                .collect();
                            let avg = if valid.is_empty() {
                                0.0
                            } else {
                                valid.iter().sum::<f64>() / valid.len() as f64
                            };
                            if m.is_nan() { f64::NAN } else { m - avg }
                        })
                        .collect();
                    Self::rolling_zscore(&deviations, deviation, self.zscore_period, i)
                } else {
                    f64::NAN
                };

                let is_diverging = !z_score.is_nan() && z_score.abs() > self.divergence_threshold;
                if is_diverging {
                    diverging_count += 1;
                }
                if !deviation.is_nan() {
                    total_deviation += deviation.abs();
                }

                asset_divergences.push(AssetClassDivergence {
                    name: name.clone(),
                    momentum,
                    deviation,
                    z_score,
                    is_diverging,
                });
            }

            // Calculate divergence breadth
            let divergence_breadth = diverging_count as f64 / self.asset_names.len().max(1) as f64;

            // Calculate divergence score
            let avg_abs_deviation = total_deviation / self.asset_names.len().max(1) as f64;
            let divergence_score = if cross_corr.is_nan() {
                avg_abs_deviation
            } else {
                // Higher score when correlation is low and deviations are high
                avg_abs_deviation * (1.0 - cross_corr.abs())
            };

            // Calculate regime change signal
            let regime_change_signal = if i >= self.correlation_period {
                // Compare current correlation to historical
                let hist_corrs: Vec<f64> = prev_correlations
                    [i.saturating_sub(self.zscore_period)..i]
                    .iter()
                    .filter(|x| !x.is_nan())
                    .copied()
                    .collect();
                if hist_corrs.len() >= 2 && !cross_corr.is_nan() {
                    let mean = hist_corrs.iter().sum::<f64>() / hist_corrs.len() as f64;
                    let variance = hist_corrs.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / hist_corrs.len() as f64;
                    let std = variance.sqrt();
                    if std > 1e-10 {
                        ((mean - cross_corr) / std).abs() // Correlation breakdown
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let divergence_type = DivergenceType::from_score(
                if divergence_breadth > 0.5 {
                    divergence_score.signum() * 2.0
                } else if divergence_breadth > 0.25 {
                    divergence_score.signum()
                } else {
                    0.0
                },
            );

            results.push(IntermarketDivergenceExtOutput {
                divergence_score,
                divergence_type,
                cross_correlation: cross_corr,
                divergence_breadth,
                asset_divergences,
                regime_change_signal,
            });
        }

        results
    }

    /// Get divergence score series.
    pub fn divergence_scores(&self) -> Vec<f64> {
        self.calculate().iter().map(|o| o.divergence_score).collect()
    }

    /// Get cross-correlation series.
    pub fn cross_correlations(&self) -> Vec<f64> {
        self.calculate().iter().map(|o| o.cross_correlation).collect()
    }

    /// Get divergence breadth series.
    pub fn divergence_breadths(&self) -> Vec<f64> {
        self.calculate().iter().map(|o| o.divergence_breadth).collect()
    }

    /// Get regime change signal series.
    pub fn regime_change_signals(&self) -> Vec<f64> {
        self.calculate()
            .iter()
            .map(|o| o.regime_change_signal)
            .collect()
    }
}

impl TechnicalIndicator for IntermarketDivergenceExt {
    fn name(&self) -> &str {
        "Intermarket Divergence Extended"
    }

    fn min_periods(&self) -> usize {
        self.correlation_period
    }

    fn compute(&self, _data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.asset_prices.len() < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "asset_prices".to_string(),
                reason: "At least 2 asset classes must be set before computing".to_string(),
            });
        }

        let min_len = self.asset_prices.iter().map(|p| p.len()).min().unwrap_or(0);
        if min_len < self.correlation_period {
            return Err(IndicatorError::InsufficientData {
                required: self.correlation_period,
                got: min_len,
            });
        }

        let scores = self.divergence_scores();
        Ok(IndicatorOutput::single(scores))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_indicator(n: usize) -> IntermarketDivergenceExt {
        // Create synthetic asset class data with different patterns
        let equities: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.3 + ((i as f64) * 0.1).sin() * 3.0)
            .collect();

        let bonds: Vec<f64> = (0..n)
            .map(|i| 100.0 - (i as f64) * 0.1 + ((i as f64) * 0.1).cos() * 2.0)
            .collect();

        let commodities: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.2 + ((i as f64) * 0.15).sin() * 4.0)
            .collect();

        let currencies: Vec<f64> = (0..n)
            .map(|i| 100.0 + ((i as f64) * 0.08).sin() * 2.0)
            .collect();

        let mut indicator = IntermarketDivergenceExt::new(20, 60, 100, 1.5).unwrap();
        indicator.add_asset("Equities", equities);
        indicator.add_asset("Bonds", bonds);
        indicator.add_asset("Commodities", commodities);
        indicator.add_asset("Currencies", currencies);

        indicator
    }

    #[test]
    fn test_divergence_basic() {
        let indicator = create_test_indicator(150);
        let outputs = indicator.calculate();

        assert_eq!(outputs.len(), 150);

        // Check last output
        let last = outputs.last().unwrap();
        assert!(last.divergence_score.is_finite());
        assert_eq!(last.asset_divergences.len(), 4);
    }

    #[test]
    fn test_cross_correlation() {
        let indicator = create_test_indicator(150);
        let correlations = indicator.cross_correlations();

        // After warmup, should have valid correlations
        for corr in correlations.iter().skip(70) {
            if !corr.is_nan() {
                assert!(*corr >= -1.0 && *corr <= 1.0);
            }
        }
    }

    #[test]
    fn test_divergence_breadth() {
        let indicator = create_test_indicator(150);
        let breadths = indicator.divergence_breadths();

        // Breadth should be between 0 and 1
        for breadth in &breadths {
            assert!(*breadth >= 0.0 && *breadth <= 1.0);
        }
    }

    #[test]
    fn test_asset_divergence_data() {
        let indicator = create_test_indicator(150);
        let outputs = indicator.calculate();

        let last = outputs.last().unwrap();
        for div in &last.asset_divergences {
            assert!(!div.name.is_empty());
            // Momentum and deviation should be finite (may be NaN early)
        }
    }

    #[test]
    fn test_regime_change_signal() {
        let indicator = create_test_indicator(150);
        let signals = indicator.regime_change_signals();

        assert_eq!(signals.len(), 150);
        // Signals should be non-negative
        for signal in &signals {
            assert!(*signal >= 0.0);
        }
    }

    #[test]
    fn test_divergence_type() {
        assert_eq!(DivergenceType::from_score(2.0), DivergenceType::StrongBullish);
        assert_eq!(DivergenceType::from_score(0.0), DivergenceType::None);
        assert_eq!(DivergenceType::from_score(-2.0), DivergenceType::StrongBearish);
    }

    #[test]
    fn test_with_builder_pattern() {
        let indicator = IntermarketDivergenceExt::new(20, 60, 100, 1.5)
            .unwrap()
            .with_assets(
                vec!["A".to_string(), "B".to_string()],
                vec![
                    (0..100).map(|i| 100.0 + i as f64).collect(),
                    (0..100).map(|i| 100.0 - i as f64 * 0.1).collect(),
                ],
            );

        let outputs = indicator.calculate();
        assert!(!outputs.is_empty());
    }

    #[test]
    fn test_technical_indicator_impl() {
        let indicator = create_test_indicator(150);
        let data = OHLCVSeries::from_close(vec![100.0; 150]);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 150);
    }

    #[test]
    fn test_insufficient_assets_error() {
        let mut indicator = IntermarketDivergenceExt::new(20, 60, 100, 1.5).unwrap();
        indicator.add_asset("Single", vec![100.0; 100]);

        let data = OHLCVSeries::from_close(vec![100.0; 100]);
        let result = indicator.compute(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_parameters() {
        let result = IntermarketDivergenceExt::new(2, 60, 100, 1.5);
        assert!(result.is_err());

        let result = IntermarketDivergenceExt::new(20, 10, 100, 1.5);
        assert!(result.is_err());

        let result = IntermarketDivergenceExt::new(20, 60, 100, 0.0);
        assert!(result.is_err());
    }
}
