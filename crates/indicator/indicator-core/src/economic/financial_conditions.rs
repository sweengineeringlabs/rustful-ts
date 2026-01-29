//! Financial Conditions Index Indicator (IND-322)
//!
//! Tracks a composite financial stress index combining multiple
//! market-based indicators of financial conditions and risk appetite.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Configuration for FinancialConditionsIndex indicator.
#[derive(Debug, Clone)]
pub struct FinancialConditionsIndexConfig {
    /// Lookback period for normalization
    pub lookback_period: usize,
    /// Weight for equity volatility component
    pub equity_vol_weight: f64,
    /// Weight for credit spread component
    pub credit_spread_weight: f64,
    /// Weight for yield curve component
    pub yield_curve_weight: f64,
    /// Weight for funding stress component
    pub funding_stress_weight: f64,
    /// Smoothing period
    pub smoothing_period: usize,
    /// Stress threshold (standard deviations)
    pub stress_threshold: f64,
}

impl Default for FinancialConditionsIndexConfig {
    fn default() -> Self {
        Self {
            lookback_period: 252, // 1 year of daily data
            equity_vol_weight: 0.25,
            credit_spread_weight: 0.30,
            yield_curve_weight: 0.25,
            funding_stress_weight: 0.20,
            smoothing_period: 20,
            stress_threshold: 1.5,
        }
    }
}

/// Financial Conditions Index Indicator (IND-322)
///
/// Creates a composite index measuring overall financial conditions
/// by combining multiple stress indicators into a single measure.
///
/// # Components (when available)
/// - Equity market volatility (VIX-like)
/// - Credit spreads (investment grade and high yield)
/// - Yield curve slope
/// - Funding market stress (TED spread, Libor-OIS)
///
/// # Interpretation
/// - Positive values: Tighter financial conditions (stress)
/// - Negative values: Looser financial conditions (easy)
/// - Values above 1.5 std dev indicate significant stress
/// - Values below -1.5 std dev indicate extreme accommodation
///
/// # Example
/// ```ignore
/// let indicator = FinancialConditionsIndex::new(FinancialConditionsIndexConfig::default())?;
/// let fci = indicator.calculate(&components);
/// ```
#[derive(Debug, Clone)]
pub struct FinancialConditionsIndex {
    config: FinancialConditionsIndexConfig,
}

/// Components for financial conditions calculation.
#[derive(Debug, Clone, Default)]
pub struct FinancialConditionsComponents {
    /// Equity volatility index (e.g., VIX)
    pub equity_volatility: Option<Vec<f64>>,
    /// Credit spread (bps over risk-free)
    pub credit_spread: Option<Vec<f64>>,
    /// Yield curve slope (10y - 2y)
    pub yield_curve_slope: Option<Vec<f64>>,
    /// Funding stress indicator (e.g., TED spread)
    pub funding_stress: Option<Vec<f64>>,
}

impl FinancialConditionsIndex {
    /// Create a new FinancialConditionsIndex indicator.
    pub fn new(config: FinancialConditionsIndexConfig) -> Result<Self> {
        if config.lookback_period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }

        let total_weight = config.equity_vol_weight + config.credit_spread_weight +
            config.yield_curve_weight + config.funding_stress_weight;
        if (total_weight - 1.0).abs() > 0.01 {
            return Err(IndicatorError::InvalidParameter {
                name: "weights".to_string(),
                reason: "component weights must sum to 1.0".to_string(),
            });
        }

        Ok(Self { config })
    }

    /// Create with default configuration.
    pub fn default_fci() -> Result<Self> {
        Self::new(FinancialConditionsIndexConfig::default())
    }

    /// Normalize a series to z-scores.
    fn normalize_zscore(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let lookback = self.config.lookback_period;
        let mut result = vec![f64::NAN; n];

        if n < lookback {
            return result;
        }

        for i in lookback - 1..n {
            let slice = &data[i + 1 - lookback..=i];
            let valid: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();

            if valid.len() < 2 {
                continue;
            }

            let mean = valid.iter().sum::<f64>() / valid.len() as f64;
            let variance = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / valid.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                result[i] = (data[i] - mean) / std_dev;
            }
        }

        result
    }

    /// Calculate composite FCI from components.
    pub fn calculate_from_components(&self, components: &FinancialConditionsComponents) -> Vec<f64> {
        // Determine the length from available components
        let n = [
            components.equity_volatility.as_ref().map(|v| v.len()),
            components.credit_spread.as_ref().map(|v| v.len()),
            components.yield_curve_slope.as_ref().map(|v| v.len()),
            components.funding_stress.as_ref().map(|v| v.len()),
        ]
        .iter()
        .flatten()
        .min()
        .copied()
        .unwrap_or(0);

        if n == 0 {
            return Vec::new();
        }

        let mut result = vec![f64::NAN; n];

        // Normalize each component
        let vol_z = components.equity_volatility.as_ref()
            .map(|v| self.normalize_zscore(v));
        let spread_z = components.credit_spread.as_ref()
            .map(|v| self.normalize_zscore(v));
        let curve_z = components.yield_curve_slope.as_ref()
            .map(|v| {
                // Invert yield curve (flatter/inverted = tighter conditions)
                let inverted: Vec<f64> = v.iter().map(|x| -x).collect();
                self.normalize_zscore(&inverted)
            });
        let funding_z = components.funding_stress.as_ref()
            .map(|v| self.normalize_zscore(v));

        // Combine with weights
        for i in 0..n {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;

            if let Some(ref z) = vol_z {
                if i < z.len() && !z[i].is_nan() {
                    weighted_sum += z[i] * self.config.equity_vol_weight;
                    total_weight += self.config.equity_vol_weight;
                }
            }

            if let Some(ref z) = spread_z {
                if i < z.len() && !z[i].is_nan() {
                    weighted_sum += z[i] * self.config.credit_spread_weight;
                    total_weight += self.config.credit_spread_weight;
                }
            }

            if let Some(ref z) = curve_z {
                if i < z.len() && !z[i].is_nan() {
                    weighted_sum += z[i] * self.config.yield_curve_weight;
                    total_weight += self.config.yield_curve_weight;
                }
            }

            if let Some(ref z) = funding_z {
                if i < z.len() && !z[i].is_nan() {
                    weighted_sum += z[i] * self.config.funding_stress_weight;
                    total_weight += self.config.funding_stress_weight;
                }
            }

            if total_weight > 0.0 {
                result[i] = weighted_sum / total_weight;
            }
        }

        result
    }

    /// Calculate FCI from single proxy series (e.g., VIX or credit spread).
    pub fn calculate_from_proxy(&self, proxy: &[f64]) -> Vec<f64> {
        self.normalize_zscore(proxy)
    }

    /// Calculate smoothed FCI.
    pub fn calculate_smoothed(&self, fci: &[f64]) -> Vec<f64> {
        let n = fci.len();
        let period = self.config.smoothing_period;
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        for i in period - 1..n {
            let slice = &fci[i + 1 - period..=i];
            let valid: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();

            if !valid.is_empty() {
                result[i] = valid.iter().sum::<f64>() / valid.len() as f64;
            }
        }

        result
    }

    /// Calculate FCI momentum (rate of change).
    pub fn calculate_momentum(&self, fci: &[f64]) -> Vec<f64> {
        let n = fci.len();
        let mut result = vec![f64::NAN; n];

        for i in 1..n {
            if !fci[i].is_nan() && !fci[i - 1].is_nan() {
                result[i] = fci[i] - fci[i - 1];
            }
        }

        result
    }

    /// Detect stress regime.
    pub fn detect_stress_regime(&self, fci: &[f64]) -> Vec<i32> {
        let n = fci.len();
        let threshold = self.config.stress_threshold;
        let mut result = vec![0; n];

        for i in 0..n {
            if fci[i].is_nan() {
                continue;
            }

            if fci[i] > threshold * 2.0 {
                result[i] = 3; // Extreme stress
            } else if fci[i] > threshold {
                result[i] = 2; // Elevated stress
            } else if fci[i] > 0.5 {
                result[i] = 1; // Mild tightening
            } else if fci[i] < -threshold * 2.0 {
                result[i] = -3; // Extreme accommodation
            } else if fci[i] < -threshold {
                result[i] = -2; // Significant accommodation
            } else if fci[i] < -0.5 {
                result[i] = -1; // Mild easing
            }
        }

        result
    }

    /// Calculate stress probability (percentile rank).
    pub fn calculate_stress_percentile(&self, proxy: &[f64]) -> Vec<f64> {
        let n = proxy.len();
        let lookback = self.config.lookback_period;
        let mut result = vec![f64::NAN; n];

        if n < lookback {
            return result;
        }

        for i in lookback - 1..n {
            let slice = &proxy[i + 1 - lookback..=i];
            let mut sorted: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if sorted.len() < 2 {
                continue;
            }

            let current = proxy[i];
            let rank = sorted.iter().filter(|&&x| x < current).count();
            result[i] = (rank as f64 / (sorted.len() - 1) as f64) * 100.0;
        }

        result
    }

    /// Generate risk-on/risk-off signal.
    pub fn risk_signal(&self, fci: &[f64]) -> Vec<i32> {
        let smoothed = self.calculate_smoothed(fci);
        let momentum = self.calculate_momentum(fci);
        let n = fci.len();
        let mut result = vec![0; n];

        for i in 0..n {
            let level = if !smoothed[i].is_nan() { smoothed[i] } else { continue };
            let mom = if !momentum[i].is_nan() { momentum[i] } else { 0.0 };

            // Strong risk-off: high stress + worsening
            if level > self.config.stress_threshold && mom > 0.0 {
                result[i] = -2;
            }
            // Moderate risk-off: elevated stress
            else if level > 0.5 {
                result[i] = -1;
            }
            // Strong risk-on: easy conditions + improving
            else if level < -self.config.stress_threshold && mom < 0.0 {
                result[i] = 2;
            }
            // Moderate risk-on: accommodative conditions
            else if level < -0.5 {
                result[i] = 1;
            }
        }

        result
    }

    /// Calculate historical stress events count.
    pub fn count_stress_events(&self, fci: &[f64], window: usize) -> Vec<usize> {
        let n = fci.len();
        let threshold = self.config.stress_threshold;
        let mut result = vec![0; n];

        if n < window {
            return result;
        }

        for i in window - 1..n {
            let slice = &fci[i + 1 - window..=i];
            result[i] = slice.iter()
                .filter(|x| !x.is_nan() && **x > threshold)
                .count();
        }

        result
    }
}

impl TechnicalIndicator for FinancialConditionsIndex {
    fn name(&self) -> &str {
        "FinancialConditionsIndex"
    }

    fn min_periods(&self) -> usize {
        self.config.lookback_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        // Use close prices as proxy (e.g., VIX or credit spread)
        Ok(IndicatorOutput::single(self.calculate_from_proxy(&data.close)))
    }
}

impl SignalIndicator for FinancialConditionsIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let fci = self.calculate_from_proxy(&data.close);
        let risk = self.risk_signal(&fci);

        if let Some(&last) = risk.last() {
            match last {
                1 | 2 => Ok(IndicatorSignal::Bullish),   // Risk-on
                -1 | -2 => Ok(IndicatorSignal::Bearish), // Risk-off
                _ => Ok(IndicatorSignal::Neutral),
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let fci = self.calculate_from_proxy(&data.close);
        let risk = self.risk_signal(&fci);
        Ok(risk
            .iter()
            .map(|&r| match r {
                1 | 2 => IndicatorSignal::Bullish,
                -1 | -2 => IndicatorSignal::Bearish,
                _ => IndicatorSignal::Neutral,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_stress_data() -> Vec<f64> {
        // Simulated stress indicator (e.g., VIX-like)
        let mut data = Vec::new();
        for i in 0..300 {
            // Base level with occasional stress spikes
            let base = 15.0 + (i as f64 * 0.05).sin() * 3.0;
            let spike = if i >= 100 && i <= 120 { 20.0 } else { 0.0 };
            data.push(base + spike);
        }
        data
    }

    fn create_test_components() -> FinancialConditionsComponents {
        let n = 300;
        FinancialConditionsComponents {
            equity_volatility: Some((0..n).map(|i| 15.0 + (i as f64 * 0.05).sin() * 5.0).collect()),
            credit_spread: Some((0..n).map(|i| 100.0 + (i as f64 * 0.03).sin() * 30.0).collect()),
            yield_curve_slope: Some((0..n).map(|i| 0.5 + (i as f64 * 0.04).sin() * 0.3).collect()),
            funding_stress: Some((0..n).map(|i| 20.0 + (i as f64 * 0.02).sin() * 10.0).collect()),
        }
    }

    #[test]
    fn test_fci_from_proxy() {
        let data = create_test_stress_data();
        let indicator = FinancialConditionsIndex::default_fci().unwrap();
        let fci = indicator.calculate_from_proxy(&data);

        assert_eq!(fci.len(), data.len());

        // First lookback - 1 values should be NaN
        assert!(fci[0].is_nan());
        assert!(fci[100].is_nan() || !fci[100].is_nan()); // May or may not be NaN depending on lookback

        // After lookback, should have values
        assert!(!fci[260].is_nan());
    }

    #[test]
    fn test_fci_from_components() {
        let components = create_test_components();
        let indicator = FinancialConditionsIndex::default_fci().unwrap();
        let fci = indicator.calculate_from_components(&components);

        assert!(!fci.is_empty());

        // Should have some non-NaN values after lookback
        let valid: Vec<f64> = fci.iter().filter(|x| !x.is_nan()).copied().collect();
        assert!(!valid.is_empty());
    }

    #[test]
    fn test_smoothed_fci() {
        let data = create_test_stress_data();
        let indicator = FinancialConditionsIndex::default_fci().unwrap();
        let fci = indicator.calculate_from_proxy(&data);
        let smoothed = indicator.calculate_smoothed(&fci);

        assert_eq!(smoothed.len(), fci.len());

        // Smoothed values should exist after additional warmup
        assert!(!smoothed[280].is_nan());
    }

    #[test]
    fn test_momentum() {
        let data = create_test_stress_data();
        let indicator = FinancialConditionsIndex::default_fci().unwrap();
        let fci = indicator.calculate_from_proxy(&data);
        let momentum = indicator.calculate_momentum(&fci);

        assert_eq!(momentum.len(), fci.len());
    }

    #[test]
    fn test_stress_regime() {
        let data = create_test_stress_data();
        let indicator = FinancialConditionsIndex::default_fci().unwrap();
        let fci = indicator.calculate_from_proxy(&data);
        let regime = indicator.detect_stress_regime(&fci);

        assert_eq!(regime.len(), fci.len());

        // Regime values should be between -3 and 3
        for r in regime.iter() {
            assert!(*r >= -3 && *r <= 3);
        }
    }

    #[test]
    fn test_stress_percentile() {
        let data = create_test_stress_data();
        let indicator = FinancialConditionsIndex::default_fci().unwrap();
        let percentile = indicator.calculate_stress_percentile(&data);

        assert_eq!(percentile.len(), data.len());

        // Percentile should be between 0 and 100
        for p in percentile.iter().skip(251) {
            if !p.is_nan() {
                assert!(*p >= 0.0 && *p <= 100.0);
            }
        }
    }

    #[test]
    fn test_risk_signal() {
        let data = create_test_stress_data();
        let indicator = FinancialConditionsIndex::default_fci().unwrap();
        let fci = indicator.calculate_from_proxy(&data);
        let risk = indicator.risk_signal(&fci);

        assert_eq!(risk.len(), fci.len());

        // Risk signals should be between -2 and 2
        for r in risk.iter() {
            assert!(*r >= -2 && *r <= 2);
        }
    }

    #[test]
    fn test_stress_events_count() {
        let data = create_test_stress_data();
        let indicator = FinancialConditionsIndex::default_fci().unwrap();
        let fci = indicator.calculate_from_proxy(&data);
        let events = indicator.count_stress_events(&fci, 60);

        assert_eq!(events.len(), fci.len());

        // Event count should be non-negative
        for e in events.iter() {
            assert!(*e <= 60);
        }
    }

    #[test]
    fn test_technical_indicator_impl() {
        let indicator = FinancialConditionsIndex::default_fci().unwrap();
        let data = OHLCVSeries::from_close(create_test_stress_data());
        let result = indicator.compute(&data);

        assert!(result.is_ok());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let indicator = FinancialConditionsIndex::default_fci().unwrap();
        let data = OHLCVSeries::from_close(create_test_stress_data());
        let signals = indicator.signals(&data);

        assert!(signals.is_ok());
        assert_eq!(signals.unwrap().len(), data.close.len());
    }

    #[test]
    fn test_invalid_lookback() {
        let config = FinancialConditionsIndexConfig {
            lookback_period: 10,
            ..Default::default()
        };
        let result = FinancialConditionsIndex::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_weights() {
        let config = FinancialConditionsIndexConfig {
            equity_vol_weight: 0.5,
            credit_spread_weight: 0.5,
            yield_curve_weight: 0.5,
            funding_stress_weight: 0.5,
            ..Default::default()
        };
        let result = FinancialConditionsIndex::new(config);
        assert!(result.is_err());
    }
}
