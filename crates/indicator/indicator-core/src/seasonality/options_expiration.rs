//! Options Expiration Effect Indicator (IND-241)
//!
//! Analyzes market behavior during options expiration (OpEx) weeks.
//! OpEx weeks often exhibit increased volatility and specific patterns.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Classification of OpEx week timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpExWeek {
    /// Regular week (not OpEx)
    Regular,
    /// Week before OpEx
    PreOpEx,
    /// Options expiration week (third Friday of month)
    OpExWeek,
    /// Triple/Quadruple witching (quarterly OpEx)
    QuadWitching,
    /// Week after OpEx
    PostOpEx,
}

/// Configuration for the Options Expiration Effect indicator.
#[derive(Debug, Clone)]
pub struct OptionsExpirationConfig {
    /// Lookback period for volatility comparison
    pub volatility_lookback: usize,
    /// Number of historical OpEx periods to analyze
    pub historical_opex_count: usize,
    /// Threshold for considering volatility elevated
    pub volatility_threshold: f64,
}

impl Default for OptionsExpirationConfig {
    fn default() -> Self {
        Self {
            volatility_lookback: 20,
            historical_opex_count: 4,
            volatility_threshold: 1.2,
        }
    }
}

/// Options Expiration Effect indicator.
///
/// This indicator helps identify:
/// - Pin risk around popular strike prices
/// - Gamma exposure effects on market makers
/// - Volatility patterns during OpEx weeks
/// - Triple/Quadruple witching effects
#[derive(Debug, Clone)]
pub struct OptionsExpirationEffect {
    config: OptionsExpirationConfig,
}

impl OptionsExpirationEffect {
    /// Create a new Options Expiration Effect indicator.
    pub fn new() -> Self {
        Self {
            config: OptionsExpirationConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: OptionsExpirationConfig) -> Self {
        Self { config }
    }

    /// Estimate OpEx week based on trading day index.
    /// In practice, this would use actual calendar data.
    /// OpEx is typically the third Friday of each month (~15-21 trading days into month).
    fn estimate_opex_week(&self, day_index: usize) -> OpExWeek {
        // Approximate: ~21 trading days per month, OpEx around day 15-16
        let day_in_month = day_index % 21;
        let month_index = day_index / 21;

        // Third Friday falls around trading day 14-16
        let is_opex_week = day_in_month >= 13 && day_in_month <= 17;
        let is_quad_witching = is_opex_week && (month_index % 3 == 2); // Quarterly

        if is_quad_witching {
            OpExWeek::QuadWitching
        } else if is_opex_week {
            OpExWeek::OpExWeek
        } else if day_in_month >= 10 && day_in_month < 13 {
            OpExWeek::PreOpEx
        } else if day_in_month > 17 && day_in_month <= 20 {
            OpExWeek::PostOpEx
        } else {
            OpExWeek::Regular
        }
    }

    /// Calculate realized volatility for a window.
    fn calculate_volatility(&self, close: &[f64], start: usize, end: usize) -> f64 {
        if end <= start || end > close.len() || end - start < 2 {
            return f64::NAN;
        }

        let returns: Vec<f64> = (start + 1..end)
            .filter_map(|i| {
                if close[i - 1] > 0.0 {
                    Some((close[i] / close[i - 1]).ln())
                } else {
                    None
                }
            })
            .collect();

        if returns.is_empty() {
            return f64::NAN;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        // Annualized volatility (assuming 252 trading days)
        (variance.sqrt() * (252.0_f64).sqrt())
    }

    /// Calculate OpEx volatility ratio (current vs historical average).
    fn calculate_opex_volatility_ratio(
        &self,
        close: &[f64],
        high: &[f64],
        low: &[f64],
        idx: usize,
    ) -> f64 {
        if idx < self.config.volatility_lookback {
            return f64::NAN;
        }

        // Current volatility (using range-based)
        let current_range: f64 = (idx.saturating_sub(5)..=idx)
            .filter_map(|i| {
                if i < high.len() && low[i] > 0.0 {
                    Some((high[i] - low[i]) / low[i])
                } else {
                    None
                }
            })
            .sum::<f64>() / 5.0;

        // Historical average volatility
        let historical_range: f64 = (idx.saturating_sub(self.config.volatility_lookback)..idx.saturating_sub(5))
            .filter_map(|i| {
                if i < high.len() && low[i] > 0.0 {
                    Some((high[i] - low[i]) / low[i])
                } else {
                    None
                }
            })
            .sum::<f64>() / (self.config.volatility_lookback - 5) as f64;

        if historical_range > 0.0 {
            current_range / historical_range
        } else {
            1.0
        }
    }

    /// Calculate the pin effect strength (tendency to close near round numbers).
    fn calculate_pin_strength(&self, close: &[f64], idx: usize) -> f64 {
        if idx < 5 || close[idx] <= 0.0 {
            return 0.0;
        }

        let price = close[idx];

        // Find nearest round strike (assuming $5 increments for simplicity)
        let strike_interval = if price > 100.0 { 5.0 } else { 2.5 };
        let nearest_strike = (price / strike_interval).round() * strike_interval;

        // Distance from nearest strike as percentage
        let distance = (price - nearest_strike).abs() / price;

        // Pin strength is higher when close to strike
        // Max at strike (1.0), decays with distance
        let pin_strength = (-distance * 100.0).exp();

        pin_strength
    }

    /// Calculate the options expiration effect indicators.
    pub fn calculate(
        &self,
        close: &[f64],
        high: &[f64],
        low: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<OpExWeek>) {
        let n = close.len();
        let mut volatility_ratio = vec![f64::NAN; n];
        let mut pin_strength = vec![f64::NAN; n];
        let mut opex_phase = vec![OpExWeek::Regular; n];

        for i in 0..n {
            opex_phase[i] = self.estimate_opex_week(i);

            if i >= self.config.volatility_lookback {
                volatility_ratio[i] = self.calculate_opex_volatility_ratio(close, high, low, i);
                pin_strength[i] = self.calculate_pin_strength(close, i);
            }
        }

        (volatility_ratio, pin_strength, opex_phase)
    }

    /// Get the primary OpEx effect signal (volatility ratio).
    pub fn calculate_signal(&self, close: &[f64], high: &[f64], low: &[f64]) -> Vec<f64> {
        self.calculate(close, high, low).0
    }
}

impl Default for OptionsExpirationEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for OptionsExpirationEffect {
    fn name(&self) -> &str {
        "OptionsExpirationEffect"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.volatility_lookback {
            return Err(IndicatorError::InsufficientData {
                required: self.config.volatility_lookback,
                got: data.close.len(),
            });
        }

        let (volatility_ratio, pin_strength, _) =
            self.calculate(&data.close, &data.high, &data.low);

        Ok(IndicatorOutput::dual(
            volatility_ratio,
            pin_strength,
        ))
    }

    fn min_periods(&self) -> usize {
        self.config.volatility_lookback
    }
}

impl SignalIndicator for OptionsExpirationEffect {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.len() < self.config.volatility_lookback {
            return Ok(IndicatorSignal::Neutral);
        }

        let (volatility_ratio, pin_strength, opex_phase) =
            self.calculate(&data.close, &data.high, &data.low);
        let n = volatility_ratio.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let vol_ratio = volatility_ratio[n - 1];
        let pin = pin_strength[n - 1];
        let phase = opex_phase[n - 1];

        // During OpEx with high pin strength and low volatility, expect mean reversion
        if matches!(phase, OpExWeek::OpExWeek | OpExWeek::QuadWitching) {
            if pin > 0.8 && vol_ratio < 1.0 {
                // Strong pin effect, low volatility - price likely to stay pinned
                return Ok(IndicatorSignal::Neutral);
            } else if vol_ratio > self.config.volatility_threshold {
                // High volatility during OpEx - caution
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (volatility_ratio, pin_strength, opex_phase) =
            self.calculate(&data.close, &data.high, &data.low);

        let signals = volatility_ratio
            .iter()
            .enumerate()
            .map(|(i, &vol_ratio)| {
                if vol_ratio.is_nan() {
                    return IndicatorSignal::Neutral;
                }

                let phase = opex_phase[i];
                let pin = pin_strength[i];

                if matches!(phase, OpExWeek::OpExWeek | OpExWeek::QuadWitching) {
                    if pin > 0.8 && vol_ratio < 1.0 {
                        IndicatorSignal::Neutral
                    } else if vol_ratio > self.config.volatility_threshold {
                        IndicatorSignal::Bearish
                    } else {
                        IndicatorSignal::Neutral
                    }
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n: usize) -> OHLCVSeries {
        OHLCVSeries {
            open: (0..n).map(|i| 100.0 + (i as f64 * 0.1)).collect(),
            high: (0..n).map(|i| 102.0 + (i as f64 * 0.1)).collect(),
            low: (0..n).map(|i| 98.0 + (i as f64 * 0.1)).collect(),
            close: (0..n).map(|i| 100.0 + (i as f64 * 0.15)).collect(),
            volume: vec![1000000.0; n],
        }
    }

    #[test]
    fn test_opex_effect_basic() {
        let indicator = OptionsExpirationEffect::new();
        let data = create_test_data(50);

        let (vol_ratio, pin_strength, opex_phase) =
            indicator.calculate(&data.close, &data.high, &data.low);

        assert_eq!(vol_ratio.len(), 50);
        assert_eq!(pin_strength.len(), 50);
        assert_eq!(opex_phase.len(), 50);
    }

    #[test]
    fn test_opex_phases() {
        let indicator = OptionsExpirationEffect::new();
        let data = create_test_data(100);

        let (_, _, opex_phase) = indicator.calculate(&data.close, &data.high, &data.low);

        // Should have OpEx weeks in the data
        let has_opex = opex_phase.iter().any(|p| matches!(p, OpExWeek::OpExWeek));
        let has_regular = opex_phase.iter().any(|p| matches!(p, OpExWeek::Regular));

        assert!(has_opex);
        assert!(has_regular);
    }

    #[test]
    fn test_opex_volatility_ratio() {
        let indicator = OptionsExpirationEffect::new();
        let data = create_test_data(50);

        let (vol_ratio, _, _) = indicator.calculate(&data.close, &data.high, &data.low);

        // Valid values should be positive
        for &ratio in vol_ratio.iter().filter(|v| !v.is_nan()) {
            assert!(ratio >= 0.0);
        }
    }

    #[test]
    fn test_opex_pin_strength() {
        let indicator = OptionsExpirationEffect::new();
        let data = create_test_data(50);

        let (_, pin_strength, _) = indicator.calculate(&data.close, &data.high, &data.low);

        // Pin strength should be between 0 and 1
        for &pin in pin_strength.iter().filter(|v| !v.is_nan()) {
            assert!(pin >= 0.0 && pin <= 1.0);
        }
    }

    #[test]
    fn test_opex_insufficient_data() {
        let indicator = OptionsExpirationEffect::new();
        let data = create_test_data(10);

        let result = indicator.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_opex_technical_indicator() {
        let indicator = OptionsExpirationEffect::new();
        let data = create_test_data(50);

        let result = indicator.compute(&data);
        assert!(result.is_ok());

        assert_eq!(indicator.name(), "OptionsExpirationEffect");
        assert_eq!(indicator.min_periods(), 20);
    }
}
