//! TED Spread indicator (IND-300).
//!
//! Measures the difference between 3-month LIBOR and 3-month T-Bill rates.
//! A key indicator of credit risk in the banking system.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// TED Spread configuration.
#[derive(Debug, Clone)]
pub struct TEDSpreadConfig {
    /// Smoothing period for the spread (default: 1, no smoothing).
    pub period: usize,
    /// Stress threshold in basis points (default: 50).
    pub stress_threshold: f64,
    /// Crisis threshold in basis points (default: 100).
    pub crisis_threshold: f64,
}

impl TEDSpreadConfig {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            stress_threshold: 50.0,
            crisis_threshold: 100.0,
        }
    }

    pub fn with_thresholds(period: usize, stress: f64, crisis: f64) -> Self {
        Self {
            period,
            stress_threshold: stress,
            crisis_threshold: crisis,
        }
    }
}

impl Default for TEDSpreadConfig {
    fn default() -> Self {
        Self {
            period: 1,
            stress_threshold: 50.0,
            crisis_threshold: 100.0,
        }
    }
}

/// TED Spread (IND-300).
///
/// TED = T-bill and Eurodollar Difference
/// Calculated as: 3-month LIBOR - 3-month T-Bill rate
///
/// Interpretation:
/// - Normal range: 10-50 basis points
/// - Elevated (>50bp): Increased credit risk perception
/// - Crisis levels (>100bp): Significant banking stress
/// - Historical max during 2008: ~460bp
///
/// Note: Since LIBOR is being phased out, this indicator may use
/// SOFR or other reference rates as alternatives.
#[derive(Debug, Clone)]
pub struct TEDSpread {
    config: TEDSpreadConfig,
}

impl TEDSpread {
    pub fn new(period: usize) -> Self {
        Self {
            config: TEDSpreadConfig::new(period),
        }
    }

    pub fn from_config(config: TEDSpreadConfig) -> Self {
        Self { config }
    }

    /// Calculate TED spread from LIBOR and T-Bill rates.
    ///
    /// # Arguments
    /// * `libor_rates` - 3-month LIBOR rates (as percentages)
    /// * `tbill_rates` - 3-month T-Bill rates (as percentages)
    ///
    /// # Returns
    /// TED spread in basis points (LIBOR - T-Bill) * 100
    pub fn calculate(&self, libor_rates: &[f64], tbill_rates: &[f64]) -> Vec<f64> {
        let n = libor_rates.len().min(tbill_rates.len());
        if n == 0 {
            return vec![];
        }

        // Calculate raw spread in basis points
        let mut spreads: Vec<f64> = (0..n)
            .map(|i| (libor_rates[i] - tbill_rates[i]) * 100.0)
            .collect();

        // Apply smoothing if period > 1
        if self.config.period > 1 && n >= self.config.period {
            spreads = self.smooth(&spreads);
        }

        spreads
    }

    /// Calculate spread from dual OHLCV series (LIBOR as primary, T-Bill as benchmark).
    pub fn calculate_from_series(&self, libor: &OHLCVSeries, tbill: &OHLCVSeries) -> Vec<f64> {
        self.calculate(&libor.close, &tbill.close)
    }

    /// Apply simple moving average smoothing.
    fn smooth(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let period = self.config.period;

        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..n {
            let sum: f64 = data[(i + 1 - period)..=i].iter().sum();
            result.push(sum / period as f64);
        }

        result
    }

    /// Classify current spread level.
    pub fn classify(&self, spread: f64) -> TEDSpreadLevel {
        if spread.is_nan() {
            TEDSpreadLevel::Unknown
        } else if spread < 0.0 {
            TEDSpreadLevel::Inverted
        } else if spread < self.config.stress_threshold {
            TEDSpreadLevel::Normal
        } else if spread < self.config.crisis_threshold {
            TEDSpreadLevel::Elevated
        } else {
            TEDSpreadLevel::Crisis
        }
    }

    /// Calculate rate of change of spread.
    pub fn spread_momentum(&self, spreads: &[f64], lookback: usize) -> Vec<f64> {
        let n = spreads.len();
        if n < lookback + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; lookback];

        for i in lookback..n {
            if spreads[i - lookback] != 0.0 && !spreads[i - lookback].is_nan() {
                result.push(spreads[i] - spreads[i - lookback]);
            } else {
                result.push(f64::NAN);
            }
        }

        result
    }
}

/// TED Spread level classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TEDSpreadLevel {
    /// Spread is negative (unusual)
    Inverted,
    /// Normal range (< stress threshold)
    Normal,
    /// Elevated risk (>= stress, < crisis)
    Elevated,
    /// Crisis level (>= crisis threshold)
    Crisis,
    /// Cannot determine
    Unknown,
}

impl TechnicalIndicator for TEDSpread {
    fn name(&self) -> &str {
        "TEDSpread"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // Convention: close contains LIBOR, open contains T-Bill
        if data.close.len() < self.config.period {
            return Err(IndicatorError::InsufficientData {
                required: self.config.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close, &data.open);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.config.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ted_spread_basic() {
        let indicator = TEDSpread::new(1);

        // LIBOR at 2.5%, T-Bill at 2.0% = 50bp spread
        let libor = vec![2.5, 2.8, 3.0, 2.7, 2.4];
        let tbill = vec![2.0, 2.1, 2.2, 2.0, 1.9];

        let result = indicator.calculate(&libor, &tbill);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 50.0).abs() < 0.001); // 2.5 - 2.0 = 0.5% = 50bp
        assert!((result[1] - 70.0).abs() < 0.001); // 2.8 - 2.1 = 0.7% = 70bp
    }

    #[test]
    fn test_ted_spread_smoothed() {
        let indicator = TEDSpread::new(3);

        let libor = vec![2.5, 2.8, 3.0, 2.7, 2.4];
        let tbill = vec![2.0, 2.1, 2.2, 2.0, 1.9];

        let result = indicator.calculate(&libor, &tbill);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Third value should be average of first 3 spreads
        let expected = (50.0 + 70.0 + 80.0) / 3.0;
        assert!((result[2] - expected).abs() < 0.001);
    }

    #[test]
    fn test_ted_spread_classification() {
        let indicator = TEDSpread::from_config(TEDSpreadConfig::with_thresholds(1, 50.0, 100.0));

        assert_eq!(indicator.classify(25.0), TEDSpreadLevel::Normal);
        assert_eq!(indicator.classify(75.0), TEDSpreadLevel::Elevated);
        assert_eq!(indicator.classify(150.0), TEDSpreadLevel::Crisis);
        assert_eq!(indicator.classify(-10.0), TEDSpreadLevel::Inverted);
    }

    #[test]
    fn test_ted_spread_momentum() {
        let indicator = TEDSpread::new(1);

        let spreads = vec![50.0, 60.0, 70.0, 65.0, 80.0];
        let momentum = indicator.spread_momentum(&spreads, 2);

        assert_eq!(momentum.len(), 5);
        assert!(momentum[0].is_nan());
        assert!(momentum[1].is_nan());
        assert!((momentum[2] - 20.0).abs() < 0.001); // 70 - 50 = 20
        assert!((momentum[3] - 5.0).abs() < 0.001);  // 65 - 60 = 5
    }

    #[test]
    fn test_ted_spread_crisis_levels() {
        let indicator = TEDSpread::new(1);

        // Simulate 2008-like crisis spreads
        let libor = vec![3.0, 4.0, 5.0, 6.0];
        let tbill = vec![2.0, 1.5, 1.0, 0.5];

        let result = indicator.calculate(&libor, &tbill);

        // Last spread should be at crisis level (550bp)
        assert!(result[3] > 100.0);
        assert_eq!(indicator.classify(result[3]), TEDSpreadLevel::Crisis);
    }
}
