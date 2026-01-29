//! Credit Spread indicator (IND-299).
//!
//! Measures the difference between corporate bond yields and Treasury yields
//! of similar maturity. Widening spreads indicate increased credit risk.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Credit Spread configuration.
#[derive(Debug, Clone)]
pub struct CreditSpreadConfig {
    /// Smoothing period for the spread (default: 1, no smoothing).
    pub period: usize,
    /// Warning threshold for widening spread (in basis points).
    pub warning_threshold: f64,
    /// Critical threshold for widening spread (in basis points).
    pub critical_threshold: f64,
}

impl CreditSpreadConfig {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            warning_threshold: 200.0,
            critical_threshold: 400.0,
        }
    }

    pub fn with_thresholds(period: usize, warning: f64, critical: f64) -> Self {
        Self {
            period,
            warning_threshold: warning,
            critical_threshold: critical,
        }
    }
}

impl Default for CreditSpreadConfig {
    fn default() -> Self {
        Self {
            period: 1,
            warning_threshold: 200.0,
            critical_threshold: 400.0,
        }
    }
}

/// Credit Spread (IND-299).
///
/// Calculates the difference between corporate bond yields and Treasury yields.
/// Corporate yield - Treasury yield = Credit Spread
///
/// Interpretation:
/// - Widening spreads indicate increased credit risk / flight to safety
/// - Narrowing spreads indicate risk-on sentiment / credit improvement
/// - Typically expressed in basis points (100 bp = 1%)
#[derive(Debug, Clone)]
pub struct CreditSpread {
    config: CreditSpreadConfig,
}

impl CreditSpread {
    pub fn new(period: usize) -> Self {
        Self {
            config: CreditSpreadConfig::new(period),
        }
    }

    pub fn from_config(config: CreditSpreadConfig) -> Self {
        Self { config }
    }

    /// Calculate credit spread from corporate and treasury yields.
    ///
    /// # Arguments
    /// * `corporate_yields` - Corporate bond yields (as percentages)
    /// * `treasury_yields` - Treasury yields (as percentages)
    ///
    /// # Returns
    /// Credit spread in basis points (corporate - treasury) * 100
    pub fn calculate(&self, corporate_yields: &[f64], treasury_yields: &[f64]) -> Vec<f64> {
        let n = corporate_yields.len().min(treasury_yields.len());
        if n == 0 {
            return vec![];
        }

        // Calculate raw spread in basis points
        let mut spreads: Vec<f64> = (0..n)
            .map(|i| (corporate_yields[i] - treasury_yields[i]) * 100.0)
            .collect();

        // Apply smoothing if period > 1
        if self.config.period > 1 && n >= self.config.period {
            spreads = self.smooth(&spreads);
        }

        spreads
    }

    /// Calculate spread from dual OHLCV series (corporate as primary, treasury as benchmark).
    pub fn calculate_from_series(&self, corporate: &OHLCVSeries, treasury: &OHLCVSeries) -> Vec<f64> {
        self.calculate(&corporate.close, &treasury.close)
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

    /// Calculate z-score of current spread relative to historical values.
    pub fn z_score(&self, spreads: &[f64], lookback: usize) -> Vec<f64> {
        let n = spreads.len();
        if n < lookback {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; lookback - 1];

        for i in (lookback - 1)..n {
            let window = &spreads[(i + 1 - lookback)..=i];
            let mean: f64 = window.iter().sum::<f64>() / lookback as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / lookback as f64;
            let std_dev = variance.sqrt();

            if std_dev > 0.0 {
                result.push((spreads[i] - mean) / std_dev);
            } else {
                result.push(0.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for CreditSpread {
    fn name(&self) -> &str {
        "CreditSpread"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // When using single series, assume close contains corporate yields
        // and open contains treasury yields (convention for spread calculation)
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
    fn test_credit_spread_basic() {
        let indicator = CreditSpread::new(1);

        // Corporate yields at 5%, Treasury at 3% = 200bp spread
        let corporate = vec![5.0, 5.5, 6.0, 5.8, 5.2];
        let treasury = vec![3.0, 3.2, 3.5, 3.3, 3.0];

        let result = indicator.calculate(&corporate, &treasury);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 200.0).abs() < 0.001); // 5.0 - 3.0 = 2.0% = 200bp
        assert!((result[1] - 230.0).abs() < 0.001); // 5.5 - 3.2 = 2.3% = 230bp
    }

    #[test]
    fn test_credit_spread_smoothed() {
        let indicator = CreditSpread::new(3);

        let corporate = vec![5.0, 5.5, 6.0, 5.8, 5.2];
        let treasury = vec![3.0, 3.2, 3.5, 3.3, 3.0];

        let result = indicator.calculate(&corporate, &treasury);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Third value should be average of first 3 spreads
        let expected = (200.0 + 230.0 + 250.0) / 3.0;
        assert!((result[2] - expected).abs() < 0.001);
    }

    #[test]
    fn test_credit_spread_z_score() {
        let indicator = CreditSpread::new(1);

        let spreads = vec![200.0, 210.0, 220.0, 230.0, 400.0]; // Last one is spike
        let z_scores = indicator.z_score(&spreads, 5);

        assert_eq!(z_scores.len(), 5);
        // Last z-score should be high (positive) since 400 is well above mean
        assert!(z_scores[4] > 1.0);
    }

    #[test]
    fn test_credit_spread_negative() {
        let indicator = CreditSpread::new(1);

        // Treasury higher than corporate (inverted, rare)
        let corporate = vec![3.0, 3.5];
        let treasury = vec![4.0, 4.2];

        let result = indicator.calculate(&corporate, &treasury);

        assert!(result[0] < 0.0); // -100bp
        assert!(result[1] < 0.0); // -70bp
    }
}
