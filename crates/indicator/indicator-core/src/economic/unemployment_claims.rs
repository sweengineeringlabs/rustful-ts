//! Unemployment Claims Trend Indicator (IND-317)
//!
//! Tracks the 4-week moving average of initial unemployment claims
//! as a leading economic indicator for labor market health.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Configuration for UnemploymentClaimsTrend indicator.
#[derive(Debug, Clone)]
pub struct UnemploymentClaimsTrendConfig {
    /// Moving average period (default: 4 weeks)
    pub ma_period: usize,
    /// Threshold for significant change (percentage)
    pub significant_change_pct: f64,
    /// Period for trend detection
    pub trend_period: usize,
}

impl Default for UnemploymentClaimsTrendConfig {
    fn default() -> Self {
        Self {
            ma_period: 4,
            significant_change_pct: 5.0,
            trend_period: 13,
        }
    }
}

/// Unemployment Claims Trend Indicator (IND-317)
///
/// Calculates the 4-week moving average of unemployment claims data
/// to smooth out weekly volatility and identify underlying trends.
///
/// # Interpretation
/// - Rising MA suggests weakening labor market (bearish economic signal)
/// - Falling MA suggests strengthening labor market (bullish economic signal)
/// - Sharp spikes often precede recessions
///
/// # Example
/// ```ignore
/// let indicator = UnemploymentClaimsTrend::new(UnemploymentClaimsTrendConfig::default())?;
/// let result = indicator.calculate(&claims_data);
/// ```
#[derive(Debug, Clone)]
pub struct UnemploymentClaimsTrend {
    config: UnemploymentClaimsTrendConfig,
}

impl UnemploymentClaimsTrend {
    /// Create a new UnemploymentClaimsTrend indicator.
    pub fn new(config: UnemploymentClaimsTrendConfig) -> Result<Self> {
        if config.ma_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "ma_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default 4-week MA.
    pub fn default_4w() -> Result<Self> {
        Self::new(UnemploymentClaimsTrendConfig::default())
    }

    /// Calculate 4-week moving average of claims.
    pub fn calculate(&self, claims: &[f64]) -> Vec<f64> {
        let n = claims.len();
        let period = self.config.ma_period;
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        // Calculate simple moving average
        let mut sum: f64 = claims[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        for i in period..n {
            sum = sum - claims[i - period] + claims[i];
            result[i] = sum / period as f64;
        }

        result
    }

    /// Calculate week-over-week change in MA.
    pub fn calculate_wow_change(&self, claims: &[f64]) -> Vec<f64> {
        let ma = self.calculate(claims);
        let n = ma.len();
        let mut result = vec![f64::NAN; n];

        for i in 1..n {
            if !ma[i].is_nan() && !ma[i - 1].is_nan() && ma[i - 1].abs() > 1e-10 {
                result[i] = (ma[i] - ma[i - 1]) / ma[i - 1] * 100.0;
            }
        }

        result
    }

    /// Calculate year-over-year change (52 weeks).
    pub fn calculate_yoy_change(&self, claims: &[f64]) -> Vec<f64> {
        let ma = self.calculate(claims);
        let n = ma.len();
        let mut result = vec![f64::NAN; n];

        for i in 52..n {
            if !ma[i].is_nan() && !ma[i - 52].is_nan() && ma[i - 52].abs() > 1e-10 {
                result[i] = (ma[i] - ma[i - 52]) / ma[i - 52] * 100.0;
            }
        }

        result
    }

    /// Detect trend direction based on slope.
    pub fn detect_trend(&self, claims: &[f64]) -> Vec<i32> {
        let ma = self.calculate(claims);
        let n = ma.len();
        let period = self.config.trend_period;
        let mut result = vec![0; n];

        if n < period {
            return result;
        }

        for i in period..n {
            if ma[i].is_nan() || ma[i - period].is_nan() {
                continue;
            }

            let change_pct = (ma[i] - ma[i - period]) / ma[i - period] * 100.0;

            if change_pct > self.config.significant_change_pct {
                result[i] = 1; // Rising claims (bearish)
            } else if change_pct < -self.config.significant_change_pct {
                result[i] = -1; // Falling claims (bullish)
            }
        }

        result
    }

    /// Calculate normalized claims level (z-score).
    pub fn calculate_zscore(&self, claims: &[f64], lookback: usize) -> Vec<f64> {
        let ma = self.calculate(claims);
        let n = ma.len();
        let mut result = vec![f64::NAN; n];

        if n < lookback {
            return result;
        }

        for i in lookback..n {
            let slice = &ma[i - lookback..i];
            let valid: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();

            if valid.len() < 2 {
                continue;
            }

            let mean = valid.iter().sum::<f64>() / valid.len() as f64;
            let variance = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / valid.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 && !ma[i].is_nan() {
                result[i] = (ma[i] - mean) / std_dev;
            }
        }

        result
    }
}

impl TechnicalIndicator for UnemploymentClaimsTrend {
    fn name(&self) -> &str {
        "UnemploymentClaimsTrend"
    }

    fn min_periods(&self) -> usize {
        self.config.ma_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // Use close prices as proxy for claims data
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

impl SignalIndicator for UnemploymentClaimsTrend {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let trend = self.detect_trend(&data.close);

        if let Some(&last) = trend.last() {
            match last {
                1 => Ok(IndicatorSignal::Bearish),  // Rising claims = bad economy
                -1 => Ok(IndicatorSignal::Bullish), // Falling claims = good economy
                _ => Ok(IndicatorSignal::Neutral),
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let trend = self.detect_trend(&data.close);
        Ok(trend
            .iter()
            .map(|&t| match t {
                1 => IndicatorSignal::Bearish,
                -1 => IndicatorSignal::Bullish,
                _ => IndicatorSignal::Neutral,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_claims() -> Vec<f64> {
        // Simulated weekly claims data (thousands)
        vec![
            200.0, 205.0, 195.0, 210.0, 215.0, 220.0, 225.0, 230.0,
            235.0, 240.0, 238.0, 242.0, 245.0, 250.0, 248.0, 252.0,
            255.0, 260.0, 258.0, 262.0, 265.0, 270.0, 268.0, 272.0,
        ]
    }

    #[test]
    fn test_unemployment_claims_ma() {
        let claims = create_test_claims();
        let indicator = UnemploymentClaimsTrend::default_4w().unwrap();
        let ma = indicator.calculate(&claims);

        assert_eq!(ma.len(), claims.len());

        // First 3 values should be NaN
        assert!(ma[0].is_nan());
        assert!(ma[1].is_nan());
        assert!(ma[2].is_nan());

        // First valid value at index 3
        let expected = (200.0 + 205.0 + 195.0 + 210.0) / 4.0;
        assert!((ma[3] - expected).abs() < 0.001);
    }

    #[test]
    fn test_wow_change() {
        let claims = create_test_claims();
        let indicator = UnemploymentClaimsTrend::default_4w().unwrap();
        let wow = indicator.calculate_wow_change(&claims);

        assert_eq!(wow.len(), claims.len());

        // WoW change should exist after MA warmup + 1
        assert!(!wow[4].is_nan());
    }

    #[test]
    fn test_trend_detection() {
        let claims = create_test_claims();
        let indicator = UnemploymentClaimsTrend::default_4w().unwrap();
        let trend = indicator.detect_trend(&claims);

        assert_eq!(trend.len(), claims.len());

        // With rising claims, should detect bearish trend
        // Check last value (trend should be positive = rising claims)
        if trend.len() > 20 {
            // The data shows rising claims, so trend should be positive at end
            assert!(trend.last().unwrap() >= &0);
        }
    }

    #[test]
    fn test_zscore() {
        let claims = create_test_claims();
        let indicator = UnemploymentClaimsTrend::default_4w().unwrap();
        let zscore = indicator.calculate_zscore(&claims, 10);

        assert_eq!(zscore.len(), claims.len());

        // Z-score should be within reasonable bounds
        for z in zscore.iter().skip(10) {
            if !z.is_nan() {
                assert!(z.abs() < 5.0, "Z-score should be reasonable");
            }
        }
    }

    #[test]
    fn test_technical_indicator_impl() {
        let indicator = UnemploymentClaimsTrend::default_4w().unwrap();
        let data = OHLCVSeries::from_close(create_test_claims());
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().primary.len(), data.close.len());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let indicator = UnemploymentClaimsTrend::default_4w().unwrap();
        let data = OHLCVSeries::from_close(create_test_claims());
        let signals = indicator.signals(&data);

        assert!(signals.is_ok());
        assert_eq!(signals.unwrap().len(), data.close.len());
    }

    #[test]
    fn test_invalid_period() {
        let config = UnemploymentClaimsTrendConfig {
            ma_period: 1,
            ..Default::default()
        };
        let result = UnemploymentClaimsTrend::new(config);
        assert!(result.is_err());
    }
}
