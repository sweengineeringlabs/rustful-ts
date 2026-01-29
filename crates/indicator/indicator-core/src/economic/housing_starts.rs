//! Housing Starts Trend Indicator (IND-318)
//!
//! Tracks year-over-year change in housing starts as a leading
//! economic indicator for construction and broader economic activity.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Configuration for HousingStartsTrend indicator.
#[derive(Debug, Clone)]
pub struct HousingStartsTrendConfig {
    /// Period for YoY comparison (default: 12 months)
    pub yoy_period: usize,
    /// Smoothing period for the trend
    pub smoothing_period: usize,
    /// Threshold for bullish signal (percentage)
    pub bullish_threshold: f64,
    /// Threshold for bearish signal (percentage)
    pub bearish_threshold: f64,
}

impl Default for HousingStartsTrendConfig {
    fn default() -> Self {
        Self {
            yoy_period: 12,
            smoothing_period: 3,
            bullish_threshold: 5.0,
            bearish_threshold: -5.0,
        }
    }
}

/// Housing Starts Trend Indicator (IND-318)
///
/// Measures the year-over-year change in housing starts, a leading
/// indicator of economic activity due to housing's multiplier effect.
///
/// # Interpretation
/// - Positive YoY change indicates expanding housing market (bullish)
/// - Negative YoY change indicates contracting housing market (bearish)
/// - Peaks often lead economic peaks by 12-18 months
/// - Troughs often lead economic troughs by 6-12 months
///
/// # Example
/// ```ignore
/// let indicator = HousingStartsTrend::new(HousingStartsTrendConfig::default())?;
/// let yoy_change = indicator.calculate_yoy(&housing_data);
/// ```
#[derive(Debug, Clone)]
pub struct HousingStartsTrend {
    config: HousingStartsTrendConfig,
}

impl HousingStartsTrend {
    /// Create a new HousingStartsTrend indicator.
    pub fn new(config: HousingStartsTrendConfig) -> Result<Self> {
        if config.yoy_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "yoy_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if config.smoothing_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration.
    pub fn default_monthly() -> Result<Self> {
        Self::new(HousingStartsTrendConfig::default())
    }

    /// Calculate year-over-year change.
    pub fn calculate_yoy(&self, starts: &[f64]) -> Vec<f64> {
        let n = starts.len();
        let period = self.config.yoy_period;
        let mut result = vec![f64::NAN; n];

        if n <= period {
            return result;
        }

        for i in period..n {
            let current = starts[i];
            let previous = starts[i - period];

            if previous.abs() > 1e-10 {
                result[i] = (current - previous) / previous * 100.0;
            }
        }

        result
    }

    /// Calculate smoothed YoY change.
    pub fn calculate_smoothed_yoy(&self, starts: &[f64]) -> Vec<f64> {
        let yoy = self.calculate_yoy(starts);
        let n = yoy.len();
        let period = self.config.smoothing_period;
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        for i in period - 1..n {
            let slice = &yoy[i + 1 - period..=i];
            let valid: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();

            if !valid.is_empty() {
                result[i] = valid.iter().sum::<f64>() / valid.len() as f64;
            }
        }

        result
    }

    /// Calculate month-over-month change.
    pub fn calculate_mom(&self, starts: &[f64]) -> Vec<f64> {
        let n = starts.len();
        let mut result = vec![f64::NAN; n];

        for i in 1..n {
            if starts[i - 1].abs() > 1e-10 {
                result[i] = (starts[i] - starts[i - 1]) / starts[i - 1] * 100.0;
            }
        }

        result
    }

    /// Calculate 3-month moving average of starts.
    pub fn calculate_3ma(&self, starts: &[f64]) -> Vec<f64> {
        let n = starts.len();
        let mut result = vec![f64::NAN; n];

        if n < 3 {
            return result;
        }

        for i in 2..n {
            result[i] = (starts[i] + starts[i - 1] + starts[i - 2]) / 3.0;
        }

        result
    }

    /// Detect trend acceleration/deceleration.
    pub fn calculate_momentum(&self, starts: &[f64]) -> Vec<f64> {
        let yoy = self.calculate_yoy(starts);
        let n = yoy.len();
        let mut result = vec![f64::NAN; n];

        for i in 1..n {
            if !yoy[i].is_nan() && !yoy[i - 1].is_nan() {
                result[i] = yoy[i] - yoy[i - 1];
            }
        }

        result
    }

    /// Calculate normalized level (relative to historical range).
    pub fn calculate_percentile(&self, starts: &[f64], lookback: usize) -> Vec<f64> {
        let n = starts.len();
        let mut result = vec![f64::NAN; n];

        if n < lookback {
            return result;
        }

        for i in lookback - 1..n {
            let slice = &starts[i + 1 - lookback..=i];
            let mut sorted: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if sorted.len() < 2 {
                continue;
            }

            let current = starts[i];
            let rank = sorted.iter().filter(|&&x| x < current).count();
            result[i] = (rank as f64 / (sorted.len() - 1) as f64) * 100.0;
        }

        result
    }

    /// Generate economic outlook based on housing trend.
    pub fn economic_outlook(&self, starts: &[f64]) -> Vec<i32> {
        let smoothed_yoy = self.calculate_smoothed_yoy(starts);
        let momentum = self.calculate_momentum(starts);
        let n = starts.len();
        let mut result = vec![0; n];

        for i in 0..n {
            let yoy_val = if !smoothed_yoy[i].is_nan() { smoothed_yoy[i] } else { continue };
            let mom_val = if i > 0 && !momentum[i].is_nan() { momentum[i] } else { 0.0 };

            // Strong bullish: positive YoY + accelerating
            if yoy_val > self.config.bullish_threshold && mom_val > 0.0 {
                result[i] = 2;
            }
            // Moderate bullish: positive YoY
            else if yoy_val > self.config.bullish_threshold {
                result[i] = 1;
            }
            // Strong bearish: negative YoY + decelerating
            else if yoy_val < self.config.bearish_threshold && mom_val < 0.0 {
                result[i] = -2;
            }
            // Moderate bearish: negative YoY
            else if yoy_val < self.config.bearish_threshold {
                result[i] = -1;
            }
        }

        result
    }
}

impl TechnicalIndicator for HousingStartsTrend {
    fn name(&self) -> &str {
        "HousingStartsTrend"
    }

    fn min_periods(&self) -> usize {
        self.config.yoy_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate_yoy(&data.close)))
    }
}

impl SignalIndicator for HousingStartsTrend {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let outlook = self.economic_outlook(&data.close);

        if let Some(&last) = outlook.last() {
            match last {
                1 | 2 => Ok(IndicatorSignal::Bullish),
                -1 | -2 => Ok(IndicatorSignal::Bearish),
                _ => Ok(IndicatorSignal::Neutral),
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let outlook = self.economic_outlook(&data.close);
        Ok(outlook
            .iter()
            .map(|&o| match o {
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

    fn create_test_housing_data() -> Vec<f64> {
        // Simulated monthly housing starts (thousands)
        vec![
            1200.0, 1220.0, 1180.0, 1250.0, 1280.0, 1300.0, 1320.0, 1350.0,
            1380.0, 1400.0, 1420.0, 1450.0, 1480.0, 1500.0, 1520.0, 1550.0,
            1580.0, 1600.0, 1620.0, 1650.0, 1680.0, 1700.0, 1720.0, 1750.0,
        ]
    }

    #[test]
    fn test_housing_starts_yoy() {
        let data = create_test_housing_data();
        let indicator = HousingStartsTrend::default_monthly().unwrap();
        let yoy = indicator.calculate_yoy(&data);

        assert_eq!(yoy.len(), data.len());

        // First 12 values should be NaN
        for i in 0..12 {
            assert!(yoy[i].is_nan());
        }

        // At index 12: (1480 - 1200) / 1200 * 100 = 23.33%
        let expected = (1480.0 - 1200.0) / 1200.0 * 100.0;
        assert!((yoy[12] - expected).abs() < 0.01);
    }

    #[test]
    fn test_smoothed_yoy() {
        let data = create_test_housing_data();
        let indicator = HousingStartsTrend::default_monthly().unwrap();
        let smoothed = indicator.calculate_smoothed_yoy(&data);

        assert_eq!(smoothed.len(), data.len());

        // Smoothed values should exist after YoY warmup + smoothing
        assert!(!smoothed[14].is_nan());
    }

    #[test]
    fn test_mom_change() {
        let data = create_test_housing_data();
        let indicator = HousingStartsTrend::default_monthly().unwrap();
        let mom = indicator.calculate_mom(&data);

        assert_eq!(mom.len(), data.len());
        assert!(mom[0].is_nan());

        // At index 1: (1220 - 1200) / 1200 * 100 = 1.67%
        let expected = (1220.0 - 1200.0) / 1200.0 * 100.0;
        assert!((mom[1] - expected).abs() < 0.01);
    }

    #[test]
    fn test_3ma() {
        let data = create_test_housing_data();
        let indicator = HousingStartsTrend::default_monthly().unwrap();
        let ma = indicator.calculate_3ma(&data);

        assert_eq!(ma.len(), data.len());
        assert!(ma[0].is_nan());
        assert!(ma[1].is_nan());

        // At index 2: (1200 + 1220 + 1180) / 3
        let expected = (1200.0 + 1220.0 + 1180.0) / 3.0;
        assert!((ma[2] - expected).abs() < 0.01);
    }

    #[test]
    fn test_momentum() {
        let data = create_test_housing_data();
        let indicator = HousingStartsTrend::default_monthly().unwrap();
        let momentum = indicator.calculate_momentum(&data);

        assert_eq!(momentum.len(), data.len());

        // Momentum should exist after YoY warmup + 1
        assert!(!momentum[13].is_nan());
    }

    #[test]
    fn test_percentile() {
        let data = create_test_housing_data();
        let indicator = HousingStartsTrend::default_monthly().unwrap();
        let percentile = indicator.calculate_percentile(&data, 12);

        assert_eq!(percentile.len(), data.len());

        // Percentile should be between 0 and 100
        for p in percentile.iter().skip(11) {
            if !p.is_nan() {
                assert!(*p >= 0.0 && *p <= 100.0);
            }
        }
    }

    #[test]
    fn test_economic_outlook() {
        let data = create_test_housing_data();
        let indicator = HousingStartsTrend::default_monthly().unwrap();
        let outlook = indicator.economic_outlook(&data);

        assert_eq!(outlook.len(), data.len());

        // With consistently rising housing starts, outlook should be positive
        assert!(outlook.last().unwrap() >= &0);
    }

    #[test]
    fn test_technical_indicator_impl() {
        let indicator = HousingStartsTrend::default_monthly().unwrap();
        let data = OHLCVSeries::from_close(create_test_housing_data());
        let result = indicator.compute(&data);

        assert!(result.is_ok());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let indicator = HousingStartsTrend::default_monthly().unwrap();
        let data = OHLCVSeries::from_close(create_test_housing_data());
        let signals = indicator.signals(&data);

        assert!(signals.is_ok());
        assert_eq!(signals.unwrap().len(), data.close.len());
    }

    #[test]
    fn test_invalid_period() {
        let config = HousingStartsTrendConfig {
            yoy_period: 0,
            ..Default::default()
        };
        let result = HousingStartsTrend::new(config);
        assert!(result.is_err());
    }
}
