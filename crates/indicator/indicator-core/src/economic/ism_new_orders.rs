//! ISM New Orders Indicator (IND-320)
//!
//! Tracks the ISM Manufacturing New Orders component as a leading
//! indicator of economic activity and manufacturing sector health.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Configuration for ISMNewOrders indicator.
#[derive(Debug, Clone)]
pub struct ISMNewOrdersConfig {
    /// Expansion threshold (typically 50)
    pub expansion_threshold: f64,
    /// Strong expansion threshold
    pub strong_expansion: f64,
    /// Contraction threshold
    pub contraction_threshold: f64,
    /// Smoothing period
    pub smoothing_period: usize,
    /// Trend lookback period
    pub trend_lookback: usize,
}

impl Default for ISMNewOrdersConfig {
    fn default() -> Self {
        Self {
            expansion_threshold: 50.0,
            strong_expansion: 55.0,
            contraction_threshold: 45.0,
            smoothing_period: 3,
            trend_lookback: 6,
        }
    }
}

/// ISM New Orders Indicator (IND-320)
///
/// Analyzes the ISM Manufacturing New Orders Index, which is considered
/// one of the most leading components of the ISM Manufacturing Index.
///
/// # Interpretation
/// - Above 50: Manufacturing new orders expanding
/// - Below 50: Manufacturing new orders contracting
/// - The new orders component leads overall manufacturing activity
/// - Often leads GDP by 2-4 quarters
///
/// # Example
/// ```ignore
/// let indicator = ISMNewOrders::new(ISMNewOrdersConfig::default())?;
/// let analysis = indicator.analyze(&ism_data);
/// ```
#[derive(Debug, Clone)]
pub struct ISMNewOrders {
    config: ISMNewOrdersConfig,
}

impl ISMNewOrders {
    /// Create a new ISMNewOrders indicator.
    pub fn new(config: ISMNewOrdersConfig) -> Result<Self> {
        if config.expansion_threshold <= 0.0 || config.expansion_threshold > 100.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "expansion_threshold".to_string(),
                reason: "must be between 0 and 100".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration.
    pub fn default_ism() -> Result<Self> {
        Self::new(ISMNewOrdersConfig::default())
    }

    /// Calculate expansion/contraction status.
    pub fn calculate_status(&self, data: &[f64]) -> Vec<i32> {
        let n = data.len();
        let mut result = vec![0; n];

        for i in 0..n {
            if data[i] > self.config.strong_expansion {
                result[i] = 2; // Strong expansion
            } else if data[i] > self.config.expansion_threshold {
                result[i] = 1; // Moderate expansion
            } else if data[i] < self.config.contraction_threshold {
                result[i] = -2; // Strong contraction
            } else if data[i] < self.config.expansion_threshold {
                result[i] = -1; // Moderate contraction
            }
        }

        result
    }

    /// Calculate smoothed new orders index.
    pub fn calculate_smoothed(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let period = self.config.smoothing_period;
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        for i in period - 1..n {
            let slice = &data[i + 1 - period..=i];
            result[i] = slice.iter().sum::<f64>() / period as f64;
        }

        result
    }

    /// Calculate month-over-month change.
    pub fn calculate_mom_change(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        for i in 1..n {
            result[i] = data[i] - data[i - 1];
        }

        result
    }

    /// Calculate deviation from expansion threshold.
    pub fn calculate_deviation(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let threshold = self.config.expansion_threshold;
        let mut result = vec![0.0; n];

        for i in 0..n {
            result[i] = data[i] - threshold;
        }

        result
    }

    /// Calculate trend direction.
    pub fn calculate_trend(&self, data: &[f64]) -> Vec<i32> {
        let n = data.len();
        let lookback = self.config.trend_lookback;
        let mut result = vec![0; n];

        if n < lookback {
            return result;
        }

        for i in lookback - 1..n {
            let start_val = data[i + 1 - lookback];
            let end_val = data[i];

            // Simple trend based on change over lookback
            let change = end_val - start_val;

            if change > 2.0 {
                result[i] = 1; // Uptrend
            } else if change < -2.0 {
                result[i] = -1; // Downtrend
            }
        }

        result
    }

    /// Calculate momentum (acceleration of change).
    pub fn calculate_momentum(&self, data: &[f64]) -> Vec<f64> {
        let mom = self.calculate_mom_change(data);
        let n = mom.len();
        let mut result = vec![f64::NAN; n];

        for i in 2..n {
            if !mom[i].is_nan() && !mom[i - 1].is_nan() {
                result[i] = mom[i] - mom[i - 1];
            }
        }

        result
    }

    /// Count consecutive months above/below threshold.
    pub fn calculate_streak(&self, data: &[f64]) -> Vec<i32> {
        let n = data.len();
        let threshold = self.config.expansion_threshold;
        let mut result = vec![0; n];

        for i in 0..n {
            if i == 0 {
                result[i] = if data[i] >= threshold { 1 } else { -1 };
            } else {
                let prev_streak = result[i - 1];
                if data[i] >= threshold {
                    result[i] = if prev_streak > 0 { prev_streak + 1 } else { 1 };
                } else {
                    result[i] = if prev_streak < 0 { prev_streak - 1 } else { -1 };
                }
            }
        }

        result
    }

    /// Calculate z-score relative to historical data.
    pub fn calculate_zscore(&self, data: &[f64], lookback: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < lookback {
            return result;
        }

        for i in lookback - 1..n {
            let slice = &data[i + 1 - lookback..=i];
            let mean = slice.iter().sum::<f64>() / lookback as f64;
            let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / lookback as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                result[i] = (data[i] - mean) / std_dev;
            }
        }

        result
    }

    /// Generate leading economic signal.
    pub fn leading_signal(&self, data: &[f64]) -> Vec<i32> {
        let smoothed = self.calculate_smoothed(data);
        let trend = self.calculate_trend(data);
        let n = data.len();
        let mut result = vec![0; n];

        for i in 0..n {
            let level = if !smoothed[i].is_nan() { smoothed[i] } else { data[i] };
            let trend_dir = trend[i];

            // Strong bullish: above expansion + uptrend
            if level > self.config.strong_expansion && trend_dir > 0 {
                result[i] = 2;
            }
            // Moderate bullish: above expansion
            else if level > self.config.expansion_threshold {
                result[i] = 1;
            }
            // Strong bearish: below contraction + downtrend
            else if level < self.config.contraction_threshold && trend_dir < 0 {
                result[i] = -2;
            }
            // Moderate bearish: below expansion
            else if level < self.config.expansion_threshold {
                result[i] = -1;
            }
        }

        result
    }

    /// Calculate breadth indicator (how far from threshold).
    pub fn calculate_breadth(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let threshold = self.config.expansion_threshold;
        let mut result = vec![0.0; n];

        // Normalize deviation to -100 to +100 scale
        for i in 0..n {
            let deviation = data[i] - threshold;
            // Map 0-50 range to 0-100
            result[i] = (deviation / threshold) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for ISMNewOrders {
    fn name(&self) -> &str {
        "ISMNewOrders"
    }

    fn min_periods(&self) -> usize {
        self.config.smoothing_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate_smoothed(&data.close)))
    }
}

impl SignalIndicator for ISMNewOrders {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let signal = self.leading_signal(&data.close);

        if let Some(&last) = signal.last() {
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
        let signal = self.leading_signal(&data.close);
        Ok(signal
            .iter()
            .map(|&s| match s {
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

    fn create_test_ism_data() -> Vec<f64> {
        // Simulated ISM New Orders data
        vec![
            52.0, 53.5, 51.0, 54.0, 56.0, 57.5, 55.0, 58.0,
            59.0, 57.0, 55.5, 53.0, 51.0, 49.0, 47.5, 46.0,
            48.0, 50.5, 52.0, 54.5, 56.0, 58.5, 60.0, 61.5,
        ]
    }

    #[test]
    fn test_ism_status() {
        let data = create_test_ism_data();
        let indicator = ISMNewOrders::default_ism().unwrap();
        let status = indicator.calculate_status(&data);

        assert_eq!(status.len(), data.len());

        // First value (52.0) should be moderate expansion (1)
        assert_eq!(status[0], 1);

        // Value 46.0 should be strong contraction (-2)
        assert_eq!(status[15], -2);
    }

    #[test]
    fn test_smoothed() {
        let data = create_test_ism_data();
        let indicator = ISMNewOrders::default_ism().unwrap();
        let smoothed = indicator.calculate_smoothed(&data);

        assert_eq!(smoothed.len(), data.len());
        assert!(smoothed[0].is_nan());
        assert!(smoothed[1].is_nan());

        // At index 2: (52.0 + 53.5 + 51.0) / 3
        let expected = (52.0 + 53.5 + 51.0) / 3.0;
        assert!((smoothed[2] - expected).abs() < 0.001);
    }

    #[test]
    fn test_mom_change() {
        let data = create_test_ism_data();
        let indicator = ISMNewOrders::default_ism().unwrap();
        let mom = indicator.calculate_mom_change(&data);

        assert_eq!(mom.len(), data.len());
        assert!(mom[0].is_nan());

        // At index 1: 53.5 - 52.0 = 1.5
        assert!((mom[1] - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_deviation() {
        let data = create_test_ism_data();
        let indicator = ISMNewOrders::default_ism().unwrap();
        let deviation = indicator.calculate_deviation(&data);

        assert_eq!(deviation.len(), data.len());

        // At index 0: 52.0 - 50.0 = 2.0
        assert!((deviation[0] - 2.0).abs() < 0.001);

        // At index 15: 46.0 - 50.0 = -4.0
        assert!((deviation[15] - (-4.0)).abs() < 0.001);
    }

    #[test]
    fn test_trend() {
        let data = create_test_ism_data();
        let indicator = ISMNewOrders::default_ism().unwrap();
        let trend = indicator.calculate_trend(&data);

        assert_eq!(trend.len(), data.len());

        // Trend values should be -1, 0, or 1
        for t in trend.iter() {
            assert!(*t >= -1 && *t <= 1);
        }
    }

    #[test]
    fn test_momentum() {
        let data = create_test_ism_data();
        let indicator = ISMNewOrders::default_ism().unwrap();
        let momentum = indicator.calculate_momentum(&data);

        assert_eq!(momentum.len(), data.len());
        assert!(momentum[0].is_nan());
        assert!(momentum[1].is_nan());
    }

    #[test]
    fn test_streak() {
        let data = create_test_ism_data();
        let indicator = ISMNewOrders::default_ism().unwrap();
        let streak = indicator.calculate_streak(&data);

        assert_eq!(streak.len(), data.len());

        // First few values are above 50, so streak should be positive
        assert!(streak[0] > 0);
        assert!(streak[5] > 0);
    }

    #[test]
    fn test_zscore() {
        let data = create_test_ism_data();
        let indicator = ISMNewOrders::default_ism().unwrap();
        let zscore = indicator.calculate_zscore(&data, 12);

        assert_eq!(zscore.len(), data.len());

        // Z-score should be reasonable (typically -3 to +3)
        for z in zscore.iter().skip(11) {
            if !z.is_nan() {
                assert!(z.abs() < 5.0);
            }
        }
    }

    #[test]
    fn test_leading_signal() {
        let data = create_test_ism_data();
        let indicator = ISMNewOrders::default_ism().unwrap();
        let signal = indicator.leading_signal(&data);

        assert_eq!(signal.len(), data.len());

        // Signals should be -2, -1, 0, 1, or 2
        for s in signal.iter() {
            assert!(*s >= -2 && *s <= 2);
        }
    }

    #[test]
    fn test_breadth() {
        let data = create_test_ism_data();
        let indicator = ISMNewOrders::default_ism().unwrap();
        let breadth = indicator.calculate_breadth(&data);

        assert_eq!(breadth.len(), data.len());

        // Breadth at 52.0: (52 - 50) / 50 * 100 = 4.0
        assert!((breadth[0] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_technical_indicator_impl() {
        let indicator = ISMNewOrders::default_ism().unwrap();
        let data = OHLCVSeries::from_close(create_test_ism_data());
        let result = indicator.compute(&data);

        assert!(result.is_ok());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let indicator = ISMNewOrders::default_ism().unwrap();
        let data = OHLCVSeries::from_close(create_test_ism_data());
        let signals = indicator.signals(&data);

        assert!(signals.is_ok());
        assert_eq!(signals.unwrap().len(), data.close.len());
    }
}
