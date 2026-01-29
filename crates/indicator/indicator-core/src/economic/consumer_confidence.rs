//! Consumer Confidence Delta Indicator (IND-319)
//!
//! Tracks month-over-month changes in consumer confidence as a
//! leading indicator of consumer spending and economic activity.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Configuration for ConsumerConfidenceDelta indicator.
#[derive(Debug, Clone)]
pub struct ConsumerConfidenceDeltaConfig {
    /// Period for MoM calculation (default: 1 month)
    pub mom_period: usize,
    /// Smoothing period
    pub smoothing_period: usize,
    /// Threshold for significant change
    pub significant_change: f64,
    /// Lookback for trend analysis
    pub trend_lookback: usize,
}

impl Default for ConsumerConfidenceDeltaConfig {
    fn default() -> Self {
        Self {
            mom_period: 1,
            smoothing_period: 3,
            significant_change: 3.0,
            trend_lookback: 6,
        }
    }
}

/// Consumer Confidence Delta Indicator (IND-319)
///
/// Measures month-over-month changes in consumer confidence index,
/// capturing shifts in consumer sentiment that often lead spending changes.
///
/// # Interpretation
/// - Positive delta indicates improving confidence (bullish for consumer discretionary)
/// - Negative delta indicates declining confidence (bearish)
/// - Sharp drops often precede recessions
/// - Consumer confidence leads actual spending by 1-3 months
///
/// # Example
/// ```ignore
/// let indicator = ConsumerConfidenceDelta::new(ConsumerConfidenceDeltaConfig::default())?;
/// let delta = indicator.calculate_mom(&confidence_data);
/// ```
#[derive(Debug, Clone)]
pub struct ConsumerConfidenceDelta {
    config: ConsumerConfidenceDeltaConfig,
}

impl ConsumerConfidenceDelta {
    /// Create a new ConsumerConfidenceDelta indicator.
    pub fn new(config: ConsumerConfidenceDeltaConfig) -> Result<Self> {
        if config.mom_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "mom_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration.
    pub fn default_monthly() -> Result<Self> {
        Self::new(ConsumerConfidenceDeltaConfig::default())
    }

    /// Calculate month-over-month change (absolute).
    pub fn calculate_mom(&self, confidence: &[f64]) -> Vec<f64> {
        let n = confidence.len();
        let period = self.config.mom_period;
        let mut result = vec![f64::NAN; n];

        if n <= period {
            return result;
        }

        for i in period..n {
            result[i] = confidence[i] - confidence[i - period];
        }

        result
    }

    /// Calculate month-over-month change (percentage).
    pub fn calculate_mom_pct(&self, confidence: &[f64]) -> Vec<f64> {
        let n = confidence.len();
        let period = self.config.mom_period;
        let mut result = vec![f64::NAN; n];

        if n <= period {
            return result;
        }

        for i in period..n {
            if confidence[i - period].abs() > 1e-10 {
                result[i] = (confidence[i] - confidence[i - period]) / confidence[i - period] * 100.0;
            }
        }

        result
    }

    /// Calculate smoothed MoM change.
    pub fn calculate_smoothed_delta(&self, confidence: &[f64]) -> Vec<f64> {
        let delta = self.calculate_mom(confidence);
        let n = delta.len();
        let period = self.config.smoothing_period;
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        for i in period - 1..n {
            let slice = &delta[i + 1 - period..=i];
            let valid: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();

            if !valid.is_empty() {
                result[i] = valid.iter().sum::<f64>() / valid.len() as f64;
            }
        }

        result
    }

    /// Calculate cumulative change over N months.
    pub fn calculate_cumulative_change(&self, confidence: &[f64], months: usize) -> Vec<f64> {
        let n = confidence.len();
        let mut result = vec![f64::NAN; n];

        if n <= months {
            return result;
        }

        for i in months..n {
            result[i] = confidence[i] - confidence[i - months];
        }

        result
    }

    /// Calculate rate of change in confidence.
    pub fn calculate_roc(&self, confidence: &[f64], period: usize) -> Vec<f64> {
        let n = confidence.len();
        let mut result = vec![f64::NAN; n];

        if n <= period {
            return result;
        }

        for i in period..n {
            if confidence[i - period].abs() > 1e-10 {
                result[i] = (confidence[i] / confidence[i - period] - 1.0) * 100.0;
            }
        }

        result
    }

    /// Detect significant changes.
    pub fn detect_significant_changes(&self, confidence: &[f64]) -> Vec<i32> {
        let delta = self.calculate_mom(confidence);
        let n = delta.len();
        let mut result = vec![0; n];

        for i in 0..n {
            if delta[i].is_nan() {
                continue;
            }

            if delta[i] > self.config.significant_change {
                result[i] = 1; // Significant positive change
            } else if delta[i] < -self.config.significant_change {
                result[i] = -1; // Significant negative change
            }
        }

        result
    }

    /// Calculate trend strength over lookback period.
    pub fn calculate_trend_strength(&self, confidence: &[f64]) -> Vec<f64> {
        let delta = self.calculate_mom(confidence);
        let n = delta.len();
        let lookback = self.config.trend_lookback;
        let mut result = vec![f64::NAN; n];

        if n < lookback {
            return result;
        }

        for i in lookback - 1..n {
            let slice = &delta[i + 1 - lookback..=i];
            let valid: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();

            if valid.is_empty() {
                continue;
            }

            // Count positive vs negative changes
            let positive = valid.iter().filter(|&&x| x > 0.0).count();
            let negative = valid.iter().filter(|&&x| x < 0.0).count();
            let total = positive + negative;

            if total > 0 {
                result[i] = (positive as f64 - negative as f64) / total as f64 * 100.0;
            }
        }

        result
    }

    /// Calculate standard deviation of changes (volatility).
    pub fn calculate_volatility(&self, confidence: &[f64], lookback: usize) -> Vec<f64> {
        let delta = self.calculate_mom(confidence);
        let n = delta.len();
        let mut result = vec![f64::NAN; n];

        if n < lookback {
            return result;
        }

        for i in lookback - 1..n {
            let slice = &delta[i + 1 - lookback..=i];
            let valid: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();

            if valid.len() < 2 {
                continue;
            }

            let mean = valid.iter().sum::<f64>() / valid.len() as f64;
            let variance = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / valid.len() as f64;
            result[i] = variance.sqrt();
        }

        result
    }

    /// Generate sentiment signal based on confidence delta.
    pub fn sentiment_signal(&self, confidence: &[f64]) -> Vec<i32> {
        let smoothed = self.calculate_smoothed_delta(confidence);
        let trend = self.calculate_trend_strength(confidence);
        let n = confidence.len();
        let mut result = vec![0; n];

        for i in 0..n {
            let smooth_val = if !smoothed[i].is_nan() { smoothed[i] } else { continue };
            let trend_val = if !trend[i].is_nan() { trend[i] } else { 0.0 };

            // Strong bullish: positive delta + positive trend
            if smooth_val > self.config.significant_change && trend_val > 50.0 {
                result[i] = 2;
            }
            // Moderate bullish: positive delta
            else if smooth_val > 0.0 && trend_val > 0.0 {
                result[i] = 1;
            }
            // Strong bearish: negative delta + negative trend
            else if smooth_val < -self.config.significant_change && trend_val < -50.0 {
                result[i] = -2;
            }
            // Moderate bearish: negative delta
            else if smooth_val < 0.0 && trend_val < 0.0 {
                result[i] = -1;
            }
        }

        result
    }
}

impl TechnicalIndicator for ConsumerConfidenceDelta {
    fn name(&self) -> &str {
        "ConsumerConfidenceDelta"
    }

    fn min_periods(&self) -> usize {
        self.config.mom_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate_mom(&data.close)))
    }
}

impl SignalIndicator for ConsumerConfidenceDelta {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let sentiment = self.sentiment_signal(&data.close);

        if let Some(&last) = sentiment.last() {
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
        let sentiment = self.sentiment_signal(&data.close);
        Ok(sentiment
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

    fn create_test_confidence_data() -> Vec<f64> {
        // Simulated consumer confidence index
        vec![
            100.0, 102.0, 101.0, 104.0, 106.0, 108.0, 107.0, 110.0,
            112.0, 114.0, 113.0, 116.0, 118.0, 120.0, 119.0, 122.0,
            124.0, 126.0, 125.0, 128.0, 130.0, 132.0, 131.0, 134.0,
        ]
    }

    #[test]
    fn test_consumer_confidence_mom() {
        let data = create_test_confidence_data();
        let indicator = ConsumerConfidenceDelta::default_monthly().unwrap();
        let delta = indicator.calculate_mom(&data);

        assert_eq!(delta.len(), data.len());
        assert!(delta[0].is_nan());

        // At index 1: 102 - 100 = 2
        assert!((delta[1] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_mom_pct() {
        let data = create_test_confidence_data();
        let indicator = ConsumerConfidenceDelta::default_monthly().unwrap();
        let pct = indicator.calculate_mom_pct(&data);

        assert_eq!(pct.len(), data.len());

        // At index 1: (102 - 100) / 100 * 100 = 2%
        assert!((pct[1] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_smoothed_delta() {
        let data = create_test_confidence_data();
        let indicator = ConsumerConfidenceDelta::default_monthly().unwrap();
        let smoothed = indicator.calculate_smoothed_delta(&data);

        assert_eq!(smoothed.len(), data.len());

        // Should have values after warmup
        assert!(!smoothed[3].is_nan());
    }

    #[test]
    fn test_cumulative_change() {
        let data = create_test_confidence_data();
        let indicator = ConsumerConfidenceDelta::default_monthly().unwrap();
        let cumulative = indicator.calculate_cumulative_change(&data, 6);

        assert_eq!(cumulative.len(), data.len());

        // At index 6: 107 - 100 = 7
        assert!((cumulative[6] - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_roc() {
        let data = create_test_confidence_data();
        let indicator = ConsumerConfidenceDelta::default_monthly().unwrap();
        let roc = indicator.calculate_roc(&data, 3);

        assert_eq!(roc.len(), data.len());
        assert!(!roc[3].is_nan());
    }

    #[test]
    fn test_significant_changes() {
        let data = create_test_confidence_data();
        let indicator = ConsumerConfidenceDelta::default_monthly().unwrap();
        let changes = indicator.detect_significant_changes(&data);

        assert_eq!(changes.len(), data.len());

        // Values should be -1, 0, or 1
        for c in changes.iter() {
            assert!(*c >= -1 && *c <= 1);
        }
    }

    #[test]
    fn test_trend_strength() {
        let data = create_test_confidence_data();
        let indicator = ConsumerConfidenceDelta::default_monthly().unwrap();
        let trend = indicator.calculate_trend_strength(&data);

        assert_eq!(trend.len(), data.len());

        // Trend strength should be between -100 and 100
        for t in trend.iter().skip(6) {
            if !t.is_nan() {
                assert!(*t >= -100.0 && *t <= 100.0);
            }
        }
    }

    #[test]
    fn test_volatility() {
        let data = create_test_confidence_data();
        let indicator = ConsumerConfidenceDelta::default_monthly().unwrap();
        let vol = indicator.calculate_volatility(&data, 6);

        assert_eq!(vol.len(), data.len());

        // Volatility should be non-negative
        for v in vol.iter() {
            if !v.is_nan() {
                assert!(*v >= 0.0);
            }
        }
    }

    #[test]
    fn test_sentiment_signal() {
        let data = create_test_confidence_data();
        let indicator = ConsumerConfidenceDelta::default_monthly().unwrap();
        let sentiment = indicator.sentiment_signal(&data);

        assert_eq!(sentiment.len(), data.len());

        // With consistently rising confidence, should be bullish
        assert!(sentiment.last().unwrap() >= &0);
    }

    #[test]
    fn test_technical_indicator_impl() {
        let indicator = ConsumerConfidenceDelta::default_monthly().unwrap();
        let data = OHLCVSeries::from_close(create_test_confidence_data());
        let result = indicator.compute(&data);

        assert!(result.is_ok());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let indicator = ConsumerConfidenceDelta::default_monthly().unwrap();
        let data = OHLCVSeries::from_close(create_test_confidence_data());
        let signals = indicator.signals(&data);

        assert!(signals.is_ok());
        assert_eq!(signals.unwrap().len(), data.close.len());
    }
}
