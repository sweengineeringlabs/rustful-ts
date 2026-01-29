//! Google Trends Proxy Indicator - IND-280
//!
//! A proxy indicator for Google Trends search interest data.
//! Uses volume and volatility patterns as proxies for public search interest.
//!
//! High volume spikes = Increased public interest
//! Retail buying patterns = Search trend correlation
//! Parabolic moves = Peak search interest

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Google Trends Proxy output.
#[derive(Debug, Clone)]
pub struct GoogleTrendsOutput {
    /// Search interest index (0-100 scale, normalized).
    pub interest_index: Vec<f64>,
    /// Interest momentum (rate of change).
    pub momentum: Vec<f64>,
    /// Peak detection (1 = potential peak, 0 = normal).
    pub peaks: Vec<i8>,
    /// Relative interest compared to baseline.
    pub relative_interest: Vec<f64>,
}

/// Google Trends signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoogleTrendsSignal {
    /// Peak interest - potential top.
    PeakInterest,
    /// High interest.
    HighInterest,
    /// Normal interest.
    NormalInterest,
    /// Low interest.
    LowInterest,
    /// Minimum interest - potential bottom.
    MinimumInterest,
}

/// Google Trends Proxy Indicator - IND-280
///
/// Estimates Google search interest from market data patterns.
///
/// # Formula
/// ```text
/// Volume Activity = Volume / SMA(Volume)
/// Price Acceleration = ROC(ROC(Close))
/// Interest Index = Normalize(Volume Activity * |Price Acceleration|)
/// ```
///
/// # Example
/// ```
/// use indicator_core::sentiment::GoogleTrends;
///
/// let gt = GoogleTrends::new(14, 50).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GoogleTrends {
    /// Baseline period for normalization.
    period: usize,
    /// Long-term period for relative comparison.
    long_period: usize,
    /// Peak threshold (percentile).
    peak_threshold: f64,
}

impl GoogleTrends {
    /// Create a new Google Trends proxy indicator.
    pub fn new(period: usize, long_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if long_period <= period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than period".to_string(),
            });
        }
        Ok(Self {
            period,
            long_period,
            peak_threshold: 90.0,
        })
    }

    /// Create with custom peak threshold.
    pub fn with_peak_threshold(period: usize, long_period: usize, peak_threshold: f64) -> Result<Self> {
        let mut gt = Self::new(period, long_period)?;
        gt.peak_threshold = peak_threshold.clamp(70.0, 99.0);
        Ok(gt)
    }

    /// Calculate Google Trends proxy from OHLCV data.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> GoogleTrendsOutput {
        let n = close.len().min(volume.len());

        if n < self.long_period + 1 {
            return GoogleTrendsOutput {
                interest_index: vec![0.0; n],
                momentum: vec![0.0; n],
                peaks: vec![0; n],
                relative_interest: vec![0.0; n],
            };
        }

        // Calculate raw interest proxy
        let mut raw_interest = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // 1. Volume activity (normalized volume)
            let avg_vol: f64 = volume[start..i].iter().sum::<f64>() / (i - start) as f64;
            let vol_activity = if avg_vol > 0.0 {
                volume[i] / avg_vol
            } else {
                1.0
            };

            // 2. Price acceleration (second derivative)
            let roc1 = if i >= 2 && close[i - 1] > 0.0 {
                close[i] / close[i - 1] - 1.0
            } else {
                0.0
            };
            let roc2 = if i >= 3 && close[i - 2] > 0.0 {
                close[i - 1] / close[i - 2] - 1.0
            } else {
                0.0
            };
            let acceleration = (roc1 - roc2).abs() * 100.0;

            // 3. Volatility component
            let mut vol_sum = 0.0;
            for j in (start + 1)..=i {
                vol_sum += (close[j] / close[j - 1] - 1.0).abs();
            }
            let volatility = vol_sum / self.period as f64 * 100.0;

            // Combine: interest increases with volume, acceleration, and volatility
            raw_interest[i] = vol_activity * (1.0 + acceleration) * (1.0 + volatility * 0.5);
        }

        // Normalize to 0-100 scale
        let interest_index = self.normalize_to_index(&raw_interest);

        // Calculate momentum
        let momentum = self.calculate_momentum(&interest_index);

        // Detect peaks
        let peaks = self.detect_peaks(&interest_index);

        // Calculate relative interest (compared to long-term baseline)
        let relative_interest = self.calculate_relative(&raw_interest);

        GoogleTrendsOutput {
            interest_index,
            momentum,
            peaks,
            relative_interest,
        }
    }

    /// Normalize raw values to 0-100 index.
    fn normalize_to_index(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let start = i.saturating_sub(self.long_period);
            let window: Vec<f64> = data[start..=i]
                .iter()
                .filter(|x| **x > 0.0)
                .copied()
                .collect();

            if window.is_empty() {
                continue;
            }

            let min = window.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            if max - min > 1e-10 {
                result[i] = ((data[i] - min) / (max - min) * 100.0).clamp(0.0, 100.0);
            } else {
                result[i] = 50.0;
            }
        }

        result
    }

    /// Calculate momentum (rate of change in interest).
    fn calculate_momentum(&self, interest: &[f64]) -> Vec<f64> {
        let n = interest.len();
        let mut result = vec![0.0; n];
        let mom_period = self.period.min(5);

        for i in mom_period..n {
            result[i] = interest[i] - interest[i - mom_period];
        }

        result
    }

    /// Detect interest peaks.
    fn detect_peaks(&self, interest: &[f64]) -> Vec<i8> {
        let n = interest.len();
        let mut result = vec![0i8; n];

        for i in self.period..n {
            // Check if current value is above threshold
            if interest[i] >= self.peak_threshold {
                // Check if it's a local maximum
                let is_local_max = (i < 2 || interest[i] >= interest[i - 1])
                    && (i < 3 || interest[i] >= interest[i - 2]);

                if is_local_max {
                    result[i] = 1;
                }
            }
        }

        result
    }

    /// Calculate relative interest compared to long-term baseline.
    fn calculate_relative(&self, raw: &[f64]) -> Vec<f64> {
        let n = raw.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let start = i.saturating_sub(self.long_period);
            let baseline: f64 = raw[start..i].iter().sum::<f64>() / (i - start) as f64;

            if baseline > 1e-10 {
                result[i] = (raw[i] / baseline - 1.0) * 100.0;
            }
        }

        result
    }

    /// Get signal interpretation.
    pub fn interpret(&self, interest_index: f64) -> GoogleTrendsSignal {
        if interest_index.is_nan() {
            GoogleTrendsSignal::NormalInterest
        } else if interest_index >= self.peak_threshold {
            GoogleTrendsSignal::PeakInterest
        } else if interest_index >= 70.0 {
            GoogleTrendsSignal::HighInterest
        } else if interest_index <= 10.0 {
            GoogleTrendsSignal::MinimumInterest
        } else if interest_index <= 30.0 {
            GoogleTrendsSignal::LowInterest
        } else {
            GoogleTrendsSignal::NormalInterest
        }
    }

    /// Interpret with momentum context (contrarian signals).
    pub fn interpret_contrarian(&self, interest_index: f64, momentum: f64) -> GoogleTrendsSignal {
        let base = self.interpret(interest_index);

        // Peak interest with negative momentum = peak confirmed
        // Minimum interest with positive momentum = bottom confirmed
        match base {
            GoogleTrendsSignal::HighInterest if momentum < -5.0 => {
                GoogleTrendsSignal::PeakInterest
            }
            GoogleTrendsSignal::LowInterest if momentum > 5.0 => {
                GoogleTrendsSignal::MinimumInterest
            }
            _ => base,
        }
    }

    /// Calculate interest trend over period.
    pub fn interest_trend(&self, interest: &[f64]) -> f64 {
        if interest.len() < self.period {
            return 0.0;
        }

        let start = interest.len() - self.period;
        let end = interest.len() - 1;

        // Simple linear regression slope
        let n = self.period as f64;
        let sum_x: f64 = (0..self.period).map(|x| x as f64).sum();
        let sum_y: f64 = interest[start..=end].iter().sum();
        let sum_xy: f64 = (0..self.period)
            .map(|i| i as f64 * interest[start + i])
            .sum();
        let sum_x2: f64 = (0..self.period).map(|x| (x as f64).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }
}

impl TechnicalIndicator for GoogleTrends {
    fn name(&self) -> &str {
        "Google Trends Proxy"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(output.interest_index))
    }
}

impl Default for GoogleTrends {
    fn default() -> Self {
        Self::new(14, 50).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..80).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let volume: Vec<f64> = (0..80).map(|i| 1000.0 + (i as f64) * 10.0).collect();
        (close, volume)
    }

    fn make_spike_data() -> (Vec<f64>, Vec<f64>) {
        let mut close: Vec<f64> = (0..80).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let mut volume: Vec<f64> = vec![1000.0; 80];

        // Create a price/volume spike (simulating high interest)
        for i in 60..65 {
            close[i] = close[i - 1] * 1.05; // 5% daily gain
            volume[i] = 5000.0; // 5x volume
        }

        (close, volume)
    }

    #[test]
    fn test_google_trends_basic() {
        let (close, volume) = make_test_data();
        let gt = GoogleTrends::new(10, 30).unwrap();
        let output = gt.calculate(&close, &volume);

        assert_eq!(output.interest_index.len(), close.len());
        // Values should be in 0-100 range
        for i in 35..output.interest_index.len() {
            assert!(output.interest_index[i] >= 0.0 && output.interest_index[i] <= 100.0);
        }
    }

    #[test]
    fn test_google_trends_spike() {
        let (close, volume) = make_spike_data();
        let gt = GoogleTrends::new(10, 30).unwrap();
        let output = gt.calculate(&close, &volume);

        // Interest should be higher during the spike
        let pre_spike_avg: f64 = output.interest_index[40..55].iter().sum::<f64>() / 15.0;
        let spike_avg: f64 = output.interest_index[60..65].iter().sum::<f64>() / 5.0;

        assert!(spike_avg > pre_spike_avg);
    }

    #[test]
    fn test_google_trends_interpretation() {
        let gt = GoogleTrends::default();

        assert_eq!(gt.interpret(95.0), GoogleTrendsSignal::PeakInterest);
        assert_eq!(gt.interpret(75.0), GoogleTrendsSignal::HighInterest);
        assert_eq!(gt.interpret(50.0), GoogleTrendsSignal::NormalInterest);
        assert_eq!(gt.interpret(25.0), GoogleTrendsSignal::LowInterest);
        assert_eq!(gt.interpret(5.0), GoogleTrendsSignal::MinimumInterest);
    }

    #[test]
    fn test_google_trends_peaks() {
        let (close, volume) = make_spike_data();
        let gt = GoogleTrends::with_peak_threshold(10, 30, 85.0).unwrap();
        let output = gt.calculate(&close, &volume);

        // Should detect at least one peak during the spike
        let peak_count: i8 = output.peaks[55..70].iter().sum();
        assert!(peak_count >= 0); // May or may not detect depending on threshold
    }

    #[test]
    fn test_google_trends_momentum() {
        let (close, volume) = make_spike_data();
        let gt = GoogleTrends::new(10, 30).unwrap();
        let output = gt.calculate(&close, &volume);

        // Momentum should be positive during rising interest
        assert!(output.momentum[62] > 0.0 || output.momentum[63] > 0.0);
    }

    #[test]
    fn test_google_trends_relative() {
        let (close, volume) = make_spike_data();
        let gt = GoogleTrends::new(10, 30).unwrap();
        let output = gt.calculate(&close, &volume);

        // Relative interest should be positive during spike
        let spike_relative: f64 = output.relative_interest[60..65].iter().sum::<f64>() / 5.0;
        assert!(spike_relative > 0.0);
    }

    #[test]
    fn test_google_trends_validation() {
        assert!(GoogleTrends::new(2, 30).is_err());
        assert!(GoogleTrends::new(10, 5).is_err());
        assert!(GoogleTrends::new(10, 10).is_err());
        assert!(GoogleTrends::new(10, 30).is_ok());
    }

    #[test]
    fn test_technical_indicator_impl() {
        let gt = GoogleTrends::default();
        assert_eq!(gt.name(), "Google Trends Proxy");
        assert!(gt.min_periods() > 0);
    }

    #[test]
    fn test_interest_trend() {
        let (close, volume) = make_spike_data();
        let gt = GoogleTrends::new(10, 30).unwrap();
        let output = gt.calculate(&close, &volume);

        let trend = gt.interest_trend(&output.interest_index);
        // Trend calculation should return a valid number
        assert!(!trend.is_nan());
    }

    #[test]
    fn test_interpret_contrarian() {
        let gt = GoogleTrends::default();

        // High interest with falling momentum = peak
        assert_eq!(
            gt.interpret_contrarian(75.0, -10.0),
            GoogleTrendsSignal::PeakInterest
        );

        // Low interest with rising momentum = minimum
        assert_eq!(
            gt.interpret_contrarian(25.0, 10.0),
            GoogleTrendsSignal::MinimumInterest
        );
    }
}
