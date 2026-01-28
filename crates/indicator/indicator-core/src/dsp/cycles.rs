//! Cycle Detection Indicators
//!
//! Indicators for detecting and measuring market cycles.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Dominant Cycle Period - Estimates dominant cycle length
#[derive(Debug, Clone)]
pub struct DominantCyclePeriod {
    min_period: usize,
    max_period: usize,
}

impl DominantCyclePeriod {
    pub fn new(min_period: usize, max_period: usize) -> Result<Self> {
        if min_period < 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_period".to_string(),
                reason: "must be at least 4".to_string(),
            });
        }
        if max_period <= min_period {
            return Err(IndicatorError::InvalidParameter {
                name: "max_period".to_string(),
                reason: "must be greater than min_period".to_string(),
            });
        }
        Ok(Self { min_period, max_period })
    }

    /// Find dominant cycle using autocorrelation
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.max_period..n {
            let start = i.saturating_sub(self.max_period);
            let window = &close[start..=i];

            // Calculate mean
            let mean = window.iter().sum::<f64>() / window.len() as f64;

            // Find period with highest autocorrelation
            let mut best_period = self.min_period;
            let mut best_corr = 0.0;

            for period in self.min_period..=self.max_period.min(window.len() / 2) {
                let mut sum_xy = 0.0;
                let mut sum_xx = 0.0;
                let mut sum_yy = 0.0;

                for j in 0..(window.len() - period) {
                    let x = window[j] - mean;
                    let y = window[j + period] - mean;
                    sum_xy += x * y;
                    sum_xx += x * x;
                    sum_yy += y * y;
                }

                let corr = if sum_xx > 0.0 && sum_yy > 0.0 {
                    sum_xy / (sum_xx * sum_yy).sqrt()
                } else {
                    0.0
                };

                if corr > best_corr {
                    best_corr = corr;
                    best_period = period;
                }
            }

            result[i] = best_period as f64;
        }
        result
    }
}

impl TechnicalIndicator for DominantCyclePeriod {
    fn name(&self) -> &str {
        "Dominant Cycle Period"
    }

    fn min_periods(&self) -> usize {
        self.max_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Cycle Amplitude - Measures cycle strength
#[derive(Debug, Clone)]
pub struct CycleAmplitude {
    period: usize,
}

impl CycleAmplitude {
    pub fn new(period: usize) -> Result<Self> {
        if period < 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 4".to_string(),
            });
        }
        Ok(Self { period })
    }

    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let window = &close[start..=i];

            let max_val = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_val = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let mean = window.iter().sum::<f64>() / window.len() as f64;

            // Amplitude as percentage of mean
            if mean > 0.0 {
                result[i] = (max_val - min_val) / mean * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for CycleAmplitude {
    fn name(&self) -> &str {
        "Cycle Amplitude"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Cycle Phase - Current position in cycle
#[derive(Debug, Clone)]
pub struct CyclePhase {
    period: usize,
}

impl CyclePhase {
    pub fn new(period: usize) -> Result<Self> {
        if period < 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 4".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate phase (0-360 degrees)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let window = &close[start..=i];

            // Calculate detrended position
            let first = window[0];
            let last = window[window.len() - 1];
            let trend_slope = (last - first) / window.len() as f64;

            // Find local extrema
            let mut last_high_idx = 0;
            let mut last_low_idx = 0;

            for j in 1..(window.len() - 1) {
                let detrended = window[j] - first - trend_slope * j as f64;
                let prev_detrended = window[j - 1] - first - trend_slope * (j - 1) as f64;
                let next_detrended = window[j + 1] - first - trend_slope * (j + 1) as f64;

                if detrended > prev_detrended && detrended > next_detrended {
                    last_high_idx = j;
                }
                if detrended < prev_detrended && detrended < next_detrended {
                    last_low_idx = j;
                }
            }

            // Estimate phase based on position relative to last extrema
            let current_pos = window.len() - 1;
            if last_high_idx > last_low_idx {
                // Coming from a high, heading to low
                let progress = (current_pos - last_high_idx) as f64 / (self.period as f64 / 2.0);
                result[i] = 180.0 + progress * 180.0;
            } else {
                // Coming from a low, heading to high
                let progress = (current_pos - last_low_idx) as f64 / (self.period as f64 / 2.0);
                result[i] = progress * 180.0;
            }

            result[i] = result[i] % 360.0;
        }
        result
    }
}

impl TechnicalIndicator for CyclePhase {
    fn name(&self) -> &str {
        "Cycle Phase"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend-Cycle Decomposition - Separates trend from cycle
#[derive(Debug, Clone)]
pub struct TrendCycleDecomposition {
    trend_period: usize,
    cycle_period: usize,
}

impl TrendCycleDecomposition {
    pub fn new(trend_period: usize, cycle_period: usize) -> Result<Self> {
        if trend_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if cycle_period < 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "cycle_period".to_string(),
                reason: "must be at least 4".to_string(),
            });
        }
        Ok(Self { trend_period, cycle_period })
    }

    /// Returns (trend, cycle) components
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut trend = vec![0.0; n];
        let mut cycle = vec![0.0; n];

        // Calculate trend (SMA)
        for i in (self.trend_period - 1)..n {
            let start = i.saturating_sub(self.trend_period - 1);
            trend[i] = close[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
        }

        // Calculate cycle (detrended)
        for i in 0..n {
            cycle[i] = close[i] - trend[i];
        }

        (trend, cycle)
    }
}

impl TechnicalIndicator for TrendCycleDecomposition {
    fn name(&self) -> &str {
        "Trend-Cycle Decomposition"
    }

    fn min_periods(&self) -> usize {
        self.trend_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (trend, cycle) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(trend, cycle))
    }
}

/// Cycle Momentum - Rate of change within cycle
#[derive(Debug, Clone)]
pub struct CycleMomentum {
    period: usize,
}

impl CycleMomentum {
    pub fn new(period: usize) -> Result<Self> {
        if period < 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 4".to_string(),
            });
        }
        Ok(Self { period })
    }

    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First detrend the data
        let half_period = self.period / 2;

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Simple moving average for trend
            let trend: f64 = close[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;

            // Detrended values
            let current_detrended = close[i] - trend;
            let past_detrended = if i >= half_period {
                let past_start = (i - half_period).saturating_sub(self.period);
                let past_trend: f64 = close[past_start..(i - half_period + 1)].iter().sum::<f64>()
                    / ((i - half_period + 1) - past_start) as f64;
                close[i - half_period] - past_trend
            } else {
                0.0
            };

            // Momentum of detrended series
            result[i] = current_detrended - past_detrended;
        }
        result
    }
}

impl TechnicalIndicator for CycleMomentum {
    fn name(&self) -> &str {
        "Cycle Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Cycle Turning Point - Detects cycle peaks and troughs
#[derive(Debug, Clone)]
pub struct CycleTurningPoint {
    period: usize,
    smoothing: usize,
}

impl CycleTurningPoint {
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 4".to_string(),
            });
        }
        if smoothing < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, smoothing })
    }

    /// Returns 1 for peak, -1 for trough, 0 otherwise
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First smooth the data
        let mut smoothed = vec![0.0; n];
        for i in (self.smoothing - 1)..n {
            let start = i.saturating_sub(self.smoothing - 1);
            smoothed[i] = close[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
        }

        // Detect turning points
        let lookback = self.period / 2;
        for i in (lookback * 2)..n {
            let current = smoothed[i - lookback];
            let mut is_peak = true;
            let mut is_trough = true;

            // Check if current is higher/lower than surrounding points
            for j in 0..=lookback {
                if j != lookback {
                    let left_idx = i - lookback - (lookback - j);
                    let right_idx = i - lookback + (j + 1);

                    if left_idx < n && smoothed[left_idx] >= current {
                        is_peak = false;
                    }
                    if right_idx < n && smoothed[right_idx] >= current {
                        is_peak = false;
                    }
                    if left_idx < n && smoothed[left_idx] <= current {
                        is_trough = false;
                    }
                    if right_idx < n && smoothed[right_idx] <= current {
                        is_trough = false;
                    }
                }
            }

            if is_peak {
                result[i - lookback] = 1.0;
            } else if is_trough {
                result[i - lookback] = -1.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for CycleTurningPoint {
    fn name(&self) -> &str {
        "Cycle Turning Point"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        let mut close = Vec::with_capacity(100);
        for i in 0..100 {
            // Sine wave with trend
            let trend = 100.0 + i as f64 * 0.1;
            let cycle = 5.0 * (i as f64 * std::f64::consts::PI / 10.0).sin();
            close.push(trend + cycle);
        }
        close
    }

    #[test]
    fn test_dominant_cycle_period() {
        let close = make_test_data();
        let dcp = DominantCyclePeriod::new(5, 30).unwrap();
        let result = dcp.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Check that we get valid period values
        let avg_period: f64 = result[50..].iter().sum::<f64>() / (result.len() - 50) as f64;
        assert!(avg_period >= 5.0 && avg_period <= 30.0, "Period {} out of range", avg_period);
    }

    #[test]
    fn test_cycle_amplitude() {
        let close = make_test_data();
        let ca = CycleAmplitude::new(20).unwrap();
        let result = ca.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[50] > 0.0);
    }

    #[test]
    fn test_cycle_phase() {
        let close = make_test_data();
        let cp = CyclePhase::new(20).unwrap();
        let result = cp.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Phase should be between 0 and 360
        for &v in result.iter().skip(25) {
            assert!(v >= 0.0 && v < 360.0, "Phase {} out of range", v);
        }
    }

    #[test]
    fn test_trend_cycle_decomposition() {
        let close = make_test_data();
        let tcd = TrendCycleDecomposition::new(20, 10).unwrap();
        let (trend, cycle) = tcd.calculate(&close);

        assert_eq!(trend.len(), close.len());
        assert_eq!(cycle.len(), close.len());
        // Trend should be increasing
        assert!(trend[80] > trend[40]);
    }

    #[test]
    fn test_cycle_momentum() {
        let close = make_test_data();
        let cm = CycleMomentum::new(20).unwrap();
        let result = cm.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_cycle_turning_point() {
        let close = make_test_data();
        let ctp = CycleTurningPoint::new(10, 3).unwrap();
        let result = ctp.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Should detect some peaks and troughs
        let peaks = result.iter().filter(|&&v| v == 1.0).count();
        let troughs = result.iter().filter(|&&v| v == -1.0).count();
        assert!(peaks > 0 && troughs > 0);
    }
}
