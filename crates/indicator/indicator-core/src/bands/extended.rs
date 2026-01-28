//! Extended Band Indicators
//!
//! Additional channel and band indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Adaptive Bands - Volatility-adaptive bands
#[derive(Debug, Clone)]
pub struct AdaptiveBands {
    period: usize,
    mult: f64,
}

impl AdaptiveBands {
    pub fn new(period: usize, mult: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, mult })
    }

    /// Calculate adaptive bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Adaptive MA (weighted by range)
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for j in start..=i {
                let range = high[j] - low[j];
                weighted_sum += close[j] * range;
                weight_sum += range;
            }

            let avg = if weight_sum > 1e-10 {
                weighted_sum / weight_sum
            } else {
                close[start..=i].iter().sum::<f64>() / self.period as f64
            };

            // ATR for band width
            let mut atr_sum = 0.0;
            for j in (start + 1)..=i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                atr_sum += tr;
            }
            let atr = atr_sum / self.period as f64;

            middle[i] = avg;
            upper[i] = avg + self.mult * atr;
            lower[i] = avg - self.mult * atr;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for AdaptiveBands {
    fn name(&self) -> &str {
        "Adaptive Bands"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Fixed Percentage Envelope - Fixed percentage bands
#[derive(Debug, Clone)]
pub struct FixedPercentageEnvelope {
    period: usize,
    percent: f64,
}

impl FixedPercentageEnvelope {
    pub fn new(period: usize, percent: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if percent <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "percent".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, percent })
    }

    /// Calculate percentage bands (middle, upper, lower)
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let ma: f64 = close[start..=i].iter().sum::<f64>() / self.period as f64;

            middle[i] = ma;
            upper[i] = ma * (1.0 + self.percent / 100.0);
            lower[i] = ma * (1.0 - self.percent / 100.0);
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for FixedPercentageEnvelope {
    fn name(&self) -> &str {
        "Fixed Percentage Envelope"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Momentum Bands - Bands based on momentum
#[derive(Debug, Clone)]
pub struct MomentumBands {
    period: usize,
    momentum_period: usize,
    mult: f64,
}

impl MomentumBands {
    pub fn new(period: usize, momentum_period: usize, mult: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, momentum_period, mult })
    }

    /// Calculate momentum bands
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        // Calculate momentum
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            momentum[i] = close[i] - close[i - self.momentum_period];
        }

        let start_idx = self.period.max(self.momentum_period);

        for i in start_idx..n {
            let start = i.saturating_sub(self.period);
            let ma: f64 = close[start..=i].iter().sum::<f64>() / self.period as f64;

            // Momentum std dev
            let mom_slice = &momentum[start..=i];
            let mom_mean: f64 = mom_slice.iter().sum::<f64>() / mom_slice.len() as f64;
            let mom_var: f64 = mom_slice.iter()
                .map(|m| (m - mom_mean).powi(2))
                .sum::<f64>() / mom_slice.len() as f64;
            let mom_std = mom_var.sqrt();

            middle[i] = ma;
            upper[i] = ma + self.mult * mom_std;
            lower[i] = ma - self.mult * mom_std;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for MomentumBands {
    fn name(&self) -> &str {
        "Momentum Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.momentum_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Volume Weighted Bands - Bands weighted by volume
#[derive(Debug, Clone)]
pub struct VolumeWeightedBands {
    period: usize,
    mult: f64,
}

impl VolumeWeightedBands {
    pub fn new(period: usize, mult: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period, mult })
    }

    /// Calculate volume weighted bands
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // VWMA
            let mut pv_sum = 0.0;
            let mut v_sum = 0.0;
            for j in start..=i {
                pv_sum += close[j] * volume[j];
                v_sum += volume[j];
            }

            let vwma = if v_sum > 1e-10 { pv_sum / v_sum } else { close[i] };

            // Volume-weighted standard deviation
            let mut weighted_var = 0.0;
            for j in start..=i {
                weighted_var += volume[j] * (close[j] - vwma).powi(2);
            }
            let vw_std = if v_sum > 1e-10 {
                (weighted_var / v_sum).sqrt()
            } else {
                0.0
            };

            middle[i] = vwma;
            upper[i] = vwma + self.mult * vw_std;
            lower[i] = vwma - self.mult * vw_std;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for VolumeWeightedBands {
    fn name(&self) -> &str {
        "Volume Weighted Bands"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Dynamic Channel - Channels that expand/contract
#[derive(Debug, Clone)]
pub struct DynamicChannel {
    period: usize,
    smoothing: usize,
}

impl DynamicChannel {
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smoothing < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smoothing })
    }

    /// Calculate dynamic channel
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len().min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        // Raw channels
        let mut raw_upper = vec![0.0; n];
        let mut raw_lower = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            raw_upper[i] = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            raw_lower[i] = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
        }

        // Smooth the channels
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in self.period..n {
            if i == self.period {
                upper[i] = raw_upper[i];
                lower[i] = raw_lower[i];
            } else {
                upper[i] = alpha * raw_upper[i] + (1.0 - alpha) * upper[i - 1];
                lower[i] = alpha * raw_lower[i] + (1.0 - alpha) * lower[i - 1];
            }
            middle[i] = (upper[i] + lower[i]) / 2.0;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for DynamicChannel {
    fn name(&self) -> &str {
        "Dynamic Channel"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Linear Regression Channel - Regression-based bands
#[derive(Debug, Clone)]
pub struct LinearRegressionChannel {
    period: usize,
    mult: f64,
}

impl LinearRegressionChannel {
    pub fn new(period: usize, mult: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period, mult })
    }

    /// Calculate linear regression channel
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let slice = &close[start..=i];

            // Linear regression
            let n_points = slice.len() as f64;
            let sum_x: f64 = (0..slice.len()).map(|x| x as f64).sum();
            let sum_y: f64 = slice.iter().sum();
            let sum_xy: f64 = slice.iter().enumerate().map(|(x, &y)| x as f64 * y).sum();
            let sum_xx: f64 = (0..slice.len()).map(|x| (x * x) as f64).sum();

            let slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_xx - sum_x * sum_x);
            let intercept = (sum_y - slope * sum_x) / n_points;

            // Regression value at current point
            let reg_value = intercept + slope * (self.period as f64 - 1.0);

            // Standard error
            let mut sse = 0.0;
            for (x, &y) in slice.iter().enumerate() {
                let pred = intercept + slope * x as f64;
                sse += (y - pred).powi(2);
            }
            let std_err = (sse / (n_points - 2.0)).sqrt();

            middle[i] = reg_value;
            upper[i] = reg_value + self.mult * std_err;
            lower[i] = reg_value - self.mult * std_err;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for LinearRegressionChannel {
    fn name(&self) -> &str {
        "Linear Regression Channel"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                       112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0];
        let low = vec![98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
                      108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0];
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                        110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0];
        let volume = vec![1000.0; 20];
        (high, low, close, volume)
    }

    #[test]
    fn test_adaptive_bands() {
        let (high, low, close, _) = make_test_data();
        let ab = AdaptiveBands::new(10, 2.0).unwrap();
        let (middle, upper, lower) = ab.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        assert!(upper[15] > middle[15]);
        assert!(lower[15] < middle[15]);
    }

    #[test]
    fn test_fixed_percentage_envelope() {
        let (_, _, close, _) = make_test_data();
        let fpe = FixedPercentageEnvelope::new(10, 5.0).unwrap();
        let (middle, upper, lower) = fpe.calculate(&close);

        assert_eq!(middle.len(), close.len());
        assert!(upper[15] > middle[15]);
        assert!(lower[15] < middle[15]);
    }

    #[test]
    fn test_momentum_bands() {
        let (_, _, close, _) = make_test_data();
        let mb = MomentumBands::new(10, 5, 2.0).unwrap();
        let (middle, upper, lower) = mb.calculate(&close);

        assert_eq!(middle.len(), close.len());
        assert!(upper[15] >= middle[15]);
        assert!(lower[15] <= middle[15]);
    }

    #[test]
    fn test_volume_weighted_bands() {
        let (_, _, close, volume) = make_test_data();
        let vwb = VolumeWeightedBands::new(10, 2.0).unwrap();
        let (middle, upper, lower) = vwb.calculate(&close, &volume);

        assert_eq!(middle.len(), close.len());
        assert!(upper[15] >= middle[15]);
        assert!(lower[15] <= middle[15]);
    }

    #[test]
    fn test_dynamic_channel() {
        let (high, low, _, _) = make_test_data();
        let dc = DynamicChannel::new(10, 5).unwrap();
        let (middle, upper, lower) = dc.calculate(&high, &low);

        assert_eq!(middle.len(), high.len());
        assert!(upper[15] > middle[15]);
        assert!(lower[15] < middle[15]);
    }

    #[test]
    fn test_linear_regression_channel() {
        let (_, _, close, _) = make_test_data();
        let lrc = LinearRegressionChannel::new(10, 2.0).unwrap();
        let (middle, upper, lower) = lrc.calculate(&close);

        assert_eq!(middle.len(), close.len());
        // Upper should be >= middle, lower should be <= middle
        assert!(upper[15] >= middle[15]);
        assert!(lower[15] <= middle[15]);
    }
}
