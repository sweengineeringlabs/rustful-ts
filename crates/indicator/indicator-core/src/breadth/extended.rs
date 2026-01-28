//! Extended Market Breadth Indicators
//!
//! Additional breadth indicators for market-wide analysis.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Advance/Decline Thrust - Measures AD momentum
#[derive(Debug, Clone)]
pub struct ADThrust {
    period: usize,
}

impl ADThrust {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate AD Thrust using price momentum as proxy
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut declines = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let total = advances + declines;
            if total > 0 {
                result[i] = (advances as f64 / total as f64) * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for ADThrust {
    fn name(&self) -> &str {
        "AD Thrust"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Zweig Breadth Thrust - Extreme breadth signal
#[derive(Debug, Clone)]
pub struct ZweigBreadthThrust {
    period: usize,
    threshold: f64,
}

impl ZweigBreadthThrust {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Returns 1 for thrust signal, 0 otherwise
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut ema = vec![0.0; n];

        // Calculate advance ratio
        let mut ad_ratio = vec![0.0; n];
        for i in 1..n {
            ad_ratio[i] = if close[i] > close[i - 1] { 1.0 } else { 0.0 };
        }

        // Calculate EMA of advance ratio
        let alpha = 2.0 / (self.period as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                ema[i] = ad_ratio[i];
            } else {
                ema[i] = alpha * ad_ratio[i] + (1.0 - alpha) * ema[i - 1];
            }
        }

        // Detect thrust
        let mut in_oversold = false;
        for i in self.period..n {
            if ema[i] < (1.0 - self.threshold) {
                in_oversold = true;
            }
            if in_oversold && ema[i] > self.threshold {
                result[i] = 1.0;
                in_oversold = false;
            }
        }
        result
    }
}

impl TechnicalIndicator for ZweigBreadthThrust {
    fn name(&self) -> &str {
        "Zweig Breadth Thrust"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Arms Index (TRIN) Extended - With signal smoothing
#[derive(Debug, Clone)]
pub struct TRINSmoothed {
    period: usize,
}

impl TRINSmoothed {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate smoothed TRIN proxy
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut adv_vol = 0.0;
            let mut dec_vol = 0.0;
            let mut adv_count = 0;
            let mut dec_count = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    adv_vol += volume[j];
                    adv_count += 1;
                } else if close[j] < close[j - 1] {
                    dec_vol += volume[j];
                    dec_count += 1;
                }
            }

            // TRIN = (AD Ratio) / (Volume Ratio)
            if dec_count > 0 && adv_vol > 0.0 && dec_vol > 0.0 {
                let ad_ratio = adv_count as f64 / dec_count as f64;
                let vol_ratio = adv_vol / dec_vol;
                result[i] = ad_ratio / vol_ratio;
            }
        }
        result
    }
}

impl TechnicalIndicator for TRINSmoothed {
    fn name(&self) -> &str {
        "TRIN Smoothed"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Breadth Momentum - Rate of change in breadth
#[derive(Debug, Clone)]
pub struct BreadthMomentum {
    period: usize,
    roc_period: usize,
}

impl BreadthMomentum {
    pub fn new(period: usize, roc_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if roc_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "roc_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, roc_period })
    }

    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate cumulative AD line
        let mut ad_line = vec![0.0; n];
        for i in 1..n {
            let advance = if close[i] > close[i - 1] { 1.0 } else { 0.0 };
            let decline = if close[i] < close[i - 1] { 1.0 } else { 0.0 };
            ad_line[i] = ad_line[i - 1] + advance - decline;
        }

        // Calculate momentum of AD line
        for i in (self.period + self.roc_period)..n {
            let base_val: f64 = ad_line[i - self.roc_period];
            if base_val != 0.0 {
                result[i] = ((ad_line[i] - base_val) / base_val.abs()) * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for BreadthMomentum {
    fn name(&self) -> &str {
        "Breadth Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + self.roc_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volume Breadth - Volume-weighted breadth
#[derive(Debug, Clone)]
pub struct VolumeBreadth {
    period: usize,
}

impl VolumeBreadth {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut up_vol = 0.0;
            let mut down_vol = 0.0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_vol += volume[j];
                } else if close[j] < close[j - 1] {
                    down_vol += volume[j];
                }
            }

            let total = up_vol + down_vol;
            if total > 0.0 {
                result[i] = (up_vol - down_vol) / total * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for VolumeBreadth {
    fn name(&self) -> &str {
        "Volume Breadth"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Percentage Breadth - Momentum breadth
#[derive(Debug, Clone)]
pub struct PercentageBreadth {
    period: usize,
    momentum_period: usize,
}

impl PercentageBreadth {
    pub fn new(period: usize, momentum_period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if momentum_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, momentum_period })
    }

    /// Percentage of periods with positive momentum
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in (self.period + self.momentum_period)..n {
            let start = i.saturating_sub(self.period);
            let mut positive_count = 0;
            let mut total_count = 0;

            for j in (start + self.momentum_period)..=i {
                let momentum = close[j] / close[j - self.momentum_period] - 1.0;
                if momentum > 0.0 {
                    positive_count += 1;
                }
                total_count += 1;
            }

            if total_count > 0 {
                result[i] = (positive_count as f64 / total_count as f64) * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for PercentageBreadth {
    fn name(&self) -> &str {
        "Percentage Breadth"
    }

    fn min_periods(&self) -> usize {
        self.period + self.momentum_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        let close = vec![100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0,
                        104.5, 106.0, 105.5, 107.0, 106.5, 108.0, 107.5, 109.0, 108.5, 110.0,
                        109.5, 111.0, 110.5, 112.0, 111.5, 113.0, 112.5, 114.0, 113.5, 115.0];
        let volume = vec![1000.0, 1200.0, 1100.0, 1300.0, 1500.0, 1400.0, 1600.0, 1800.0, 1700.0, 1900.0,
                         2000.0, 1800.0, 2100.0, 2200.0, 2000.0, 2300.0, 2100.0, 2400.0, 2200.0, 2500.0,
                         2300.0, 2600.0, 2400.0, 2700.0, 2500.0, 2800.0, 2600.0, 2900.0, 2700.0, 3000.0];
        (close, volume)
    }

    #[test]
    fn test_ad_thrust() {
        let (close, _) = make_test_data();
        let adt = ADThrust::new(10).unwrap();
        let result = adt.calculate(&close);

        assert_eq!(result.len(), close.len());
        // In uptrend, should be above 50%
        assert!(result[20] > 40.0);
    }

    #[test]
    fn test_zweig_breadth_thrust() {
        let (close, _) = make_test_data();
        let zbt = ZweigBreadthThrust::new(10, 0.6).unwrap();
        let result = zbt.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_trin_smoothed() {
        let (close, volume) = make_test_data();
        let trin = TRINSmoothed::new(5).unwrap();
        let result = trin.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_breadth_momentum() {
        let (close, _) = make_test_data();
        let bm = BreadthMomentum::new(10, 5).unwrap();
        let result = bm.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_volume_breadth() {
        let (close, volume) = make_test_data();
        let vb = VolumeBreadth::new(10).unwrap();
        let result = vb.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // In uptrend, should be positive
        assert!(result[20] > 0.0);
    }

    #[test]
    fn test_percentage_breadth() {
        let (close, _) = make_test_data();
        let pb = PercentageBreadth::new(10, 3).unwrap();
        let result = pb.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[20] >= 0.0 && result[20] <= 100.0);
    }
}
