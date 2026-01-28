//! Extended Moving Average Indicators
//!
//! Additional moving average implementations.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Volume Adjusted Moving Average
#[derive(Debug, Clone)]
pub struct VolumeAdjustedMA {
    period: usize,
}

impl VolumeAdjustedMA {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume-adjusted moving average
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period - 1);

            let vol_sum: f64 = volume[start..=i].iter().sum();
            if vol_sum > 1e-10 {
                let weighted_sum: f64 = (start..=i)
                    .map(|j| close[j] * volume[j])
                    .sum();
                result[i] = weighted_sum / vol_sum;
            } else {
                result[i] = close[start..=i].iter().sum::<f64>() / self.period as f64;
            }
        }
        result
    }
}

impl TechnicalIndicator for VolumeAdjustedMA {
    fn name(&self) -> &str {
        "Volume Adjusted MA"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Range Weighted Moving Average
#[derive(Debug, Clone)]
pub struct RangeWeightedMA {
    period: usize,
}

impl RangeWeightedMA {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate range-weighted moving average (weights by bar range)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period - 1);

            // Calculate ranges
            let ranges: Vec<f64> = (start..=i)
                .map(|j| (high[j] - low[j]).max(0.001))
                .collect();
            let range_sum: f64 = ranges.iter().sum();

            if range_sum > 1e-10 {
                let weighted_sum: f64 = (start..=i)
                    .zip(ranges.iter())
                    .map(|(j, &r)| close[j] * r)
                    .sum();
                result[i] = weighted_sum / range_sum;
            } else {
                result[i] = close[start..=i].iter().sum::<f64>() / self.period as f64;
            }
        }
        result
    }
}

impl TechnicalIndicator for RangeWeightedMA {
    fn name(&self) -> &str {
        "Range Weighted MA"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Momentum Weighted Moving Average
#[derive(Debug, Clone)]
pub struct MomentumWeightedMA {
    period: usize,
}

impl MomentumWeightedMA {
    pub fn new(period: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate momentum-weighted moving average
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period - 1);

            // Calculate momentum weights (abs change from previous)
            let weights: Vec<f64> = (start..=i)
                .map(|j| {
                    if j > 0 {
                        (close[j] - close[j - 1]).abs().max(0.001)
                    } else {
                        0.001
                    }
                })
                .collect();
            let weight_sum: f64 = weights.iter().sum();

            if weight_sum > 1e-10 {
                let weighted_sum: f64 = (start..=i)
                    .zip(weights.iter())
                    .map(|(j, &w)| close[j] * w)
                    .sum();
                result[i] = weighted_sum / weight_sum;
            } else {
                result[i] = close[start..=i].iter().sum::<f64>() / self.period as f64;
            }
        }
        result
    }
}

impl TechnicalIndicator for MomentumWeightedMA {
    fn name(&self) -> &str {
        "Momentum Weighted MA"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Adaptive Moving Average (Efficiency Ratio based)
#[derive(Debug, Clone)]
pub struct AdaptiveMA {
    period: usize,
    fast_period: usize,
    slow_period: usize,
}

impl AdaptiveMA {
    pub fn new(period: usize, fast_period: usize, slow_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if fast_period < 2 || fast_period >= slow_period {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be at least 2 and less than slow_period".to_string(),
            });
        }
        Ok(Self { period, fast_period, slow_period })
    }

    /// Calculate adaptive moving average
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        let fast_sc = 2.0 / (self.fast_period as f64 + 1.0);
        let slow_sc = 2.0 / (self.slow_period as f64 + 1.0);

        // Initialize first value
        if n > 0 {
            result[0] = close[0];
        }

        for i in 1..n {
            // Calculate efficiency ratio
            let er = if i >= self.period {
                let change = (close[i] - close[i - self.period]).abs();
                let volatility: f64 = (i - self.period + 1..=i)
                    .map(|j| (close[j] - close[j - 1]).abs())
                    .sum();
                if volatility > 1e-10 {
                    change / volatility
                } else {
                    0.0
                }
            } else {
                0.5
            };

            // Calculate smoothing constant
            let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);

            // Update AMA
            result[i] = result[i - 1] + sc * (close[i] - result[i - 1]);
        }
        result
    }
}

impl TechnicalIndicator for AdaptiveMA {
    fn name(&self) -> &str {
        "Adaptive MA"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Double Smoothed Moving Average
#[derive(Debug, Clone)]
pub struct DoubleSmoothedMA {
    period: usize,
}

impl DoubleSmoothedMA {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate double-smoothed moving average (EMA of EMA)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n == 0 {
            return vec![];
        }

        // First EMA
        let alpha = 2.0 / (self.period as f64 + 1.0);
        let mut ema1 = vec![0.0; n];
        ema1[0] = close[0];
        for i in 1..n {
            ema1[i] = alpha * close[i] + (1.0 - alpha) * ema1[i - 1];
        }

        // Second EMA (EMA of EMA)
        let mut result = vec![0.0; n];
        result[0] = ema1[0];
        for i in 1..n {
            result[i] = alpha * ema1[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }
}

impl TechnicalIndicator for DoubleSmoothedMA {
    fn name(&self) -> &str {
        "Double Smoothed MA"
    }

    fn min_periods(&self) -> usize {
        self.period * 2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Triple Exponential Moving Average (TRIX variant)
#[derive(Debug, Clone)]
pub struct TripleSmoothedMA {
    period: usize,
}

impl TripleSmoothedMA {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate triple-smoothed moving average
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n == 0 {
            return vec![];
        }

        let alpha = 2.0 / (self.period as f64 + 1.0);

        // First EMA
        let mut ema1 = vec![0.0; n];
        ema1[0] = close[0];
        for i in 1..n {
            ema1[i] = alpha * close[i] + (1.0 - alpha) * ema1[i - 1];
        }

        // Second EMA
        let mut ema2 = vec![0.0; n];
        ema2[0] = ema1[0];
        for i in 1..n {
            ema2[i] = alpha * ema1[i] + (1.0 - alpha) * ema2[i - 1];
        }

        // Third EMA
        let mut result = vec![0.0; n];
        result[0] = ema2[0];
        for i in 1..n {
            result[i] = alpha * ema2[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }
}

impl TechnicalIndicator for TripleSmoothedMA {
    fn name(&self) -> &str {
        "Triple Smoothed MA"
    }

    fn min_periods(&self) -> usize {
        self.period * 3
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![102.0, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0, 108.0, 107.5,
                       109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0, 114.0, 113.5, 115.0,
                       116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0, 119.5, 121.0, 122.0];
        let low = vec![98.0, 99.0, 100.0, 99.5, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5,
                      105.0, 106.0, 105.5, 107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0,
                      112.0, 111.5, 113.0, 114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0];
        let close = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
                        107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
                        114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0];
        let volume = vec![1000.0; 30];
        (high, low, close, volume)
    }

    #[test]
    fn test_volume_adjusted_ma() {
        let (_, _, close, volume) = make_test_data();
        let vama = VolumeAdjustedMA::new(10).unwrap();
        let result = vama.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_range_weighted_ma() {
        let (high, low, close, _) = make_test_data();
        let rwma = RangeWeightedMA::new(10).unwrap();
        let result = rwma.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_momentum_weighted_ma() {
        let (_, _, close, _) = make_test_data();
        let mwma = MomentumWeightedMA::new(10).unwrap();
        let result = mwma.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_adaptive_ma() {
        let (_, _, close, _) = make_test_data();
        let ama = AdaptiveMA::new(10, 2, 30).unwrap();
        let result = ama.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_double_smoothed_ma() {
        let (_, _, close, _) = make_test_data();
        let dsma = DoubleSmoothedMA::new(10).unwrap();
        let result = dsma.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_triple_smoothed_ma() {
        let (_, _, close, _) = make_test_data();
        let tsma = TripleSmoothedMA::new(5).unwrap();
        let result = tsma.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0);
    }
}
