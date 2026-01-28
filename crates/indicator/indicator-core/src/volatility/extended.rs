//! Extended Volatility Indicators
//!
//! Additional volatility and range analysis indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Volatility Ratio - Range vs average range
#[derive(Debug, Clone)]
pub struct VolatilityRatio {
    period: usize,
}

impl VolatilityRatio {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volatility ratio
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            // True range for current bar
            let tr = high[i] - low[i];

            // Average range over period
            let start = i.saturating_sub(self.period);
            let avg_range: f64 = (start..i)
                .map(|j| high[j] - low[j])
                .sum::<f64>() / self.period as f64;

            if avg_range > 1e-10 {
                result[i] = tr / avg_range;
            }
        }
        result
    }
}

impl TechnicalIndicator for VolatilityRatio {
    fn name(&self) -> &str {
        "Volatility Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low)))
    }
}

/// Range Expansion Index - Measures range expansion
#[derive(Debug, Clone)]
pub struct RangeExpansionIndex {
    period: usize,
    smoothing: usize,
}

impl RangeExpansionIndex {
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

    /// Calculate Range Expansion Index
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len()).min(close.len());
        let mut result = vec![0.0; n];

        for i in (self.period + self.smoothing)..n {
            let mut expansion_sum = 0.0;

            for j in (i - self.smoothing + 1)..=i {
                let start = j.saturating_sub(self.period);

                // Current range
                let current_range = high[j] - low[j];

                // Previous highest high and lowest low
                let prev_high = high[start..j].iter().cloned().fold(f64::MIN, f64::max);
                let prev_low = low[start..j].iter().cloned().fold(f64::MAX, f64::min);

                // Range expansion score
                let expansion = if high[j] > prev_high { 1.0 } else { 0.0 }
                    + if low[j] < prev_low { 1.0 } else { 0.0 };

                expansion_sum += expansion;
            }

            result[i] = expansion_sum / (self.smoothing as f64 * 2.0) * 100.0;
        }
        result
    }
}

impl TechnicalIndicator for RangeExpansionIndex {
    fn name(&self) -> &str {
        "Range Expansion Index"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Intraday Intensity Volatility - Volume-weighted range
#[derive(Debug, Clone)]
pub struct IntradayIntensityVolatility {
    period: usize,
}

impl IntradayIntensityVolatility {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume-weighted volatility
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len()).min(close.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            let mut weighted_range_sum = 0.0;
            let mut volume_sum = 0.0;

            for j in start..=i {
                let range = high[j] - low[j];
                // Intraday intensity: where close is within range
                let close_position = if range > 1e-10 {
                    ((close[j] - low[j]) / range * 2.0 - 1.0).abs()
                } else {
                    0.0
                };

                weighted_range_sum += range * close_position * volume[j];
                volume_sum += volume[j];
            }

            if volume_sum > 1e-10 {
                result[i] = weighted_range_sum / volume_sum;
            }
        }
        result
    }
}

impl TechnicalIndicator for IntradayIntensityVolatility {
    fn name(&self) -> &str {
        "Intraday Intensity Volatility"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Normalized Volatility - Z-score of volatility
#[derive(Debug, Clone)]
pub struct NormalizedVolatility {
    volatility_period: usize,
    zscore_period: usize,
}

impl NormalizedVolatility {
    pub fn new(volatility_period: usize, zscore_period: usize) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if zscore_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "zscore_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { volatility_period, zscore_period })
    }

    /// Calculate z-score normalized volatility
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First calculate rolling volatility
        let mut volatility = vec![0.0; n];
        for i in self.volatility_period..n {
            let start = i.saturating_sub(self.volatility_period);
            let returns: Vec<f64> = (start + 1..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();

            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let var: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / returns.len() as f64;
            volatility[i] = var.sqrt() * (252.0_f64).sqrt();
        }

        // Then normalize with z-score
        for i in (self.volatility_period + self.zscore_period)..n {
            let start = i.saturating_sub(self.zscore_period);
            let vol_slice = &volatility[start..=i];

            let mean: f64 = vol_slice.iter().sum::<f64>() / vol_slice.len() as f64;
            let var: f64 = vol_slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / vol_slice.len() as f64;
            let std = var.sqrt();

            if std > 1e-10 {
                result[i] = (volatility[i] - mean) / std;
            }
        }
        result
    }
}

impl TechnicalIndicator for NormalizedVolatility {
    fn name(&self) -> &str {
        "Normalized Volatility"
    }

    fn min_periods(&self) -> usize {
        self.volatility_period + self.zscore_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volatility Breakout - Detects volatility breakouts
#[derive(Debug, Clone)]
pub struct VolatilityBreakout {
    period: usize,
    threshold: f64,
}

impl VolatilityBreakout {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate breakout signals (1 = upside, -1 = downside, 0 = none)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len()).min(close.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate average true range
            let mut atr_sum = 0.0;
            for j in (start + 1)..i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                atr_sum += tr;
            }
            let atr = atr_sum / (self.period - 1) as f64;

            // Check for breakout
            let prev_close = close[i - 1];
            let upper_band = prev_close + atr * self.threshold;
            let lower_band = prev_close - atr * self.threshold;

            if close[i] > upper_band {
                result[i] = 1.0;
            } else if close[i] < lower_band {
                result[i] = -1.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for VolatilityBreakout {
    fn name(&self) -> &str {
        "Volatility Breakout"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Volatility Regime Classifier - Current volatility regime classification
#[derive(Debug, Clone)]
pub struct VolatilityRegimeClassifier {
    short_period: usize,
    long_period: usize,
}

impl VolatilityRegimeClassifier {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self { short_period, long_period })
    }

    /// Calculate regime (positive = high vol, negative = low vol, magnitude = strength)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term volatility
            let short_start = i.saturating_sub(self.short_period);
            let short_returns: Vec<f64> = (short_start + 1..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();
            let short_mean: f64 = short_returns.iter().sum::<f64>() / short_returns.len() as f64;
            let short_var: f64 = short_returns.iter()
                .map(|r| (r - short_mean).powi(2))
                .sum::<f64>() / short_returns.len() as f64;
            let short_vol = short_var.sqrt();

            // Long-term volatility
            let long_start = i.saturating_sub(self.long_period);
            let long_returns: Vec<f64> = (long_start + 1..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();
            let long_mean: f64 = long_returns.iter().sum::<f64>() / long_returns.len() as f64;
            let long_var: f64 = long_returns.iter()
                .map(|r| (r - long_mean).powi(2))
                .sum::<f64>() / long_returns.len() as f64;
            let long_vol = long_var.sqrt();

            // Regime: ratio of short to long volatility
            if long_vol > 1e-10 {
                let ratio = short_vol / long_vol;
                // Above 1 = high vol regime, below 1 = low vol regime
                result[i] = (ratio - 1.0) * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for VolatilityRegimeClassifier {
    fn name(&self) -> &str {
        "Volatility Regime Classifier"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![102.0, 103.0, 104.0, 103.0, 105.0, 106.0, 105.0, 107.0, 108.0, 107.0,
                       109.0, 110.0, 109.0, 111.0, 112.0, 111.0, 113.0, 114.0, 113.0, 115.0,
                       116.0, 115.0, 117.0, 118.0, 117.0, 119.0, 120.0, 119.0, 121.0, 122.0];
        let low = vec![98.0, 99.0, 100.0, 99.0, 101.0, 102.0, 101.0, 103.0, 104.0, 103.0,
                      105.0, 106.0, 105.0, 107.0, 108.0, 107.0, 109.0, 110.0, 109.0, 111.0,
                      112.0, 111.0, 113.0, 114.0, 113.0, 115.0, 116.0, 115.0, 117.0, 118.0];
        let close = vec![100.0, 101.0, 102.0, 101.0, 103.0, 104.0, 103.0, 105.0, 106.0, 105.0,
                        107.0, 108.0, 107.0, 109.0, 110.0, 109.0, 111.0, 112.0, 111.0, 113.0,
                        114.0, 113.0, 115.0, 116.0, 115.0, 117.0, 118.0, 117.0, 119.0, 120.0];
        let volume = vec![1000.0, 1100.0, 1200.0, 1100.0, 1300.0, 1400.0, 1300.0, 1500.0, 1600.0, 1500.0,
                         1700.0, 1800.0, 1700.0, 1900.0, 2000.0, 1900.0, 2100.0, 2200.0, 2100.0, 2300.0,
                         2400.0, 2300.0, 2500.0, 2600.0, 2500.0, 2700.0, 2800.0, 2700.0, 2900.0, 3000.0];
        (high, low, close, volume)
    }

    #[test]
    fn test_volatility_ratio() {
        let (high, low, _, _) = make_test_data();
        let vr = VolatilityRatio::new(10).unwrap();
        let result = vr.calculate(&high, &low);

        assert_eq!(result.len(), high.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_range_expansion_index() {
        let (high, low, close, _) = make_test_data();
        let rei = RangeExpansionIndex::new(10, 5).unwrap();
        let result = rei.calculate(&high, &low, &close);

        assert_eq!(result.len(), high.len());
        assert!(result[20] >= 0.0 && result[20] <= 100.0);
    }

    #[test]
    fn test_intraday_intensity_volatility() {
        let (high, low, close, volume) = make_test_data();
        let iiv = IntradayIntensityVolatility::new(10).unwrap();
        let result = iiv.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), high.len());
        assert!(result[15] >= 0.0);
    }

    #[test]
    fn test_normalized_volatility() {
        let (_, _, close, _) = make_test_data();
        let nv = NormalizedVolatility::new(5, 15).unwrap();
        let result = nv.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_volatility_breakout() {
        let (high, low, close, _) = make_test_data();
        let vb = VolatilityBreakout::new(10, 1.5).unwrap();
        let result = vb.calculate(&high, &low, &close);

        assert_eq!(result.len(), high.len());
        // Values should be -1, 0, or 1
        assert!(result.iter().all(|&v| v == -1.0 || v == 0.0 || v == 1.0));
    }

    #[test]
    fn test_volatility_regime_classifier() {
        let (_, _, close, _) = make_test_data();
        let vr = VolatilityRegimeClassifier::new(5, 20).unwrap();
        let result = vr.calculate(&close);

        assert_eq!(result.len(), close.len());
    }
}
