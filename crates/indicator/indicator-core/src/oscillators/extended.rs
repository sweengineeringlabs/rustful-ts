//! Extended Oscillator Indicators
//!
//! Additional oscillator-type indicators for momentum analysis.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Adaptive RSI - RSI with adaptive period based on volatility
#[derive(Debug, Clone)]
pub struct AdaptiveRSI {
    base_period: usize,
    volatility_period: usize,
}

impl AdaptiveRSI {
    pub fn new(base_period: usize, volatility_period: usize) -> Result<Self> {
        if base_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { base_period, volatility_period })
    }

    /// Calculate adaptive RSI
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate volatility for adaptive period
        let mut volatility = vec![0.0; n];
        for i in self.volatility_period..n {
            let start = i.saturating_sub(self.volatility_period);
            let mean: f64 = close[start..=i].iter().sum::<f64>() / self.volatility_period as f64;
            let var: f64 = close[start..=i].iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.volatility_period as f64;
            volatility[i] = var.sqrt();
        }

        // Calculate average volatility
        let avg_vol: f64 = volatility.iter().filter(|&&v| v > 0.0).sum::<f64>()
            / volatility.iter().filter(|&&v| v > 0.0).count().max(1) as f64;

        let min_period = self.base_period.max(self.volatility_period);

        for i in min_period..n {
            // Adaptive period based on volatility
            let vol_ratio = if avg_vol > 1e-10 { volatility[i] / avg_vol } else { 1.0 };
            let adaptive_period = ((self.base_period as f64 / vol_ratio.max(0.5).min(2.0)) as usize)
                .max(3)
                .min(self.base_period * 2);

            // Calculate RSI with adaptive period
            let start = i.saturating_sub(adaptive_period);
            let mut gains = 0.0;
            let mut losses = 0.0;

            for j in (start + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses += change.abs();
                }
            }

            let period_len = (i - start) as f64;
            let avg_gain = gains / period_len;
            let avg_loss = losses / period_len;

            if avg_loss > 1e-10 {
                let rs = avg_gain / avg_loss;
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            } else if gains > 0.0 {
                result[i] = 100.0;
            } else {
                result[i] = 50.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for AdaptiveRSI {
    fn name(&self) -> &str {
        "Adaptive RSI"
    }

    fn min_periods(&self) -> usize {
        self.base_period.max(self.volatility_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volume Weighted RSI - RSI weighted by volume
#[derive(Debug, Clone)]
pub struct VolumeWeightedRSI {
    period: usize,
}

impl VolumeWeightedRSI {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume-weighted RSI
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut vol_weighted_gains = 0.0;
            let mut vol_weighted_losses = 0.0;
            let mut total_vol = 0.0;

            for j in (start + 1)..=i {
                let change = close[j] - close[j - 1];
                let vol = volume[j];
                total_vol += vol;

                if change > 0.0 {
                    vol_weighted_gains += change * vol;
                } else {
                    vol_weighted_losses += change.abs() * vol;
                }
            }

            if total_vol > 1e-10 {
                let avg_gain = vol_weighted_gains / total_vol;
                let avg_loss = vol_weighted_losses / total_vol;

                if avg_loss > 1e-10 {
                    let rs = avg_gain / avg_loss;
                    result[i] = 100.0 - (100.0 / (1.0 + rs));
                } else if avg_gain > 0.0 {
                    result[i] = 100.0;
                } else {
                    result[i] = 50.0;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for VolumeWeightedRSI {
    fn name(&self) -> &str {
        "Volume Weighted RSI"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Stochastic Momentum - Combines stochastic and momentum concepts
#[derive(Debug, Clone)]
pub struct StochasticMomentum {
    k_period: usize,
    d_period: usize,
}

impl StochasticMomentum {
    pub fn new(k_period: usize, d_period: usize) -> Result<Self> {
        if k_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "k_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if d_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "d_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { k_period, d_period })
    }

    /// Calculate stochastic momentum (K and D lines)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut k_line = vec![0.0; n];
        let mut d_line = vec![0.0; n];

        for i in self.k_period..n {
            let start = i.saturating_sub(self.k_period);

            // Find high-low range
            let mut highest = high[start];
            let mut lowest = low[start];
            for j in start..=i {
                if high[j] > highest { highest = high[j]; }
                if low[j] < lowest { lowest = low[j]; }
            }

            // Stochastic momentum: position of close relative to midpoint
            let midpoint = (highest + lowest) / 2.0;
            let half_range = (highest - lowest) / 2.0;

            if half_range > 1e-10 {
                // -100 to +100 scale based on deviation from midpoint
                k_line[i] = ((close[i] - midpoint) / half_range) * 100.0;
            }
        }

        // D-line: SMA of K-line
        for i in (self.k_period + self.d_period - 1)..n {
            let start = i.saturating_sub(self.d_period - 1);
            d_line[i] = k_line[start..=i].iter().sum::<f64>() / self.d_period as f64;
        }

        (k_line, d_line)
    }
}

impl TechnicalIndicator for StochasticMomentum {
    fn name(&self) -> &str {
        "Stochastic Momentum"
    }

    fn min_periods(&self) -> usize {
        self.k_period + self.d_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (k, d) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(k, d))
    }
}

/// Trend Intensity Oscillator - Measures trend strength as oscillator
#[derive(Debug, Clone)]
pub struct TrendIntensityOscillator {
    period: usize,
}

impl TrendIntensityOscillator {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate trend intensity oscillator (-100 to +100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate SMA
            let sma: f64 = close[start..=i].iter().sum::<f64>() / self.period as f64;

            // Count closes above and below SMA
            let mut above = 0;
            let mut below = 0;
            for j in start..=i {
                if close[j] > sma {
                    above += 1;
                } else if close[j] < sma {
                    below += 1;
                }
            }

            // Intensity: ratio of dominant direction
            let total = above + below;
            if total > 0 {
                result[i] = ((above as f64 - below as f64) / total as f64) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for TrendIntensityOscillator {
    fn name(&self) -> &str {
        "Trend Intensity Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Range Oscillator - Measures price position within recent range
#[derive(Debug, Clone)]
pub struct RangeOscillator {
    period: usize,
}

impl RangeOscillator {
    pub fn new(period: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate range oscillator (0 to 100)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            let mut highest = high[start];
            let mut lowest = low[start];
            for j in start..=i {
                if high[j] > highest { highest = high[j]; }
                if low[j] < lowest { lowest = low[j]; }
            }

            let range = highest - lowest;
            if range > 1e-10 {
                result[i] = ((close[i] - lowest) / range) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for RangeOscillator {
    fn name(&self) -> &str {
        "Range Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Momentum Divergence Oscillator - Detects price-momentum divergences
#[derive(Debug, Clone)]
pub struct MomentumDivergenceOscillator {
    price_period: usize,
    momentum_period: usize,
}

impl MomentumDivergenceOscillator {
    pub fn new(price_period: usize, momentum_period: usize) -> Result<Self> {
        if price_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "price_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { price_period, momentum_period })
    }

    /// Calculate divergence oscillator
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let max_period = self.price_period.max(self.momentum_period);

        for i in max_period..n {
            // Price trend (normalized)
            let price_change = if close[i - self.price_period] > 1e-10 {
                (close[i] / close[i - self.price_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Momentum (normalized)
            let momentum = if close[i - self.momentum_period] > 1e-10 {
                (close[i] / close[i - self.momentum_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Divergence: difference between normalized momentum and price change
            // Positive = momentum leading price up
            // Negative = momentum leading price down
            result[i] = momentum - price_change;
        }

        result
    }
}

impl TechnicalIndicator for MomentumDivergenceOscillator {
    fn name(&self) -> &str {
        "Momentum Divergence Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.price_period.max(self.momentum_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.3).sin() * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = (0..40).map(|i| 1000.0 + (i as f64 * 0.5).sin() * 500.0).collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_adaptive_rsi() {
        let data = make_test_data();
        let arsi = AdaptiveRSI::new(14, 10).unwrap();
        let result = arsi.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // RSI should be 0-100
        for i in 20..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_volume_weighted_rsi() {
        let data = make_test_data();
        let vwrsi = VolumeWeightedRSI::new(14).unwrap();
        let result = vwrsi.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
        for i in 20..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_stochastic_momentum() {
        let data = make_test_data();
        let sm = StochasticMomentum::new(14, 3).unwrap();
        let (k, d) = sm.calculate(&data.high, &data.low, &data.close);

        assert_eq!(k.len(), data.close.len());
        assert_eq!(d.len(), data.close.len());
        // K line should be -100 to +100
        for i in 20..k.len() {
            assert!(k[i] >= -100.0 && k[i] <= 100.0);
        }
    }

    #[test]
    fn test_trend_intensity_oscillator() {
        let data = make_test_data();
        let tio = TrendIntensityOscillator::new(14).unwrap();
        let result = tio.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        for i in 20..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_range_oscillator() {
        let data = make_test_data();
        let ro = RangeOscillator::new(14).unwrap();
        let result = ro.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        for i in 20..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_momentum_divergence_oscillator() {
        let data = make_test_data();
        let mdo = MomentumDivergenceOscillator::new(10, 14).unwrap();
        let result = mdo.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_validation() {
        assert!(AdaptiveRSI::new(2, 10).is_err());
        assert!(VolumeWeightedRSI::new(2).is_err());
        assert!(StochasticMomentum::new(1, 3).is_err());
        assert!(TrendIntensityOscillator::new(2).is_err());
        assert!(RangeOscillator::new(1).is_err());
        assert!(MomentumDivergenceOscillator::new(2, 10).is_err());
    }
}
