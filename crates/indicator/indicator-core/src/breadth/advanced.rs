//! Advanced Market Breadth Indicators
//!
//! Additional breadth indicators for comprehensive market analysis.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Market Momentum Breadth - Breadth-based momentum indicator
#[derive(Debug, Clone)]
pub struct MarketMomentumBreadth {
    short_period: usize,
    long_period: usize,
}

impl MarketMomentumBreadth {
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

    /// Calculate momentum breadth using price proxy
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term advancing days
            let short_start = i.saturating_sub(self.short_period);
            let mut short_advances = 0;
            for j in (short_start + 1)..=i {
                if close[j] > close[j - 1] {
                    short_advances += 1;
                }
            }
            let short_ratio = short_advances as f64 / self.short_period as f64;

            // Long-term advancing days
            let long_start = i.saturating_sub(self.long_period);
            let mut long_advances = 0;
            for j in (long_start + 1)..=i {
                if close[j] > close[j - 1] {
                    long_advances += 1;
                }
            }
            let long_ratio = long_advances as f64 / self.long_period as f64;

            // Momentum breadth: short vs long ratio
            result[i] = (short_ratio - long_ratio) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for MarketMomentumBreadth {
    fn name(&self) -> &str {
        "Market Momentum Breadth"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Breadth Oscillator - Smoothed breadth ratio oscillator
#[derive(Debug, Clone)]
pub struct BreadthOscillator {
    period: usize,
    smooth_period: usize,
}

impl BreadthOscillator {
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smooth_period })
    }

    /// Calculate breadth oscillator (-100 to +100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut raw = vec![0.0; n];
        let mut result = vec![0.0; n];

        // Calculate raw breadth ratio
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
                raw[i] = ((advances - declines) as f64 / total as f64) * 100.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in self.period..n {
            if i == self.period {
                result[i] = raw[i];
            } else {
                result[i] = alpha * raw[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for BreadthOscillator {
    fn name(&self) -> &str {
        "Breadth Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Cumulative Breadth Index - Running sum of advance/decline differences
#[derive(Debug, Clone)]
pub struct CumulativeBreadthIndex {
    period: usize,
}

impl CumulativeBreadthIndex {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate cumulative breadth index
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut cumulative = 0.0;

        for i in 1..n {
            if close[i] > close[i - 1] {
                cumulative += 1.0;
            } else if close[i] < close[i - 1] {
                cumulative -= 1.0;
            }

            if i >= self.period {
                result[i] = cumulative;
            }
        }

        result
    }
}

impl TechnicalIndicator for CumulativeBreadthIndex {
    fn name(&self) -> &str {
        "Cumulative Breadth Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volume Breadth Ratio - Volume-weighted breadth
#[derive(Debug, Clone)]
pub struct VolumeBreadthRatio {
    period: usize,
}

impl VolumeBreadthRatio {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume-weighted breadth ratio
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut up_volume = 0.0;
            let mut down_volume = 0.0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_volume += volume[j];
                } else if close[j] < close[j - 1] {
                    down_volume += volume[j];
                }
            }

            let total_volume = up_volume + down_volume;
            if total_volume > 1e-10 {
                result[i] = ((up_volume - down_volume) / total_volume) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for VolumeBreadthRatio {
    fn name(&self) -> &str {
        "Volume Breadth Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Breadth Divergence - Detects price/breadth divergence
#[derive(Debug, Clone)]
pub struct BreadthDivergence {
    period: usize,
}

impl BreadthDivergence {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate breadth divergence score
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Price trend
            let price_change = if close[start] > 1e-10 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Breadth trend (cumulative AD)
            let mut breadth_change = 0.0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    breadth_change += 1.0;
                } else if close[j] < close[j - 1] {
                    breadth_change -= 1.0;
                }
            }

            // Normalize breadth
            let normalized_breadth = breadth_change / self.period as f64 * 50.0;

            // Divergence: difference between price direction and breadth direction
            // Positive = breadth leads price (bullish)
            // Negative = breadth lags price (bearish)
            result[i] = normalized_breadth - price_change;
        }

        result
    }
}

impl TechnicalIndicator for BreadthDivergence {
    fn name(&self) -> &str {
        "Breadth Divergence"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Participation Rate - Measures market participation
#[derive(Debug, Clone)]
pub struct ParticipationRate {
    period: usize,
    threshold: f64,
}

impl ParticipationRate {
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

    /// Calculate participation rate (% of bars with significant movement)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut participating = 0;

            for j in (start + 1)..=i {
                let pct_change = if close[j - 1] > 1e-10 {
                    ((close[j] / close[j - 1]) - 1.0).abs() * 100.0
                } else {
                    0.0
                };

                if pct_change >= self.threshold {
                    participating += 1;
                }
            }

            result[i] = (participating as f64 / self.period as f64) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for ParticipationRate {
    fn name(&self) -> &str {
        "Participation Rate"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.3 + (i as f64 * 0.4).sin() * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 0.3).sin() * 500.0).collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_market_momentum_breadth() {
        let data = make_test_data();
        let mmb = MarketMomentumBreadth::new(7, 21).unwrap();
        let result = mmb.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_breadth_oscillator() {
        let data = make_test_data();
        let bo = BreadthOscillator::new(10, 5).unwrap();
        let result = bo.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Oscillator should be -100 to +100
        for i in 20..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_cumulative_breadth_index() {
        let data = make_test_data();
        let cbi = CumulativeBreadthIndex::new(10).unwrap();
        let result = cbi.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_volume_breadth_ratio() {
        let data = make_test_data();
        let vbr = VolumeBreadthRatio::new(10).unwrap();
        let result = vbr.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
        // Ratio should be -100 to +100
        for i in 15..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_breadth_divergence() {
        let data = make_test_data();
        let bd = BreadthDivergence::new(14).unwrap();
        let result = bd.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_participation_rate() {
        let data = make_test_data();
        let pr = ParticipationRate::new(10, 0.5).unwrap();
        let result = pr.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Rate should be 0-100
        for i in 15..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_validation() {
        assert!(MarketMomentumBreadth::new(2, 21).is_err());
        assert!(MarketMomentumBreadth::new(10, 5).is_err()); // long <= short
        assert!(BreadthOscillator::new(2, 5).is_err());
        assert!(CumulativeBreadthIndex::new(2).is_err());
        assert!(VolumeBreadthRatio::new(2).is_err());
        assert!(BreadthDivergence::new(5).is_err());
        assert!(ParticipationRate::new(2, 0.5).is_err());
    }
}
