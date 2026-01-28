//! Extended Volume Indicators
//!
//! Additional volume analysis indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Volume Momentum - Rate of change in volume
#[derive(Debug, Clone)]
pub struct VolumeMomentum {
    period: usize,
}

impl VolumeMomentum {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume momentum
    pub fn calculate(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            if volume[i - self.period] > 1e-10 {
                result[i] = (volume[i] / volume[i - self.period] - 1.0) * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for VolumeMomentum {
    fn name(&self) -> &str {
        "Volume Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.volume)))
    }
}

/// Relative Volume - Volume compared to average
#[derive(Debug, Clone)]
pub struct RelativeVolume {
    period: usize,
}

impl RelativeVolume {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate relative volume
    pub fn calculate(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / self.period as f64;

            if avg_volume > 1e-10 {
                result[i] = volume[i] / avg_volume;
            }
        }
        result
    }
}

impl TechnicalIndicator for RelativeVolume {
    fn name(&self) -> &str {
        "Relative Volume"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.volume)))
    }
}

/// Volume Weighted Price Momentum - VWAP momentum
#[derive(Debug, Clone)]
pub struct VolumeWeightedPriceMomentum {
    period: usize,
}

impl VolumeWeightedPriceMomentum {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate VWAP momentum
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        // Calculate cumulative VWAP
        let mut cum_vwap = vec![0.0; n];
        let mut cum_pv = 0.0;
        let mut cum_v = 0.0;

        for i in 0..n {
            let typical = (high[i] + low[i] + close[i]) / 3.0;
            cum_pv += typical * volume[i];
            cum_v += volume[i];

            if cum_v > 1e-10 {
                cum_vwap[i] = cum_pv / cum_v;
            }
        }

        // VWAP momentum
        for i in self.period..n {
            if cum_vwap[i - self.period] > 1e-10 {
                result[i] = (cum_vwap[i] / cum_vwap[i - self.period] - 1.0) * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for VolumeWeightedPriceMomentum {
    fn name(&self) -> &str {
        "Volume Weighted Price Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Volume Price Trend Extended - Enhanced VPT
#[derive(Debug, Clone)]
pub struct VPTExtended {
    signal_period: usize,
}

impl VPTExtended {
    pub fn new(signal_period: usize) -> Result<Self> {
        if signal_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "signal_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { signal_period })
    }

    /// Calculate VPT and signal line, returns (vpt, signal)
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len().min(volume.len());
        let mut vpt = vec![0.0; n];
        let mut signal = vec![0.0; n];

        // Calculate VPT
        for i in 1..n {
            let price_change = (close[i] - close[i - 1]) / close[i - 1];
            vpt[i] = vpt[i - 1] + volume[i] * price_change;
        }

        // Calculate EMA signal
        let alpha = 2.0 / (self.signal_period as f64 + 1.0);
        for i in 0..n {
            if i == 0 {
                signal[i] = vpt[i];
            } else {
                signal[i] = alpha * vpt[i] + (1.0 - alpha) * signal[i - 1];
            }
        }

        (vpt, signal)
    }
}

impl TechnicalIndicator for VPTExtended {
    fn name(&self) -> &str {
        "VPT Extended"
    }

    fn min_periods(&self) -> usize {
        self.signal_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (vpt, signal) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(vpt, signal))
    }
}

/// Volume Buying Pressure - Measures buying vs selling pressure
#[derive(Debug, Clone)]
pub struct VolumeBuyingPressure {
    period: usize,
}

impl VolumeBuyingPressure {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume buying pressure
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        // Calculate signed volume
        let mut signed_vol = vec![0.0; n];
        for i in 1..n {
            signed_vol[i] = if close[i] > close[i - 1] {
                volume[i]
            } else if close[i] < close[i - 1] {
                -volume[i]
            } else {
                0.0
            };
        }

        // Calculate EMA of signed volume
        let alpha = 2.0 / (self.period as f64 + 1.0);
        let mut ema_sv = vec![0.0; n];
        let mut ema_v = vec![0.0; n];

        for i in 0..n {
            if i == 0 {
                ema_sv[i] = signed_vol[i];
                ema_v[i] = volume[i];
            } else {
                ema_sv[i] = alpha * signed_vol[i] + (1.0 - alpha) * ema_sv[i - 1];
                ema_v[i] = alpha * volume[i] + (1.0 - alpha) * ema_v[i - 1];
            }

            if ema_v[i] > 1e-10 {
                result[i] = ema_sv[i] / ema_v[i] * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for VolumeBuyingPressure {
    fn name(&self) -> &str {
        "Volume Buying Pressure"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Price Volume Rank - Ranks price and volume changes
#[derive(Debug, Clone)]
pub struct PriceVolumeRank {
    period: usize,
}

impl PriceVolumeRank {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate PVR score
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Price change rank
            let price_change = close[i] / close[i - 1] - 1.0;
            let mut price_changes: Vec<f64> = (start + 1..=i)
                .map(|j| close[j] / close[j - 1] - 1.0)
                .collect();
            price_changes.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let price_rank = price_changes.iter()
                .position(|&x| x >= price_change)
                .unwrap_or(price_changes.len()) as f64 / price_changes.len() as f64;

            // Volume change rank
            let vol_change = volume[i] / volume[i - 1] - 1.0;
            let mut vol_changes: Vec<f64> = (start + 1..=i)
                .map(|j| volume[j] / volume[j - 1] - 1.0)
                .collect();
            vol_changes.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let vol_rank = vol_changes.iter()
                .position(|&x| x >= vol_change)
                .unwrap_or(vol_changes.len()) as f64 / vol_changes.len() as f64;

            // Combined score
            result[i] = (price_rank + vol_rank) / 2.0 * 100.0;
        }
        result
    }
}

impl TechnicalIndicator for PriceVolumeRank {
    fn name(&self) -> &str {
        "Price Volume Rank"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                       112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
                       122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0];
        let low = vec![98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
                      108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0,
                      118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0];
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                        110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
                        120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0];
        let volume = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0,
                         2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0,
                         3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0];
        (high, low, close, volume)
    }

    #[test]
    fn test_volume_momentum() {
        let (_, _, _, volume) = make_test_data();
        let vm = VolumeMomentum::new(5).unwrap();
        let result = vm.calculate(&volume);

        assert_eq!(result.len(), volume.len());
        assert!(result[10] > 0.0); // Increasing volume
    }

    #[test]
    fn test_relative_volume() {
        let (_, _, _, volume) = make_test_data();
        let rv = RelativeVolume::new(10).unwrap();
        let result = rv.calculate(&volume);

        assert_eq!(result.len(), volume.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_volume_weighted_price_momentum() {
        let (high, low, close, volume) = make_test_data();
        let vwpm = VolumeWeightedPriceMomentum::new(10).unwrap();
        let result = vwpm.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_vpt_extended() {
        let (_, _, close, volume) = make_test_data();
        let vpt = VPTExtended::new(10).unwrap();
        let (vpt_line, signal) = vpt.calculate(&close, &volume);

        assert_eq!(vpt_line.len(), close.len());
        assert_eq!(signal.len(), close.len());
    }

    #[test]
    fn test_volume_buying_pressure() {
        let (_, _, close, volume) = make_test_data();
        let vbp = VolumeBuyingPressure::new(10).unwrap();
        let result = vbp.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Should be between -100 and 100
        assert!(result.iter().all(|&v| v >= -100.0 && v <= 100.0));
    }

    #[test]
    fn test_price_volume_rank() {
        let (_, _, close, volume) = make_test_data();
        let pvr = PriceVolumeRank::new(10).unwrap();
        let result = pvr.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Rank should be between 0 and 100
        assert!(result[15] >= 0.0 && result[15] <= 100.0);
    }
}
