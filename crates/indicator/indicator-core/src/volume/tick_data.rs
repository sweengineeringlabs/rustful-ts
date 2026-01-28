//! Tick Data Indicators
//!
//! Indicators for analyzing tick-level market data.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Tick Volume - Count per bar approximation
/// Uses volume/average trade size as proxy
#[derive(Debug, Clone)]
pub struct TickVolume {
    avg_trade_size: f64,
}

impl TickVolume {
    pub fn new(avg_trade_size: f64) -> Result<Self> {
        if avg_trade_size <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "avg_trade_size".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { avg_trade_size })
    }

    pub fn calculate(&self, volume: &[f64]) -> Vec<f64> {
        volume.iter()
            .map(|v| v / self.avg_trade_size)
            .collect()
    }
}

impl TechnicalIndicator for TickVolume {
    fn name(&self) -> &str {
        "Tick Volume"
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.volume)))
    }
}

/// Tick Speed - Ticks per second approximation
/// Uses volume changes to estimate tick activity
#[derive(Debug, Clone)]
pub struct TickSpeed {
    period: usize,
    avg_trade_size: f64,
    bar_seconds: f64,
}

impl TickSpeed {
    pub fn new(period: usize, avg_trade_size: f64, bar_seconds: f64) -> Result<Self> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if avg_trade_size <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "avg_trade_size".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if bar_seconds <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "bar_seconds".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, avg_trade_size, bar_seconds })
    }

    pub fn calculate(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(self.period - 1);
            let avg_volume = volume[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
            let ticks = avg_volume / self.avg_trade_size;
            result[i] = ticks / self.bar_seconds;
        }
        result
    }
}

impl TechnicalIndicator for TickSpeed {
    fn name(&self) -> &str {
        "Tick Speed"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.volume)))
    }
}

/// Trade Intensity - Arrival rate using volume patterns
#[derive(Debug, Clone)]
pub struct TradeIntensity {
    period: usize,
    avg_trade_size: f64,
}

impl TradeIntensity {
    pub fn new(period: usize, avg_trade_size: f64) -> Result<Self> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if avg_trade_size <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "avg_trade_size".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, avg_trade_size })
    }

    pub fn calculate(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![0.0; n];
        let mut vol_sma = vec![0.0; n];

        // Calculate volume SMA
        for i in 0..n {
            let start = i.saturating_sub(self.period - 1);
            vol_sma[i] = volume[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
        }

        // Intensity = current volume / average volume normalized
        for i in 0..n {
            if vol_sma[i] > 0.0 {
                result[i] = (volume[i] / self.avg_trade_size) / (vol_sma[i] / self.avg_trade_size);
            }
        }
        result
    }
}

impl TechnicalIndicator for TradeIntensity {
    fn name(&self) -> &str {
        "Trade Intensity"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.volume)))
    }
}

/// Trade Size Distribution - Analyze trade lot sizes
/// Returns mean, std, skewness of trade sizes
#[derive(Debug, Clone)]
pub struct TradeSizeDistribution {
    period: usize,
}

impl TradeSizeDistribution {
    pub fn new(period: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period })
    }

    pub fn calculate(&self, volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = volume.len();
        let mut mean = vec![0.0; n];
        let mut std = vec![0.0; n];
        let mut skew = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let window: Vec<f64> = volume[start..=i].to_vec();

            // Mean
            let m = window.iter().sum::<f64>() / window.len() as f64;
            mean[i] = m;

            // Standard deviation
            let variance = window.iter()
                .map(|x| (x - m).powi(2))
                .sum::<f64>() / window.len() as f64;
            let s = variance.sqrt();
            std[i] = s;

            // Skewness
            if s > 0.0 {
                let m3 = window.iter()
                    .map(|x| ((x - m) / s).powi(3))
                    .sum::<f64>() / window.len() as f64;
                skew[i] = m3;
            }
        }

        (mean, std, skew)
    }
}

impl TechnicalIndicator for TradeSizeDistribution {
    fn name(&self) -> &str {
        "Trade Size Distribution"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (mean, std, skew) = self.calculate(&data.volume);
        Ok(IndicatorOutput::triple(mean, std, skew))
    }
}

/// Dollar Volume - Price x Volume
#[derive(Debug, Clone)]
pub struct DollarVolume {
    period: usize,
}

impl DollarVolume {
    pub fn new(period: usize) -> Result<Self> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period })
    }

    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in 0..n {
            result[i] = close[i] * volume[i];
        }

        // Apply SMA if period > 1
        if self.period > 1 {
            let mut smoothed = vec![0.0; n];
            for i in 0..n {
                let start = i.saturating_sub(self.period - 1);
                smoothed[i] = result[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
            }
            return smoothed;
        }

        result
    }
}

impl TechnicalIndicator for DollarVolume {
    fn name(&self) -> &str {
        "Dollar Volume"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// VWAP Deviation - Distance from VWAP
#[derive(Debug, Clone)]
pub struct VWAPDeviation {
    period: usize,
}

impl VWAPDeviation {
    pub fn new(period: usize) -> Result<Self> {
        if period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period })
    }

    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(self.period - 1);

            // Calculate VWAP for period
            let mut sum_pv = 0.0;
            let mut sum_v = 0.0;

            for j in start..=i {
                let typical = (high[j] + low[j] + close[j]) / 3.0;
                sum_pv += typical * volume[j];
                sum_v += volume[j];
            }

            let vwap = if sum_v > 0.0 { sum_pv / sum_v } else { close[i] };

            // Deviation as percentage
            if vwap > 0.0 {
                result[i] = (close[i] - vwap) / vwap * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for VWAPDeviation {
    fn name(&self) -> &str {
        "VWAP Deviation"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(
            &data.high, &data.low, &data.close, &data.volume
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![105.0, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 109.5, 111.0,
                       110.5, 112.0, 111.5, 113.0, 112.5, 114.0, 113.5, 115.0, 114.5, 116.0];
        let low = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 104.5, 106.0,
                      105.5, 107.0, 106.5, 108.0, 107.5, 109.0, 108.5, 110.0, 109.5, 111.0];
        let close = vec![102.0, 104.0, 105.0, 104.0, 106.0, 107.0, 106.0, 108.0, 107.0, 109.0,
                        108.0, 110.0, 109.0, 111.0, 110.0, 112.0, 111.0, 113.0, 112.0, 114.0];
        let volume = vec![1000.0, 1200.0, 1100.0, 1300.0, 1500.0, 1400.0, 1600.0, 1800.0, 1700.0, 1900.0,
                         2000.0, 1800.0, 2100.0, 2200.0, 2000.0, 2300.0, 2100.0, 2400.0, 2200.0, 2500.0];
        (high, low, close, volume)
    }

    #[test]
    fn test_tick_volume() {
        let (_, _, _, volume) = make_test_data();
        let tv = TickVolume::new(100.0).unwrap();
        let result = tv.calculate(&volume);

        assert_eq!(result.len(), volume.len());
        assert!((result[0] - 10.0).abs() < 0.01); // 1000 / 100 = 10
        assert!((result[4] - 15.0).abs() < 0.01); // 1500 / 100 = 15
    }

    #[test]
    fn test_tick_speed() {
        let (_, _, _, volume) = make_test_data();
        let ts = TickSpeed::new(5, 100.0, 300.0).unwrap(); // 5-min bars
        let result = ts.calculate(&volume);

        assert_eq!(result.len(), volume.len());
        assert!(result[4] > 0.0);
    }

    #[test]
    fn test_trade_intensity() {
        let (_, _, _, volume) = make_test_data();
        let ti = TradeIntensity::new(5, 100.0).unwrap();
        let result = ti.calculate(&volume);

        assert_eq!(result.len(), volume.len());
        // Intensity should be around 1.0 on average
        let avg: f64 = result[5..].iter().sum::<f64>() / (result.len() - 5) as f64;
        assert!(avg > 0.5 && avg < 2.0);
    }

    #[test]
    fn test_trade_size_distribution() {
        let (_, _, _, volume) = make_test_data();
        let tsd = TradeSizeDistribution::new(5).unwrap();
        let (mean, std, skew) = tsd.calculate(&volume);

        assert_eq!(mean.len(), volume.len());
        assert_eq!(std.len(), volume.len());
        assert_eq!(skew.len(), volume.len());
        assert!(mean[10] > 0.0);
        assert!(std[10] > 0.0);
    }

    #[test]
    fn test_dollar_volume() {
        let (_, _, close, volume) = make_test_data();
        let dv = DollarVolume::new(1).unwrap();
        let result = dv.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        assert!((result[0] - 102000.0).abs() < 0.01); // 102 * 1000
    }

    #[test]
    fn test_vwap_deviation() {
        let (high, low, close, volume) = make_test_data();
        let vd = VWAPDeviation::new(10).unwrap();
        let result = vd.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Deviation should be small percentage
        assert!(result[15].abs() < 5.0);
    }
}
