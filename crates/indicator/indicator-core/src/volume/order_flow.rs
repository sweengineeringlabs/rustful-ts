//! Order Flow Indicators
//!
//! Indicators for analyzing order flow and trade dynamics.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Delta Volume - Buy vs Sell volume approximation
#[derive(Debug, Clone)]
pub struct DeltaVolume {
    period: usize,
}

impl DeltaVolume {
    pub fn new(period: usize) -> Result<Self> {
        if period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate delta volume using close position within bar
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in 0..n {
            let range = high[i] - low[i];
            if range > 0.0 {
                // Close position in bar: 0 = bottom, 1 = top
                let position = (close[i] - low[i]) / range;
                // Delta: positive if close near top, negative if near bottom
                result[i] = volume[i] * (2.0 * position - 1.0);
            }
        }

        // Apply smoothing if period > 1
        if self.period > 1 {
            let mut smoothed = vec![0.0; n];
            for i in 0..n {
                let start = i.saturating_sub(self.period - 1);
                smoothed[i] = result[start..=i].iter().sum::<f64>();
            }
            return smoothed;
        }

        result
    }
}

impl TechnicalIndicator for DeltaVolume {
    fn name(&self) -> &str {
        "Delta Volume"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Cumulative Delta - Running sum of delta
#[derive(Debug, Clone)]
pub struct CumulativeDelta;

impl CumulativeDelta {
    pub fn new() -> Self {
        Self
    }

    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut result = vec![0.0; n];
        let mut cumsum = 0.0;

        for i in 0..n {
            let range = high[i] - low[i];
            if range > 0.0 {
                let position = (close[i] - low[i]) / range;
                let delta = volume[i] * (2.0 * position - 1.0);
                cumsum += delta;
            }
            result[i] = cumsum;
        }
        result
    }
}

impl TechnicalIndicator for CumulativeDelta {
    fn name(&self) -> &str {
        "Cumulative Delta"
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

impl Default for CumulativeDelta {
    fn default() -> Self {
        Self::new()
    }
}

/// Imbalance Ratio - Buy vs Sell imbalance
#[derive(Debug, Clone)]
pub struct ImbalanceRatio {
    period: usize,
}

impl ImbalanceRatio {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate imbalance ratio: (buy - sell) / (buy + sell)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let mut buy_vol = 0.0;
            let mut sell_vol = 0.0;

            for j in start..=i {
                let range = high[j] - low[j];
                if range > 0.0 {
                    let position = (close[j] - low[j]) / range;
                    buy_vol += volume[j] * position;
                    sell_vol += volume[j] * (1.0 - position);
                }
            }

            let total = buy_vol + sell_vol;
            if total > 0.0 {
                result[i] = (buy_vol - sell_vol) / total * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for ImbalanceRatio {
    fn name(&self) -> &str {
        "Imbalance Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Absorption - Large volume near highs/lows
#[derive(Debug, Clone)]
pub struct Absorption {
    period: usize,
    threshold: f64,
}

impl Absorption {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
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

    /// Detect absorption: 1 = bullish absorption, -1 = bearish, 0 = none
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let avg_vol = volume[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;

            // High volume bar
            if volume[i] > avg_vol * self.threshold {
                let range = high[i] - low[i];
                if range > 0.0 {
                    let position = (close[i] - low[i]) / range;

                    // Bullish absorption: high volume at lows with close near high
                    if position > 0.7 {
                        result[i] = 1.0;
                    }
                    // Bearish absorption: high volume at highs with close near low
                    else if position < 0.3 {
                        result[i] = -1.0;
                    }
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for Absorption {
    fn name(&self) -> &str {
        "Absorption"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Volume Profile POC - Point of Control approximation
#[derive(Debug, Clone)]
pub struct VolumePOC {
    period: usize,
    bins: usize,
}

impl VolumePOC {
    pub fn new(period: usize, bins: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if bins < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "bins".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period, bins })
    }

    /// Calculate POC price level
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);

            // Find price range
            let min_price = low[start..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_price = high[start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let range = max_price - min_price;

            if range <= 0.0 {
                result[i] = close[i];
                continue;
            }

            let bin_size = range / self.bins as f64;
            let mut bin_volume = vec![0.0; self.bins];

            // Distribute volume to bins
            for j in start..=i {
                let typical = (high[j] + low[j] + close[j]) / 3.0;
                let bin_idx = ((typical - min_price) / bin_size).floor() as usize;
                let bin_idx = bin_idx.min(self.bins - 1);
                bin_volume[bin_idx] += volume[j];
            }

            // Find POC (highest volume bin)
            let poc_bin = bin_volume.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            // Convert bin to price level
            result[i] = min_price + (poc_bin as f64 + 0.5) * bin_size;
        }
        result
    }
}

impl TechnicalIndicator for VolumePOC {
    fn name(&self) -> &str {
        "Volume POC"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Footprint Imbalance - Bid/Ask imbalance detection
#[derive(Debug, Clone)]
pub struct FootprintImbalance {
    period: usize,
    imbalance_threshold: f64,
}

impl FootprintImbalance {
    pub fn new(period: usize, imbalance_threshold: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if imbalance_threshold <= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "imbalance_threshold".to_string(),
                reason: "must be greater than 1.0".to_string(),
            });
        }
        Ok(Self { period, imbalance_threshold })
    }

    /// Detect imbalances: count of stacked imbalances
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let mut imbalance_count = 0i32;

            for j in start..=i {
                let range = high[j] - low[j];
                if range > 0.0 {
                    let position = (close[j] - low[j]) / range;
                    let buy_est = position;
                    let sell_est = 1.0 - position;

                    // Check for imbalance
                    if buy_est > sell_est * self.imbalance_threshold {
                        imbalance_count += 1;
                    } else if sell_est > buy_est * self.imbalance_threshold {
                        imbalance_count -= 1;
                    }
                }
            }

            result[i] = imbalance_count as f64;
        }
        result
    }
}

impl TechnicalIndicator for FootprintImbalance {
    fn name(&self) -> &str {
        "Footprint Imbalance"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
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
        let close = vec![104.0, 105.0, 106.0, 105.0, 107.0, 108.0, 107.0, 109.0, 108.0, 110.0,
                        109.0, 111.0, 110.0, 112.0, 111.0, 113.0, 112.0, 114.0, 113.0, 115.0];
        let volume = vec![1000.0, 1200.0, 1100.0, 1300.0, 1500.0, 1400.0, 1600.0, 1800.0, 1700.0, 1900.0,
                         2000.0, 1800.0, 2100.0, 2200.0, 2000.0, 2300.0, 2100.0, 2400.0, 2200.0, 2500.0];
        (high, low, close, volume)
    }

    #[test]
    fn test_delta_volume() {
        let (high, low, close, volume) = make_test_data();
        let dv = DeltaVolume::new(1).unwrap();
        let result = dv.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Closes near highs should have positive delta
        assert!(result[0] > 0.0);
    }

    #[test]
    fn test_cumulative_delta() {
        let (high, low, close, volume) = make_test_data();
        let cd = CumulativeDelta::new();
        let result = cd.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Cumulative should be increasing in uptrend
        assert!(result[10] > result[5]);
    }

    #[test]
    fn test_imbalance_ratio() {
        let (high, low, close, volume) = make_test_data();
        let ir = ImbalanceRatio::new(5).unwrap();
        let result = ir.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Should be between -100 and 100
        assert!(result[10] >= -100.0 && result[10] <= 100.0);
    }

    #[test]
    fn test_absorption() {
        let (high, low, close, volume) = make_test_data();
        let abs = Absorption::new(5, 1.5).unwrap();
        let result = abs.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Should be -1, 0, or 1
        for &v in result.iter().skip(5) {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_volume_poc() {
        let (high, low, close, volume) = make_test_data();
        let poc = VolumePOC::new(10, 10).unwrap();
        let result = poc.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // POC should be within the price range
        assert!(result[15] > 100.0 && result[15] < 120.0);
    }

    #[test]
    fn test_footprint_imbalance() {
        let (high, low, close, volume) = make_test_data();
        let fi = FootprintImbalance::new(5, 2.0).unwrap();
        let result = fi.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Check values exist
        assert!(result[10].abs() < 20.0);
    }
}
