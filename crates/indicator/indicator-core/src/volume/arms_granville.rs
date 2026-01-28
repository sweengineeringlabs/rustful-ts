//! Richard Arms and Joe Granville Extended Indicators
//!
//! Advanced volume analysis indicators from these legendary technicians.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use crate::{SMA, EMA};

/// Ease of Movement Moving Average - Smoothed EMV.
///
/// EMV MA = SMA(EMV, period)
///
/// Smooths the Ease of Movement indicator for clearer signals.
#[derive(Debug, Clone)]
pub struct EaseOfMovementMA {
    period: usize,
    ma_period: usize,
}

impl EaseOfMovementMA {
    pub fn new(period: usize, ma_period: usize) -> Self {
        Self { period, ma_period }
    }

    /// Calculate EMV.
    fn calculate_emv(high: &[f64], low: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n < 2 {
            return vec![f64::NAN; n];
        }

        let mut emv = vec![f64::NAN];

        for i in 1..n {
            let distance_moved = ((high[i] + low[i]) / 2.0) - ((high[i-1] + low[i-1]) / 2.0);
            let box_ratio = if (high[i] - low[i]).abs() > 1e-10 {
                (volume[i] / 1_000_000.0) / (high[i] - low[i])
            } else {
                0.0
            };

            if box_ratio.abs() > 1e-10 {
                emv.push(distance_moved / box_ratio);
            } else {
                emv.push(0.0);
            }
        }

        emv
    }

    /// Calculate smoothed EMV.
    pub fn calculate(&self, high: &[f64], low: &[f64], volume: &[f64]) -> Vec<f64> {
        let emv = Self::calculate_emv(high, low, volume);
        let sma = SMA::new(self.ma_period);
        sma.calculate(&emv)
    }
}

impl TechnicalIndicator for EaseOfMovementMA {
    fn name(&self) -> &str {
        "Ease of Movement MA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.ma_period + 1;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.volume);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.ma_period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for EaseOfMovementMA {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let emv_ma = self.calculate(&data.high, &data.low, &data.volume);

        if emv_ma.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = emv_ma.len();
        let current = emv_ma[n - 1];
        let prev = emv_ma[n - 2];

        if current.is_nan() || prev.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: EMV MA crosses above 0
        if prev <= 0.0 && current > 0.0 {
            Ok(IndicatorSignal::Bullish)
        }
        // Bearish: EMV MA crosses below 0
        else if prev >= 0.0 && current < 0.0 {
            Ok(IndicatorSignal::Bearish)
        }
        else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let emv_ma = self.calculate(&data.high, &data.low, &data.volume);
        let n = emv_ma.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            if emv_ma[i].is_nan() || emv_ma[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            if emv_ma[i - 1] <= 0.0 && emv_ma[i] > 0.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if emv_ma[i - 1] >= 0.0 && emv_ma[i] < 0.0 {
                signals.push(IndicatorSignal::Bearish);
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

/// Volume-Adjusted Moving Average (VAMA) - Arms' volume-weighted MA.
///
/// Weights each price by volume relative to average volume.
#[derive(Debug, Clone)]
pub struct VAMA {
    period: usize,
}

impl VAMA {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate VAMA.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let prices = &close[start..=i];
            let volumes = &volume[start..=i];

            // Calculate average volume
            let avg_vol: f64 = volumes.iter().sum::<f64>() / self.period as f64;

            if avg_vol.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            // Calculate volume-adjusted weights
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for j in 0..self.period {
                let weight = volumes[j] / avg_vol;
                weighted_sum += prices[j] * weight;
                weight_sum += weight;
            }

            if weight_sum.abs() > 1e-10 {
                result.push(weighted_sum / weight_sum);
            } else {
                result.push(f64::NAN);
            }
        }

        result
    }
}

impl TechnicalIndicator for VAMA {
    fn name(&self) -> &str {
        "VAMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Equivolume Width - Measures relative volume for Equivolume charts.
///
/// Width = Volume / Average Volume
///
/// Used to determine the width of Equivolume bars.
#[derive(Debug, Clone)]
pub struct EquivolumeWidth {
    period: usize,
}

impl EquivolumeWidth {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate relative volume width.
    pub fn calculate(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &volume[start..=i];

            let avg_vol: f64 = window.iter().sum::<f64>() / self.period as f64;

            if avg_vol.abs() > 1e-10 {
                result.push(volume[i] / avg_vol);
            } else {
                result.push(f64::NAN);
            }
        }

        result
    }
}

impl TechnicalIndicator for EquivolumeWidth {
    fn name(&self) -> &str {
        "Equivolume Width"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.volume);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// OBV Trend - On Balance Volume with MA crossover signals.
///
/// Generates signals when OBV crosses its moving average.
#[derive(Debug, Clone)]
pub struct OBVTrend {
    ma_period: usize,
}

impl OBVTrend {
    pub fn new(ma_period: usize) -> Self {
        Self { ma_period }
    }

    /// Calculate OBV.
    fn calculate_obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n == 0 {
            return vec![];
        }

        let mut obv = vec![0.0];

        for i in 1..n {
            let prev_obv = obv[i - 1];
            if close[i] > close[i - 1] {
                obv.push(prev_obv + volume[i]);
            } else if close[i] < close[i - 1] {
                obv.push(prev_obv - volume[i]);
            } else {
                obv.push(prev_obv);
            }
        }

        obv
    }

    /// Calculate OBV and its MA.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let obv = Self::calculate_obv(close, volume);
        let sma = SMA::new(self.ma_period);
        let obv_ma = sma.calculate(&obv);
        (obv, obv_ma)
    }
}

impl TechnicalIndicator for OBVTrend {
    fn name(&self) -> &str {
        "OBV Trend"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.ma_period {
            return Err(IndicatorError::InsufficientData {
                required: self.ma_period,
                got: data.close.len(),
            });
        }

        let (obv, obv_ma) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(obv, obv_ma))
    }

    fn min_periods(&self) -> usize {
        self.ma_period
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for OBVTrend {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (obv, obv_ma) = self.calculate(&data.close, &data.volume);

        if obv.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = obv.len();
        let curr_obv = obv[n - 1];
        let curr_ma = obv_ma[n - 1];
        let prev_obv = obv[n - 2];
        let prev_ma = obv_ma[n - 2];

        if curr_ma.is_nan() || prev_ma.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: OBV crosses above MA
        if prev_obv <= prev_ma && curr_obv > curr_ma {
            Ok(IndicatorSignal::Bullish)
        }
        // Bearish: OBV crosses below MA
        else if prev_obv >= prev_ma && curr_obv < curr_ma {
            Ok(IndicatorSignal::Bearish)
        }
        else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (obv, obv_ma) = self.calculate(&data.close, &data.volume);
        let n = obv.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            if obv_ma[i].is_nan() || obv_ma[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            if obv[i - 1] <= obv_ma[i - 1] && obv[i] > obv_ma[i] {
                signals.push(IndicatorSignal::Bullish);
            } else if obv[i - 1] >= obv_ma[i - 1] && obv[i] < obv_ma[i] {
                signals.push(IndicatorSignal::Bearish);
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

/// OBV Divergence - Detects divergence between price and OBV.
///
/// Bullish divergence: Price makes lower low, OBV makes higher low
/// Bearish divergence: Price makes higher high, OBV makes lower high
#[derive(Debug, Clone)]
pub struct OBVDivergence {
    lookback: usize,
}

/// Divergence type for OBV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OBVDivergenceType {
    None,
    Bullish,
    Bearish,
}

impl OBVDivergence {
    pub fn new(lookback: usize) -> Self {
        Self { lookback }
    }

    /// Calculate OBV divergence.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<OBVDivergenceType> {
        let n = close.len();
        if n < self.lookback + 1 {
            return vec![OBVDivergenceType::None; n];
        }

        let obv = OBVTrend::calculate_obv(close, volume);
        let mut divergence = vec![OBVDivergenceType::None; n];

        for i in self.lookback..n {
            let start = i - self.lookback;

            // Find recent swing points
            let price_min_idx = (start..=i).min_by(|&a, &b|
                close[a].partial_cmp(&close[b]).unwrap_or(std::cmp::Ordering::Equal)
            ).unwrap();
            let price_max_idx = (start..=i).max_by(|&a, &b|
                close[a].partial_cmp(&close[b]).unwrap_or(std::cmp::Ordering::Equal)
            ).unwrap();

            let obv_min_idx = (start..=i).min_by(|&a, &b|
                obv[a].partial_cmp(&obv[b]).unwrap_or(std::cmp::Ordering::Equal)
            ).unwrap();
            let obv_max_idx = (start..=i).max_by(|&a, &b|
                obv[a].partial_cmp(&obv[b]).unwrap_or(std::cmp::Ordering::Equal)
            ).unwrap();

            // Bullish: Price at recent low but OBV not at low
            if price_min_idx == i && obv_min_idx != i && obv[i] > obv[obv_min_idx] * 1.01 {
                divergence[i] = OBVDivergenceType::Bullish;
            }
            // Bearish: Price at recent high but OBV not at high
            else if price_max_idx == i && obv_max_idx != i && obv[i] < obv[obv_max_idx] * 0.99 {
                divergence[i] = OBVDivergenceType::Bearish;
            }
        }

        divergence
    }
}

impl TechnicalIndicator for OBVDivergence {
    fn name(&self) -> &str {
        "OBV Divergence"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.lookback + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.lookback + 1,
                got: data.close.len(),
            });
        }

        let divergence = self.calculate(&data.close, &data.volume);
        let values: Vec<f64> = divergence.iter().map(|d| match d {
            OBVDivergenceType::Bullish => 1.0,
            OBVDivergenceType::Bearish => -1.0,
            OBVDivergenceType::None => 0.0,
        }).collect();

        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for OBVDivergence {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let divergence = self.calculate(&data.close, &data.volume);

        if divergence.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        match divergence.last().unwrap() {
            OBVDivergenceType::Bullish => Ok(IndicatorSignal::Bullish),
            OBVDivergenceType::Bearish => Ok(IndicatorSignal::Bearish),
            OBVDivergenceType::None => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let divergence = self.calculate(&data.close, &data.volume);

        Ok(divergence.iter().map(|d| match d {
            OBVDivergenceType::Bullish => IndicatorSignal::Bullish,
            OBVDivergenceType::Bearish => IndicatorSignal::Bearish,
            OBVDivergenceType::None => IndicatorSignal::Neutral,
        }).collect())
    }
}

/// Volume Climax Indicator - Granville's climax detection.
///
/// Detects unusual volume spikes that may indicate trend exhaustion.
#[derive(Debug, Clone)]
pub struct VolumeClimax {
    period: usize,
    threshold: f64,
}

impl VolumeClimax {
    pub fn new(period: usize) -> Self {
        Self { period, threshold: 2.0 }
    }

    pub fn with_threshold(period: usize, threshold: f64) -> Self {
        Self { period, threshold }
    }

    /// Calculate volume climax signals.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period || self.period == 0 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let vol_window = &volume[start..i]; // Exclude current

            let avg_vol: f64 = vol_window.iter().sum::<f64>() / vol_window.len() as f64;
            let vol_std: f64 = (vol_window.iter()
                .map(|v| (v - avg_vol).powi(2))
                .sum::<f64>() / vol_window.len() as f64).sqrt();

            if avg_vol.abs() < 1e-10 || vol_std.abs() < 1e-10 {
                result.push(0.0);
                continue;
            }

            let z_score = (volume[i] - avg_vol) / vol_std;

            // Climax: volume spike with price direction
            if z_score > self.threshold {
                if i > 0 && close[i] > close[i - 1] {
                    result.push(1.0); // Buying climax
                } else if i > 0 && close[i] < close[i - 1] {
                    result.push(-1.0); // Selling climax
                } else {
                    result.push(0.0);
                }
            } else {
                result.push(0.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for VolumeClimax {
    fn name(&self) -> &str {
        "Volume Climax"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for VolumeClimax {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let climax = self.calculate(&data.close, &data.volume);

        if climax.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let last = *climax.last().unwrap();
        if last > 0.5 {
            Ok(IndicatorSignal::Bearish) // Buying climax often precedes reversal
        } else if last < -0.5 {
            Ok(IndicatorSignal::Bullish) // Selling climax often precedes reversal
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let climax = self.calculate(&data.close, &data.volume);

        Ok(climax.iter().map(|&c| {
            if c > 0.5 {
                IndicatorSignal::Bearish
            } else if c < -0.5 {
                IndicatorSignal::Bullish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..50).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..50).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0 + i as f64 * 0.1).collect();
        let volume: Vec<f64> = (0..50).map(|i| 1000000.0 + (i as f64 * 0.2).sin() * 200000.0).collect();
        (high, low, close, volume)
    }

    #[test]
    fn test_emv_ma() {
        let (high, low, close, volume) = make_test_data();
        let emv_ma = EaseOfMovementMA::new(14, 10);
        let result = emv_ma.calculate(&high, &low, &volume);

        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_vama() {
        let (_, _, close, volume) = make_test_data();
        let vama = VAMA::new(14);
        let result = vama.calculate(&close, &volume);

        assert_eq!(result.len(), 50);

        // VAMA should be close to regular price
        for i in 13..50 {
            if !result[i].is_nan() {
                assert!((result[i] - close[i]).abs() < 20.0);
            }
        }
    }

    #[test]
    fn test_equivolume_width() {
        let (_, _, _, volume) = make_test_data();
        let eq = EquivolumeWidth::new(20);
        let result = eq.calculate(&volume);

        assert_eq!(result.len(), 50);

        // Width should be around 1.0 for average volume
        for i in 19..50 {
            if !result[i].is_nan() {
                assert!(result[i] > 0.0 && result[i] < 5.0);
            }
        }
    }

    #[test]
    fn test_obv_trend() {
        let (_, _, close, volume) = make_test_data();
        let obv_trend = OBVTrend::new(14);
        let (obv, obv_ma) = obv_trend.calculate(&close, &volume);

        assert_eq!(obv.len(), 50);
        assert_eq!(obv_ma.len(), 50);
    }

    #[test]
    fn test_obv_divergence() {
        let (_, _, close, volume) = make_test_data();
        let divergence = OBVDivergence::new(10);
        let result = divergence.calculate(&close, &volume);

        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_volume_climax() {
        let (_, _, close, volume) = make_test_data();
        let climax = VolumeClimax::new(20);
        let result = climax.calculate(&close, &volume);

        assert_eq!(result.len(), 50);

        // Climax values should be -1, 0, or 1
        for c in &result {
            assert!(*c >= -1.0 && *c <= 1.0);
        }
    }
}
