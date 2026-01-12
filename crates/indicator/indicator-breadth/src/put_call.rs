//! Put/Call Ratio indicator.

use crate::BreadthIndicator;
use indicator_spi::{IndicatorError, IndicatorOutput, Result};

/// Put/Call Ratio
///
/// A sentiment indicator measuring the ratio of put options volume to
/// call options volume. It gauges investor sentiment and can signal
/// potential market reversals when at extreme levels.
///
/// # Formula
/// Put/Call Ratio = Put Volume / Call Volume
///
/// # Types
/// - Total Put/Call: All options on all assets
/// - Equity Put/Call: Stock options only
/// - Index Put/Call: Index options only (often used for hedging)
/// - VIX Put/Call: VIX options
///
/// # Interpretation
/// - Ratio < 0.7: Bullish sentiment (complacency, potentially overbought)
/// - Ratio 0.7-1.0: Normal range
/// - Ratio > 1.0: Bearish sentiment (fear, potentially oversold)
/// - Ratio > 1.2: Extreme fear (contrarian buy signal)
///
/// Note: This is a contrarian indicator - extreme readings often
/// signal potential reversals.
#[derive(Debug, Clone)]
pub struct PutCallRatio {
    /// Smoothing period (0 = no smoothing)
    smoothing_period: usize,
    /// Whether to use EMA instead of SMA
    use_ema: bool,
    /// Overbought threshold (low ratio = bullish sentiment)
    overbought_threshold: f64,
    /// Oversold threshold (high ratio = bearish sentiment)
    oversold_threshold: f64,
    /// Extreme overbought threshold
    extreme_overbought: f64,
    /// Extreme oversold threshold
    extreme_oversold: f64,
}

impl Default for PutCallRatio {
    fn default() -> Self {
        Self::new()
    }
}

impl PutCallRatio {
    pub fn new() -> Self {
        Self {
            smoothing_period: 0,
            use_ema: false,
            overbought_threshold: 0.7,
            oversold_threshold: 1.0,
            extreme_overbought: 0.5,
            extreme_oversold: 1.2,
        }
    }

    /// Create with 10-day moving average (common setting)
    pub fn smoothed_10() -> Self {
        Self {
            smoothing_period: 10,
            use_ema: false,
            overbought_threshold: 0.7,
            oversold_threshold: 1.0,
            extreme_overbought: 0.5,
            extreme_oversold: 1.2,
        }
    }

    /// Create with 21-day EMA (common setting)
    pub fn ema_21() -> Self {
        Self {
            smoothing_period: 21,
            use_ema: true,
            overbought_threshold: 0.7,
            oversold_threshold: 1.0,
            extreme_overbought: 0.5,
            extreme_oversold: 1.2,
        }
    }

    pub fn with_smoothing(mut self, period: usize) -> Self {
        self.smoothing_period = period;
        self
    }

    pub fn with_ema(mut self) -> Self {
        self.use_ema = true;
        self
    }

    pub fn with_thresholds(
        mut self,
        overbought: f64,
        oversold: f64,
        extreme_ob: f64,
        extreme_os: f64,
    ) -> Self {
        self.overbought_threshold = overbought;
        self.oversold_threshold = oversold;
        self.extreme_overbought = extreme_ob;
        self.extreme_oversold = extreme_os;
        self
    }

    /// Calculate SMA
    fn calculate_sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().filter(|v| !v.is_nan()).sum();
        result.push(sum / period as f64);

        for i in period..data.len() {
            if !data[i - period].is_nan() {
                sum -= data[i - period];
            }
            if !data[i].is_nan() {
                sum += data[i];
            }
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculate EMA
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; period - 1];
        let multiplier = 2.0 / (period as f64 + 1.0);

        // Initial SMA as first EMA
        let valid_count = data[..period].iter().filter(|v| !v.is_nan()).count();
        let sum: f64 = data[..period].iter().filter(|v| !v.is_nan()).sum();
        let mut ema = sum / valid_count.max(1) as f64;
        result.push(ema);

        // Calculate EMA
        for i in period..data.len() {
            if !data[i].is_nan() {
                ema = (data[i] - ema) * multiplier + ema;
            }
            result.push(ema);
        }

        result
    }

    /// Calculate put/call ratio from volume data
    pub fn calculate(&self, put_volume: &[f64], call_volume: &[f64]) -> Vec<f64> {
        let raw: Vec<f64> = put_volume
            .iter()
            .zip(call_volume.iter())
            .map(|(p, c)| {
                if *c == 0.0 {
                    if *p > 0.0 {
                        f64::INFINITY
                    } else {
                        1.0 // Both zero = neutral
                    }
                } else {
                    p / c
                }
            })
            .collect();

        if self.smoothing_period > 0 {
            if self.use_ema {
                self.calculate_ema(&raw, self.smoothing_period)
            } else {
                self.calculate_sma(&raw, self.smoothing_period)
            }
        } else {
            raw
        }
    }

    /// Calculate single put/call ratio value
    pub fn calculate_single(&self, put_volume: f64, call_volume: f64) -> f64 {
        if call_volume == 0.0 {
            if put_volume > 0.0 {
                f64::INFINITY
            } else {
                1.0
            }
        } else {
            put_volume / call_volume
        }
    }

    /// Interpret put/call ratio value (contrarian)
    pub fn interpret(&self, ratio: f64) -> PutCallSignal {
        if ratio.is_nan() || ratio.is_infinite() {
            PutCallSignal::Unknown
        } else if ratio <= self.extreme_overbought {
            PutCallSignal::ExtremeComplacency
        } else if ratio <= self.overbought_threshold {
            PutCallSignal::Complacency
        } else if ratio >= self.extreme_oversold {
            PutCallSignal::ExtremeFear
        } else if ratio >= self.oversold_threshold {
            PutCallSignal::Fear
        } else {
            PutCallSignal::Neutral
        }
    }

    /// Generate contrarian signal
    pub fn contrarian_signal(&self, ratio: f64) -> ContrarianSignal {
        match self.interpret(ratio) {
            PutCallSignal::ExtremeComplacency => ContrarianSignal::StrongSell,
            PutCallSignal::Complacency => ContrarianSignal::Sell,
            PutCallSignal::ExtremeFear => ContrarianSignal::StrongBuy,
            PutCallSignal::Fear => ContrarianSignal::Buy,
            _ => ContrarianSignal::Neutral,
        }
    }
}

/// Put/Call sentiment signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PutCallSignal {
    /// Extremely low ratio: Extreme bullish sentiment (complacency)
    ExtremeComplacency,
    /// Low ratio: Bullish sentiment
    Complacency,
    /// Normal range
    Neutral,
    /// High ratio: Bearish sentiment
    Fear,
    /// Extremely high ratio: Extreme bearish sentiment
    ExtremeFear,
    /// Invalid data
    Unknown,
}

/// Contrarian trading signal based on put/call ratio
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContrarianSignal {
    /// Extreme complacency suggests sell
    StrongSell,
    /// Complacency suggests caution
    Sell,
    /// Neutral sentiment
    Neutral,
    /// Fear suggests opportunity
    Buy,
    /// Extreme fear suggests strong buy opportunity
    StrongBuy,
}

/// Put/Call data series
#[derive(Debug, Clone, Default)]
pub struct PutCallSeries {
    pub put_volume: Vec<f64>,
    pub call_volume: Vec<f64>,
    /// Optional: open interest data
    pub put_oi: Vec<f64>,
    pub call_oi: Vec<f64>,
}

impl PutCallSeries {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, put_vol: f64, call_vol: f64) {
        self.put_volume.push(put_vol);
        self.call_volume.push(call_vol);
    }

    pub fn push_with_oi(&mut self, put_vol: f64, call_vol: f64, put_oi: f64, call_oi: f64) {
        self.put_volume.push(put_vol);
        self.call_volume.push(call_vol);
        self.put_oi.push(put_oi);
        self.call_oi.push(call_oi);
    }

    pub fn len(&self) -> usize {
        self.put_volume.len()
    }

    pub fn is_empty(&self) -> bool {
        self.put_volume.is_empty()
    }

    /// Calculate volume-based put/call ratio
    pub fn volume_ratio(&self) -> Vec<f64> {
        self.put_volume
            .iter()
            .zip(self.call_volume.iter())
            .map(|(p, c)| if *c == 0.0 { f64::NAN } else { p / c })
            .collect()
    }

    /// Calculate open interest-based put/call ratio
    pub fn oi_ratio(&self) -> Vec<f64> {
        if self.put_oi.is_empty() || self.call_oi.is_empty() {
            return Vec::new();
        }

        self.put_oi
            .iter()
            .zip(self.call_oi.iter())
            .map(|(p, c)| if *c == 0.0 { f64::NAN } else { p / c })
            .collect()
    }
}

impl BreadthIndicator for PutCallRatio {
    fn name(&self) -> &str {
        "Put/Call Ratio"
    }

    fn compute_breadth(&self, data: &crate::BreadthSeries) -> Result<IndicatorOutput> {
        // Use advances as proxy for call volume and declines as put volume
        // This is a rough approximation when actual options data isn't available
        let min_required = if self.smoothing_period > 0 {
            self.smoothing_period
        } else {
            1
        };

        if data.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.len(),
            });
        }

        // Invert: declines as "put-like" activity, advances as "call-like" activity
        let values = self.calculate(&data.declines, &data.advances);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        if self.smoothing_period > 0 {
            self.smoothing_period
        } else {
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_call_basic() {
        let pcr = PutCallRatio::new();

        // More puts than calls = high ratio
        let put_vol = vec![1_000_000.0, 800_000.0, 1_200_000.0];
        let call_vol = vec![1_000_000.0, 1_000_000.0, 800_000.0];

        let result = pcr.calculate(&put_vol, &call_vol);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 0.8).abs() < 1e-10);
        assert!((result[2] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_put_call_interpretation() {
        let pcr = PutCallRatio::new();

        assert_eq!(pcr.interpret(0.4), PutCallSignal::ExtremeComplacency);
        assert_eq!(pcr.interpret(0.6), PutCallSignal::Complacency);
        assert_eq!(pcr.interpret(0.85), PutCallSignal::Neutral);
        assert_eq!(pcr.interpret(1.1), PutCallSignal::Fear);
        assert_eq!(pcr.interpret(1.5), PutCallSignal::ExtremeFear);
    }

    #[test]
    fn test_contrarian_signals() {
        let pcr = PutCallRatio::new();

        assert_eq!(pcr.contrarian_signal(0.4), ContrarianSignal::StrongSell);
        assert_eq!(pcr.contrarian_signal(0.6), ContrarianSignal::Sell);
        assert_eq!(pcr.contrarian_signal(0.85), ContrarianSignal::Neutral);
        assert_eq!(pcr.contrarian_signal(1.1), ContrarianSignal::Buy);
        assert_eq!(pcr.contrarian_signal(1.5), ContrarianSignal::StrongBuy);
    }

    #[test]
    fn test_put_call_smoothed() {
        let pcr = PutCallRatio::smoothed_10();

        let put_vol: Vec<f64> = (0..15).map(|i| 900_000.0 + (i as f64 * 20_000.0)).collect();
        let call_vol: Vec<f64> = (0..15).map(|_| 1_000_000.0).collect();

        let result = pcr.calculate(&put_vol, &call_vol);

        assert_eq!(result.len(), 15);
        // First 9 should be NaN
        for i in 0..9 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }
        assert!(!result[9].is_nan());
    }

    #[test]
    fn test_put_call_ema() {
        let pcr = PutCallRatio::ema_21();

        let put_vol: Vec<f64> = (0..30).map(|i| 900_000.0 + (i as f64 * 10_000.0)).collect();
        let call_vol: Vec<f64> = (0..30).map(|_| 1_000_000.0).collect();

        let result = pcr.calculate(&put_vol, &call_vol);

        assert_eq!(result.len(), 30);
        assert!(!result[20].is_nan());
    }

    #[test]
    fn test_put_call_single() {
        let pcr = PutCallRatio::new();

        assert!((pcr.calculate_single(1_000_000.0, 1_000_000.0) - 1.0).abs() < 1e-10);
        assert!((pcr.calculate_single(500_000.0, 1_000_000.0) - 0.5).abs() < 1e-10);
        assert!((pcr.calculate_single(1_000_000.0, 0.0)).is_infinite());
    }

    #[test]
    fn test_put_call_series() {
        let mut series = PutCallSeries::new();
        series.push(1_000_000.0, 1_000_000.0);
        series.push(800_000.0, 1_000_000.0);
        series.push_with_oi(900_000.0, 1_000_000.0, 5_000_000.0, 4_000_000.0);

        let vol_ratio = series.volume_ratio();
        let oi_ratio = series.oi_ratio();

        assert_eq!(vol_ratio.len(), 3);
        assert!((vol_ratio[0] - 1.0).abs() < 1e-10);
        assert!((vol_ratio[1] - 0.8).abs() < 1e-10);

        assert_eq!(oi_ratio.len(), 1); // Only one entry has OI
        assert!((oi_ratio[0] - 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_zero_call_volume() {
        let pcr = PutCallRatio::new();

        let put_vol = vec![1_000_000.0, 0.0];
        let call_vol = vec![0.0, 0.0];

        let result = pcr.calculate(&put_vol, &call_vol);

        assert!(result[0].is_infinite()); // Put > 0, Call = 0
        assert!((result[1] - 1.0).abs() < 1e-10); // Both zero = neutral
    }
}
