//! TD Pressure - Buying and Selling Pressure indicator.
//!
//! TD Pressure measures the buying and selling pressure in the market by
//! analyzing the relationship between price closes and the day's range.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// TD Pressure output.
#[derive(Debug, Clone)]
pub struct TDPressureOutput {
    /// Buying pressure values
    pub buying_pressure: Vec<f64>,
    /// Selling pressure values
    pub selling_pressure: Vec<f64>,
    /// Net pressure (buying - selling)
    pub net_pressure: Vec<f64>,
    /// Cumulative buying pressure
    pub cumulative_buying: Vec<f64>,
    /// Cumulative selling pressure
    pub cumulative_selling: Vec<f64>,
    /// Pressure ratio (buying / (buying + selling))
    pub pressure_ratio: Vec<f64>,
}

/// TD Pressure configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDPressureConfig {
    /// Period for smoothing (default: 14)
    pub period: usize,
    /// Use volume weighting (default: true)
    pub volume_weighted: bool,
    /// Overbought threshold for ratio (default: 0.7)
    pub overbought: f64,
    /// Oversold threshold for ratio (default: 0.3)
    pub oversold: f64,
}

impl Default for TDPressureConfig {
    fn default() -> Self {
        Self {
            period: 14,
            volume_weighted: true,
            overbought: 0.7,
            oversold: 0.3,
        }
    }
}

/// TD Pressure Indicator.
///
/// Measures buying and selling pressure based on where the close falls
/// within the day's range.
///
/// # Calculation
/// - Buying Pressure = Close - True Low
/// - Selling Pressure = True High - Close
/// - True High = max(High, Previous Close)
/// - True Low = min(Low, Previous Close)
///
/// # Interpretation
/// - High buying pressure: Bulls in control
/// - High selling pressure: Bears in control
/// - Pressure ratio approaching extremes indicates potential reversals
#[derive(Debug, Clone)]
pub struct TDPressure {
    config: TDPressureConfig,
}

impl TDPressure {
    pub fn new() -> Self {
        Self {
            config: TDPressureConfig::default(),
        }
    }

    pub fn with_config(config: TDPressureConfig) -> Self {
        Self { config }
    }

    pub fn with_period(mut self, period: usize) -> Self {
        self.config.period = period;
        self
    }

    pub fn volume_weighted(mut self, enabled: bool) -> Self {
        self.config.volume_weighted = enabled;
        self
    }

    /// Calculate TD Pressure from OHLCV data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDPressureOutput {
        let n = data.close.len();

        let mut buying_pressure = vec![0.0; n];
        let mut selling_pressure = vec![0.0; n];
        let mut net_pressure = vec![0.0; n];
        let mut cumulative_buying = vec![0.0; n];
        let mut cumulative_selling = vec![0.0; n];
        let mut pressure_ratio = vec![0.5; n];

        if n == 0 {
            return TDPressureOutput {
                buying_pressure,
                selling_pressure,
                net_pressure,
                cumulative_buying,
                cumulative_selling,
                pressure_ratio,
            };
        }

        // First bar
        let range0 = data.high[0] - data.low[0];
        if range0 > 0.0 {
            buying_pressure[0] = data.close[0] - data.low[0];
            selling_pressure[0] = data.high[0] - data.close[0];

            if self.config.volume_weighted {
                buying_pressure[0] *= data.volume[0];
                selling_pressure[0] *= data.volume[0];
            }
        }

        cumulative_buying[0] = buying_pressure[0];
        cumulative_selling[0] = selling_pressure[0];

        let total = buying_pressure[0] + selling_pressure[0];
        if total > 0.0 {
            pressure_ratio[0] = buying_pressure[0] / total;
        }

        net_pressure[0] = buying_pressure[0] - selling_pressure[0];

        // Subsequent bars
        for i in 1..n {
            let prev_close = data.close[i - 1];

            // True High and True Low
            let true_high = data.high[i].max(prev_close);
            let true_low = data.low[i].min(prev_close);
            let true_range = true_high - true_low;

            if true_range > 0.0 {
                let bp = data.close[i] - true_low;
                let sp = true_high - data.close[i];

                if self.config.volume_weighted {
                    buying_pressure[i] = bp * data.volume[i];
                    selling_pressure[i] = sp * data.volume[i];
                } else {
                    buying_pressure[i] = bp;
                    selling_pressure[i] = sp;
                }
            }

            cumulative_buying[i] = cumulative_buying[i - 1] + buying_pressure[i];
            cumulative_selling[i] = cumulative_selling[i - 1] + selling_pressure[i];

            let total = buying_pressure[i] + selling_pressure[i];
            if total > 0.0 {
                pressure_ratio[i] = buying_pressure[i] / total;
            }

            net_pressure[i] = buying_pressure[i] - selling_pressure[i];
        }

        // Apply period smoothing to ratio
        if self.config.period > 1 && n >= self.config.period {
            let smoothed_ratio = self.smooth_ratio(&buying_pressure, &selling_pressure);
            for i in 0..n {
                if !smoothed_ratio[i].is_nan() {
                    pressure_ratio[i] = smoothed_ratio[i];
                }
            }
        }

        TDPressureOutput {
            buying_pressure,
            selling_pressure,
            net_pressure,
            cumulative_buying,
            cumulative_selling,
            pressure_ratio,
        }
    }

    /// Smooth the pressure ratio over the configured period.
    fn smooth_ratio(&self, buying: &[f64], selling: &[f64]) -> Vec<f64> {
        let n = buying.len();
        let period = self.config.period;
        let mut ratio = vec![f64::NAN; n];

        if n < period {
            return ratio;
        }

        let mut bp_sum: f64 = buying[..period].iter().sum();
        let mut sp_sum: f64 = selling[..period].iter().sum();

        let total = bp_sum + sp_sum;
        if total > 0.0 {
            ratio[period - 1] = bp_sum / total;
        }

        for i in period..n {
            bp_sum = bp_sum - buying[i - period] + buying[i];
            sp_sum = sp_sum - selling[i - period] + selling[i];

            let total = bp_sum + sp_sum;
            if total > 0.0 {
                ratio[i] = bp_sum / total;
            }
        }

        ratio
    }
}

impl Default for TDPressure {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TDPressure {
    fn name(&self) -> &str {
        "TD Pressure"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let result = self.calculate(data);
        Ok(IndicatorOutput::triple(
            result.buying_pressure,
            result.selling_pressure,
            result.pressure_ratio,
        ))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for TDPressure {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.pressure_ratio.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let ratio = result.pressure_ratio[n - 1];

        // High buying pressure (ratio > overbought) could mean bullish
        // But could also mean overbought reversal coming
        // We interpret high buying pressure as currently bullish
        if ratio > self.config.overbought {
            Ok(IndicatorSignal::Bullish)
        } else if ratio < self.config.oversold {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);

        let signals = result.pressure_ratio.iter()
            .map(|&ratio| {
                if ratio.is_nan() {
                    IndicatorSignal::Neutral
                } else if ratio > self.config.overbought {
                    IndicatorSignal::Bullish
                } else if ratio < self.config.oversold {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_bullish_data(bars: usize) -> OHLCVSeries {
        // Closes near highs (bullish)
        let mut opens = Vec::with_capacity(bars);
        let mut highs = Vec::with_capacity(bars);
        let mut lows = Vec::with_capacity(bars);
        let mut closes = Vec::with_capacity(bars);

        for i in 0..bars {
            let base = 100.0 + (i as f64);
            opens.push(base);
            highs.push(base + 3.0);
            lows.push(base - 1.0);
            closes.push(base + 2.5); // Close near high
        }

        OHLCVSeries {
            open: opens,
            high: highs,
            low: lows,
            close: closes,
            volume: vec![1000.0; bars],
        }
    }

    fn create_bearish_data(bars: usize) -> OHLCVSeries {
        // Closes near lows (bearish)
        let mut opens = Vec::with_capacity(bars);
        let mut highs = Vec::with_capacity(bars);
        let mut lows = Vec::with_capacity(bars);
        let mut closes = Vec::with_capacity(bars);

        for i in 0..bars {
            let base = 100.0 - (i as f64);
            opens.push(base);
            highs.push(base + 1.0);
            lows.push(base - 3.0);
            closes.push(base - 2.5); // Close near low
        }

        OHLCVSeries {
            open: opens,
            high: highs,
            low: lows,
            close: closes,
            volume: vec![1000.0; bars],
        }
    }

    #[test]
    fn test_pressure_initialization() {
        let pressure = TDPressure::new();
        assert_eq!(pressure.name(), "TD Pressure");
        assert_eq!(pressure.config.period, 14);
        assert!(pressure.config.volume_weighted);
    }

    #[test]
    fn test_bullish_pressure() {
        let data = create_bullish_data(20);
        let pressure = TDPressure::new().volume_weighted(false);
        let result = pressure.calculate(&data);

        // Buying pressure should be higher than selling
        for i in 0..result.buying_pressure.len() {
            assert!(result.buying_pressure[i] > result.selling_pressure[i],
                "At bar {}: buying {} should > selling {}",
                i, result.buying_pressure[i], result.selling_pressure[i]);
        }

        // Pressure ratio should be > 0.5
        let last_ratio = result.pressure_ratio.last().unwrap();
        assert!(*last_ratio > 0.5);
    }

    #[test]
    fn test_bearish_pressure() {
        let data = create_bearish_data(20);
        let pressure = TDPressure::new().volume_weighted(false);
        let result = pressure.calculate(&data);

        // Selling pressure should be higher than buying
        for i in 0..result.selling_pressure.len() {
            assert!(result.selling_pressure[i] > result.buying_pressure[i],
                "At bar {}: selling {} should > buying {}",
                i, result.selling_pressure[i], result.buying_pressure[i]);
        }

        // Pressure ratio should be < 0.5
        let last_ratio = result.pressure_ratio.last().unwrap();
        assert!(*last_ratio < 0.5);
    }

    #[test]
    fn test_cumulative_values() {
        let data = create_bullish_data(10);
        let pressure = TDPressure::new().volume_weighted(false);
        let result = pressure.calculate(&data);

        // Cumulative should always increase
        for i in 1..result.cumulative_buying.len() {
            assert!(result.cumulative_buying[i] >= result.cumulative_buying[i - 1]);
            assert!(result.cumulative_selling[i] >= result.cumulative_selling[i - 1]);
        }
    }

    #[test]
    fn test_pressure_ratio_bounds() {
        let data = create_bullish_data(20);
        let pressure = TDPressure::new();
        let result = pressure.calculate(&data);

        // Ratio should be between 0 and 1
        for &ratio in &result.pressure_ratio {
            if !ratio.is_nan() {
                assert!(ratio >= 0.0 && ratio <= 1.0,
                    "Ratio {} should be in [0, 1]", ratio);
            }
        }
    }

    #[test]
    fn test_volume_weighting() {
        let mut data = create_bullish_data(10);
        data.volume = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0];

        let weighted = TDPressure::new().volume_weighted(true);
        let unweighted = TDPressure::new().volume_weighted(false);

        let result_w = weighted.calculate(&data);
        let result_u = unweighted.calculate(&data);

        // Weighted values should be different (scaled by volume)
        assert!(result_w.buying_pressure[5] != result_u.buying_pressure[5]);
    }

    #[test]
    fn test_net_pressure() {
        let data = create_bullish_data(10);
        let pressure = TDPressure::new().volume_weighted(false);
        let result = pressure.calculate(&data);

        // Net = buying - selling
        for i in 0..result.net_pressure.len() {
            let expected = result.buying_pressure[i] - result.selling_pressure[i];
            assert!((result.net_pressure[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_signal_generation() {
        let data = create_bullish_data(20);
        let pressure = TDPressure::new().volume_weighted(false);
        let signals = pressure.signals(&data).unwrap();

        assert_eq!(signals.len(), 20);

        // Most signals should be bullish or neutral in bullish data
        let bearish_count = signals.iter().filter(|s| **s == IndicatorSignal::Bearish).count();
        assert!(bearish_count < signals.len() / 2);
    }

    #[test]
    fn test_empty_data() {
        let data = OHLCVSeries {
            open: vec![],
            high: vec![],
            low: vec![],
            close: vec![],
            volume: vec![],
        };

        let pressure = TDPressure::new();
        let result = pressure.compute(&data);
        assert!(result.is_err());
    }
}
