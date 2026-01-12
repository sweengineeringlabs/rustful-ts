//! Laguerre RSI indicator.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Laguerre RSI (Relative Strength Index)
///
/// John Ehlers' Laguerre RSI uses a 4-element Laguerre filter to smooth
/// price data, resulting in an RSI that responds faster to price changes
/// with less lag than traditional RSI.
///
/// The gamma parameter controls the smoothing (0 = most responsive, 1 = smoothest).
/// Output ranges from 0 to 1.
#[derive(Debug, Clone)]
pub struct LaguerreRSI {
    /// Gamma (damping factor) - range 0 to 1 (default: 0.5)
    pub gamma: f64,
    /// Overbought level (default: 0.8)
    pub overbought: f64,
    /// Oversold level (default: 0.2)
    pub oversold: f64,
}

impl LaguerreRSI {
    pub fn new(gamma: f64) -> Self {
        Self {
            gamma: gamma.max(0.0).min(1.0),
            overbought: 0.8,
            oversold: 0.2,
        }
    }

    pub fn with_levels(gamma: f64, overbought: f64, oversold: f64) -> Self {
        Self {
            gamma: gamma.max(0.0).min(1.0),
            overbought,
            oversold,
        }
    }

    /// Calculate Laguerre RSI
    /// Returns RSI values (0 to 1 range)
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 2 {
            return vec![f64::NAN; n];
        }

        let mut rsi = vec![f64::NAN; n];

        // Laguerre filter coefficients
        let g = self.gamma;
        let one_minus_g = 1.0 - g;

        // 4-element Laguerre filter state
        let mut l0 = 0.0;
        let mut l1 = 0.0;
        let mut l2 = 0.0;
        let mut l3 = 0.0;

        for i in 0..n {
            // Save previous values
            let l0_prev = l0;
            let l1_prev = l1;
            let l2_prev = l2;

            // Update Laguerre filter
            l0 = one_minus_g * data[i] + g * l0_prev;
            l1 = -g * l0 + l0_prev + g * l1_prev;
            l2 = -g * l1 + l1_prev + g * l2_prev;
            l3 = -g * l2 + l2_prev + g * l3;

            // Calculate cumulative up and down
            let mut cu = 0.0;
            let mut cd = 0.0;

            if l0 >= l1 {
                cu += l0 - l1;
            } else {
                cd += l1 - l0;
            }

            if l1 >= l2 {
                cu += l1 - l2;
            } else {
                cd += l2 - l1;
            }

            if l2 >= l3 {
                cu += l2 - l3;
            } else {
                cd += l3 - l2;
            }

            // Calculate RSI
            if cu + cd != 0.0 {
                rsi[i] = cu / (cu + cd);
            } else {
                rsi[i] = 0.5; // Neutral when no change
            }
        }

        rsi
    }
}

impl Default for LaguerreRSI {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl TechnicalIndicator for LaguerreRSI {
    fn name(&self) -> &str {
        "LaguerreRSI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 2 {
            return Err(IndicatorError::InsufficientData {
                required: 2,
                got: data.close.len(),
            });
        }

        let rsi = self.calculate(&data.close);
        Ok(IndicatorOutput::single(rsi))
    }

    fn min_periods(&self) -> usize {
        2
    }
}

impl SignalIndicator for LaguerreRSI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let rsi = self.calculate(&data.close);
        let n = rsi.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let rsi_last = rsi[n - 1];
        let rsi_prev = rsi[n - 2];

        if rsi_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Oversold crossing above oversold level = bullish
        if rsi_last > self.oversold && rsi_prev <= self.oversold {
            Ok(IndicatorSignal::Bullish)
        }
        // Overbought crossing below overbought level = bearish
        else if rsi_last < self.overbought && rsi_prev >= self.overbought {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let rsi = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..rsi.len() {
            if rsi[i].is_nan() || rsi[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if rsi[i] > self.oversold && rsi[i - 1] <= self.oversold {
                signals.push(IndicatorSignal::Bullish);
            } else if rsi[i] < self.overbought && rsi[i - 1] >= self.overbought {
                signals.push(IndicatorSignal::Bearish);
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laguerre_rsi_basic() {
        let lrsi = LaguerreRSI::default();
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();

        let rsi = lrsi.calculate(&data);

        assert_eq!(rsi.len(), n);

        // All values should be valid (no warmup period for Laguerre)
        for i in 1..n {
            assert!(!rsi[i].is_nan());
        }
    }

    #[test]
    fn test_laguerre_rsi_range() {
        let lrsi = LaguerreRSI::new(0.7);
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.2).sin() * 15.0).collect();

        let rsi = lrsi.calculate(&data);

        // RSI should be bounded between 0 and 1
        for i in 5..n {
            assert!(rsi[i] >= 0.0 && rsi[i] <= 1.0,
                "RSI[{}] = {} out of bounds", i, rsi[i]);
        }
    }

    #[test]
    fn test_laguerre_rsi_trending() {
        let lrsi = LaguerreRSI::default();

        // Strong uptrend
        let uptrend: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let rsi_up = lrsi.calculate(&uptrend);

        // Strong downtrend
        let downtrend: Vec<f64> = (0..50).map(|i| 200.0 - i as f64 * 2.0).collect();
        let rsi_down = lrsi.calculate(&downtrend);

        // Uptrend should have higher RSI than downtrend
        let avg_up: f64 = rsi_up[40..50].iter().sum::<f64>() / 10.0;
        let avg_down: f64 = rsi_down[40..50].iter().sum::<f64>() / 10.0;

        assert!(avg_up > avg_down, "Uptrend RSI ({}) should be > downtrend RSI ({})", avg_up, avg_down);
    }

    #[test]
    fn test_laguerre_rsi_gamma_effect() {
        // Lower gamma = more responsive
        let lrsi_fast = LaguerreRSI::new(0.2);
        let lrsi_slow = LaguerreRSI::new(0.8);

        let n = 50;
        let data: Vec<f64> = (0..n).map(|i| {
            if i < 25 { 100.0 } else { 120.0 }  // Step change
        }).collect();

        let rsi_fast = lrsi_fast.calculate(&data);
        let rsi_slow = lrsi_slow.calculate(&data);

        // Fast RSI should react quicker to the step change
        // Check response at bar 30 (5 bars after step)
        let fast_response = (rsi_fast[30] - rsi_fast[24]).abs();
        let slow_response = (rsi_slow[30] - rsi_slow[24]).abs();

        assert!(fast_response > slow_response,
            "Fast RSI response ({}) should be > slow RSI response ({})", fast_response, slow_response);
    }

    #[test]
    fn test_laguerre_rsi_custom_levels() {
        let lrsi = LaguerreRSI::with_levels(0.5, 0.7, 0.3);
        assert_eq!(lrsi.overbought, 0.7);
        assert_eq!(lrsi.oversold, 0.3);
    }

    #[test]
    fn test_laguerre_rsi_trait_impl() {
        let lrsi = LaguerreRSI::default();
        assert_eq!(lrsi.name(), "LaguerreRSI");
        assert_eq!(lrsi.min_periods(), 2);
        assert_eq!(lrsi.output_features(), 1);
    }
}
