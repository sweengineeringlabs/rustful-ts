//! Chaikin Oscillator implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Chaikin Oscillator.
///
/// The Chaikin Oscillator is the difference between the 3-day and 10-day
/// EMAs of the Accumulation/Distribution Line.
///
/// AD = ((Close - Low) - (High - Close)) / (High - Low) * Volume
/// Chaikin Oscillator = EMA(3) of AD - EMA(10) of AD
///
/// - Positive: Bullish momentum
/// - Negative: Bearish momentum
/// - Zero crossings: Trend changes
#[derive(Debug, Clone)]
pub struct ChaikinOscillator {
    fast_period: usize,
    slow_period: usize,
}

impl ChaikinOscillator {
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
        }
    }

    /// Calculate A/D Line values.
    fn calculate_ad(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);

        // Calculate first Money Flow Volume
        let mfv_0 = self.money_flow_volume(high[0], low[0], close[0], volume[0]);
        result.push(mfv_0);

        // Cumulative sum
        for i in 1..n {
            let mfv = self.money_flow_volume(high[i], low[i], close[i], volume[i]);
            result.push(result[i - 1] + mfv);
        }

        result
    }

    /// Calculate Money Flow Volume for a single bar.
    fn money_flow_volume(&self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let hl_range = high - low;
        if hl_range > 0.0 {
            let mfm = ((close - low) - (high - close)) / hl_range;
            mfm * volume
        } else {
            0.0
        }
    }

    /// Calculate EMA.
    fn ema(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let alpha = 2.0 / (period as f64 + 1.0);

        // Calculate initial SMA
        let sum: f64 = values[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        // Apply EMA
        for i in period..n {
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate Chaikin Oscillator values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        let ad = self.calculate_ad(high, low, close, volume);
        let n = ad.len();
        let mut result = vec![f64::NAN; n];

        if n < self.slow_period {
            return result;
        }

        let fast_ema = self.ema(&ad, self.fast_period);
        let slow_ema = self.ema(&ad, self.slow_period);

        for i in 0..n {
            if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
                result[i] = fast_ema[i] - slow_ema[i];
            }
        }

        result
    }
}

impl Default for ChaikinOscillator {
    fn default() -> Self {
        Self {
            fast_period: 3,
            slow_period: 10,
        }
    }
}

impl TechnicalIndicator for ChaikinOscillator {
    fn name(&self) -> &str {
        "Chaikin Oscillator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.slow_period {
            return Err(IndicatorError::InsufficientData {
                required: self.slow_period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.slow_period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for ChaikinOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);

        if values.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = values.len();
        let current = values[n - 1];
        let previous = values[n - 2];

        if current.is_nan() || previous.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: crosses above zero
        if previous <= 0.0 && current > 0.0 {
            return Ok(IndicatorSignal::Bullish);
        }
        // Bearish: crosses below zero
        if previous >= 0.0 && current < 0.0 {
            return Ok(IndicatorSignal::Bearish);
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..values.len() {
            if values[i].is_nan() || values[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if values[i - 1] <= 0.0 && values[i] > 0.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if values[i - 1] >= 0.0 && values[i] < 0.0 {
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
    fn test_chaikin_oscillator() {
        let co = ChaikinOscillator::new(3, 5);
        let n = 15;
        // Uptrend with closes near highs
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 104.0 + i as f64).collect();
        let volume: Vec<f64> = (0..n).map(|_| 1000.0).collect();

        let result = co.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), n);
        // First values should be NaN until slow_period
        assert!(result[0].is_nan());
        // After slow_period, should have values
        assert!(!result[n - 1].is_nan());
    }

    #[test]
    fn test_chaikin_oscillator_trait() {
        let co = ChaikinOscillator::default();

        assert_eq!(co.name(), "Chaikin Oscillator");
        assert_eq!(co.min_periods(), 10);
        assert_eq!(co.output_features(), 1);
    }
}
