//! Stochastic RSI indicator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Stochastic RSI - IND-037
///
/// Applies Stochastic formula to RSI values for more sensitivity.
/// StochRSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)
#[derive(Debug, Clone)]
pub struct StochasticRSI {
    rsi_period: usize,
    stoch_period: usize,
    k_smooth: usize,
    d_smooth: usize,
    overbought: f64,
    oversold: f64,
}

impl StochasticRSI {
    pub fn new(rsi_period: usize, stoch_period: usize, k_smooth: usize, d_smooth: usize) -> Self {
        Self {
            rsi_period,
            stoch_period,
            k_smooth,
            d_smooth,
            overbought: 80.0,
            oversold: 20.0,
        }
    }

    fn rsi(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period + 1 {
            return vec![f64::NAN; n];
        }

        let mut gains = Vec::with_capacity(n - 1);
        let mut losses = Vec::with_capacity(n - 1);

        for i in 1..n {
            let change = data[i] - data[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        let mut result = vec![f64::NAN; period];

        let mut avg_gain: f64 = gains[0..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[0..period].iter().sum::<f64>() / period as f64;

        let rsi = if avg_loss == 0.0 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        };
        result.push(rsi);

        for i in period..(n - 1) {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            let rsi = if avg_loss == 0.0 {
                100.0
            } else {
                100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            };
            result.push(rsi);
        }

        result
    }

    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().filter(|x| !x.is_nan()).sum();
        result.push(sum / period as f64);

        for i in period..n {
            let old = if data[i - period].is_nan() { 0.0 } else { data[i - period] };
            let new = if data[i].is_nan() { 0.0 } else { data[i] };
            sum = sum - old + new;
            result.push(sum / period as f64);
        }

        result
    }

    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        let min_req = self.rsi_period + self.stoch_period + self.k_smooth + self.d_smooth;

        if n < min_req {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate RSI
        let rsi = Self::rsi(data, self.rsi_period);

        // Apply Stochastic formula to RSI
        let mut stoch_rsi = vec![f64::NAN; self.rsi_period + self.stoch_period - 1];

        for i in (self.rsi_period + self.stoch_period - 1)..n {
            let window = &rsi[(i - self.stoch_period + 1)..=i];
            let valid: Vec<f64> = window.iter().filter(|x| !x.is_nan()).copied().collect();

            if valid.is_empty() {
                stoch_rsi.push(f64::NAN);
                continue;
            }

            let lowest = valid.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let highest = valid.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let current_rsi = rsi[i];
            if current_rsi.is_nan() || (highest - lowest) == 0.0 {
                stoch_rsi.push(50.0);
            } else {
                let value = ((current_rsi - lowest) / (highest - lowest)) * 100.0;
                // Clamp to [0, 100] to handle floating point precision issues
                stoch_rsi.push(value.clamp(0.0, 100.0));
            }
        }

        // K line: SMA of StochRSI
        let k_raw = Self::sma(&stoch_rsi, self.k_smooth);
        // Clamp K values to [0, 100] to handle floating point precision issues
        let k: Vec<f64> = k_raw.iter()
            .map(|&v| if v.is_nan() { f64::NAN } else { v.clamp(0.0, 100.0) })
            .collect();

        // D line: SMA of K
        let d_raw = Self::sma(&k, self.d_smooth);
        // Clamp D values to [0, 100] to handle floating point precision issues
        let d: Vec<f64> = d_raw.iter()
            .map(|&v| if v.is_nan() { f64::NAN } else { v.clamp(0.0, 100.0) })
            .collect();

        (k, d)
    }
}

impl Default for StochasticRSI {
    fn default() -> Self {
        Self::new(14, 14, 3, 3)
    }
}

impl TechnicalIndicator for StochasticRSI {
    fn name(&self) -> &str {
        "StochRSI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_req = self.rsi_period + self.stoch_period + self.k_smooth + self.d_smooth;

        if data.close.len() < min_req {
            return Err(IndicatorError::InsufficientData {
                required: min_req,
                got: data.close.len(),
            });
        }

        let (k, d) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(k, d))
    }

    fn min_periods(&self) -> usize {
        self.rsi_period + self.stoch_period + self.k_smooth + self.d_smooth
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for StochasticRSI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (k, _) = self.calculate(&data.close);
        let last = k.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if last >= self.overbought {
            Ok(IndicatorSignal::Bearish)
        } else if last <= self.oversold {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (k, _) = self.calculate(&data.close);
        let signals = k.iter().map(|&val| {
            if val.is_nan() {
                IndicatorSignal::Neutral
            } else if val >= self.overbought {
                IndicatorSignal::Bearish
            } else if val <= self.oversold {
                IndicatorSignal::Bullish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect();
        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stoch_rsi_range() {
        let stoch_rsi = StochasticRSI::default();
        let data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let (k, d) = stoch_rsi.calculate(&data);

        // StochRSI should be in range [0, 100]
        for val in k.iter().chain(d.iter()) {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 100.0, "Value {} out of range", val);
            }
        }
    }
}
