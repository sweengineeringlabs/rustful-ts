//! Double Stochastic.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Double Stochastic - IND-158
///
/// Stochastic applied to stochastic values.
/// Provides smoother signals than single stochastic.
#[derive(Debug, Clone)]
pub struct DoubleStochastic {
    period: usize,
    smooth_k: usize,
    smooth_d: usize,
    overbought: f64,
    oversold: f64,
}

impl DoubleStochastic {
    pub fn new(period: usize, smooth_k: usize, smooth_d: usize) -> Self {
        Self {
            period,
            smooth_k,
            smooth_d,
            overbought: 80.0,
            oversold: 20.0,
        }
    }

    fn stochastic(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
        let n = close.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..n {
            let window_high = &high[(i - period + 1)..=i];
            let window_low = &low[(i - period + 1)..=i];

            let highest = window_high.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = window_low.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            let range = highest - lowest;
            if range == 0.0 {
                result.push(50.0);
            } else {
                result.push(((close[i] - lowest) / range) * 100.0);
            }
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

    fn stochastic_of_values(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..n {
            let window: Vec<f64> = data[(i - period + 1)..=i]
                .iter()
                .filter(|x| !x.is_nan())
                .copied()
                .collect();

            if window.is_empty() {
                result.push(f64::NAN);
                continue;
            }

            let highest = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            let current = data[i];
            if current.is_nan() {
                result.push(f64::NAN);
            } else if (highest - lowest) == 0.0 {
                result.push(50.0);
            } else {
                result.push(((current - lowest) / (highest - lowest)) * 100.0);
            }
        }

        result
    }

    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        if n < self.period * 2 + self.smooth_k + self.smooth_d {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // First stochastic
        let stoch1 = Self::stochastic(high, low, close, self.period);

        // Second stochastic (stochastic of stochastic)
        let stoch2 = Self::stochastic_of_values(&stoch1, self.period);

        // Smooth %K
        let k = Self::sma(&stoch2, self.smooth_k);

        // %D is SMA of %K
        let d = Self::sma(&k, self.smooth_d);

        (k, d)
    }
}

impl Default for DoubleStochastic {
    fn default() -> Self {
        Self::new(14, 3, 3)
    }
}

impl TechnicalIndicator for DoubleStochastic {
    fn name(&self) -> &str {
        "DoubleStoch"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_req = self.period * 2 + self.smooth_k + self.smooth_d;

        if data.close.len() < min_req {
            return Err(IndicatorError::InsufficientData {
                required: min_req,
                got: data.close.len(),
            });
        }

        let (k, d) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(k, d))
    }

    fn min_periods(&self) -> usize {
        self.period * 2 + self.smooth_k + self.smooth_d
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for DoubleStochastic {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (k, d) = self.calculate(&data.high, &data.low, &data.close);

        if k.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let k_last = k[k.len() - 1];
        let d_last = d[d.len() - 1];
        let k_prev = k[k.len() - 2];
        let d_prev = d[d.len() - 2];

        if k_last.is_nan() || d_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Overbought/oversold with crossover
        if k_last > d_last && k_prev <= d_prev && k_last < self.oversold {
            Ok(IndicatorSignal::Bullish)
        } else if k_last < d_last && k_prev >= d_prev && k_last > self.overbought {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (k, d) = self.calculate(&data.high, &data.low, &data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..k.len().min(d.len()) {
            if k[i].is_nan() || d[i].is_nan() || k[i-1].is_nan() || d[i-1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if k[i] > d[i] && k[i-1] <= d[i-1] && k[i] < self.oversold {
                signals.push(IndicatorSignal::Bullish);
            } else if k[i] < d[i] && k[i-1] >= d[i-1] && k[i] > self.overbought {
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
    fn test_double_stoch_range() {
        let ds = DoubleStochastic::default();
        let n = 60;
        let high: Vec<f64> = (0..n).map(|i| 110.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (k, d) = ds.calculate(&high, &low, &close);

        // Double Stochastic should be in range [0, 100]
        for val in k.iter().chain(d.iter()) {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 100.0, "Value {} out of range", val);
            }
        }
    }
}
