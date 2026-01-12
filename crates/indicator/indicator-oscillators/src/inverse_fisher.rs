//! Inverse Fisher Transform indicator.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Inverse Fisher Transform - IND-043
///
/// Smooths oscillator signals by applying inverse Fisher transform.
/// IFT = (exp(2x) - 1) / (exp(2x) + 1)
#[derive(Debug, Clone)]
pub struct InverseFisherTransform {
    period: usize,
    wma_period: usize,
    overbought: f64,
    oversold: f64,
}

impl InverseFisherTransform {
    pub fn new(period: usize, wma_period: usize) -> Self {
        Self {
            period,
            wma_period,
            overbought: 0.5,
            oversold: -0.5,
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

        let rsi = if avg_loss == 0.0 { 100.0 } else { 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) };
        result.push(rsi);

        for i in period..(n - 1) {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            let rsi = if avg_loss == 0.0 { 100.0 } else { 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) };
            result.push(rsi);
        }

        result
    }

    fn wma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];
        let weight_sum: f64 = (1..=period).map(|x| x as f64).sum();

        for i in (period - 1)..n {
            let window = &data[(i - period + 1)..=i];
            let valid: Vec<(usize, f64)> = window.iter()
                .enumerate()
                .filter(|(_, x)| !x.is_nan())
                .map(|(j, &x)| (j, x))
                .collect();

            if valid.is_empty() {
                result.push(f64::NAN);
            } else {
                let weighted_sum: f64 = valid.iter()
                    .map(|(j, x)| (j + 1) as f64 * x)
                    .sum();
                result.push(weighted_sum / weight_sum);
            }
        }

        result
    }

    fn inverse_fisher(x: f64) -> f64 {
        let exp_2x = (2.0 * x).exp();
        (exp_2x - 1.0) / (exp_2x + 1.0)
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period + self.wma_period {
            return vec![f64::NAN; n];
        }

        // Calculate RSI
        let rsi = Self::rsi(data, self.period);

        // Normalize RSI to -5 to +5 range
        let normalized: Vec<f64> = rsi.iter()
            .map(|&r| if r.is_nan() { f64::NAN } else { 0.1 * (r - 50.0) })
            .collect();

        // Apply WMA smoothing
        let smoothed = Self::wma(&normalized, self.wma_period);

        // Apply Inverse Fisher Transform
        smoothed.iter()
            .map(|&x| if x.is_nan() { f64::NAN } else { Self::inverse_fisher(x) })
            .collect()
    }
}

impl Default for InverseFisherTransform {
    fn default() -> Self {
        Self::new(5, 9)
    }
}

impl TechnicalIndicator for InverseFisherTransform {
    fn name(&self) -> &str {
        "IFT"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + self.wma_period {
            return Err(IndicatorError::InsufficientData {
                required: self.period + self.wma_period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + self.wma_period
    }
}

impl SignalIndicator for InverseFisherTransform {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

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
        let values = self.calculate(&data.close);
        let signals = values.iter().map(|&val| {
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
    fn test_ift_range() {
        let ift = InverseFisherTransform::default();
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let result = ift.calculate(&data);

        // IFT should be in range [-1, 1]
        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= -1.0 && *val <= 1.0, "IFT value {} out of range", val);
            }
        }
    }
}
