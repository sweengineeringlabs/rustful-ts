//! Relative Volatility Index (RVI - volatility version).

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Relative Volatility Index - IND-157
///
/// RSI applied to standard deviation instead of price.
/// Measures direction of volatility.
#[derive(Debug, Clone)]
pub struct RelativeVolatilityIndex {
    std_period: usize,
    rvi_period: usize,
    overbought: f64,
    oversold: f64,
}

impl RelativeVolatilityIndex {
    pub fn new(std_period: usize, rvi_period: usize) -> Self {
        Self {
            std_period,
            rvi_period,
            overbought: 60.0,
            oversold: 40.0,
        }
    }

    fn std_dev(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..n {
            let start_idx = i + 1 - period;
            let window = &data[start_idx..=i];
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / period as f64;
            result.push(variance.sqrt());
        }

        result
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.std_period + self.rvi_period {
            return vec![f64::NAN; n];
        }

        // Calculate standard deviation
        let std = Self::std_dev(data, self.std_period);

        // Calculate up/down volatility
        let mut up_vol = vec![0.0; n];
        let mut down_vol = vec![0.0; n];

        for i in 1..n {
            if std[i].is_nan() || std[i - 1].is_nan() {
                continue;
            }

            if data[i] > data[i - 1] {
                up_vol[i] = std[i];
            } else if data[i] < data[i - 1] {
                down_vol[i] = std[i];
            }
        }

        // Apply Wilder's smoothing (like RSI)
        let mut result = vec![f64::NAN; self.std_period + self.rvi_period - 1];

        let start = self.std_period;
        let mut avg_up: f64 = up_vol[start..(start + self.rvi_period)].iter().sum::<f64>() / self.rvi_period as f64;
        let mut avg_down: f64 = down_vol[start..(start + self.rvi_period)].iter().sum::<f64>() / self.rvi_period as f64;

        let rvi = if (avg_up + avg_down) == 0.0 {
            50.0
        } else {
            // Clamp to [0, 100] to handle floating point precision issues
            ((avg_up / (avg_up + avg_down)) * 100.0).clamp(0.0, 100.0)
        };
        result.push(rvi);

        for i in (start + self.rvi_period)..n {
            avg_up = (avg_up * (self.rvi_period - 1) as f64 + up_vol[i]) / self.rvi_period as f64;
            avg_down = (avg_down * (self.rvi_period - 1) as f64 + down_vol[i]) / self.rvi_period as f64;

            let rvi = if (avg_up + avg_down) == 0.0 {
                50.0
            } else {
                // Clamp to [0, 100] to handle floating point precision issues
                ((avg_up / (avg_up + avg_down)) * 100.0).clamp(0.0, 100.0)
            };
            result.push(rvi);
        }

        result
    }
}

impl Default for RelativeVolatilityIndex {
    fn default() -> Self {
        Self::new(10, 14)
    }
}

impl TechnicalIndicator for RelativeVolatilityIndex {
    fn name(&self) -> &str {
        "RVIVol"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.std_period + self.rvi_period {
            return Err(IndicatorError::InsufficientData {
                required: self.std_period + self.rvi_period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.std_period + self.rvi_period
    }
}

impl SignalIndicator for RelativeVolatilityIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if last >= self.overbought {
            Ok(IndicatorSignal::Bullish)
        } else if last <= self.oversold {
            Ok(IndicatorSignal::Bearish)
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
                IndicatorSignal::Bullish
            } else if val <= self.oversold {
                IndicatorSignal::Bearish
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
    fn test_rvi_vol_range() {
        let rvi = RelativeVolatilityIndex::default();
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.2).sin() * 10.0).collect();
        let result = rvi.calculate(&data);

        // RVI should be in range [0, 100]
        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 100.0, "RVI value {} out of range", val);
            }
        }
    }
}
