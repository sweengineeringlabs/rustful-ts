//! Money Flow Index (MFI) implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};
use indicator_api::MFIConfig;

/// Money Flow Index.
///
/// MFI is a volume-weighted RSI.
/// - Above 80: Overbought
/// - Below 20: Oversold
#[derive(Debug, Clone)]
pub struct MFI {
    period: usize,
    overbought: f64,
    oversold: f64,
}

impl MFI {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            overbought: 80.0,
            oversold: 20.0,
        }
    }

    pub fn from_config(config: MFIConfig) -> Self {
        Self {
            period: config.period,
            overbought: config.overbought,
            oversold: config.oversold,
        }
    }

    /// Calculate MFI values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n <= self.period {
            return result;
        }

        // Calculate Typical Price and Money Flow for each bar
        let mut typical_prices: Vec<f64> = Vec::with_capacity(n);
        let mut money_flows: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            let tp = (high[i] + low[i] + close[i]) / 3.0;
            typical_prices.push(tp);
            money_flows.push(tp * volume[i]);
        }

        // Calculate MFI
        for i in self.period..n {
            let mut pos_flow = 0.0;
            let mut neg_flow = 0.0;

            for j in (i - self.period + 1)..=i {
                if typical_prices[j] > typical_prices[j - 1] {
                    pos_flow += money_flows[j];
                } else if typical_prices[j] < typical_prices[j - 1] {
                    neg_flow += money_flows[j];
                }
            }

            result[i] = if neg_flow > 0.0 {
                100.0 - 100.0 / (1.0 + pos_flow / neg_flow)
            } else {
                100.0
            };
        }

        result
    }
}

impl Default for MFI {
    fn default() -> Self {
        Self::from_config(MFIConfig::default())
    }
}

impl TechnicalIndicator for MFI {
    fn name(&self) -> &str {
        "MFI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() <= self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for MFI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last > self.overbought {
                    return Ok(IndicatorSignal::Bearish);  // Overbought
                } else if last < self.oversold {
                    return Ok(IndicatorSignal::Bullish);  // Oversold
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(values.iter().map(|&v| {
            if v.is_nan() {
                IndicatorSignal::Neutral
            } else if v > self.overbought {
                IndicatorSignal::Bearish
            } else if v < self.oversold {
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

    #[test]
    fn test_mfi() {
        let mfi = MFI::new(14);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let volume: Vec<f64> = (0..n).map(|_| 1000.0).collect();

        let result = mfi.calculate(&high, &low, &close, &volume);

        // Should have NaN initially
        assert!(result[0].is_nan());
        // After period, should have values between 0 and 100
        assert!(!result[29].is_nan());
        assert!(result[29] >= 0.0 && result[29] <= 100.0);
    }
}
