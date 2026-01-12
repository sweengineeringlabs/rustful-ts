//! Chaikin Money Flow (CMF) implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};
use indicator_api::CMFConfig;

/// Chaikin Money Flow.
///
/// Measures buying and selling pressure over a period.
/// CMF = Sum(Money Flow Volume) / Sum(Volume)
/// - Positive: Buying pressure
/// - Negative: Selling pressure
#[derive(Debug, Clone)]
pub struct CMF {
    period: usize,
}

impl CMF {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: CMFConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate CMF values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < self.period {
            return result;
        }

        // Calculate Money Flow Multiplier and Money Flow Volume for each bar
        let mut mfv: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let hl_range = high[i] - low[i];
            let mf_multiplier = if hl_range > 0.0 {
                ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
            } else {
                0.0
            };
            mfv.push(mf_multiplier * volume[i]);
        }

        // Calculate CMF for each period
        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let sum_mfv: f64 = mfv[start..=i].iter().sum();
            let sum_vol: f64 = volume[start..=i].iter().sum();

            result[i] = if sum_vol > 0.0 { sum_mfv / sum_vol } else { 0.0 };
        }

        result
    }
}

impl Default for CMF {
    fn default() -> Self {
        Self::from_config(CMFConfig::default())
    }
}

impl TechnicalIndicator for CMF {
    fn name(&self) -> &str {
        "CMF"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for CMF {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last > 0.05 {
                    return Ok(IndicatorSignal::Bullish);  // Strong buying pressure
                } else if last < -0.05 {
                    return Ok(IndicatorSignal::Bearish);  // Strong selling pressure
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
            } else if v > 0.05 {
                IndicatorSignal::Bullish
            } else if v < -0.05 {
                IndicatorSignal::Bearish
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
    fn test_cmf() {
        let cmf = CMF::new(20);
        let n = 30;
        // Uptrend with closes near highs (bullish)
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 104.0 + i as f64).collect();  // Close near high
        let volume: Vec<f64> = (0..n).map(|_| 1000.0).collect();

        let result = cmf.calculate(&high, &low, &close, &volume);

        // Should have NaN initially
        assert!(result[0].is_nan());
        // After period, should have values between -1 and 1
        assert!(!result[29].is_nan());
        assert!(result[29] >= -1.0 && result[29] <= 1.0);
        // With closes near highs, CMF should be positive
        assert!(result[29] > 0.0);
    }
}
