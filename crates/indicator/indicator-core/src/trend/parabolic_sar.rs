//! Parabolic SAR implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_api::ParabolicSARConfig;

/// Parabolic SAR output.
#[derive(Debug, Clone)]
pub struct ParabolicSAROutput {
    pub sar: Vec<f64>,
    pub trend: Vec<f64>, // 1.0 = bullish, -1.0 = bearish
}

/// Parabolic SAR (Stop and Reverse).
///
/// Trend-following indicator that provides entry/exit points.
/// Uses an acceleration factor that increases as the trend continues.
#[derive(Debug, Clone)]
pub struct ParabolicSAR {
    af_start: f64,
    af_step: f64,
    af_max: f64,
}

impl ParabolicSAR {
    pub fn new(af_start: f64, af_step: f64, af_max: f64) -> Self {
        Self { af_start, af_step, af_max }
    }

    pub fn from_config(config: ParabolicSARConfig) -> Self {
        Self {
            af_start: config.af_start,
            af_step: config.af_step,
            af_max: config.af_max,
        }
    }

    /// Calculate Parabolic SAR values.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> ParabolicSAROutput {
        let n = high.len();
        if n < 2 {
            return ParabolicSAROutput {
                sar: vec![f64::NAN; n],
                trend: vec![f64::NAN; n],
            };
        }

        let mut sar = vec![0.0; n];
        let mut trend = vec![0.0; n];

        // Initialize with bullish trend
        let mut is_bullish = high[1] > high[0];
        let mut af = self.af_start;
        let mut ep = if is_bullish { high[0] } else { low[0] };
        let mut sar_val = if is_bullish { low[0] } else { high[0] };

        sar[0] = sar_val;
        trend[0] = if is_bullish { 1.0 } else { -1.0 };

        for i in 1..n {
            let prev_sar = sar_val;

            if is_bullish {
                // Bullish trend
                sar_val = prev_sar + af * (ep - prev_sar);

                // SAR cannot be above prior two lows
                if i >= 2 {
                    sar_val = sar_val.min(low[i - 1]).min(low[i - 2]);
                } else {
                    sar_val = sar_val.min(low[i - 1]);
                }

                // Check for trend reversal
                if low[i] < sar_val {
                    is_bullish = false;
                    sar_val = ep;
                    ep = low[i];
                    af = self.af_start;
                } else {
                    // Update EP and AF
                    if high[i] > ep {
                        ep = high[i];
                        af = (af + self.af_step).min(self.af_max);
                    }
                }
            } else {
                // Bearish trend
                sar_val = prev_sar + af * (ep - prev_sar);

                // SAR cannot be below prior two highs
                if i >= 2 {
                    sar_val = sar_val.max(high[i - 1]).max(high[i - 2]);
                } else {
                    sar_val = sar_val.max(high[i - 1]);
                }

                // Check for trend reversal
                if high[i] > sar_val {
                    is_bullish = true;
                    sar_val = ep;
                    ep = high[i];
                    af = self.af_start;
                } else {
                    // Update EP and AF
                    if low[i] < ep {
                        ep = low[i];
                        af = (af + self.af_step).min(self.af_max);
                    }
                }
            }

            sar[i] = sar_val;
            trend[i] = if is_bullish { 1.0 } else { -1.0 };
        }

        ParabolicSAROutput { sar, trend }
    }
}

impl Default for ParabolicSAR {
    fn default() -> Self {
        Self::from_config(ParabolicSARConfig::default())
    }
}

impl TechnicalIndicator for ParabolicSAR {
    fn name(&self) -> &str {
        "ParabolicSAR"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < 2 {
            return Err(IndicatorError::InsufficientData {
                required: 2,
                got: data.high.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::dual(result.sar, result.trend))
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for ParabolicSAR {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low);

        if result.trend.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = result.trend.len();
        let curr = result.trend[n - 1];
        let prev = result.trend[n - 2];

        // Signal on trend change
        if prev < 0.0 && curr > 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if prev > 0.0 && curr < 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low);
        let n = result.trend.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            let curr = result.trend[i];
            let prev = result.trend[i - 1];

            if prev < 0.0 && curr > 0.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if prev > 0.0 && curr < 0.0 {
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
    fn test_parabolic_sar() {
        let sar = ParabolicSAR::default();
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64 * 0.5).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64 * 0.5).collect();

        let result = sar.calculate(&high, &low);

        assert_eq!(result.sar.len(), n);
        assert_eq!(result.trend.len(), n);

        // In uptrend, SAR should be below price
        for i in 5..n {
            if result.trend[i] > 0.0 {
                assert!(result.sar[i] < low[i]);
            }
        }
    }
}
