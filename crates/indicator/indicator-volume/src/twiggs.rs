//! Twiggs Money Flow implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Twiggs Money Flow.
///
/// A variation of Chaikin Money Flow that uses True Range instead of
/// High-Low range, making it more sensitive to gaps.
///
/// True Range High = max(High, Previous Close)
/// True Range Low = min(Low, Previous Close)
/// AD = ((Close - TRL) - (TRH - Close)) / (TRH - TRL) * Volume
///
/// TMF = EMA(AD) / EMA(Volume)
#[derive(Debug, Clone)]
pub struct TwiggsMoneyFlow {
    period: usize,
}

impl TwiggsMoneyFlow {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate Twiggs Money Flow values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < 2 {
            return result;
        }

        // Calculate AD values using True Range
        let mut ad = vec![0.0_f64; n];
        ad[0] = 0.0;

        for i in 1..n {
            let prev_close = close[i - 1];
            let trh = high[i].max(prev_close);
            let trl = low[i].min(prev_close);
            let tr = trh - trl;

            if tr > 0.0 {
                let mfm = ((close[i] - trl) - (trh - close[i])) / tr;
                ad[i] = mfm * volume[i];
            }
        }

        // Apply Wilder's smoothing (EMA equivalent)
        let alpha = 2.0 / (self.period as f64 + 1.0);

        if n < self.period + 1 {
            return result;
        }

        // Calculate initial averages
        let mut ad_sum = 0.0;
        let mut vol_sum = 0.0;
        for i in 1..=self.period {
            ad_sum += ad[i];
            vol_sum += volume[i];
        }
        let mut ema_ad = ad_sum / self.period as f64;
        let mut ema_vol = vol_sum / self.period as f64;

        result[self.period] = if ema_vol > 0.0 {
            ema_ad / ema_vol
        } else {
            0.0
        };

        // Apply EMA
        for i in (self.period + 1)..n {
            ema_ad = alpha * ad[i] + (1.0 - alpha) * ema_ad;
            ema_vol = alpha * volume[i] + (1.0 - alpha) * ema_vol;

            result[i] = if ema_vol > 0.0 {
                ema_ad / ema_vol
            } else {
                0.0
            };
        }

        result
    }
}

impl Default for TwiggsMoneyFlow {
    fn default() -> Self {
        Self { period: 21 }
    }
}

impl TechnicalIndicator for TwiggsMoneyFlow {
    fn name(&self) -> &str {
        "Twiggs Money Flow"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
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

impl SignalIndicator for TwiggsMoneyFlow {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last > 0.1 {
                    return Ok(IndicatorSignal::Bullish);
                } else if last < -0.1 {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(values
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    IndicatorSignal::Neutral
                } else if v > 0.1 {
                    IndicatorSignal::Bullish
                } else if v < -0.1 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twiggs_money_flow() {
        let tmf = TwiggsMoneyFlow::new(5);
        let n = 15;
        // Uptrend with closes near highs (bullish)
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 104.0 + i as f64).collect(); // Close near high
        let volume: Vec<f64> = (0..n).map(|_| 1000.0).collect();

        let result = tmf.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), n);
        // Early values should be NaN
        assert!(result[0].is_nan());
        // Later values should be valid and positive (bullish)
        assert!(!result[n - 1].is_nan());
        // With closes near highs, TMF should be positive
        assert!(result[n - 1] > 0.0);
    }
}
