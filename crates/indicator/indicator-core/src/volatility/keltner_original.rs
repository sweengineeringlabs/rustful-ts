//! Original Keltner Channel implementation.
//!
//! The original Keltner Channel developed by Chester Keltner in the 1960s,
//! which uses a 10-day simple moving average of typical price with ATR-based bands.
//! This differs from the modern Keltner Channel which uses EMA and ATR multipliers.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

/// Original Keltner Channel.
///
/// Chester Keltner's original channel uses:
/// - Middle Band: 10-day SMA of Typical Price (High + Low + Close) / 3
/// - Upper Band: Middle Band + Average True Range
/// - Lower Band: Middle Band - Average True Range
///
/// The original version uses a simple ATR (average of True Range) rather than
/// Wilder's smoothed ATR used in modern implementations.
///
/// # Trading Signals
/// - Price above upper band: Strong uptrend (Bullish)
/// - Price below lower band: Strong downtrend (Bearish)
/// - Price within bands: Normal conditions (Neutral)
#[derive(Debug, Clone)]
pub struct KeltnerOriginal {
    /// Period for SMA and ATR calculations (traditionally 10).
    period: usize,
}

impl KeltnerOriginal {
    /// Create a new Original Keltner Channel indicator.
    ///
    /// # Arguments
    /// * `period` - Period for calculations (Chester Keltner used 10)
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Create with default parameters (10-day).
    pub fn default_params() -> Self {
        Self::new(10)
    }

    /// Calculate True Range for each bar.
    fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n == 0 {
            return vec![];
        }

        let mut tr = Vec::with_capacity(n);
        tr.push(high[0] - low[0]); // First TR is just high-low

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr.push(hl.max(hc).max(lc));
        }

        tr
    }

    /// Calculate Original Keltner Channel values.
    ///
    /// Returns (middle_band, upper_band, lower_band).
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n < self.period || self.period == 0 {
            return (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate Typical Price: (High + Low + Close) / 3
        let typical_price: Vec<f64> = high
            .iter()
            .zip(low.iter())
            .zip(close.iter())
            .map(|((&h, &l), &c)| (h + l + c) / 3.0)
            .collect();

        // Calculate True Range
        let tr = Self::true_range(high, low, close);

        let mut middle = vec![f64::NAN; self.period - 1];
        let mut upper = vec![f64::NAN; self.period - 1];
        let mut lower = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;

            // SMA of Typical Price
            let tp_sum: f64 = typical_price[start..=i].iter().sum();
            let middle_band = tp_sum / self.period as f64;

            // Simple average of True Range (not Wilder's smoothed ATR)
            let tr_sum: f64 = tr[start..=i].iter().sum();
            let avg_tr = tr_sum / self.period as f64;

            middle.push(middle_band);
            upper.push(middle_band + avg_tr);
            lower.push(middle_band - avg_tr);
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for KeltnerOriginal {
    fn name(&self) -> &str {
        "KeltnerOriginal"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        3 // middle, upper, lower
    }
}

impl SignalIndicator for KeltnerOriginal {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (_, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        let n = data.close.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let close = data.close[n - 1];
        let upper_val = upper[n - 1];
        let lower_val = lower[n - 1];

        if upper_val.is_nan() || lower_val.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Price above upper band = strong uptrend
        if close > upper_val {
            Ok(IndicatorSignal::Bullish)
        }
        // Price below lower band = strong downtrend
        else if close < lower_val {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (_, upper, lower) = self.calculate(&data.high, &data.low, &data.close);

        let signals = data
            .close
            .iter()
            .zip(upper.iter())
            .zip(lower.iter())
            .map(|((&close, &up), &lo)| {
                if up.is_nan() || lo.is_nan() {
                    IndicatorSignal::Neutral
                } else if close > up {
                    IndicatorSignal::Bullish
                } else if close < lo {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keltner_original() {
        let keltner = KeltnerOriginal::new(10);

        // Generate sample OHLC data
        let high: Vec<f64> = (0..50)
            .map(|i| 105.0 + (i as f64 * 0.1).sin() * 3.0)
            .collect();
        let low: Vec<f64> = (0..50)
            .map(|i| 95.0 + (i as f64 * 0.1).sin() * 3.0)
            .collect();
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 3.0)
            .collect();

        let (middle, upper, lower) = keltner.calculate(&high, &low, &close);

        assert_eq!(middle.len(), 50);
        assert_eq!(upper.len(), 50);
        assert_eq!(lower.len(), 50);

        // First 9 values should be NaN
        for i in 0..9 {
            assert!(middle[i].is_nan());
            assert!(upper[i].is_nan());
            assert!(lower[i].is_nan());
        }

        // Upper should be above middle, lower should be below middle
        for i in 9..50 {
            assert!(
                upper[i] > middle[i],
                "Upper band should be above middle at index {}",
                i
            );
            assert!(
                lower[i] < middle[i],
                "Lower band should be below middle at index {}",
                i
            );
        }
    }

    #[test]
    fn test_keltner_original_default() {
        let keltner = KeltnerOriginal::default_params();
        assert_eq!(keltner.period, 10);
    }

    #[test]
    fn test_keltner_original_bands_symmetric() {
        let keltner = KeltnerOriginal::new(10);

        // Generate sample OHLC data
        let high: Vec<f64> = (0..30).map(|i| 102.0 + i as f64 * 0.1).collect();
        let low: Vec<f64> = (0..30).map(|i| 98.0 + i as f64 * 0.1).collect();
        let close: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.1).collect();

        let (middle, upper, lower) = keltner.calculate(&high, &low, &close);

        // Bands should be symmetric around middle
        for i in 9..30 {
            let upper_dist = upper[i] - middle[i];
            let lower_dist = middle[i] - lower[i];
            assert!(
                (upper_dist - lower_dist).abs() < 1e-10,
                "Bands should be symmetric at index {}",
                i
            );
        }
    }

    #[test]
    fn test_keltner_original_signal() {
        let keltner = KeltnerOriginal::new(10);

        // Generate data where price breaks above upper band
        let mut high: Vec<f64> = vec![102.0; 20];
        let mut low: Vec<f64> = vec![98.0; 20];
        let mut close: Vec<f64> = vec![100.0; 20];

        // Last value has price spike above upper band
        high[19] = 120.0;
        close[19] = 115.0;

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 20],
        };

        let signal = keltner.signal(&series).unwrap();
        assert_eq!(signal, IndicatorSignal::Bullish);
    }

    #[test]
    fn test_keltner_original_technical_indicator() {
        let keltner = KeltnerOriginal::new(10);
        assert_eq!(keltner.name(), "KeltnerOriginal");
        assert_eq!(keltner.min_periods(), 10);
        assert_eq!(keltner.output_features(), 3);
    }

    #[test]
    fn test_keltner_original_insufficient_data() {
        let keltner = KeltnerOriginal::new(10);

        let series = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![102.0; 5],
            low: vec![98.0; 5],
            close: vec![100.0; 5],
            volume: vec![1000.0; 5],
        };

        let result = keltner.compute(&series);
        assert!(result.is_err());
    }
}
