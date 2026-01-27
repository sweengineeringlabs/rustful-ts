//! VWAP Bands implementation.
//!
//! Standard deviation bands around Volume Weighted Average Price (VWAP).

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// VWAP Bands indicator.
///
/// Creates standard deviation bands around VWAP (Volume Weighted Average Price).
/// Similar to Bollinger Bands but centered on VWAP instead of SMA.
///
/// - Middle Band: Cumulative VWAP
/// - Upper Band: VWAP + (std_dev_multiplier * rolling standard deviation)
/// - Lower Band: VWAP - (std_dev_multiplier * rolling standard deviation)
///
/// VWAP represents the average price weighted by volume, commonly used by
/// institutional traders as a benchmark. The bands help identify when price
/// deviates significantly from this fair value.
#[derive(Debug, Clone)]
pub struct VWAPBands {
    /// Multiplier for standard deviation bands.
    std_dev_multiplier: f64,
    /// Number of bands to calculate (1, 2, or 3 standard deviations).
    num_bands: usize,
}

impl VWAPBands {
    /// Create a new VWAPBands indicator.
    ///
    /// # Arguments
    /// * `std_dev_multiplier` - Multiplier for the standard deviation (typically 1.0, 2.0, or 3.0)
    pub fn new(std_dev_multiplier: f64) -> Self {
        Self {
            std_dev_multiplier,
            num_bands: 1,
        }
    }

    /// Create with multiple band levels (e.g., 1, 2, and 3 standard deviations).
    ///
    /// # Arguments
    /// * `num_bands` - Number of band levels (1-3)
    pub fn with_multiple_bands(num_bands: usize) -> Self {
        Self {
            std_dev_multiplier: 1.0,
            num_bands: num_bands.min(3).max(1),
        }
    }

    /// Calculate VWAP.
    fn calculate_vwap(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut vwap = Vec::with_capacity(n);

        let mut cum_tp_vol = 0.0;
        let mut cum_vol = 0.0;

        for i in 0..n {
            let typical_price = (high[i] + low[i] + close[i]) / 3.0;
            cum_tp_vol += typical_price * volume[i];
            cum_vol += volume[i];

            vwap.push(if cum_vol > 0.0 {
                cum_tp_vol / cum_vol
            } else {
                typical_price
            });
        }

        vwap
    }

    /// Calculate rolling standard deviation of typical price from VWAP.
    fn calculate_std_dev(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64], vwap: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut std_dev = Vec::with_capacity(n);

        let mut cum_vol = 0.0;
        let mut cum_sq_dev_vol = 0.0;

        for i in 0..n {
            let typical_price = (high[i] + low[i] + close[i]) / 3.0;
            let deviation = typical_price - vwap[i];

            cum_sq_dev_vol += deviation * deviation * volume[i];
            cum_vol += volume[i];

            if cum_vol > 0.0 {
                let variance = cum_sq_dev_vol / cum_vol;
                std_dev.push(variance.sqrt());
            } else {
                std_dev.push(0.0);
            }
        }

        std_dev
    }

    /// Calculate VWAP Bands (vwap, upper, lower).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        if n == 0 {
            return (vec![], vec![], vec![]);
        }

        let vwap = self.calculate_vwap(high, low, close, volume);
        let std_dev = self.calculate_std_dev(high, low, close, volume, &vwap);

        let mut upper = Vec::with_capacity(n);
        let mut lower = Vec::with_capacity(n);

        for i in 0..n {
            let band_width = self.std_dev_multiplier * std_dev[i];
            upper.push(vwap[i] + band_width);
            lower.push(vwap[i] - band_width);
        }

        (vwap, upper, lower)
    }

    /// Calculate multiple band levels (1, 2, 3 standard deviations).
    /// Returns (vwap, upper1, lower1, upper2, lower2, upper3, lower3).
    pub fn calculate_multi(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64])
        -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)
    {
        let n = close.len();
        if n == 0 {
            return (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        }

        let vwap = self.calculate_vwap(high, low, close, volume);
        let std_dev = self.calculate_std_dev(high, low, close, volume, &vwap);

        let mut upper1 = Vec::with_capacity(n);
        let mut lower1 = Vec::with_capacity(n);
        let mut upper2 = Vec::with_capacity(n);
        let mut lower2 = Vec::with_capacity(n);
        let mut upper3 = Vec::with_capacity(n);
        let mut lower3 = Vec::with_capacity(n);

        for i in 0..n {
            let sd = std_dev[i];
            upper1.push(vwap[i] + sd);
            lower1.push(vwap[i] - sd);
            upper2.push(vwap[i] + 2.0 * sd);
            lower2.push(vwap[i] - 2.0 * sd);
            upper3.push(vwap[i] + 3.0 * sd);
            lower3.push(vwap[i] - 3.0 * sd);
        }

        (vwap, upper1, lower1, upper2, lower2, upper3, lower3)
    }

    /// Calculate %B (position within bands).
    pub fn percent_b(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let (_, upper, lower) = self.calculate(high, low, close, volume);
        close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if (u - l).abs() < 1e-10 {
                    0.5
                } else {
                    (price - l) / (u - l)
                }
            })
            .collect()
    }
}

impl Default for VWAPBands {
    fn default() -> Self {
        Self::new(2.0)
    }
}

impl TechnicalIndicator for VWAPBands {
    fn name(&self) -> &str {
        "VWAPBands"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let (vwap, upper, lower) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(vwap, upper, lower))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for VWAPBands {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (vwap, upper, lower) = self.calculate(&data.high, &data.low, &data.close, &data.volume);

        let n = data.close.len();
        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let price = data.close[n - 1];
        let v = vwap[n - 1];
        let u = upper[n - 1];
        let l = lower[n - 1];

        // Strong signals at band extremes
        if price <= l {
            Ok(IndicatorSignal::Bullish)
        } else if price >= u {
            Ok(IndicatorSignal::Bearish)
        }
        // Mild signals based on VWAP crossing
        else if price < v {
            Ok(IndicatorSignal::Neutral) // Below VWAP but above lower band
        } else {
            Ok(IndicatorSignal::Neutral) // Above VWAP but below upper band
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (_, upper, lower) = self.calculate(&data.high, &data.low, &data.close, &data.volume);

        let signals: Vec<_> = data.close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if price <= l {
                    IndicatorSignal::Bullish
                } else if price >= u {
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

    fn create_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let volume: Vec<f64> = (0..n).map(|i| 1000.0 + (i as f64 * 0.2).sin() * 500.0).collect();
        (high, low, close, volume)
    }

    #[test]
    fn test_vwap_bands_basic() {
        let vb = VWAPBands::new(2.0);
        let (high, low, close, volume) = create_test_data(30);

        let (vwap, upper, lower) = vb.calculate(&high, &low, &close, &volume);

        assert_eq!(vwap.len(), 30);
        assert_eq!(upper.len(), 30);
        assert_eq!(lower.len(), 30);

        // Check band relationships
        for i in 0..30 {
            assert!(upper[i] >= vwap[i], "Upper should be >= VWAP at index {}", i);
            assert!(lower[i] <= vwap[i], "Lower should be <= VWAP at index {}", i);
        }
    }

    #[test]
    fn test_vwap_bands_first_value() {
        let vb = VWAPBands::new(2.0);
        let high = vec![105.0];
        let low = vec![95.0];
        let close = vec![100.0];
        let volume = vec![1000.0];

        let (vwap, upper, lower) = vb.calculate(&high, &low, &close, &volume);

        // First VWAP equals first typical price
        let tp = (105.0 + 95.0 + 100.0) / 3.0;
        assert!((vwap[0] - tp).abs() < 1e-10);

        // First std_dev is 0, so bands equal VWAP
        assert!((upper[0] - vwap[0]).abs() < 1e-10);
        assert!((lower[0] - vwap[0]).abs() < 1e-10);
    }

    #[test]
    fn test_vwap_bands_multi() {
        let vb = VWAPBands::with_multiple_bands(3);
        let (high, low, close, volume) = create_test_data(30);

        let (vwap, u1, l1, u2, l2, u3, l3) = vb.calculate_multi(&high, &low, &close, &volume);

        // Check that bands expand with each level
        for i in 1..30 {
            // Skip first value where std_dev is 0
            if (u1[i] - vwap[i]).abs() > 1e-10 {
                assert!(u2[i] > u1[i], "2nd upper band should be above 1st at index {}", i);
                assert!(u3[i] > u2[i], "3rd upper band should be above 2nd at index {}", i);
                assert!(l2[i] < l1[i], "2nd lower band should be below 1st at index {}", i);
                assert!(l3[i] < l2[i], "3rd lower band should be below 2nd at index {}", i);
            }
        }
    }

    #[test]
    fn test_vwap_bands_percent_b() {
        let vb = VWAPBands::new(2.0);
        let (high, low, close, volume) = create_test_data(30);

        let percent_b = vb.percent_b(&high, &low, &close, &volume);

        assert_eq!(percent_b.len(), 30);

        // %B should be finite
        for pb in &percent_b {
            assert!(pb.is_finite());
        }
    }

    #[test]
    fn test_vwap_bands_default() {
        let vb = VWAPBands::default();
        assert!((vb.std_dev_multiplier - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_vwap_bands_empty() {
        let vb = VWAPBands::new(2.0);
        let (vwap, upper, lower) = vb.calculate(&[], &[], &[], &[]);

        assert!(vwap.is_empty());
        assert!(upper.is_empty());
        assert!(lower.is_empty());
    }

    #[test]
    fn test_vwap_bands_signal_bullish() {
        let vb = VWAPBands::new(1.0);

        // Create data where price drops significantly below VWAP
        let high = vec![105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 90.0];
        let low = vec![95.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0, 87.0, 80.0];
        let close = vec![100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 85.0];
        let volume = vec![1000.0; 10];

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let signals = vb.signals(&data).unwrap();
        assert_eq!(signals.len(), 10);

        // Last price is far below VWAP, should trigger bullish at some point
        let (_, _, lower) = vb.calculate(&data.high, &data.low, &data.close, &data.volume);
        let last_price = data.close[9];
        let last_lower = lower[9];

        if last_price <= last_lower {
            assert_eq!(signals[9], IndicatorSignal::Bullish);
        }
    }

    #[test]
    fn test_vwap_bands_signal_bearish() {
        let vb = VWAPBands::new(1.0);

        // Create data where price rises significantly above VWAP
        let high = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 115.0];
        let low = vec![90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 105.0];
        let close = vec![95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 110.0];
        let volume = vec![1000.0; 10];

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let signals = vb.signals(&data).unwrap();

        // Last price is far above VWAP, should trigger bearish
        let (_, upper, _) = vb.calculate(&data.high, &data.low, &data.close, &data.volume);
        let last_price = data.close[9];
        let last_upper = upper[9];

        if last_price >= last_upper {
            assert_eq!(signals[9], IndicatorSignal::Bearish);
        }
    }

    #[test]
    fn test_vwap_bands_compute() {
        let vb = VWAPBands::new(2.0);
        let (high, low, close, volume) = create_test_data(30);

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let output = vb.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_vwap_bands_insufficient_data() {
        let vb = VWAPBands::new(2.0);

        let data = OHLCVSeries {
            open: vec![],
            high: vec![],
            low: vec![],
            close: vec![],
            volume: vec![],
        };

        let result = vb.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_vwap_bands_zero_volume() {
        let vb = VWAPBands::new(2.0);
        let high = vec![105.0, 106.0, 107.0];
        let low = vec![95.0, 96.0, 97.0];
        let close = vec![100.0, 101.0, 102.0];
        let volume = vec![0.0, 0.0, 0.0];

        let (vwap, _, _) = vb.calculate(&high, &low, &close, &volume);

        // With zero volume, VWAP falls back to typical price
        for i in 0..3 {
            let tp = (high[i] + low[i] + close[i]) / 3.0;
            assert!((vwap[i] - tp).abs() < 1e-10);
        }
    }
}
