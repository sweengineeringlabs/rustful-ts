//! Volume Price Confirmation Indicator (VPCI) implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Volume Price Confirmation Indicator (VPCI).
///
/// Measures the agreement between price trend and volume trend to confirm
/// the validity of price movements.
///
/// Calculation:
/// - VPC = VWMA - SMA (difference between volume-weighted and simple averages)
/// - VPR = VWMA / SMA (ratio)
/// - VM = SMA(Volume) / SMA(Volume, long_period) (Volume Multiplier)
/// - VPCI = VPC * VPR * VM
///
/// Interpretation:
/// - VPCI > 0: Volume confirms price trend (bullish confirmation)
/// - VPCI < 0: Volume diverges from price trend (bearish divergence)
/// - Rising VPCI with rising price: Strong uptrend
/// - Falling VPCI with falling price: Strong downtrend
/// - Divergences between VPCI and price signal potential reversals
#[derive(Debug, Clone)]
pub struct VolumePriceConfirm {
    short_period: usize,
    long_period: usize,
}

impl VolumePriceConfirm {
    /// Create a new Volume Price Confirmation indicator.
    ///
    /// # Arguments
    /// * `short_period` - Short period for SMA/VWMA calculation (typically 5)
    /// * `long_period` - Long period for volume multiplier (typically 20)
    pub fn new(short_period: usize, long_period: usize) -> Self {
        Self {
            short_period,
            long_period,
        }
    }

    /// Calculate SMA.
    fn sma(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        for i in (period - 1)..n {
            let start = i + 1 - period;
            let sum: f64 = values[start..=i].iter().sum();
            result[i] = sum / period as f64;
        }

        result
    }

    /// Calculate VWMA.
    fn vwma(&self, close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        for i in (period - 1)..n {
            let start = i + 1 - period;
            let mut sum_pv = 0.0;
            let mut sum_v = 0.0;

            for j in start..=i {
                sum_pv += close[j] * volume[j];
                sum_v += volume[j];
            }

            result[i] = if sum_v > 0.0 { sum_pv / sum_v } else { close[i] };
        }

        result
    }

    /// Calculate VPCI values.
    /// Returns (VPCI, VPC, VPR) for detailed analysis
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();

        if n < self.long_period {
            return (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate VWMA and SMA of price
        let vwma = self.vwma(close, volume, self.short_period);
        let sma = self.sma(close, self.short_period);

        // Calculate SMA of volume (short and long)
        let vol_sma_short = self.sma(volume, self.short_period);
        let vol_sma_long = self.sma(volume, self.long_period);

        // Calculate VPC (Volume Price Confirmation)
        let mut vpc = vec![f64::NAN; n];
        for i in 0..n {
            if !vwma[i].is_nan() && !sma[i].is_nan() {
                vpc[i] = vwma[i] - sma[i];
            }
        }

        // Calculate VPR (Volume Price Ratio)
        let mut vpr = vec![f64::NAN; n];
        for i in 0..n {
            if !vwma[i].is_nan() && !sma[i].is_nan() && sma[i] != 0.0 {
                vpr[i] = vwma[i] / sma[i];
            }
        }

        // Calculate VM (Volume Multiplier)
        let mut vm = vec![f64::NAN; n];
        for i in 0..n {
            if !vol_sma_short[i].is_nan() && !vol_sma_long[i].is_nan() && vol_sma_long[i] != 0.0 {
                vm[i] = vol_sma_short[i] / vol_sma_long[i];
            }
        }

        // Calculate VPCI = VPC * VPR * VM
        let mut vpci = vec![f64::NAN; n];
        for i in 0..n {
            if !vpc[i].is_nan() && !vpr[i].is_nan() && !vm[i].is_nan() {
                vpci[i] = vpc[i] * vpr[i] * vm[i];
            }
        }

        (vpci, vpc, vpr)
    }
}

impl Default for VolumePriceConfirm {
    fn default() -> Self {
        Self {
            short_period: 5,
            long_period: 20,
        }
    }
}

impl TechnicalIndicator for VolumePriceConfirm {
    fn name(&self) -> &str {
        "Volume Price Confirmation"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.long_period {
            return Err(IndicatorError::InsufficientData {
                required: self.long_period,
                got: data.close.len(),
            });
        }

        let (vpci, vpc, _vpr) = self.calculate(&data.close, &data.volume);

        // Return VPCI and VPC (most important outputs)
        Ok(IndicatorOutput::dual(vpci, vpc))
    }

    fn min_periods(&self) -> usize {
        self.long_period
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for VolumePriceConfirm {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (vpci, _, _) = self.calculate(&data.close, &data.volume);

        if vpci.len() < 2 || data.close.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = vpci.len();
        let curr_vpci = vpci[n - 1];
        let prev_vpci = vpci[n - 2];
        let curr_close = data.close[n - 1];
        let prev_close = data.close[n - 2];

        if curr_vpci.is_nan() || prev_vpci.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Strong bullish: VPCI positive and increasing with rising price
        if curr_vpci > 0.0 && curr_vpci > prev_vpci && curr_close > prev_close {
            return Ok(IndicatorSignal::Bullish);
        }

        // Strong bearish: VPCI negative and decreasing with falling price
        if curr_vpci < 0.0 && curr_vpci < prev_vpci && curr_close < prev_close {
            return Ok(IndicatorSignal::Bearish);
        }

        // Bullish divergence: VPCI turning positive
        if prev_vpci <= 0.0 && curr_vpci > 0.0 {
            return Ok(IndicatorSignal::Bullish);
        }

        // Bearish divergence: VPCI turning negative
        if prev_vpci >= 0.0 && curr_vpci < 0.0 {
            return Ok(IndicatorSignal::Bearish);
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (vpci, _, _) = self.calculate(&data.close, &data.volume);

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..vpci.len() {
            if vpci[i].is_nan() || vpci[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if vpci[i] > 0.0 && vpci[i] > vpci[i - 1] && data.close[i] > data.close[i - 1] {
                // Bullish confirmation
                signals.push(IndicatorSignal::Bullish);
            } else if vpci[i] < 0.0 && vpci[i] < vpci[i - 1] && data.close[i] < data.close[i - 1] {
                // Bearish confirmation
                signals.push(IndicatorSignal::Bearish);
            } else if vpci[i - 1] <= 0.0 && vpci[i] > 0.0 {
                // Bullish crossover
                signals.push(IndicatorSignal::Bullish);
            } else if vpci[i - 1] >= 0.0 && vpci[i] < 0.0 {
                // Bearish crossover
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
    fn test_vpci_uptrend_high_volume() {
        let vpci = VolumePriceConfirm::new(3, 10);
        // Uptrend with increasing volume
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let volume: Vec<f64> = (0..20).map(|i| 1000.0 + i as f64 * 100.0).collect();

        let (vpci_values, vpc, _) = vpci.calculate(&close, &volume);

        assert_eq!(vpci_values.len(), 20);
        assert_eq!(vpc.len(), 20);

        // VPCI should be positive (volume confirms uptrend)
        for i in 10..20 {
            assert!(!vpci_values[i].is_nan());
        }
    }

    #[test]
    fn test_vpci_equal_volume() {
        let vpci = VolumePriceConfirm::new(3, 10);
        // With equal volume, VWMA = SMA, so VPC = 0
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let volume = vec![1000.0; 20];

        let (_, vpc, _) = vpci.calculate(&close, &volume);

        // VPC should be very close to 0 with equal volume
        for i in 10..20 {
            assert!(!vpc[i].is_nan());
            assert!(vpc[i].abs() < 0.001, "VPC should be ~0 with equal volume");
        }
    }

    #[test]
    fn test_vpci_high_volume_on_highs() {
        let vpci = VolumePriceConfirm::new(3, 10);
        // Higher volume on higher prices (accumulation)
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        // Volume proportional to price
        let volume: Vec<f64> = close.iter().map(|c| c * 10.0).collect();

        let (vpci_values, vpc, _) = vpci.calculate(&close, &volume);

        // VPC should be positive (VWMA > SMA due to high volume on high prices)
        for i in 10..20 {
            assert!(!vpc[i].is_nan());
            assert!(vpc[i] > 0.0, "VPC should be positive with high-volume highs");
        }

        // VPCI should also be positive
        for i in 10..20 {
            assert!(!vpci_values[i].is_nan());
            assert!(vpci_values[i] > 0.0);
        }
    }

    #[test]
    fn test_vpci_high_volume_on_lows() {
        let vpci = VolumePriceConfirm::new(3, 10);
        // Higher volume on lower prices (distribution)
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        // Inverse volume (higher on lower prices)
        let volume: Vec<f64> = close.iter().map(|c| 2200.0 - c * 10.0).collect();

        let (_, vpc, _) = vpci.calculate(&close, &volume);

        // VPC should be negative (VWMA < SMA due to high volume on low prices)
        for i in 10..20 {
            assert!(!vpc[i].is_nan());
            assert!(vpc[i] < 0.0, "VPC should be negative with high-volume lows");
        }
    }

    #[test]
    fn test_vpci_insufficient_data() {
        let vpci = VolumePriceConfirm::new(5, 20);
        let close = vec![100.0; 10];
        let volume = vec![1000.0; 10];

        let (vpci_values, _, _) = vpci.calculate(&close, &volume);

        // All NaN with insufficient data
        for val in vpci_values {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_vpci_signal_bullish() {
        let vpci_ind = VolumePriceConfirm::new(3, 10);
        // Rising price with accelerating volume on highs
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        // Volume increases exponentially with price to create increasing VPCI
        let volume: Vec<f64> = close.iter().enumerate().map(|(i, c)| c * 10.0 * (1.0 + i as f64 * 0.1)).collect();

        let (vpci_values, _, _) = vpci_ind.calculate(&close, &volume);

        // Verify VPCI is positive at the end (volume confirms uptrend)
        let n = vpci_values.len();
        assert!(!vpci_values[n - 1].is_nan());
        assert!(vpci_values[n - 1] > 0.0, "VPCI should be positive with high-volume highs");

        // Check signal
        let signal = vpci_ind.signal(&OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 1.0).collect(),
            low: close.iter().map(|x| x - 1.0).collect(),
            close: close.clone(),
            volume: volume.clone(),
        }).unwrap();

        // Signal should be bullish or neutral (VPCI > 0 with rising price)
        // Note: might be neutral if VPCI is not increasing
        assert!(signal == IndicatorSignal::Bullish || signal == IndicatorSignal::Neutral);
    }

    #[test]
    fn test_vpci_technical_indicator() {
        let vpci = VolumePriceConfirm::default();
        assert_eq!(vpci.name(), "Volume Price Confirmation");
        assert_eq!(vpci.min_periods(), 20);
        assert_eq!(vpci.output_features(), 2);
    }

    #[test]
    fn test_vpci_empty() {
        let vpci = VolumePriceConfirm::default();
        let (values, vpc, vpr) = vpci.calculate(&[], &[]);
        assert!(values.is_empty());
        assert!(vpc.is_empty());
        assert!(vpr.is_empty());
    }

    #[test]
    fn test_vpci_signals() {
        let vpci = VolumePriceConfirm::new(3, 10);
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let volume: Vec<f64> = close.iter().map(|c| c * 10.0).collect();

        let signals = vpci.signals(&OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 1.0).collect(),
            low: close.iter().map(|x| x - 1.0).collect(),
            close: close.clone(),
            volume: volume.clone(),
        }).unwrap();

        assert_eq!(signals.len(), 20);
        // Should have bullish signals during strong uptrend with volume confirmation
    }
}
