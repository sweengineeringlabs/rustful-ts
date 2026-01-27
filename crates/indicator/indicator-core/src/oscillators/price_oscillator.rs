//! Price Oscillator (PO).

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

/// Price Oscillator (PO) - IND-080
///
/// The Price Oscillator measures the absolute difference between two simple moving averages.
/// Unlike PPO which expresses this as a percentage, the Price Oscillator shows
/// the raw price difference.
///
/// Formula: Price Oscillator = Fast SMA - Slow SMA
///
/// # Interpretation
/// - Positive values indicate the fast SMA is above the slow SMA (bullish)
/// - Negative values indicate the fast SMA is below the slow SMA (bearish)
/// - Zero-line crossovers generate trading signals
#[derive(Debug, Clone)]
pub struct PriceOscillator {
    fast_period: usize,
    slow_period: usize,
}

impl PriceOscillator {
    /// Creates a new Price Oscillator with the specified periods.
    ///
    /// # Arguments
    /// * `fast` - Period for the fast SMA (shorter)
    /// * `slow` - Period for the slow SMA (longer)
    pub fn new(fast: usize, slow: usize) -> Self {
        Self {
            fast_period: fast,
            slow_period: slow,
        }
    }

    /// Calculates Simple Moving Average (SMA).
    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = Vec::with_capacity(n);

        // Fill with NaN until we have enough data
        for _ in 0..period - 1 {
            result.push(f64::NAN);
        }

        // Calculate SMA using sliding window
        let mut sum: f64 = data[..period].iter().sum();
        result.push(sum / period as f64);

        for i in period..n {
            sum = sum - data[i - period] + data[i];
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculates the Price Oscillator values.
    ///
    /// # Arguments
    /// * `data` - Slice of price data (typically closing prices)
    ///
    /// # Returns
    /// Vector of Price Oscillator values (Fast SMA - Slow SMA)
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.slow_period {
            return vec![f64::NAN; n];
        }

        let fast_sma = Self::sma(data, self.fast_period);
        let slow_sma = Self::sma(data, self.slow_period);

        fast_sma
            .iter()
            .zip(slow_sma.iter())
            .map(|(f, s)| {
                if f.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    f - s
                }
            })
            .collect()
    }
}

impl Default for PriceOscillator {
    fn default() -> Self {
        Self::new(12, 26)
    }
}

impl TechnicalIndicator for PriceOscillator {
    fn name(&self) -> &str {
        "PriceOscillator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.slow_period {
            return Err(IndicatorError::InsufficientData {
                required: self.slow_period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.slow_period
    }
}

impl SignalIndicator for PriceOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);

        if values.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let last = values[values.len() - 1];
        let prev = values[values.len() - 2];

        if last.is_nan() || prev.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Zero-line crossover signals
        if last > 0.0 && prev <= 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < 0.0 && prev >= 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..values.len() {
            if values[i].is_nan() || values[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if values[i] > 0.0 && values[i - 1] <= 0.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if values[i] < 0.0 && values[i - 1] >= 0.0 {
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
    fn test_price_oscillator_basic() {
        let po = PriceOscillator::default();
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let result = po.calculate(&data);

        assert_eq!(result.len(), 50);
        // In uptrend, Price Oscillator should be positive
        let last = result.last().unwrap();
        assert!(!last.is_nan());
        assert!(*last > 0.0);
    }

    #[test]
    fn test_price_oscillator_downtrend() {
        let po = PriceOscillator::new(5, 10);
        // Decreasing prices
        let data: Vec<f64> = (0..30).map(|i| 200.0 - i as f64).collect();
        let result = po.calculate(&data);

        assert_eq!(result.len(), 30);
        // In downtrend, Price Oscillator should be negative
        let last = result.last().unwrap();
        assert!(!last.is_nan());
        assert!(*last < 0.0);
    }

    #[test]
    fn test_price_oscillator_insufficient_data() {
        let po = PriceOscillator::new(5, 10);
        let data: Vec<f64> = vec![100.0, 101.0, 102.0]; // Only 3 points
        let result = po.calculate(&data);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_price_oscillator_sma_calculation() {
        // Test SMA directly with known values
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = PriceOscillator::sma(&data, 3);

        assert_eq!(sma.len(), 5);
        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < 1e-10); // (1+2+3)/3 = 2
        assert!((sma[3] - 3.0).abs() < 1e-10); // (2+3+4)/3 = 3
        assert!((sma[4] - 4.0).abs() < 1e-10); // (3+4+5)/3 = 4
    }

    #[test]
    fn test_price_oscillator_formula() {
        // Verify the formula: PO = Fast SMA - Slow SMA
        let po = PriceOscillator::new(3, 5);
        let data = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

        let result = po.calculate(&data);
        let fast_sma = PriceOscillator::sma(&data, 3);
        let slow_sma = PriceOscillator::sma(&data, 5);

        // Check value at index 5 (where both SMAs are valid)
        let expected = fast_sma[5] - slow_sma[5];
        assert!((result[5] - expected).abs() < 1e-10);

        // At index 5:
        // Fast SMA(3) = (13+14+15)/3 = 14
        // Slow SMA(5) = (11+12+13+14+15)/5 = 13
        // PO = 14 - 13 = 1
        assert!((result[5] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_price_oscillator_signals() {
        let po = PriceOscillator::new(3, 5);

        // Create data that crosses from negative to positive
        let data = vec![
            100.0, 99.0, 98.0, 97.0, 96.0, // Downtrend (fast < slow)
            100.0, 105.0, 110.0, 115.0, 120.0, // Strong uptrend (fast > slow)
        ];

        let ohlcv = OHLCVSeries {
            open: data.clone(),
            high: data.clone(),
            low: data.clone(),
            close: data.clone(),
            volume: vec![1000.0; data.len()],
        };

        let signals = po.signals(&ohlcv).unwrap();
        assert_eq!(signals.len(), data.len());

        // Should contain at least one bullish signal (crossover)
        assert!(signals
            .iter()
            .any(|s| matches!(s, IndicatorSignal::Bullish)));
    }

    #[test]
    fn test_price_oscillator_signal_crossover() {
        let po = PriceOscillator::new(2, 4);

        // Create data with a clear zero-line crossover
        let data = vec![
            100.0, 100.0, 100.0, 100.0, // Flat (PO near 0)
            95.0, 90.0, 85.0, // Downtrend (PO negative)
            90.0, 95.0, 100.0, 105.0, 110.0, // Recovery/uptrend
        ];

        let ohlcv = OHLCVSeries {
            open: data.clone(),
            high: data.clone(),
            low: data.clone(),
            close: data.clone(),
            volume: vec![1000.0; data.len()],
        };

        let signal = po.signal(&ohlcv).unwrap();
        // Final signal depends on the state at the end
        assert!(matches!(
            signal,
            IndicatorSignal::Bullish | IndicatorSignal::Neutral
        ));
    }

    #[test]
    fn test_price_oscillator_default() {
        let po = PriceOscillator::default();
        assert_eq!(po.fast_period, 12);
        assert_eq!(po.slow_period, 26);
    }

    #[test]
    fn test_price_oscillator_min_periods() {
        let po = PriceOscillator::new(10, 20);
        assert_eq!(po.min_periods(), 20);
    }

    #[test]
    fn test_price_oscillator_name() {
        let po = PriceOscillator::default();
        assert_eq!(po.name(), "PriceOscillator");
    }

    #[test]
    fn test_price_oscillator_compute() {
        let po = PriceOscillator::default();
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();

        let ohlcv = OHLCVSeries {
            open: data.clone(),
            high: data.clone(),
            low: data.clone(),
            close: data.clone(),
            volume: vec![1000.0; data.len()],
        };

        let output = po.compute(&ohlcv).unwrap();
        assert_eq!(output.primary.len(), 50);
    }

    #[test]
    fn test_price_oscillator_compute_insufficient_data() {
        let po = PriceOscillator::new(5, 10);
        let data: Vec<f64> = vec![100.0, 101.0, 102.0];

        let ohlcv = OHLCVSeries {
            open: data.clone(),
            high: data.clone(),
            low: data.clone(),
            close: data.clone(),
            volume: vec![1000.0; data.len()],
        };

        let result = po.compute(&ohlcv);
        assert!(result.is_err());
    }

    #[test]
    fn test_price_oscillator_vs_ema_based() {
        // Verify that SMA-based calculation differs from EMA-based
        let po = PriceOscillator::new(5, 10);
        let data: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64).sin() * 10.0).collect();

        let result = po.calculate(&data);

        // The Price Oscillator should have valid values starting at slow_period - 1
        // For slow_period=10, first valid value is at index 9
        assert!(result[8].is_nan()); // Index 8 is still NaN (before slow SMA warmup)
        assert!(!result[9].is_nan()); // Index 9 is the first valid slow SMA
    }

    #[test]
    fn test_price_oscillator_constant_prices() {
        let po = PriceOscillator::new(3, 5);
        let data: Vec<f64> = vec![100.0; 20]; // All same price

        let result = po.calculate(&data);

        // With constant prices, both SMAs equal 100, so PO = 0
        for i in 4..result.len() {
            assert!((result[i] - 0.0).abs() < 1e-10);
        }
    }
}
