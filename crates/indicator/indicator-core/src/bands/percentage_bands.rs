//! Percentage Bands implementation.
//!
//! Fixed percentage bands above and below a moving average.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Percentage Bands indicator.
///
/// Creates bands at fixed percentage distances above and below a moving average.
/// Similar to Envelope but allows for different percentages above and below.
///
/// - Middle Band: SMA or EMA of close
/// - Upper Band: Middle * (1 + upper_percent)
/// - Lower Band: Middle * (1 - lower_percent)
///
/// This indicator is useful for identifying overbought/oversold conditions
/// and potential mean reversion opportunities.
#[derive(Debug, Clone)]
pub struct PercentageBands {
    /// Period for the moving average.
    period: usize,
    /// Upper band percentage (e.g., 0.03 for 3%).
    upper_percent: f64,
    /// Lower band percentage (e.g., 0.03 for 3%).
    lower_percent: f64,
    /// Use EMA instead of SMA.
    use_ema: bool,
}

impl PercentageBands {
    /// Create a new PercentageBands indicator with symmetric bands.
    ///
    /// # Arguments
    /// * `period` - Period for the moving average
    /// * `percent` - Percentage distance from MA (same for upper and lower)
    /// * `use_ema` - Use EMA if true, SMA if false
    pub fn new(period: usize, percent: f64, use_ema: bool) -> Self {
        Self {
            period,
            upper_percent: percent,
            lower_percent: percent,
            use_ema,
        }
    }

    /// Create with asymmetric bands.
    ///
    /// # Arguments
    /// * `period` - Period for the moving average
    /// * `upper_percent` - Upper band percentage
    /// * `lower_percent` - Lower band percentage
    /// * `use_ema` - Use EMA if true, SMA if false
    pub fn asymmetric(period: usize, upper_percent: f64, lower_percent: f64, use_ema: bool) -> Self {
        Self {
            period,
            upper_percent,
            lower_percent,
            use_ema,
        }
    }

    /// Create with SMA and symmetric bands.
    pub fn with_sma(period: usize, percent: f64) -> Self {
        Self::new(period, percent, false)
    }

    /// Create with EMA and symmetric bands.
    pub fn with_ema(period: usize, percent: f64) -> Self {
        Self::new(period, percent, true)
    }

    /// Calculate SMA values.
    fn calculate_sma(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];
        let mut sum: f64 = data[0..self.period].iter().sum();
        result.push(sum / self.period as f64);

        for i in self.period..n {
            sum = sum - data[i - self.period] + data[i];
            result.push(sum / self.period as f64);
        }

        result
    }

    /// Calculate EMA values.
    fn calculate_ema(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let alpha = 2.0 / (self.period as f64 + 1.0);
        let mut result = vec![f64::NAN; self.period - 1];

        // Initial SMA as seed
        let initial_sma: f64 = data[0..self.period].iter().sum::<f64>() / self.period as f64;
        result.push(initial_sma);

        // EMA calculation
        let mut ema = initial_sma;
        for i in self.period..n {
            ema = alpha * data[i] + (1.0 - alpha) * ema;
            result.push(ema);
        }

        result
    }

    /// Calculate Percentage Bands (middle, upper, lower).
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();

        // Calculate moving average (middle band)
        let middle = if self.use_ema {
            self.calculate_ema(close)
        } else {
            self.calculate_sma(close)
        };

        // Calculate upper and lower bands
        let mut upper = Vec::with_capacity(n);
        let mut lower = Vec::with_capacity(n);

        for i in 0..n {
            if middle[i].is_nan() {
                upper.push(f64::NAN);
                lower.push(f64::NAN);
            } else {
                upper.push(middle[i] * (1.0 + self.upper_percent));
                lower.push(middle[i] * (1.0 - self.lower_percent));
            }
        }

        (middle, upper, lower)
    }

    /// Calculate %B (position within bands).
    pub fn percent_b(&self, close: &[f64]) -> Vec<f64> {
        let (_, upper, lower) = self.calculate(close);
        close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if u.is_nan() || l.is_nan() || (u - l).abs() < 1e-10 {
                    f64::NAN
                } else {
                    (price - l) / (u - l)
                }
            })
            .collect()
    }
}

impl Default for PercentageBands {
    fn default() -> Self {
        Self::with_sma(20, 0.025)
    }
}

impl TechnicalIndicator for PercentageBands {
    fn name(&self) -> &str {
        "PercentageBands"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for PercentageBands {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (_, upper, lower) = self.calculate(&data.close);

        let n = data.close.len();
        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let price = data.close[n - 1];
        let u = upper[n - 1];
        let l = lower[n - 1];

        if u.is_nan() || l.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Price at/below lower band = potential buy (oversold)
        if price <= l {
            Ok(IndicatorSignal::Bullish)
        }
        // Price at/above upper band = potential sell (overbought)
        else if price >= u {
            Ok(IndicatorSignal::Bearish)
        }
        else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (_, upper, lower) = self.calculate(&data.close);

        let signals: Vec<_> = data.close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if u.is_nan() || l.is_nan() {
                    IndicatorSignal::Neutral
                } else if price <= l {
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

    #[test]
    fn test_percentage_bands_sma() {
        let pb = PercentageBands::with_sma(10, 0.02);
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (middle, upper, lower) = pb.calculate(&close);

        assert_eq!(middle.len(), 30);
        assert_eq!(upper.len(), 30);
        assert_eq!(lower.len(), 30);

        // Check warmup period
        for i in 0..9 {
            assert!(middle[i].is_nan());
        }

        // Check bands after warmup
        for i in 9..30 {
            assert!(!middle[i].is_nan(), "Middle should not be NaN at index {}", i);
            assert!(upper[i] > middle[i], "Upper should be above middle at index {}", i);
            assert!(lower[i] < middle[i], "Lower should be below middle at index {}", i);
            // Check percentage relationship
            let expected_upper = middle[i] * 1.02;
            let expected_lower = middle[i] * 0.98;
            assert!((upper[i] - expected_upper).abs() < 1e-10);
            assert!((lower[i] - expected_lower).abs() < 1e-10);
        }
    }

    #[test]
    fn test_percentage_bands_ema() {
        let pb = PercentageBands::with_ema(10, 0.03);
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (middle, upper, lower) = pb.calculate(&close);

        // Check bands after warmup
        for i in 9..30 {
            assert!(!middle[i].is_nan());
            assert!(upper[i] > middle[i]);
            assert!(lower[i] < middle[i]);
        }
    }

    #[test]
    fn test_percentage_bands_asymmetric() {
        let pb = PercentageBands::asymmetric(10, 0.03, 0.02, false);
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (middle, upper, lower) = pb.calculate(&close);

        // Check asymmetric bands after warmup
        for i in 9..30 {
            if !middle[i].is_nan() {
                let expected_upper = middle[i] * 1.03;
                let expected_lower = middle[i] * 0.98;
                assert!((upper[i] - expected_upper).abs() < 1e-10);
                assert!((lower[i] - expected_lower).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_percentage_bands_percent_b() {
        let pb = PercentageBands::with_sma(10, 0.05);
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let percent_b = pb.percent_b(&close);

        assert_eq!(percent_b.len(), 30);

        // After warmup, %B should be between reasonable bounds
        for i in 9..30 {
            if !percent_b[i].is_nan() {
                // %B can be outside 0-1 if price is outside bands
                assert!(percent_b[i].is_finite());
            }
        }
    }

    #[test]
    fn test_percentage_bands_default() {
        let pb = PercentageBands::default();
        assert_eq!(pb.period, 20);
        assert!((pb.upper_percent - 0.025).abs() < 1e-10);
        assert!((pb.lower_percent - 0.025).abs() < 1e-10);
        assert!(!pb.use_ema);
    }

    #[test]
    fn test_percentage_bands_signal() {
        let pb = PercentageBands::with_sma(5, 0.02);

        // Create data where price is at lower band
        let close = vec![100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0];
        let (_, _, lower) = pb.calculate(&close);

        // Create OHLCVSeries
        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 1.0).collect(),
            low: close.iter().map(|x| x - 1.0).collect(),
            close: close.clone(),
            volume: vec![1000.0; 8],
        };

        let signal = pb.signal(&data).unwrap();

        // Last price should be close to or below lower band (strong downtrend)
        let last_price = close[close.len() - 1];
        let last_lower = lower[lower.len() - 1];

        if last_price <= last_lower {
            assert_eq!(signal, IndicatorSignal::Bullish);
        }
    }

    #[test]
    fn test_percentage_bands_signals() {
        let pb = PercentageBands::with_sma(3, 0.02);
        let close = vec![100.0, 100.0, 100.0, 100.0, 100.0];

        let data = OHLCVSeries {
            open: close.clone(),
            high: close.clone(),
            low: close.clone(),
            close: close.clone(),
            volume: vec![1000.0; 5],
        };

        let signals = pb.signals(&data).unwrap();
        assert_eq!(signals.len(), 5);

        // With constant prices, signals should be neutral after warmup
        for signal in &signals[2..] {
            assert_eq!(*signal, IndicatorSignal::Neutral);
        }
    }

    #[test]
    fn test_percentage_bands_insufficient_data() {
        let pb = PercentageBands::with_sma(20, 0.02);

        let data = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![101.0; 10],
            low: vec![99.0; 10],
            close: vec![100.0; 10],
            volume: vec![1000.0; 10],
        };

        let result = pb.compute(&data);
        assert!(result.is_err());

        match result {
            Err(IndicatorError::InsufficientData { required, got }) => {
                assert_eq!(required, 20);
                assert_eq!(got, 10);
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }
}
