//! Contango/Backwardation Indicator (IND-413)
//!
//! Measures the shape of the futures curve by analyzing the relationship between
//! near-term and far-term prices. Contango indicates futures prices above spot,
//! while backwardation indicates futures prices below spot.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Futures curve shape classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveShape {
    /// Futures prices above spot (normal market).
    Contango,
    /// Futures prices below spot (inverted market).
    Backwardation,
    /// Flat curve or insufficient data.
    Flat,
}

/// Output for ContangoBackwardation indicator.
#[derive(Debug, Clone, Copy)]
pub struct ContangoBackwardationOutput {
    /// Curve slope value (positive = contango, negative = backwardation).
    pub slope: f64,
    /// Normalized curve shape indicator (-1 to 1).
    pub normalized: f64,
    /// Current curve shape classification.
    pub shape: CurveShape,
    /// Strength of the curve shape (0 to 100).
    pub strength: f64,
}

/// Contango/Backwardation Indicator (IND-413)
///
/// Analyzes the futures curve shape by comparing current prices to a moving average
/// as a proxy for the term structure. In commodity markets:
/// - Contango: Near-month futures < far-month futures (curve slopes upward)
/// - Backwardation: Near-month futures > far-month futures (curve slopes downward)
///
/// # Algorithm
/// 1. Calculate short-term and long-term moving averages
/// 2. Compute slope as the difference between current price and long-term MA
/// 3. Normalize the slope relative to price volatility
/// 4. Classify curve shape based on slope direction and magnitude
///
/// # Interpretation
/// - Positive slope indicates contango (carry cost > convenience yield)
/// - Negative slope indicates backwardation (convenience yield > carry cost)
/// - Stronger readings suggest more pronounced curve shape
///
/// # Example
/// ```ignore
/// let cb = ContangoBackwardation::new(5, 20, 0.5)?;
/// let output = cb.compute(&data)?;
/// ```
#[derive(Debug, Clone)]
pub struct ContangoBackwardation {
    /// Short-term period for near-month proxy.
    short_period: usize,
    /// Long-term period for far-month proxy.
    long_period: usize,
    /// Threshold for classifying as flat (percentage).
    flat_threshold: f64,
}

impl ContangoBackwardation {
    /// Create a new ContangoBackwardation indicator.
    ///
    /// # Arguments
    /// * `short_period` - Period for short-term average (near-month proxy)
    /// * `long_period` - Period for long-term average (far-month proxy)
    /// * `flat_threshold` - Percentage threshold for flat classification
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid.
    pub fn new(short_period: usize, long_period: usize, flat_threshold: f64) -> Result<Self> {
        if short_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if flat_threshold < 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "flat_threshold".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        Ok(Self {
            short_period,
            long_period,
            flat_threshold,
        })
    }

    /// Create with default parameters (5, 20, 0.5).
    pub fn default_params() -> Result<Self> {
        Self::new(5, 20, 0.5)
    }

    /// Calculate the contango/backwardation values.
    pub fn calculate(&self, close: &[f64]) -> Vec<Option<ContangoBackwardationOutput>> {
        let n = close.len();
        let mut result = vec![None; n];

        if n < self.long_period {
            return result;
        }

        // Calculate moving averages
        let short_ma = self.calculate_ma(close, self.short_period);
        let long_ma = self.calculate_ma(close, self.long_period);

        // Calculate volatility for normalization
        let volatility = self.calculate_volatility(close, self.long_period);

        for i in (self.long_period - 1)..n {
            let short = short_ma[i];
            let long = long_ma[i];
            let vol = volatility[i];

            if short.is_nan() || long.is_nan() || long == 0.0 {
                continue;
            }

            // Slope: positive = contango (current > historical avg)
            let slope = ((short - long) / long) * 100.0;

            // Normalize by volatility
            let normalized = if vol > 0.0 {
                (slope / vol).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            // Classify shape
            let shape = if slope.abs() < self.flat_threshold {
                CurveShape::Flat
            } else if slope > 0.0 {
                CurveShape::Contango
            } else {
                CurveShape::Backwardation
            };

            // Calculate strength (0-100)
            let strength = (slope.abs() / (self.flat_threshold * 10.0) * 100.0).min(100.0);

            result[i] = Some(ContangoBackwardationOutput {
                slope,
                normalized,
                shape,
                strength,
            });
        }

        result
    }

    /// Calculate simple moving average.
    fn calculate_ma(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let mut sum = 0.0;
        for i in 0..period {
            sum += data[i];
        }
        result[period - 1] = sum / period as f64;

        for i in period..n {
            sum = sum - data[i - period] + data[i];
            result[i] = sum / period as f64;
        }

        result
    }

    /// Calculate rolling volatility (standard deviation of returns).
    fn calculate_volatility(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < period + 1 {
            return result;
        }

        for i in period..n {
            let mut returns = Vec::with_capacity(period);
            for j in (i - period + 1)..=i {
                if data[j - 1] > 0.0 {
                    returns.push((data[j] / data[j - 1] - 1.0) * 100.0);
                }
            }

            if returns.is_empty() {
                continue;
            }

            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / returns.len() as f64;
            result[i] = variance.sqrt();
        }

        result
    }
}

impl Default for ContangoBackwardation {
    fn default() -> Self {
        Self::default_params().unwrap()
    }
}

impl TechnicalIndicator for ContangoBackwardation {
    fn name(&self) -> &str {
        "ContangoBackwardation"
    }

    fn min_periods(&self) -> usize {
        self.long_period
    }

    fn output_features(&self) -> usize {
        2 // slope, strength
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }

        let outputs = self.calculate(&data.close);

        let primary: Vec<f64> = outputs
            .iter()
            .map(|o| o.map(|v| v.slope).unwrap_or(f64::NAN))
            .collect();

        let secondary: Vec<f64> = outputs
            .iter()
            .map(|o| o.map(|v| v.strength).unwrap_or(f64::NAN))
            .collect();

        Ok(IndicatorOutput::dual(primary, secondary))
    }
}

impl SignalIndicator for ContangoBackwardation {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let outputs = self.calculate(&data.close);

        match outputs.last().and_then(|o| *o) {
            Some(out) => match out.shape {
                CurveShape::Contango => Ok(IndicatorSignal::Bearish), // Contango = negative roll yield
                CurveShape::Backwardation => Ok(IndicatorSignal::Bullish), // Backwardation = positive roll yield
                CurveShape::Flat => Ok(IndicatorSignal::Neutral),
            },
            None => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let outputs = self.calculate(&data.close);

        let signals = outputs
            .iter()
            .map(|o| match o {
                Some(out) => match out.shape {
                    CurveShape::Contango => IndicatorSignal::Bearish,
                    CurveShape::Backwardation => IndicatorSignal::Bullish,
                    CurveShape::Flat => IndicatorSignal::Neutral,
                },
                None => IndicatorSignal::Neutral,
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uptrend_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + i as f64 * 0.5).collect()
    }

    fn make_downtrend_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| 150.0 - i as f64 * 0.5).collect()
    }

    fn make_flat_data(n: usize) -> Vec<f64> {
        vec![100.0; n]
    }

    #[test]
    fn test_new_valid_params() {
        let cb = ContangoBackwardation::new(5, 20, 0.5);
        assert!(cb.is_ok());
    }

    #[test]
    fn test_new_invalid_short_period() {
        let cb = ContangoBackwardation::new(0, 20, 0.5);
        assert!(cb.is_err());
    }

    #[test]
    fn test_new_invalid_long_period() {
        let cb = ContangoBackwardation::new(20, 10, 0.5);
        assert!(cb.is_err());
    }

    #[test]
    fn test_new_invalid_threshold() {
        let cb = ContangoBackwardation::new(5, 20, -0.5);
        assert!(cb.is_err());
    }

    #[test]
    fn test_contango_detection() {
        let data = make_uptrend_data(50);
        let cb = ContangoBackwardation::new(5, 20, 0.5).unwrap();
        let outputs = cb.calculate(&data);

        // After warmup, should detect contango in uptrend
        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Most should be contango
        let contango_count = valid_outputs
            .iter()
            .filter(|o| o.shape == CurveShape::Contango)
            .count();
        assert!(contango_count > valid_outputs.len() / 2);
    }

    #[test]
    fn test_backwardation_detection() {
        let data = make_downtrend_data(50);
        let cb = ContangoBackwardation::new(5, 20, 0.5).unwrap();
        let outputs = cb.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Most should be backwardation
        let backwardation_count = valid_outputs
            .iter()
            .filter(|o| o.shape == CurveShape::Backwardation)
            .count();
        assert!(backwardation_count > valid_outputs.len() / 2);
    }

    #[test]
    fn test_flat_detection() {
        let data = make_flat_data(50);
        let cb = ContangoBackwardation::new(5, 20, 0.5).unwrap();
        let outputs = cb.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();

        // All should be flat
        for out in valid_outputs {
            assert_eq!(out.shape, CurveShape::Flat);
        }
    }

    #[test]
    fn test_technical_indicator_impl() {
        let data = make_uptrend_data(50);
        let cb = ContangoBackwardation::default();
        let ohlcv = OHLCVSeries::from_close(data);

        let result = cb.compute(&ohlcv);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let data = make_uptrend_data(50);
        let cb = ContangoBackwardation::default();
        let ohlcv = OHLCVSeries::from_close(data);

        let signal = cb.signal(&ohlcv);
        assert!(signal.is_ok());
    }

    #[test]
    fn test_insufficient_data() {
        let data = vec![100.0; 10];
        let cb = ContangoBackwardation::new(5, 20, 0.5).unwrap();
        let ohlcv = OHLCVSeries::from_close(data);

        let result = cb.compute(&ohlcv);
        assert!(result.is_err());
    }

    #[test]
    fn test_strength_bounds() {
        let data = make_uptrend_data(50);
        let cb = ContangoBackwardation::default();
        let outputs = cb.calculate(&data);

        for out in outputs.iter().filter_map(|o| *o) {
            assert!(out.strength >= 0.0 && out.strength <= 100.0);
        }
    }

    #[test]
    fn test_normalized_bounds() {
        let data = make_uptrend_data(50);
        let cb = ContangoBackwardation::default();
        let outputs = cb.calculate(&data);

        for out in outputs.iter().filter_map(|o| *o) {
            assert!(out.normalized >= -1.0 && out.normalized <= 1.0);
        }
    }

    #[test]
    fn test_default_impl() {
        let cb = ContangoBackwardation::default();
        assert_eq!(cb.short_period, 5);
        assert_eq!(cb.long_period, 20);
        assert!((cb.flat_threshold - 0.5).abs() < 0.001);
    }
}
