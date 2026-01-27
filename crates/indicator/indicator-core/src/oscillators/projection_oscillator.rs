//! Projection Oscillator (IND-079) implementation.
//!
//! The Projection Oscillator uses linear regression to create a bounded oscillator
//! that measures where the current close price falls within the projected price channel.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

/// Projection Oscillator.
///
/// The Projection Oscillator uses linear regression to create an oscillator that
/// measures where the current close price lies relative to the projected price channel.
///
/// ## Formula
/// 1. Calculate linear regression line of close prices over the period
/// 2. Calculate upper projection = highest deviation of high above regression line
/// 3. Calculate lower projection = lowest deviation of low below regression line
/// 4. Projection Oscillator = 100 * (Close - Lower) / (Upper - Lower)
///
/// ## Interpretation
/// - Values above 80 indicate overbought conditions
/// - Values below 20 indicate oversold conditions
/// - Crossovers with the 50 level indicate trend changes
///
/// ## Example
/// ```
/// use indicator_core::oscillators::projection_oscillator::ProjectionOscillator;
///
/// let po = ProjectionOscillator::new(14);
/// let high = vec![105.0, 106.0, 107.0, 106.5, 108.0, 107.5, 109.0, 108.5, 110.0, 109.5,
///                 111.0, 110.5, 112.0, 111.5, 113.0];
/// let low = vec![100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5,
///                106.0, 105.5, 107.0, 106.5, 108.0];
/// let close = vec![102.0, 103.0, 104.0, 103.5, 105.0, 104.5, 106.0, 105.5, 107.0, 106.5,
///                  108.0, 107.5, 109.0, 108.5, 110.0];
/// let po_values = po.calculate(&high, &low, &close);
/// ```
#[derive(Debug, Clone)]
pub struct ProjectionOscillator {
    /// Period for linear regression calculation.
    period: usize,
    /// Overbought threshold (typically 80).
    overbought: f64,
    /// Oversold threshold (typically 20).
    oversold: f64,
}

impl ProjectionOscillator {
    /// Create a new Projection Oscillator with the specified period.
    ///
    /// # Arguments
    /// * `period` - Period for linear regression (typically 14)
    pub fn new(period: usize) -> Self {
        Self {
            period,
            overbought: 80.0,
            oversold: 20.0,
        }
    }

    /// Create with custom overbought/oversold thresholds.
    ///
    /// # Arguments
    /// * `period` - Period for linear regression
    /// * `overbought` - Overbought threshold (typically 80)
    /// * `oversold` - Oversold threshold (typically 20)
    pub fn with_thresholds(period: usize, overbought: f64, oversold: f64) -> Self {
        Self {
            period,
            overbought,
            oversold,
        }
    }

    /// Create with default parameters (14-period).
    pub fn default_params() -> Self {
        Self::new(14)
    }

    /// Get the period.
    pub fn period(&self) -> usize {
        self.period
    }

    /// Get the overbought threshold.
    pub fn overbought(&self) -> f64 {
        self.overbought
    }

    /// Get the oversold threshold.
    pub fn oversold(&self) -> f64 {
        self.oversold
    }

    /// Calculate linear regression parameters for a window.
    /// Returns (slope, intercept).
    fn calculate_regression(&self, data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        if n < 2.0 {
            return (f64::NAN, f64::NAN);
        }

        // Calculate sums for least squares regression
        // Using 0-indexed: y = intercept + slope * x
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return (0.0, sum_y / n);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }

    /// Calculate the upper projection (highest high above regression line projected to end).
    fn calculate_upper_projection(&self, high: &[f64], close: &[f64]) -> f64 {
        let (slope, intercept) = self.calculate_regression(close);
        if slope.is_nan() || intercept.is_nan() {
            return f64::NAN;
        }

        let n = close.len();
        let mut max_deviation = f64::NEG_INFINITY;

        // Find maximum deviation of high above regression line
        for (i, &h) in high.iter().enumerate() {
            let regression_val = intercept + slope * (i as f64);
            let deviation = h - regression_val;
            if deviation > max_deviation {
                max_deviation = deviation;
            }
        }

        // Project to the end point
        let end_regression = intercept + slope * ((n - 1) as f64);
        end_regression + max_deviation
    }

    /// Calculate the lower projection (lowest low below regression line projected to end).
    fn calculate_lower_projection(&self, low: &[f64], close: &[f64]) -> f64 {
        let (slope, intercept) = self.calculate_regression(close);
        if slope.is_nan() || intercept.is_nan() {
            return f64::NAN;
        }

        let n = close.len();
        let mut min_deviation = f64::INFINITY;

        // Find minimum deviation of low below regression line
        for (i, &l) in low.iter().enumerate() {
            let regression_val = intercept + slope * (i as f64);
            let deviation = l - regression_val;
            if deviation < min_deviation {
                min_deviation = deviation;
            }
        }

        // Project to the end point
        let end_regression = intercept + slope * ((n - 1) as f64);
        end_regression + min_deviation
    }

    /// Calculate the Projection Oscillator values.
    ///
    /// # Arguments
    /// * `high` - High prices
    /// * `low` - Low prices
    /// * `close` - Close prices
    ///
    /// # Returns
    /// Vector of oscillator values (0-100 scale)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period || self.period < 2 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let high_window = &high[start..=i];
            let low_window = &low[start..=i];
            let close_window = &close[start..=i];

            let upper = self.calculate_upper_projection(high_window, close_window);
            let lower = self.calculate_lower_projection(low_window, close_window);

            if upper.is_nan() || lower.is_nan() {
                result.push(f64::NAN);
                continue;
            }

            let range = upper - lower;
            if range.abs() < 1e-10 {
                // No range, return neutral value
                result.push(50.0);
            } else {
                // Projection Oscillator = 100 * (Close - Lower) / (Upper - Lower)
                let po = 100.0 * (close[i] - lower) / range;
                result.push(po);
            }
        }

        result
    }

    /// Calculate the Projection Oscillator with upper and lower projections.
    ///
    /// # Returns
    /// Tuple of (oscillator, upper_projection, lower_projection)
    pub fn calculate_with_bands(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        if n < self.period || self.period < 2 {
            return (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let mut oscillator = vec![f64::NAN; self.period - 1];
        let mut upper_proj = vec![f64::NAN; self.period - 1];
        let mut lower_proj = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let high_window = &high[start..=i];
            let low_window = &low[start..=i];
            let close_window = &close[start..=i];

            let upper = self.calculate_upper_projection(high_window, close_window);
            let lower = self.calculate_lower_projection(low_window, close_window);

            upper_proj.push(upper);
            lower_proj.push(lower);

            if upper.is_nan() || lower.is_nan() {
                oscillator.push(f64::NAN);
                continue;
            }

            let range = upper - lower;
            if range.abs() < 1e-10 {
                oscillator.push(50.0);
            } else {
                let po = 100.0 * (close[i] - lower) / range;
                oscillator.push(po);
            }
        }

        (oscillator, upper_proj, lower_proj)
    }
}

impl Default for ProjectionOscillator {
    fn default() -> Self {
        Self::default_params()
    }
}

impl TechnicalIndicator for ProjectionOscillator {
    fn name(&self) -> &str {
        "ProjectionOscillator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for ProjectionOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close);

        if values.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = values.len();
        let current = values[n - 1];
        let prev = values[n - 2];

        if current.is_nan() || prev.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: crossing above oversold from below
        if prev <= self.oversold && current > self.oversold {
            Ok(IndicatorSignal::Bullish)
        }
        // Bearish: crossing below overbought from above
        else if prev >= self.overbought && current < self.overbought {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close);
        let n = values.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            let current = values[i];
            let prev = values[i - 1];

            if current.is_nan() || prev.is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            // Bullish: crossing above oversold from below
            if prev <= self.oversold && current > self.oversold {
                signals.push(IndicatorSignal::Bullish);
            }
            // Bearish: crossing below overbought from above
            else if prev >= self.overbought && current < self.overbought {
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
    fn test_projection_oscillator_basic() {
        let po = ProjectionOscillator::new(14);

        // Create sample OHLC data with upward trend
        let high: Vec<f64> = (0..30).map(|i| 105.0 + i as f64 * 0.5).collect();
        let low: Vec<f64> = (0..30).map(|i| 95.0 + i as f64 * 0.5).collect();
        let close: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.5).collect();

        let values = po.calculate(&high, &low, &close);

        assert_eq!(values.len(), 30);

        // First period-1 values should be NaN
        for i in 0..13 {
            assert!(values[i].is_nan(), "Value at {} should be NaN", i);
        }

        // Values after warmup should be valid and in 0-100 range
        for i in 13..30 {
            assert!(!values[i].is_nan(), "Value at {} should not be NaN", i);
            assert!(
                values[i] >= 0.0 && values[i] <= 100.0,
                "Value {} at index {} should be between 0 and 100",
                values[i],
                i
            );
        }
    }

    #[test]
    fn test_projection_oscillator_overbought_oversold() {
        let po = ProjectionOscillator::new(10);

        // Create data where close is near highs (overbought)
        let high: Vec<f64> = (0..20)
            .map(|i| 110.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let low: Vec<f64> = (0..20)
            .map(|i| 90.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let close: Vec<f64> = (0..20)
            .map(|i| 108.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();

        let values = po.calculate(&high, &low, &close);

        // When close is near highs, oscillator should be high (>50)
        for i in 9..20 {
            assert!(!values[i].is_nan());
            assert!(values[i] > 50.0, "Expected overbought, got {}", values[i]);
        }

        // Create data where close is near lows (oversold)
        let close_low: Vec<f64> = (0..20)
            .map(|i| 92.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let values_low = po.calculate(&high, &low, &close_low);

        for i in 9..20 {
            assert!(!values_low[i].is_nan());
            assert!(
                values_low[i] < 50.0,
                "Expected oversold, got {}",
                values_low[i]
            );
        }
    }

    #[test]
    fn test_projection_oscillator_with_bands() {
        let po = ProjectionOscillator::new(10);

        let high: Vec<f64> = (0..20)
            .map(|i| 105.0 + (i as f64 * 0.2).sin() * 5.0)
            .collect();
        let low: Vec<f64> = (0..20)
            .map(|i| 95.0 + (i as f64 * 0.2).sin() * 5.0)
            .collect();
        let close: Vec<f64> = (0..20)
            .map(|i| 100.0 + (i as f64 * 0.2).sin() * 5.0)
            .collect();

        let (oscillator, upper, lower) = po.calculate_with_bands(&high, &low, &close);

        assert_eq!(oscillator.len(), 20);
        assert_eq!(upper.len(), 20);
        assert_eq!(lower.len(), 20);

        // After warmup, upper should be >= lower
        for i in 9..20 {
            if !upper[i].is_nan() && !lower[i].is_nan() {
                assert!(
                    upper[i] >= lower[i],
                    "Upper {} should be >= Lower {} at index {}",
                    upper[i],
                    lower[i],
                    i
                );
            }
        }
    }

    #[test]
    fn test_projection_oscillator_insufficient_data() {
        let po = ProjectionOscillator::new(14);

        let high = vec![105.0; 10];
        let low = vec![95.0; 10];
        let close = vec![100.0; 10];

        let values = po.calculate(&high, &low, &close);

        // All values should be NaN since data length < period
        for val in values.iter() {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_projection_oscillator_signal() {
        let po = ProjectionOscillator::with_thresholds(10, 80.0, 20.0);

        // Create OHLCV data that will cross oversold threshold
        let mut high: Vec<f64> = (0..25).map(|i| 105.0 - i as f64 * 0.3).collect();
        let mut low: Vec<f64> = (0..25).map(|i| 95.0 - i as f64 * 0.3).collect();
        let mut close: Vec<f64> = (0..25).map(|i| 100.0 - i as f64 * 0.3).collect();

        // Add reversal at the end
        for i in 20..25 {
            high[i] = 90.0 + (i - 20) as f64 * 2.0;
            low[i] = 80.0 + (i - 20) as f64 * 2.0;
            close[i] = 85.0 + (i - 20) as f64 * 2.0;
        }

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 25],
        };

        let signals = po.signals(&data).unwrap();
        assert_eq!(signals.len(), 25);
    }

    #[test]
    fn test_projection_oscillator_default() {
        let po = ProjectionOscillator::default();
        assert_eq!(po.period(), 14);
        assert_eq!(po.overbought(), 80.0);
        assert_eq!(po.oversold(), 20.0);
    }

    #[test]
    fn test_projection_oscillator_technical_indicator() {
        let po = ProjectionOscillator::new(14);

        assert_eq!(po.name(), "ProjectionOscillator");
        assert_eq!(po.min_periods(), 14);
        assert_eq!(po.output_features(), 1);
    }

    #[test]
    fn test_projection_oscillator_flat_data() {
        let po = ProjectionOscillator::new(10);

        // Completely flat data
        let high = vec![100.0; 20];
        let low = vec![100.0; 20];
        let close = vec![100.0; 20];

        let values = po.calculate(&high, &low, &close);

        // When range is zero, should return neutral value (50)
        for i in 9..20 {
            assert!(!values[i].is_nan());
            assert!(
                (values[i] - 50.0).abs() < 1e-10,
                "Expected 50.0, got {}",
                values[i]
            );
        }
    }

    #[test]
    fn test_projection_oscillator_trending_market() {
        let po = ProjectionOscillator::new(14);

        // Strong uptrend where close follows the highs
        let high: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 2.0 + 5.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 2.0 - 5.0).collect();
        let close: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 2.0 + 3.0).collect();

        let values = po.calculate(&high, &low, &close);

        // In a strong trend where close is near highs, oscillator should be elevated
        for i in 13..30 {
            assert!(!values[i].is_nan());
            // Close is near high, so value should be > 50
            assert!(
                values[i] > 50.0,
                "Expected > 50 in uptrend, got {}",
                values[i]
            );
        }
    }

    #[test]
    fn test_projection_oscillator_compute() {
        let po = ProjectionOscillator::new(10);

        let data = OHLCVSeries {
            open: (0..20)
                .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0)
                .collect(),
            high: (0..20)
                .map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0)
                .collect(),
            low: (0..20)
                .map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0)
                .collect(),
            close: (0..20)
                .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0)
                .collect(),
            volume: vec![1000.0; 20],
        };

        let result = po.compute(&data);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.primary.len(), 20);
    }

    #[test]
    fn test_projection_oscillator_compute_insufficient_data() {
        let po = ProjectionOscillator::new(14);

        let data = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![105.0; 10],
            low: vec![95.0; 10],
            close: vec![100.0; 10],
            volume: vec![1000.0; 10],
        };

        let result = po.compute(&data);
        assert!(result.is_err());

        if let Err(IndicatorError::InsufficientData { required, got }) = result {
            assert_eq!(required, 14);
            assert_eq!(got, 10);
        } else {
            panic!("Expected InsufficientData error");
        }
    }

    #[test]
    fn test_regression_calculation() {
        let po = ProjectionOscillator::new(5);

        // Perfect linear data: y = 2x + 100
        let close: Vec<f64> = (0..5).map(|i| 100.0 + i as f64 * 2.0).collect();
        let (slope, intercept) = po.calculate_regression(&close);

        assert!(
            (slope - 2.0).abs() < 1e-10,
            "Slope should be 2.0, got {}",
            slope
        );
        assert!(
            (intercept - 100.0).abs() < 1e-10,
            "Intercept should be 100.0, got {}",
            intercept
        );
    }
}
