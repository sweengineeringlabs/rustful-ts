//! Squeeze Momentum Indicator implementation.
//!
//! LazyBear's version of the squeeze indicator with enhanced momentum visualization.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_core::{BollingerBands, KeltnerChannels, SMA};

/// Squeeze Momentum output.
#[derive(Debug, Clone)]
pub struct SqueezeMomentumOutput {
    /// Momentum histogram values.
    pub momentum: Vec<f64>,
    /// Squeeze state: true = squeeze on (low volatility).
    pub squeeze_on: Vec<bool>,
    /// Momentum color: 1 = lime, 2 = green, -1 = red, -2 = maroon.
    pub momentum_color: Vec<i8>,
}

/// Squeeze Momentum configuration.
#[derive(Debug, Clone)]
pub struct SqueezeMomentumConfig {
    /// Bollinger Bands period (default: 20).
    pub bb_period: usize,
    /// Bollinger Bands multiplier (default: 2.0).
    pub bb_mult: f64,
    /// Keltner Channels period (default: 20).
    pub kc_period: usize,
    /// Keltner Channels multiplier (default: 1.5).
    pub kc_mult: f64,
    /// Momentum period (default: 20).
    pub momentum_period: usize,
}

impl Default for SqueezeMomentumConfig {
    fn default() -> Self {
        Self {
            bb_period: 20,
            bb_mult: 2.0,
            kc_period: 20,
            kc_mult: 1.5,
            momentum_period: 20,
        }
    }
}

/// Squeeze Momentum Indicator (LazyBear).
///
/// Enhanced version of the TTM Squeeze with better momentum visualization:
/// - Squeeze Detection: BB inside KC indicates low volatility squeeze
/// - Momentum: Linear regression of (close - average(highest, lowest, close))
///
/// Momentum colors indicate trend strength and direction:
/// - Lime: Positive and increasing momentum (strong bullish)
/// - Green: Positive but decreasing momentum (weakening bullish)
/// - Maroon: Negative and decreasing momentum (strong bearish)
/// - Red: Negative but increasing momentum (weakening bearish)
#[derive(Debug, Clone)]
pub struct SqueezeMomentum {
    bb: BollingerBands,
    kc: KeltnerChannels,
    momentum_period: usize,
}

impl SqueezeMomentum {
    pub fn new(config: SqueezeMomentumConfig) -> Self {
        Self {
            bb: BollingerBands::new(config.bb_period, config.bb_mult),
            kc: KeltnerChannels::new(config.kc_period, config.kc_period, config.kc_mult),
            momentum_period: config.momentum_period,
        }
    }

    /// Calculate Squeeze Momentum values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> SqueezeMomentumOutput {
        let n = close.len();

        // Calculate Bollinger Bands
        let (_, bb_upper, bb_lower) = self.bb.calculate(close);

        // Calculate Keltner Channels
        let (_, kc_upper, kc_lower) = self.kc.calculate(high, low, close);

        // Determine squeeze state
        let mut squeeze_on = vec![false; n];
        for i in 0..n {
            if !bb_upper[i].is_nan() && !kc_upper[i].is_nan() {
                squeeze_on[i] = bb_lower[i] > kc_lower[i] && bb_upper[i] < kc_upper[i];
            }
        }

        // Calculate momentum
        let momentum = self.calculate_momentum(high, low, close);

        // Determine momentum color
        let mut momentum_color = vec![0i8; n];
        for i in 1..n {
            if momentum[i].is_nan() || momentum[i - 1].is_nan() {
                continue;
            }

            if momentum[i] > 0.0 {
                if momentum[i] > momentum[i - 1] {
                    momentum_color[i] = 1; // Lime - positive and increasing
                } else {
                    momentum_color[i] = 2; // Green - positive but decreasing
                }
            } else {
                if momentum[i] < momentum[i - 1] {
                    momentum_color[i] = -2; // Maroon - negative and decreasing
                } else {
                    momentum_color[i] = -1; // Red - negative but increasing
                }
            }
        }

        SqueezeMomentumOutput {
            momentum,
            squeeze_on,
            momentum_color,
        }
    }

    /// Calculate momentum using linear regression.
    fn calculate_momentum(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut momentum = vec![f64::NAN; n];

        if n < self.momentum_period {
            return momentum;
        }

        // Calculate highest high and lowest low for each bar
        let mut highest = vec![f64::NAN; n];
        let mut lowest = vec![f64::NAN; n];

        for i in (self.momentum_period - 1)..n {
            let start = i + 1 - self.momentum_period;
            highest[i] = high[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            lowest[i] = low[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);
        }

        // Calculate SMA of close
        let sma = SMA::new(self.momentum_period);
        let sma_close = sma.calculate(close);

        // Calculate source: close - average(highest/lowest midline, sma)
        let mut source = vec![f64::NAN; n];
        for i in 0..n {
            if !highest[i].is_nan() && !lowest[i].is_nan() && !sma_close[i].is_nan() {
                let donchian_mid = (highest[i] + lowest[i]) / 2.0;
                let avg = (donchian_mid + sma_close[i]) / 2.0;
                source[i] = close[i] - avg;
            }
        }

        // Apply linear regression
        for i in (self.momentum_period - 1)..n {
            let start = i + 1 - self.momentum_period;
            let window = &source[start..=i];

            if window.iter().any(|x| x.is_nan()) {
                continue;
            }

            momentum[i] = self.linear_regression_value(window);
        }

        momentum
    }

    /// Calculate linear regression endpoint value.
    fn linear_regression_value(&self, data: &[f64]) -> f64 {
        let n = data.len() as f64;
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

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return data[data.len() - 1];
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        slope * (n - 1.0) + intercept
    }
}

impl Default for SqueezeMomentum {
    fn default() -> Self {
        Self::new(SqueezeMomentumConfig::default())
    }
}

impl TechnicalIndicator for SqueezeMomentum {
    fn name(&self) -> &str {
        "SqueezeMomentum"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.momentum_period.max(20);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);

        let squeeze_values: Vec<f64> = result.squeeze_on.iter()
            .map(|&on| if on { 1.0 } else { 0.0 })
            .collect();

        let color_values: Vec<f64> = result.momentum_color.iter()
            .map(|&c| c as f64)
            .collect();

        Ok(IndicatorOutput::triple(result.momentum, squeeze_values, color_values))
    }

    fn min_periods(&self) -> usize {
        self.momentum_period.max(20)
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for SqueezeMomentum {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let n = result.momentum.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let mom = result.momentum[n - 1];
        let prev_mom = result.momentum[n - 2];
        let squeeze_released = n >= 2 && result.squeeze_on[n - 2] && !result.squeeze_on[n - 1];

        if mom.is_nan() || prev_mom.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Strong signal when squeeze releases
        if squeeze_released || !result.squeeze_on[n - 1] {
            // Bullish: positive momentum increasing (lime)
            if mom > 0.0 && mom > prev_mom {
                return Ok(IndicatorSignal::Bullish);
            }
            // Bearish: negative momentum decreasing (maroon)
            else if mom < 0.0 && mom < prev_mom {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let n = result.momentum.len();

        let mut signals = vec![IndicatorSignal::Neutral; n.min(1)];

        for i in 1..n {
            let mom = result.momentum[i];
            let prev_mom = result.momentum[i - 1];
            let squeeze_released = i >= 1 && result.squeeze_on[i - 1] && !result.squeeze_on[i];

            if mom.is_nan() || prev_mom.is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            if squeeze_released || !result.squeeze_on[i] {
                if mom > 0.0 && mom > prev_mom {
                    signals.push(IndicatorSignal::Bullish);
                } else if mom < 0.0 && mom < prev_mom {
                    signals.push(IndicatorSignal::Bearish);
                } else {
                    signals.push(IndicatorSignal::Neutral);
                }
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

    fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.15).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.15).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.15).sin() * 5.0).collect();
        (high, low, close)
    }

    #[test]
    fn test_squeeze_momentum_basic() {
        let sm = SqueezeMomentum::default();
        let (high, low, close) = generate_test_data(50);

        let result = sm.calculate(&high, &low, &close);

        assert_eq!(result.momentum.len(), 50);
        assert_eq!(result.squeeze_on.len(), 50);
        assert_eq!(result.momentum_color.len(), 50);
    }

    #[test]
    fn test_squeeze_momentum_colors() {
        let sm = SqueezeMomentum::default();
        let (high, low, close) = generate_test_data(50);

        let result = sm.calculate(&high, &low, &close);

        // Momentum colors should be -2, -1, 0, 1, or 2
        for &color in &result.momentum_color {
            assert!(color >= -2 && color <= 2);
        }
    }

    #[test]
    fn test_squeeze_momentum_compute() {
        let sm = SqueezeMomentum::default();
        let (high, low, close) = generate_test_data(50);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let output = sm.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_squeeze_momentum_config() {
        let config = SqueezeMomentumConfig {
            bb_period: 15,
            bb_mult: 1.5,
            kc_period: 15,
            kc_mult: 1.0,
            momentum_period: 15,
        };

        let sm = SqueezeMomentum::new(config);
        assert_eq!(sm.min_periods(), 20);
    }
}
