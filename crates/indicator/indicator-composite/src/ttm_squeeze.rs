//! TTM Squeeze implementation.
//!
//! The TTM Squeeze indicator combines Bollinger Bands and Keltner Channels
//! to identify low volatility conditions and momentum direction.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_core::{BollingerBands, KeltnerChannels};

/// TTM Squeeze output.
#[derive(Debug, Clone)]
pub struct TTMSqueezeOutput {
    /// Momentum values (positive = bullish, negative = bearish).
    pub momentum: Vec<f64>,
    /// Squeeze state: true when Bollinger inside Keltner (low volatility).
    pub squeeze_on: Vec<bool>,
    /// Momentum direction change.
    pub momentum_increasing: Vec<bool>,
}

/// TTM Squeeze configuration.
#[derive(Debug, Clone)]
pub struct TTMSqueezeConfig {
    /// Bollinger Bands period (default: 20).
    pub bb_period: usize,
    /// Bollinger Bands standard deviation multiplier (default: 2.0).
    pub bb_std_dev: f64,
    /// Keltner Channels EMA period (default: 20).
    pub kc_ema_period: usize,
    /// Keltner Channels ATR period (default: 20).
    pub kc_atr_period: usize,
    /// Keltner Channels multiplier (default: 1.5).
    pub kc_multiplier: f64,
    /// Momentum length for linear regression (default: 20).
    pub momentum_length: usize,
}

impl Default for TTMSqueezeConfig {
    fn default() -> Self {
        Self {
            bb_period: 20,
            bb_std_dev: 2.0,
            kc_ema_period: 20,
            kc_atr_period: 20,
            kc_multiplier: 1.5,
            momentum_length: 20,
        }
    }
}

/// TTM Squeeze indicator.
///
/// Identifies low volatility squeezes and momentum direction.
/// - Squeeze ON: Bollinger Bands inside Keltner Channels (low volatility)
/// - Squeeze OFF: Bollinger Bands outside Keltner Channels (volatility expanding)
/// - Momentum: Linear regression of price deviation from midline
#[derive(Debug, Clone)]
pub struct TTMSqueeze {
    bb: BollingerBands,
    kc: KeltnerChannels,
    momentum_length: usize,
}

impl TTMSqueeze {
    pub fn new(config: TTMSqueezeConfig) -> Self {
        Self {
            bb: BollingerBands::new(config.bb_period, config.bb_std_dev),
            kc: KeltnerChannels::new(config.kc_ema_period, config.kc_atr_period, config.kc_multiplier),
            momentum_length: config.momentum_length,
        }
    }

    /// Calculate TTM Squeeze values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> TTMSqueezeOutput {
        let n = close.len();

        // Calculate Bollinger Bands
        let (bb_mid, bb_upper, bb_lower) = self.bb.calculate(close);

        // Calculate Keltner Channels
        let (_kc_mid, kc_upper, kc_lower) = self.kc.calculate(high, low, close);

        // Determine squeeze state
        let mut squeeze_on = Vec::with_capacity(n);
        for i in 0..n {
            if bb_upper[i].is_nan() || kc_upper[i].is_nan() {
                squeeze_on.push(false);
            } else {
                // Squeeze is ON when BB is inside KC
                squeeze_on.push(bb_lower[i] > kc_lower[i] && bb_upper[i] < kc_upper[i]);
            }
        }

        // Calculate momentum using linear regression of price deviation
        let momentum = self.calculate_momentum(high, low, close, &bb_mid);

        // Determine if momentum is increasing
        let mut momentum_increasing = Vec::with_capacity(n);
        momentum_increasing.push(false);
        for i in 1..n {
            if momentum[i].is_nan() || momentum[i - 1].is_nan() {
                momentum_increasing.push(false);
            } else {
                momentum_increasing.push(momentum[i].abs() > momentum[i - 1].abs()
                    || (momentum[i] > 0.0 && momentum[i] > momentum[i - 1])
                    || (momentum[i] < 0.0 && momentum[i] < momentum[i - 1]));
            }
        }

        TTMSqueezeOutput {
            momentum,
            squeeze_on,
            momentum_increasing,
        }
    }

    /// Calculate momentum using linear regression.
    fn calculate_momentum(&self, high: &[f64], low: &[f64], close: &[f64], midline: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut momentum = vec![f64::NAN; n];

        if n < self.momentum_length {
            return momentum;
        }

        // Calculate (high + low + close) / 3 - midline as the deviation
        let mut deviation = Vec::with_capacity(n);
        for i in 0..n {
            if midline[i].is_nan() {
                deviation.push(f64::NAN);
            } else {
                let hlc3 = (high[i] + low[i] + close[i]) / 3.0;
                deviation.push(hlc3 - midline[i]);
            }
        }

        // Apply linear regression to get momentum
        for i in (self.momentum_length - 1)..n {
            let start = i + 1 - self.momentum_length;
            let window = &deviation[start..=i];

            if window.iter().any(|x| x.is_nan()) {
                continue;
            }

            // Linear regression value (endpoint of regression line)
            momentum[i] = self.linear_reg_value(window);
        }

        momentum
    }

    /// Calculate the linear regression endpoint value.
    fn linear_reg_value(&self, data: &[f64]) -> f64 {
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

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Return the value at the last point
        slope * (n - 1.0) + intercept
    }
}

impl Default for TTMSqueeze {
    fn default() -> Self {
        Self::new(TTMSqueezeConfig::default())
    }
}

impl TechnicalIndicator for TTMSqueeze {
    fn name(&self) -> &str {
        "TTMSqueeze"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.momentum_length.max(20);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);

        // Convert squeeze_on to f64 for output
        let squeeze_values: Vec<f64> = result.squeeze_on.iter()
            .map(|&on| if on { 1.0 } else { 0.0 })
            .collect();

        Ok(IndicatorOutput::dual(result.momentum, squeeze_values))
    }

    fn min_periods(&self) -> usize {
        self.momentum_length.max(20)
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for TTMSqueeze {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let n = result.momentum.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let mom = result.momentum[n - 1];
        let prev_mom = result.momentum[n - 2];
        let squeeze_off = !result.squeeze_on[n - 1];

        if mom.is_nan() || prev_mom.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal when squeeze releases and momentum is increasing
        if squeeze_off {
            if mom > 0.0 && mom > prev_mom {
                return Ok(IndicatorSignal::Bullish);
            } else if mom < 0.0 && mom < prev_mom {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let n = result.momentum.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            let mom = result.momentum[i];
            let prev_mom = result.momentum[i - 1];
            let squeeze_off = !result.squeeze_on[i];

            if mom.is_nan() || prev_mom.is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            if squeeze_off {
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
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        (high, low, close)
    }

    #[test]
    fn test_ttm_squeeze_basic() {
        let squeeze = TTMSqueeze::default();
        let (high, low, close) = generate_test_data(50);

        let result = squeeze.calculate(&high, &low, &close);

        assert_eq!(result.momentum.len(), 50);
        assert_eq!(result.squeeze_on.len(), 50);
        assert_eq!(result.momentum_increasing.len(), 50);

        // Check that we have valid momentum values after warmup
        let valid_count = result.momentum.iter().filter(|x| !x.is_nan()).count();
        assert!(valid_count > 0);
    }

    #[test]
    fn test_ttm_squeeze_output() {
        let squeeze = TTMSqueeze::default();
        let (high, low, close) = generate_test_data(50);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let output = squeeze.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_ttm_squeeze_config() {
        let config = TTMSqueezeConfig {
            bb_period: 15,
            bb_std_dev: 1.5,
            kc_ema_period: 15,
            kc_atr_period: 15,
            kc_multiplier: 1.0,
            momentum_length: 15,
        };

        let squeeze = TTMSqueeze::new(config);
        assert_eq!(squeeze.min_periods(), 20.max(15));
    }
}
